#include "engine/models/omnivoice/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::omnivoice {
namespace {

constexpr size_t kDefaultAudioTokenizerGraphArenaBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorPrefillGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorDecodeGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultAudioTokenizerWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorWeightContextBytes = 256ull * 1024ull * 1024ull;

// k2-fsa/OmniVoice#77 (model maintainer): 16 inference steps "also performs
// well" and is meaningfully faster than the offline path's default (32) --
// used only when the streaming caller doesn't explicitly set num_inference_
// steps, since streaming's whole point is lower per-chunk latency.
constexpr const char * kNumInferenceStepsKey = "num_inference_steps";
constexpr const char * kStreamingDefaultNumInferenceSteps = "16";

// Chunk sizing for streaming keeps the offline path's normal duration
// target (audio_chunk_duration_seconds) -- shrinking that target uniformly
// turned out not to help (see the first-sentence peel below for why) and
// would only add per-chunk fixed overhead (reference/prompt setup already
// happens once per request, not per chunk, but each chunk is still its own
// generate()+decode() call). Instead, only the very FIRST chunk is
// deliberately kept small (one sentence), and every chunk after it uses the
// normal, larger target -- see the peel step in run_streaming_request. By
// the time chunk 0 finishes *playing* (~its own audio duration), chunk 1
// should already be generated: measured RTF at 16 inference steps is well
// under 1 (~0.2-0.3), so generation runs several times faster than realtime
// and comfortably outpaces playback consumption even for a normal-sized
// chunk 1 while a short chunk 0 is still playing.

std::shared_ptr<const OmniVoiceAssets> require_assets(std::shared_ptr<const OmniVoiceAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("OmniVoice session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType fallback) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return fallback;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"omnivoice.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "omnivoice.mem_saver");
    }
    return false;
}

OmniVoiceGeneratorPerfMode perf_mode_from_options(const runtime::SessionOptions & options, bool mem_saver) {
    if (const auto value = runtime::find_option(options.options, {"omnivoice.perf_mode"})) {
        if (*value == "off" || *value == "standard") {
            return OmniVoiceGeneratorPerfMode::Standard;
        }
        if (*value == "flash_attention") {
            if (mem_saver) {
                throw std::runtime_error("omnivoice.perf_mode=flash_attention is not supported with omnivoice.mem_saver=true");
            }
            return OmniVoiceGeneratorPerfMode::FlashAttention;
        }
        throw std::runtime_error("Invalid omnivoice.perf_mode: " + *value);
    }
    return OmniVoiceGeneratorPerfMode::Standard;
}

using Clock = std::chrono::steady_clock;

OmniVoiceGenerationOptions generation_options_from_options(const std::unordered_map<std::string, std::string> & options_map) {
    OmniVoiceGenerationOptions options;
    if (const auto seed = runtime::parse_u32_option(options_map, {"seed"})) {
        options.seed = *seed;
    }
    options.num_inference_steps = runtime::parse_int_option(options_map, {"num_inference_steps"})
        .value_or(options.num_inference_steps);
    options.guidance_scale = runtime::parse_float_option(options_map, {"guidance_scale"})
        .value_or(options.guidance_scale);
    options.speed = runtime::parse_float_option(options_map, {"speed"}).value_or(options.speed);
    options.duration_seconds = runtime::parse_float_option(options_map, {"duration"});
    options.t_shift = runtime::parse_float_option(options_map, {"t_shift"}).value_or(options.t_shift);
    if (const auto value = runtime::find_option(options_map, {"denoise"})) {
        options.denoise = runtime::parse_bool_option(*value, "denoise");
    }
    if (const auto value = runtime::find_option(options_map, {"preprocess_prompt"})) {
        options.preprocess_prompt = runtime::parse_bool_option(*value, "preprocess_prompt");
    }
    if (const auto value = runtime::find_option(options_map, {"postprocess_output"})) {
        options.postprocess_output = runtime::parse_bool_option(*value, "postprocess_output");
    }
    options.layer_penalty_factor = runtime::parse_float_option(
        options_map,
        {"layer_penalty_factor"})
        .value_or(options.layer_penalty_factor);
    options.position_temperature = runtime::parse_float_option(
        options_map,
        {"position_temperature"})
        .value_or(options.position_temperature);
    options.class_temperature = runtime::parse_float_option(
        options_map,
        {"class_temperature"})
        .value_or(options.class_temperature);
    options.audio_chunk_duration_seconds = runtime::parse_float_option(
        options_map,
        {"audio_chunk_duration"})
        .value_or(options.audio_chunk_duration_seconds);
    options.audio_chunk_threshold_seconds = runtime::parse_float_option(
        options_map,
        {"audio_chunk_threshold"})
        .value_or(options.audio_chunk_threshold_seconds);
    options.sentence_chunk_pause_ms = runtime::parse_int_option(
        options_map,
        {"sentence_chunk_pause_ms"})
        .value_or(options.sentence_chunk_pause_ms);
    options.first_sentence_num_inference_steps = runtime::parse_int_option(
        options_map,
        {"first_sentence_num_inference_steps"})
        .value_or(options.first_sentence_num_inference_steps);
    options.text_chunk_size = engine::text::parse_text_chunk_size_override(options_map);
    options.text_chunk_mode = engine::text::parse_text_chunk_mode_override(options_map)
        .value_or(engine::text::TextChunkMode::TagAware);
    return options;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(sample));
        std::memcpy(&bits, &sample, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>((bits >> shift) & 0xFFu);
            hash *= kPrime;
        }
    }
    return hash;
}

std::optional<std::string> request_option(
    const std::unordered_map<std::string, std::string> & options,
    const char * primary,
    const char * secondary = nullptr) {
    if (const auto it = options.find(primary); it != options.end()) {
        return it->second;
    }
    if (secondary != nullptr) {
        if (const auto it = options.find(secondary); it != options.end()) {
            return it->second;
        }
    }
    return std::nullopt;
}

float fade_weight(size_t index, size_t count, float start, float end) {
    if (count <= 1) {
        return start;
    }
    const float t = static_cast<float>(index) / static_cast<float>(count - 1);
    return start + (end - start) * t;
}

void append_cross_faded_chunk(
    runtime::AudioBuffer & merged,
    const runtime::AudioBuffer & chunk,
    float silence_duration_seconds = 0.3F) {
    if (chunk.sample_rate <= 0 || chunk.channels != 1) {
        throw std::runtime_error("OmniVoice append_cross_faded_chunk requires mono chunk audio");
    }
    if (merged.sample_rate == 0) {
        merged = chunk;
        return;
    }
    if (merged.sample_rate != chunk.sample_rate || merged.channels != 1) {
        throw std::runtime_error("OmniVoice append_cross_faded_chunk requires matching mono chunk audio");
    }

    const int sample_rate = chunk.sample_rate;
    const size_t total_n = static_cast<size_t>(std::max(0.0F, silence_duration_seconds) * static_cast<float>(sample_rate));
    const size_t fade_n = total_n / 3;
    const size_t silence_n = fade_n;
    const size_t fout_n = std::min(fade_n, merged.samples.size());
    if (fout_n > 0) {
        for (size_t i = 0; i < fout_n; ++i) {
            const float w_out = fade_weight(i, fout_n, 1.0F, 0.0F);
            merged.samples[merged.samples.size() - fout_n + i] *= w_out;
        }
    }
    merged.samples.insert(merged.samples.end(), silence_n, 0.0F);
    const size_t chunk_start = merged.samples.size();
    merged.samples.insert(merged.samples.end(), chunk.samples.begin(), chunk.samples.end());
    const size_t fin_n = std::min(fade_n, chunk.samples.size());
    if (fin_n > 0) {
        for (size_t i = 0; i < fin_n; ++i) {
            const float w_in = fade_weight(i, fin_n, 0.0F, 1.0F);
            merged.samples[chunk_start + i] *= w_in;
        }
    }
}

// Streaming counterpart to append_cross_faded_chunk: fades only the
// INCOMING chunk's own head, in place, no silence insertion, no dependency
// on (or mutation of) whatever was emitted before it. A full two-sided
// crossfade would need to hold the previous chunk back until this one
// exists so their edges can blend -- once a chunk has already gone out over
// the wire to a streaming client, its tail can't be un-sent. This trades a
// slightly less seamless boundary for zero added latency, which is the
// entire point of the streaming path.
void apply_fade_in(runtime::AudioBuffer & chunk, float fade_seconds) {
    if (chunk.sample_rate <= 0 || chunk.samples.empty()) {
        return;
    }
    const size_t fade_n = std::min(
        chunk.samples.size(),
        static_cast<size_t>(std::max(0.0F, fade_seconds) * static_cast<float>(chunk.sample_rate)));
    for (size_t i = 0; i < fade_n; ++i) {
        chunk.samples[i] *= fade_weight(i, fade_n, 0.0F, 1.0F);
    }
}

// EXPERIMENTAL (see OmniVoiceGenerationOptions::sentence_chunk_pause_ms):
// prepends silence to the INCOMING chunk only, after its own fade-in, so it
// never touches audio already emitted over the wire -- zero added time-to-
// first-chunk, same reasoning as apply_fade_in's own comment above about
// why streaming can't do append_cross_faded_chunk's two-sided version.
void prepend_silence(runtime::AudioBuffer & chunk, int silence_ms) {
    if (chunk.sample_rate <= 0 || silence_ms <= 0) {
        return;
    }
    const size_t silence_n = static_cast<size_t>(silence_ms) * static_cast<size_t>(chunk.sample_rate) / 1000;
    chunk.samples.insert(chunk.samples.begin(), silence_n, 0.0F);
}

}  // namespace

OmniVoiceSessionBase::OmniVoiceSessionBase(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const OmniVoiceAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      audio_tokenizer_graph_arena_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.audio_tokenizer_graph_arena_mb"}, kDefaultAudioTokenizerGraphArenaBytes)),
      generator_prefill_graph_arena_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.generator_prefill_graph_arena_mb"}, kDefaultGeneratorPrefillGraphArenaBytes)),
      generator_decode_graph_arena_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.generator_decode_graph_arena_mb"}, kDefaultGeneratorDecodeGraphArenaBytes)),
      audio_tokenizer_weight_context_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.audio_tokenizer_weight_context_mb"}, kDefaultAudioTokenizerWeightContextBytes)),
      generator_weight_context_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.generator_weight_context_mb"}, kDefaultGeneratorWeightContextBytes)),
      audio_tokenizer_weight_storage_type_(
          option_weight_type(options, "omnivoice.audio_tokenizer_weight_type", engine::assets::TensorStorageType::Native)),
      generator_weight_storage_type_(
          option_weight_type(
              options,
              "omnivoice.generator_weight_type",
              option_weight_type(options, "omnivoice.weight_type", engine::assets::TensorStorageType::Native))),
      mem_saver_(mem_saver_from_options(options)),
      generator_perf_mode_(perf_mode_from_options(options, mem_saver_)),
      tokenizer_(assets_),
      audio_tokenizer_(
          assets_,
          execution_context(),
          audio_tokenizer_graph_arena_bytes_,
          audio_tokenizer_weight_context_bytes_,
          audio_tokenizer_weight_storage_type_),
      prompt_builder_(assets_, tokenizer_),
      generator_(
          assets_,
          execution_context(),
          generator_prefill_graph_arena_bytes_,
          generator_decode_graph_arena_bytes_,
          generator_weight_context_bytes_,
          generator_weight_storage_type_,
          mem_saver_,
          generator_perf_mode_) {
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("OmniVoice only supports VoiceTaskKind::Tts");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("OmniVoice only supports offline and streaming sessions");
    }
    assets_->model_weights->release_storage();
    assets_->audio_tokenizer_weights->release_storage();
}

std::string OmniVoiceSessionBase::family_impl() const {
    return "omnivoice";
}

runtime::VoiceTaskKind OmniVoiceSessionBase::task_kind_impl() const {
    return task_.task;
}

runtime::RunMode OmniVoiceSessionBase::run_mode_impl() const {
    return task_.mode;
}

void OmniVoiceSessionBase::prepare_impl(const runtime::SessionPreparationRequest & request) {
    session_defaults_ = {};
    reference_prompt_cache_.reset();
    if (request.text.has_value()) {
        session_defaults_.text = request.text;
    }
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->audio.has_value()) {
        session_defaults_.reference_audio = *request.voice->speaker->audio;
    }
    session_defaults_.options = request.options;
    if (const auto reference_text = request_option(request.options, "reference_text"); reference_text.has_value()) {
        session_defaults_.reference_text = std::move(reference_text);
    }
    if (const auto instruct = request_option(request.options, "instruct"); instruct.has_value()) {
        session_defaults_.instruct = std::move(instruct);
    } else if (request.voice.has_value() && request.voice->style.has_value()) {
        const auto it = request.voice->style->tags.find("instruct");
        if (it != request.voice->style->tags.end()) {
            session_defaults_.instruct = it->second;
        }
    }
    if (session_defaults_.reference_audio.has_value()) {
        const auto merged_options = session_defaults_.options;
        const auto generation = generation_options_from_options(merged_options);
        const bool reference_text_provided =
            session_defaults_.reference_text.has_value() && !session_defaults_.reference_text->empty();
        (void) resolve_reference_audio_tokens(
            *session_defaults_.reference_audio,
            generation.preprocess_prompt,
            reference_text_provided);
        if (mem_saver_) {
            audio_tokenizer_.release_runtime_graphs();
        }
    }
    mark_prepared();
}

std::vector<std::string> OmniVoiceSessionBase::plan_text_chunks(
    const OmniVoiceRequest & request,
    const OmniVoicePrompt & prompt) const {
    const int64_t chunk_threshold = static_cast<int64_t>(std::llround(
        static_cast<double>(request.generation.audio_chunk_threshold_seconds) *
        static_cast<double>(prompt_builder_.frame_rate())));
    const bool use_explicit_text_chunking = request.generation.text_chunk_size.has_value();
    if (!use_explicit_text_chunking && prompt.target_audio_tokens <= chunk_threshold) {
        engine::debug::trace_log_scalar("omnivoice.text_chunk_count", int64_t{1});
        return {prompt.text};
    }

    std::vector<std::string> text_chunks;
    int64_t text_chunk_size = 0;
    if (use_explicit_text_chunking) {
        text_chunk_size = *request.generation.text_chunk_size;
        text_chunks = engine::text::split_text_chunks(
            prompt.text,
            text_chunk_size,
            request.generation.text_chunk_mode);
    } else {
        const double avg_tokens_per_char =
            static_cast<double>(prompt.target_audio_tokens) /
            static_cast<double>(std::max<size_t>(1, prompt.text.size()));
        text_chunk_size = std::max<int64_t>(
            1,
            static_cast<int64_t>(request.generation.audio_chunk_duration_seconds *
                                 static_cast<float>(prompt_builder_.frame_rate()) / avg_tokens_per_char));
        text_chunks = prompt_builder_.chunk_text_punctuation(prompt.text, text_chunk_size, 3);
    }
    engine::debug::trace_log_scalar("omnivoice.text_chunk_size", text_chunk_size);
    if (use_explicit_text_chunking) {
        engine::debug::trace_log_scalar(
            "omnivoice.text_chunk_mode",
            engine::text::text_chunk_mode_name(request.generation.text_chunk_mode));
    }
    engine::debug::trace_log_scalar("omnivoice.text_chunk_count", static_cast<int64_t>(text_chunks.size()));
    return text_chunks;
}

runtime::TaskResult OmniVoiceSessionBase::run_offline_request(const runtime::TaskRequest & request) {
    require_prepared("OmniVoice run()");
    const auto wall_start = Clock::now();
    auto omni_request = make_request(request);
    if (omni_request.generation.seed.has_value()) {
        generator_.seed_rng(*omni_request.generation.seed);
    }
    runtime::TaskResult task_result;
    std::optional<double> reference_encode_ms = std::nullopt;
    bool encoder_graph_rebuilt = false;
    if (omni_request.reference_audio.has_value()) {
        const auto encode_start = Clock::now();
        const auto reference_tokens = resolve_reference_audio_tokens(
            *omni_request.reference_audio,
            omni_request.generation.preprocess_prompt,
            !omni_request.reference_text.empty());
        const auto encode_end = Clock::now();
        omni_request.reference_rms = reference_tokens.reference_rms;
        omni_request.reference_audio_tokens = std::move(reference_tokens);
        omni_request.reference_audio.reset();
        reference_encode_ms = engine::debug::elapsed_ms(encode_start, encode_end);
        const auto & tokenizer_stats = audio_tokenizer_.last_stats();
        encoder_graph_rebuilt = tokenizer_stats.encoder_graph_rebuilt;
        if (mem_saver_) {
            const auto release_start = Clock::now();
            audio_tokenizer_.release_runtime_graphs();
            engine::debug::timing_log_scalar(
                "omnivoice.reference_release.runtime_graphs_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
        }
    }
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(omni_request);
    const auto prompt_end = Clock::now();
    OmniVoiceGeneratedAudioTokens generated_tokens;
    runtime::AudioBuffer audio = {};
    bool any_generator_graph_rebuilt = false;
    bool any_decoder_graph_rebuilt = false;
    double generator_rebuild_ms = 0.0;
    double decoder_rebuild_ms = 0.0;
    double chunk_generate_ms = 0.0;
    double chunk_decode_ms = 0.0;
    const auto generate_start = Clock::now();
    double measured_generate_ms = 0.0;
    double measured_decode_ms = 0.0;
    auto text_chunks = plan_text_chunks(omni_request, prompt);
    if (text_chunks.size() <= 1) {
        generated_tokens = generator_.generate(prompt, omni_request.generation);
        engine::debug::timing_log_scalar("omnivoice.generator.forward_ms", generated_tokens.forward_ms);
        engine::debug::timing_log_scalar("omnivoice.generator.upload_ms", generated_tokens.upload_ms);
        engine::debug::timing_log_scalar("omnivoice.generator.compute_ms", generated_tokens.compute_ms);
        engine::debug::timing_log_scalar("omnivoice.generator.readback_ms", generated_tokens.readback_ms);
        engine::debug::timing_log_scalar("omnivoice.generator.scoring_ms", generated_tokens.scoring_ms);
        engine::debug::timing_log_scalar("omnivoice.generator.update_ms", generated_tokens.update_ms);
        engine::debug::timing_log_scalar("omnivoice.generator.decode_steps", generated_tokens.decode_steps);
        {
            const auto & generator_stats = generator_.last_stats();
            any_generator_graph_rebuilt = any_generator_graph_rebuilt || generator_stats.graph_rebuilt;
            generator_rebuild_ms += generator_stats.rebuild_ms;
        }
        const auto generate_end = Clock::now();
        if (mem_saver_) {
            const auto release_start = Clock::now();
            generator_.release_runtime_graphs();
            engine::debug::timing_log_scalar(
                "omnivoice.generator_release.runtime_graphs_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
        }
        measured_generate_ms = engine::debug::elapsed_ms(generate_start, generate_end);
        const auto decode_start = Clock::now();
        audio = audio_tokenizer_.decode_audio_tokens(generated_tokens);
        {
            const auto & tokenizer_stats = audio_tokenizer_.last_stats();
            any_decoder_graph_rebuilt = any_decoder_graph_rebuilt || tokenizer_stats.decoder_graph_rebuilt;
            decoder_rebuild_ms += tokenizer_stats.decoder_rebuild_ms;
        }
        const auto decode_end = Clock::now();
        if (mem_saver_) {
            const auto release_start = Clock::now();
            audio_tokenizer_.release_runtime_graphs();
            engine::debug::timing_log_scalar(
                "omnivoice.decoder_release.runtime_graphs_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
        }
        measured_decode_ms = engine::debug::elapsed_ms(decode_start, decode_end);
    } else {
        const bool has_reference_audio = omni_request.reference_audio_tokens.has_value();
        std::optional<OmniVoiceAudioTokens> first_chunk_reference = std::nullopt;
        std::string first_chunk_text;
        OmniVoiceRequest chunk_request = omni_request;
        int64_t chunk_codebooks = 0;
        chunk_request.generation.duration_seconds.reset();
        if (omni_request.generation.duration_seconds.has_value()) {
            const int64_t full_estimate = prompt_builder_.estimate_target_tokens(
                prompt.text,
                prompt.reference_text.empty() ? std::optional<std::string>() : std::make_optional(prompt.reference_text),
                prompt.reference_audio_tokens.has_value()
                    ? std::make_optional(prompt.reference_audio_tokens->frames)
                    : std::optional<int64_t>(),
                1.0F);
            const int64_t duration_target = std::max<int64_t>(
                1,
                static_cast<int64_t>(std::llround(
                static_cast<double>(*omni_request.generation.duration_seconds) *
                    static_cast<double>(prompt_builder_.frame_rate()))));
            chunk_request.generation.speed = static_cast<float>(full_estimate) / static_cast<float>(duration_target);
        }
        const int sample_rate = assets_->config.audio_tokenizer.sample_rate;
        const int64_t hop_length = assets_->config.audio_tokenizer.hop_length;
        if (sample_rate > 0 && hop_length > 0) {
            const size_t estimated_audio_samples =
                static_cast<size_t>(prompt.target_audio_tokens) * static_cast<size_t>(hop_length);
            const size_t chunk_gap_samples = text_chunks.size() > 1
                ? static_cast<size_t>(std::llround(0.3 * static_cast<double>(sample_rate))) *
                    static_cast<size_t>(text_chunks.size() - 1)
                : 0;
            audio.samples.reserve(estimated_audio_samples + chunk_gap_samples);
        }
        for (size_t chunk_index = 0; chunk_index < text_chunks.size(); ++chunk_index) {
            chunk_request.text = text_chunks[chunk_index];
            if (!has_reference_audio && chunk_index > 0) {
                chunk_request.reference_audio_tokens = *first_chunk_reference;
                chunk_request.reference_text = first_chunk_text;
            }
            const auto chunk_prompt = prompt_builder_.build(chunk_request);
            const auto chunk_generate_start = Clock::now();
            auto chunk_tokens = generator_.generate(chunk_prompt, chunk_request.generation);
            const auto chunk_generate_end = Clock::now();
            {
                const auto & generator_stats = generator_.last_stats();
                any_generator_graph_rebuilt = any_generator_graph_rebuilt || generator_stats.graph_rebuilt;
                generator_rebuild_ms += generator_stats.rebuild_ms;
            }
            if (mem_saver_) {
                const auto release_start = Clock::now();
                generator_.release_runtime_graphs();
                engine::debug::timing_log_scalar(
                    "omnivoice.generator_release.runtime_graphs_ms",
                    engine::debug::elapsed_ms(release_start, Clock::now()));
            }
            chunk_generate_ms += engine::debug::elapsed_ms(chunk_generate_start, chunk_generate_end);
            if (chunk_codebooks == 0) {
                chunk_codebooks = chunk_tokens.codebooks;
            } else if (chunk_codebooks != chunk_tokens.codebooks) {
                throw std::runtime_error("OmniVoice chunk token codebook mismatch");
            }
            if (!has_reference_audio && chunk_index == 0) {
                OmniVoiceAudioTokens chunk_ref = {};
                chunk_ref.frames = chunk_tokens.frames;
                chunk_ref.codebooks = chunk_tokens.codebooks;
                chunk_ref.token_ids = chunk_tokens.token_ids;
                first_chunk_reference = std::move(chunk_ref);
                first_chunk_text = text_chunks.front();
            }
            const auto chunk_decode_start = Clock::now();
            auto chunk_audio = audio_tokenizer_.decode_audio_tokens(chunk_tokens);
            const auto chunk_decode_end = Clock::now();
            {
                const auto & tokenizer_stats = audio_tokenizer_.last_stats();
                any_decoder_graph_rebuilt = any_decoder_graph_rebuilt || tokenizer_stats.decoder_graph_rebuilt;
                decoder_rebuild_ms += tokenizer_stats.decoder_rebuild_ms;
            }
            if (mem_saver_) {
                const auto release_start = Clock::now();
                audio_tokenizer_.release_runtime_graphs();
                engine::debug::timing_log_scalar(
                    "omnivoice.decoder_release.runtime_graphs_ms",
                    engine::debug::elapsed_ms(release_start, Clock::now()));
            }
            chunk_decode_ms += engine::debug::elapsed_ms(chunk_decode_start, chunk_decode_end);
            append_cross_faded_chunk(audio, chunk_audio);
        }
        measured_generate_ms = chunk_generate_ms;
        measured_decode_ms = chunk_decode_ms;
    }
    const auto postprocess_start = Clock::now();
    const auto result = postprocessor_.finalize(audio, omni_request);
    const auto postprocess_end = Clock::now();
    task_result.audio_output = result.audio;
    if (reference_encode_ms.has_value()) {
        engine::debug::timing_log_scalar("omnivoice.reference.encode_ms", *reference_encode_ms);
        engine::debug::timing_log_scalar("omnivoice.encoder.graph.rebuilt", encoder_graph_rebuilt);
    }
    engine::debug::timing_log_scalar("omnivoice.session.prompt_ms", engine::debug::elapsed_ms(prompt_start, prompt_end));
    engine::debug::timing_log_scalar("omnivoice.session.generate_ms", measured_generate_ms);
    engine::debug::timing_log_scalar("omnivoice.generator.graph.rebuilt", any_generator_graph_rebuilt);
    engine::debug::timing_log_scalar("omnivoice.generator.graph.rebuild_ms", generator_rebuild_ms);
    engine::debug::timing_log_scalar("omnivoice.session.decode_ms", measured_decode_ms);
    engine::debug::timing_log_scalar("omnivoice.decoder.graph.rebuilt", any_decoder_graph_rebuilt);
    engine::debug::timing_log_scalar("omnivoice.decoder.graph.rebuild_ms", decoder_rebuild_ms);
    engine::debug::timing_log_scalar("omnivoice.session.postprocess_ms", engine::debug::elapsed_ms(postprocess_start, postprocess_end));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, postprocess_end));
    return task_result;
}

runtime::TaskResult OmniVoiceSessionBase::run_streaming_request(
    const runtime::TaskRequest & request,
    const runtime::StreamEventCallback & stream_event_sink) {
    require_prepared("OmniVoice run_streaming");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("OmniVoice run_streaming requires a streaming session");
    }

    // Lower default num_inference_steps for streaming, unless the caller
    // asked for something else -- see kStreamingDefaultNumInferenceSteps.
    runtime::TaskRequest streaming_request = request;
    const auto merged_defaults = merged_request_options(streaming_request);
    if (merged_defaults.find(kNumInferenceStepsKey) == merged_defaults.end()) {
        streaming_request.options[kNumInferenceStepsKey] = kStreamingDefaultNumInferenceSteps;
    }

    const auto wall_start = Clock::now();
    auto omni_request = make_request(streaming_request);
    if (omni_request.generation.seed.has_value()) {
        generator_.seed_rng(*omni_request.generation.seed);
    }
    runtime::TaskResult task_result;
    if (omni_request.reference_audio.has_value()) {
        const auto reference_tokens = resolve_reference_audio_tokens(
            *omni_request.reference_audio,
            omni_request.generation.preprocess_prompt,
            !omni_request.reference_text.empty());
        omni_request.reference_rms = reference_tokens.reference_rms;
        omni_request.reference_audio_tokens = std::move(reference_tokens);
        omni_request.reference_audio.reset();
        if (mem_saver_) {
            audio_tokenizer_.release_runtime_graphs();
        }
    }
    const auto prompt = prompt_builder_.build(omni_request);

    // Always chunk (even a short single-chunk request just yields one
    // stream event) -- streaming's whole point is progressive emission, so
    // there's no separate "small enough, do it in one shot" branch here the
    // way the offline path has.
    //
    // Two layers, kept separate on purpose (see OmniVoiceGenerationOptions::
    // sentence_chunk_pause_ms):
    //   OUTER -- splits the request into pieces a pause belongs between.
    //            Default path: plan_text_chunks (unchanged, exactly as
    //            before -- same helper run_offline_request uses).
    //            sentence_chunk_pause_ms path: one TRUE sentence per outer
    //            piece (split_true_sentences -- splits only at . ! ?,
    //            never mid-sentence at a comma).
    //   INNER  -- ONLY the first outer piece gets peeled further, into a
    //            small opening fragment + the rest, purely to keep TTFC
    //            low (the first fragment reaches the model/client sooner).
    //            Always chunk_text_punctuation(..., 1, ...) -- unchanged
    //            from before this option existed, and the same mechanism
    //            the default path's own peel below uses. Every other outer
    //            piece passes through with no inner chunking.
    // Stitching mirrors the split: inner pieces (the first outer piece's
    // own peel) get fade-in only, same as always. Only the boundary INTO a
    // new outer piece (i.e. NOT the first outer piece's own inner split)
    // gets the extra pause -- so pauses land between sentences, never
    // inside one.
    std::vector<std::string> text_chunks;
    std::vector<bool> chunk_starts_outer_piece;
    std::vector<bool> chunk_is_first_sentence;
    if (omni_request.generation.sentence_chunk_pause_ms > 0) {
        const auto outer_pieces = prompt_builder_.split_true_sentences(prompt.text);
        for (size_t outer_index = 0; outer_index < outer_pieces.size(); ++outer_index) {
            const bool is_first_outer_piece = (outer_index == 0);
            std::vector<std::string> inner_pieces;
            if (is_first_outer_piece) {
                // Peel a small opening fragment for TTFC, same as the
                // default path's peel below -- exactly 2 inner pieces (or
                // 1, if the sentence has no internal punctuation to break
                // on), NOT one piece per punctuation mark: everything after
                // the first fragment is merged back into a single "rest"
                // piece, matching first_chunk_sentences' own merge below.
                const auto split_first = prompt_builder_.chunk_text_punctuation(
                    outer_pieces[outer_index], 1, std::nullopt);
                if (split_first.size() > 1) {
                    std::string rest;
                    for (size_t k = 1; k < split_first.size(); ++k) {
                        rest += split_first[k];
                    }
                    inner_pieces = {split_first.front(), std::move(rest)};
                } else {
                    inner_pieces = split_first;
                }
            } else {
                inner_pieces = {outer_pieces[outer_index]};
            }
            for (size_t inner_index = 0; inner_index < inner_pieces.size(); ++inner_index) {
                text_chunks.push_back(inner_pieces[inner_index]);
                chunk_starts_outer_piece.push_back(inner_index == 0 && !is_first_outer_piece);
                chunk_is_first_sentence.push_back(is_first_outer_piece);
            }
        }
    } else {
        text_chunks = plan_text_chunks(omni_request, prompt);
        if (text_chunks.empty()) {
            text_chunks.push_back(prompt.text);
        }

        // Peel just the first SENTENCE off text_chunks[0] into its own
        // chunk -- see the layering comment above for why. Only for the
        // default (non-sentence_chunk_pause_ms) path -- that path already
        // built its own inner peel above, scoped to exactly one true
        // sentence, which chunk_text_punctuation(text_chunks[0], 1, ...)
        // here is NOT a safe substitute for: unlike this block's original
        // assumption, it also splits on , ; : (see is_split_punctuation),
        // so run again on an ALREADY single-sentence text_chunks[0] it
        // would still fragment at a comma, corrupting it and inserting a
        // bogus extra chunk.
        auto first_chunk_sentences = prompt_builder_.chunk_text_punctuation(text_chunks[0], 1, std::nullopt);
        if (first_chunk_sentences.size() > 1) {
            std::string rest_of_first_chunk;
            for (size_t i = 1; i < first_chunk_sentences.size(); ++i) {
                rest_of_first_chunk += first_chunk_sentences[i];
            }
            text_chunks[0] = first_chunk_sentences.front();
            text_chunks.insert(text_chunks.begin() + 1, std::move(rest_of_first_chunk));
        }
    }
    if (chunk_starts_outer_piece.size() != text_chunks.size()) {
        chunk_starts_outer_piece.assign(text_chunks.size(), false);  // default path: no pauses at all
    }
    if (chunk_is_first_sentence.size() != text_chunks.size()) {
        chunk_is_first_sentence.assign(text_chunks.size(), false);  // default path: no step override anywhere
    }

    engine::debug::trace_log_scalar("omnivoice.text_chunk_count", static_cast<int64_t>(text_chunks.size()));

    const bool has_reference_audio = omni_request.reference_audio_tokens.has_value();
    std::optional<OmniVoiceAudioTokens> first_chunk_reference = std::nullopt;
    std::string first_chunk_text;
    OmniVoiceRequest chunk_request = omni_request;
    int64_t chunk_codebooks = 0;
    chunk_request.generation.duration_seconds.reset();
    // See OmniVoiceGenerationOptions::first_sentence_num_inference_steps --
    // captured once here (before chunk_request.generation gets overridden
    // below) so later, non-first-sentence chunks can be restored to it.
    const int64_t normal_num_inference_steps = chunk_request.generation.num_inference_steps;

    runtime::AudioBuffer merged;
    double generator_ms = 0.0;
    double decoder_ms = 0.0;
    constexpr float kChunkFadeInSeconds = 0.05F;
    for (size_t chunk_index = 0; chunk_index < text_chunks.size(); ++chunk_index) {
        chunk_request.text = text_chunks[chunk_index];
        if (!has_reference_audio && chunk_index > 0) {
            chunk_request.reference_audio_tokens = *first_chunk_reference;
            chunk_request.reference_text = first_chunk_text;
        }
        chunk_request.generation.num_inference_steps =
            (chunk_is_first_sentence[chunk_index] && omni_request.generation.first_sentence_num_inference_steps > 0)
                ? omni_request.generation.first_sentence_num_inference_steps
                : normal_num_inference_steps;
        const auto chunk_prompt = prompt_builder_.build(chunk_request);
        const auto generate_start = Clock::now();
        auto chunk_tokens = generator_.generate(chunk_prompt, chunk_request.generation);
        generator_ms += engine::debug::elapsed_ms(generate_start, Clock::now());
        if (mem_saver_) {
            generator_.release_runtime_graphs();
        }
        if (chunk_codebooks == 0) {
            chunk_codebooks = chunk_tokens.codebooks;
        } else if (chunk_codebooks != chunk_tokens.codebooks) {
            throw std::runtime_error("OmniVoice streaming chunk token codebook mismatch");
        }
        if (!has_reference_audio && chunk_index == 0) {
            OmniVoiceAudioTokens chunk_ref = {};
            chunk_ref.frames = chunk_tokens.frames;
            chunk_ref.codebooks = chunk_tokens.codebooks;
            chunk_ref.token_ids = chunk_tokens.token_ids;
            first_chunk_reference = std::move(chunk_ref);
            first_chunk_text = text_chunks.front();
        }

        const auto decode_start = Clock::now();
        auto chunk_audio = audio_tokenizer_.decode_audio_tokens(chunk_tokens);
        decoder_ms += engine::debug::elapsed_ms(decode_start, Clock::now());
        if (mem_saver_) {
            audio_tokenizer_.release_runtime_graphs();
        }

        if (chunk_index > 0) {
            apply_fade_in(chunk_audio, kChunkFadeInSeconds);
            // Only a boundary INTO a new outer piece (a new sentence) gets
            // the pause -- an inner boundary (within the first sentence's
            // own TTFC peel) stitches with just the fade above, same as
            // the default path always has.
            if (chunk_starts_outer_piece[chunk_index]) {
                prepend_silence(chunk_audio, omni_request.generation.sentence_chunk_pause_ms);
            }
        }

        runtime::NamedAudioBuffer named;
        named.id = "chunk_" + std::to_string(chunk_index);
        named.audio = chunk_audio;
        named.meta.insert_or_assign("chunk_index", std::to_string(chunk_index));
        named.meta.insert_or_assign("chunk_count", std::to_string(text_chunks.size()));
        if (stream_event_sink) {
            runtime::StreamEvent event;
            event.is_final = (chunk_index + 1 == text_chunks.size());
            event.named_audio_outputs.push_back(named);
            stream_event_sink(event);
        }
        task_result.named_audio_outputs.push_back(std::move(named));

        if (merged.sample_rate == 0) {
            merged.sample_rate = chunk_audio.sample_rate;
            merged.channels = chunk_audio.channels;
        }
        merged.samples.insert(merged.samples.end(), chunk_audio.samples.begin(), chunk_audio.samples.end());
    }
    task_result.audio_output = std::move(merged);

    engine::debug::timing_log_scalar("omnivoice.streaming.generator_ms", generator_ms);
    engine::debug::timing_log_scalar("omnivoice.streaming.decoder_ms", decoder_ms);
    engine::debug::timing_log_scalar("omnivoice.streaming.chunks", static_cast<double>(text_chunks.size()));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return task_result;
}

OmniVoiceRequest OmniVoiceSessionBase::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() && !session_defaults_.text.has_value()) {
        throw std::runtime_error("OmniVoice run() requires text_input");
    }
    OmniVoiceRequest out;
    if (request.text_input.has_value()) {
        out.text = request.text_input->text;
        out.language = request.text_input->language;
    } else {
        out.text = session_defaults_.text->text;
        out.language = session_defaults_.text->language;
    }
    const auto merged_options = merged_request_options(request);
    out.generation = generation_options_from_options(merged_options);
    out.reference_audio = resolve_reference_audio(request);
    if (const auto reference_text = resolve_reference_text(request); reference_text.has_value()) {
        out.reference_text = *reference_text;
    }
    if (const auto instruct = resolve_instruct(request); instruct.has_value()) {
        out.instruct = *instruct;
    }
    return out;
}

std::optional<runtime::AudioBuffer> OmniVoiceSessionBase::resolve_reference_audio(const runtime::TaskRequest & request) const {
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->audio.has_value()) {
        return *request.voice->speaker->audio;
    }
    if (request.audio_input.has_value()) {
        return *request.audio_input;
    }
    return session_defaults_.reference_audio;
}

std::optional<std::string> OmniVoiceSessionBase::resolve_reference_text(const runtime::TaskRequest & request) const {
    if (const auto value = request_option(request.options, "reference_text"); value.has_value()) {
        return value;
    }
    return session_defaults_.reference_text;
}

std::optional<std::string> OmniVoiceSessionBase::resolve_instruct(const runtime::TaskRequest & request) const {
    if (const auto value = request_option(request.options, "instruct"); value.has_value()) {
        return value;
    }
    if (request.voice.has_value() && request.voice->style.has_value()) {
        const auto it = request.voice->style->tags.find("instruct");
        if (it != request.voice->style->tags.end()) {
            return it->second;
        }
    }
    return session_defaults_.instruct;
}

std::unordered_map<std::string, std::string> OmniVoiceSessionBase::merged_request_options(const runtime::TaskRequest & request) const {
    auto merged = session_defaults_.options;
    for (const auto & [key, value] : request.options) {
        merged[key] = value;
    }
    return merged;
}

OmniVoiceAudioTokens OmniVoiceSessionBase::resolve_reference_audio_tokens(
    const runtime::AudioBuffer & audio,
    bool preprocess_prompt,
    bool reference_text_provided) {
    const uint64_t sample_count = static_cast<uint64_t>(audio.samples.size());
    const uint64_t sample_hash = hash_audio_samples(audio);
    const bool cache_hit = reference_prompt_cache_.has_value() &&
        reference_prompt_cache_->preprocess_prompt == preprocess_prompt &&
        reference_prompt_cache_->reference_text_provided == reference_text_provided &&
        reference_prompt_cache_->sample_rate == audio.sample_rate &&
        reference_prompt_cache_->channels == audio.channels &&
        reference_prompt_cache_->sample_count == sample_count &&
        reference_prompt_cache_->sample_hash == sample_hash;
    if (!cache_hit) {
        const OmniVoiceReferenceAudioOptions reference_options = {
            preprocess_prompt,
            reference_text_provided,
        };
        ReferencePromptCacheEntry entry;
        entry.preprocess_prompt = preprocess_prompt;
        entry.reference_text_provided = reference_text_provided;
        entry.sample_rate = audio.sample_rate;
        entry.channels = audio.channels;
        entry.sample_count = sample_count;
        entry.sample_hash = sample_hash;
        entry.tokens = audio_tokenizer_.encode_reference_audio(audio, reference_options);
        reference_prompt_cache_ = std::move(entry);
    }
    return reference_prompt_cache_->tokens;
}

OmniVoiceOfflineSession::OmniVoiceOfflineSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const OmniVoiceAssets> assets)
    : OmniVoiceSessionBase(task, std::move(options), std::move(assets)) {}

std::string OmniVoiceOfflineSession::family() const {
    return family_impl();
}

runtime::VoiceTaskKind OmniVoiceOfflineSession::task_kind() const {
    return task_kind_impl();
}

runtime::RunMode OmniVoiceOfflineSession::run_mode() const {
    return run_mode_impl();
}

void OmniVoiceOfflineSession::prepare(const runtime::SessionPreparationRequest & request) {
    prepare_impl(request);
}

runtime::TaskResult OmniVoiceOfflineSession::run(const runtime::TaskRequest & request) {
    return run_offline_request(request);
}

OmniVoiceStreamingSession::OmniVoiceStreamingSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const OmniVoiceAssets> assets)
    : OmniVoiceSessionBase(task, std::move(options), std::move(assets)) {}

std::string OmniVoiceStreamingSession::family() const {
    return family_impl();
}

runtime::VoiceTaskKind OmniVoiceStreamingSession::task_kind() const {
    return task_kind_impl();
}

runtime::RunMode OmniVoiceStreamingSession::run_mode() const {
    return run_mode_impl();
}

void OmniVoiceStreamingSession::prepare(const runtime::SessionPreparationRequest & request) {
    prepare_impl(request);
}

runtime::StreamingPolicy OmniVoiceStreamingSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    return policy;
}

void OmniVoiceStreamingSession::start_stream(const runtime::TaskRequest & request) {
    reset();
    result_ = run_streaming_request(request, stream_event_sink_);
    started_ = true;
}

void OmniVoiceStreamingSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

std::optional<runtime::StreamEvent> OmniVoiceStreamingSession::next_stream_event() {
    if (!started_) {
        throw std::runtime_error("OmniVoice streaming has not been started");
    }
    if (next_chunk_index_ >= result_.named_audio_outputs.size()) {
        return std::nullopt;
    }
    const auto & named = result_.named_audio_outputs[next_chunk_index_++];
    runtime::StreamEvent event;
    event.is_final = (next_chunk_index_ == result_.named_audio_outputs.size());
    event.named_audio_outputs.push_back(named);
    return event;
}

runtime::TaskResult OmniVoiceStreamingSession::finish_stream() {
    if (!started_) {
        throw std::runtime_error("OmniVoice streaming has not been started");
    }
    started_ = false;
    next_chunk_index_ = 0;
    return std::move(result_);
}

void OmniVoiceStreamingSession::reset() {
    result_ = runtime::TaskResult{};
    next_chunk_index_ = 0;
    started_ = false;
}

runtime::StreamEvent OmniVoiceStreamingSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    (void) chunk;
    throw std::runtime_error("OmniVoice streaming does not consume audio chunks");
}

runtime::TaskResult OmniVoiceStreamingSession::finalize() {
    return finish_stream();
}

}  // namespace engine::models::omnivoice
