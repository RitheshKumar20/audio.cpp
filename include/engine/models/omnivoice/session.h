#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/audio_tokenizer.h"
#include "engine/models/omnivoice/generator.h"
#include "engine/models/omnivoice/postprocess.h"
#include "engine/models/omnivoice/prompt_builder.h"
#include "engine/models/omnivoice/tokenizer_text.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::omnivoice {

// Shared state + logic for both the offline and streaming sessions: model
// runtimes, reference-prompt caching, request resolution, text-chunk
// planning, and the per-chunk generate+decode loop (run_offline_request
// accumulates everything into one crossfaded buffer before returning,
// matching upstream's original single-session OmniVoiceSession::run();
// run_streaming_request runs a similar per-chunk loop but emits each chunk
// through a StreamEventCallback as soon as it's decoded, instead of
// collecting silently and only supporting pull-based replay afterward).
// Splitting this out mirrors engine::models::voxcpm2::VoxCPM2SessionBase's
// Offline/Streaming split.
class OmniVoiceSessionBase : public runtime::RuntimeSessionBase {
public:
    OmniVoiceSessionBase(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const OmniVoiceAssets> assets);

protected:
    std::string family_impl() const;
    runtime::VoiceTaskKind task_kind_impl() const;
    runtime::RunMode run_mode_impl() const;
    void prepare_impl(const runtime::SessionPreparationRequest & request);

    struct SessionDefaults {
        std::optional<runtime::Transcript> text = std::nullopt;
        std::optional<runtime::AudioBuffer> reference_audio = std::nullopt;
        std::optional<std::string> reference_text = std::nullopt;
        std::optional<std::string> instruct = std::nullopt;
        std::unordered_map<std::string, std::string> options;
    };

    struct ReferencePromptCacheEntry {
        bool preprocess_prompt = true;
        bool reference_text_provided = false;
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
        OmniVoiceAudioTokens tokens;
    };

    OmniVoiceRequest make_request(const runtime::TaskRequest & request) const;
    std::optional<runtime::AudioBuffer> resolve_reference_audio(const runtime::TaskRequest & request) const;
    std::optional<std::string> resolve_reference_text(const runtime::TaskRequest & request) const;
    std::optional<std::string> resolve_instruct(const runtime::TaskRequest & request) const;
    std::unordered_map<std::string, std::string> merged_request_options(const runtime::TaskRequest & request) const;
    OmniVoiceAudioTokens resolve_reference_audio_tokens(
        const runtime::AudioBuffer & audio,
        bool preprocess_prompt,
        bool reference_text_provided);
    std::vector<std::string> plan_text_chunks(const OmniVoiceRequest & request, const OmniVoicePrompt & prompt) const;

    // Non-chunked or explicitly-chunked-but-still-collected-into-one-buffer
    // path -- byte-identical to upstream's original OmniVoiceSession::run().
    runtime::TaskResult run_offline_request(const runtime::TaskRequest & request);

    // Same per-chunk generate+decode loop as run_offline_request, but each
    // chunk's decoded audio is pushed through stream_event_sink the moment
    // it's ready. Two deliberate differences from the offline chunked path,
    // both aimed at low time-to-first-audio (see session.cpp for the full
    // reasoning): the first chunk is peeled down to a single sentence so
    // it's ready fast, and num_inference_steps defaults lower than the
    // offline path's. Chunk boundaries are smoothed with a fade-in on just
    // the incoming chunk's head (apply_fade_in) rather than a full two-sided
    // crossfade, which would require buffering one chunk to blend both
    // sides and add a full chunk's latency to time-to-first-audio.
    runtime::TaskResult run_streaming_request(
        const runtime::TaskRequest & request,
        const runtime::StreamEventCallback & stream_event_sink = nullptr);

    runtime::TaskSpec task_;
    std::shared_ptr<const OmniVoiceAssets> assets_;
    size_t audio_tokenizer_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
    size_t generator_prefill_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t generator_decode_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t audio_tokenizer_weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    size_t generator_weight_context_bytes_ = 256ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType audio_tokenizer_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType generator_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    bool mem_saver_ = false;
    OmniVoiceGeneratorPerfMode generator_perf_mode_ = OmniVoiceGeneratorPerfMode::Standard;
    OmniVoiceTextTokenizer tokenizer_;
    OmniVoiceAudioTokenizerRuntime audio_tokenizer_;
    OmniVoicePromptBuilder prompt_builder_;
    OmniVoiceGeneratorRuntime generator_;
    OmniVoicePostprocessor postprocessor_;
    SessionDefaults session_defaults_;
    std::optional<ReferencePromptCacheEntry> reference_prompt_cache_;
};

class OmniVoiceOfflineSession final
    : public OmniVoiceSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    OmniVoiceOfflineSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const OmniVoiceAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
};

class OmniVoiceStreamingSession final
    : public OmniVoiceSessionBase
    , public runtime::IStreamingVoiceTaskSession {
public:
    OmniVoiceStreamingSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const OmniVoiceAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    std::optional<runtime::StreamEvent> next_stream_event() override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    runtime::TaskResult finish_stream() override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    runtime::TaskResult result_;
    size_t next_chunk_index_ = 0;
    bool started_ = false;
    runtime::StreamEventCallback stream_event_sink_;
};

}  // namespace engine::models::omnivoice
