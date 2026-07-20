#include "engine/models/fish_audio/prompt_builder.h"

#include <regex>
#include <stdexcept>
#include <utility>

namespace engine::models::fish_audio {
namespace {

void append_tokens(std::vector<int32_t> & out, const std::vector<int32_t> & tokens) {
    out.insert(out.end(), tokens.begin(), tokens.end());
}

std::string reference_text_with_speakers(const std::string & text) {
    static const std::regex speaker_re(R"(<\|speaker:\d+\|>)");
    if (std::regex_search(text, speaker_re)) {
        return text;
    }
    return "<|speaker:0|>" + text;
}

}  // namespace

FishAudioPromptBuilder::FishAudioPromptBuilder(
    std::shared_ptr<const FishAudioAssets> assets,
    FishAudioTextTokenizer tokenizer)
    : assets_(std::move(assets)),
      tokenizer_(std::move(tokenizer)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Fish Audio prompt builder requires assets");
    }
}

FishAudioPrompt FishAudioPromptBuilder::build(
    const FishAudioRequest & request,
    const std::optional<FishAudioCodes> & reference_codes) const {
    if (request.text.empty()) {
        throw std::runtime_error("Fish Audio request text must not be empty");
    }
    const int64_t rows = assets_->config.fast.num_codebooks + 1;
    if (rows <= 1) {
        throw std::runtime_error("Fish Audio prompt rows are invalid");
    }

    std::vector<int32_t> row0;
    if (request.reference.has_value()) {
        if (!reference_codes.has_value()) {
            throw std::runtime_error("Fish Audio reference request requires encoded reference codes");
        }
        if (reference_codes->codebooks != assets_->config.fast.num_codebooks) {
            throw std::runtime_error("Fish Audio reference codebook count mismatch");
        }
        append_tokens(row0, tokenizer_.encode("<|im_start|>system\n"));
        append_tokens(row0, tokenizer_.encode("convert the provided text to speech reference to the following:\n\nText:\n"));
        append_tokens(row0, tokenizer_.encode(reference_text_with_speakers(request.reference->text)));
        append_tokens(row0, tokenizer_.encode("\n\nSpeech:\n"));
        const int32_t semantic_begin = tokenizer_.semantic_begin_id();
        for (int64_t frame = 0; frame < reference_codes->frames; ++frame) {
            const int32_t code = reference_codes->codes[static_cast<size_t>(frame)];
            row0.push_back(semantic_begin + code);
        }
        append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
    } else {
        append_tokens(row0, tokenizer_.encode("<|im_start|>system\n"));
        append_tokens(row0, tokenizer_.encode("convert the provided text to speech"));
        append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
    }
    append_tokens(row0, tokenizer_.encode("<|im_start|>user\n"));
    append_tokens(row0, tokenizer_.encode(request.text));
    append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
    append_tokens(row0, tokenizer_.encode("<|im_start|>assistant\n<|voice|>"));

    FishAudioPrompt prompt;
    prompt.codebook_rows = rows;
    prompt.steps = static_cast<int64_t>(row0.size());
    prompt.text = request.text;
    prompt.matrix.assign(static_cast<size_t>(rows * prompt.steps), 0);
    for (int64_t step = 0; step < prompt.steps; ++step) {
        prompt.matrix[static_cast<size_t>(step)] = row0[static_cast<size_t>(step)];
    }
    if (reference_codes.has_value()) {
        int64_t semantic_index = 0;
        for (int64_t step = 0; step < prompt.steps; ++step) {
            const int32_t token = prompt.matrix[static_cast<size_t>(step)];
            if (token < tokenizer_.semantic_begin_id() || token > tokenizer_.semantic_end_id()) {
                continue;
            }
            if (semantic_index >= reference_codes->frames) {
                break;
            }
            for (int64_t codebook = 0; codebook < reference_codes->codebooks; ++codebook) {
                prompt.matrix[static_cast<size_t>((codebook + 1) * prompt.steps + step)] =
                    reference_codes->codes[static_cast<size_t>(codebook * reference_codes->frames + semantic_index)];
            }
            ++semantic_index;
        }
    }
    return prompt;
}

}  // namespace engine::models::fish_audio
