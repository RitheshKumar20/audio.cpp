#pragma once

#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/tokenizer_text.h"
#include "engine/models/omnivoice/types.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::omnivoice {

class OmniVoicePromptBuilder {
public:
    OmniVoicePromptBuilder(std::shared_ptr<const OmniVoiceAssets> assets, const OmniVoiceTextTokenizer & tokenizer);

    OmniVoicePrompt build(const OmniVoiceRequest & request) const;
    int64_t frame_rate() const;
    int64_t estimate_target_tokens(
        std::string_view text,
        const std::optional<std::string> & ref_text,
        const std::optional<int64_t> & ref_audio_tokens,
        float speed) const;
    std::vector<std::string> chunk_text_punctuation(
        std::string_view text,
        int64_t chunk_len,
        std::optional<int64_t> min_chunk_len = 3) const;
    // Splits ONLY at true sentence-ending punctuation (. ! ?), unlike
    // chunk_text_punctuation whose is_split_punctuation also treats , ; :
    // as split points -- appropriate for that function's chunk-size-packing
    // use (commas become internal merge points at chunk_len>1), wrong for
    // "one chunk per sentence" (chunk_len=1 there fragments at every comma
    // too, since nothing can be merged when chunk_len is that small). Keeps
    // trailing closing quotes/brackets attached to the sentence they close,
    // e.g. a period inside a quoted "...?" does not end the sentence if a
    // comma immediately follows the closing quote (still mid-sentence).
    std::vector<std::string> split_true_sentences(std::string_view text) const;

private:
    std::shared_ptr<const OmniVoiceAssets> assets_;
    const OmniVoiceTextTokenizer & tokenizer_;
};

}  // namespace engine::models::omnivoice
