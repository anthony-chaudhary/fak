package agent

import "github.com/anthony-chaudhary/fak/internal/tokenizer"

// populateReasoningTokensUsage sets comp.Usage.CompletionTokensDetails.ReasoningTokens
// whenever comp.Message.ReasoningContent is non-empty, using tok to count reasoning
// tokens (clamped to [1, CompletionTokens] when CompletionTokens > 0).
func populateReasoningTokensUsage(comp *Completion, tok *tokenizer.Tokenizer) {
	if comp == nil || comp.Message.ReasoningContent == "" {
		return
	}
	if comp.Usage.CompletionTokensDetails != nil && comp.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
		return
	}
	reasoningTokens := 0
	if tok != nil {
		if rIDs, err := tok.Encode(comp.Message.ReasoningContent); err == nil {
			reasoningTokens = len(rIDs)
		}
	}
	if reasoningTokens == 0 {
		reasoningTokens = 1
	}
	if comp.Usage.CompletionTokens > 0 && reasoningTokens > comp.Usage.CompletionTokens {
		reasoningTokens = comp.Usage.CompletionTokens
	}
	if comp.Usage.CompletionTokensDetails == nil {
		comp.Usage.CompletionTokensDetails = &UsageCompletionTokenDetails{}
	}
	comp.Usage.CompletionTokensDetails.ReasoningTokens = reasoningTokens
}
