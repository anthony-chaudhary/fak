package agent

import "github.com/anthony-chaudhary/fak/internal/tokenizer"

const (
	tokenCycleMinPeriod  = 8
	tokenCycleMaxPeriod  = 64
	tokenCycleMinRepeats = 4
)

// tokenCycleBreaker tracks recent emitted token IDs and halts generation when
// any suffix of length [tokenCycleMinPeriod, tokenCycleMaxPeriod] repeats at
// least tokenCycleMinRepeats consecutive times. Shared across native planners
// (InKernelPlanner, Inkling2LoopPlanner, inkling2RunPlanner) so no model can
// burn a full completion budget in a verbatim repetition loop.
type tokenCycleBreaker struct {
	recent []int
	broken bool
	period int
}

func (b *tokenCycleBreaker) reset() {
	b.recent = b.recent[:0]
	b.broken = false
	b.period = 0
}

func (b *tokenCycleBreaker) push(id int) bool {
	const maxWindow = tokenCycleMaxPeriod * tokenCycleMinRepeats
	b.recent = append(b.recent, id)
	if len(b.recent) > 2*maxWindow {
		copy(b.recent, b.recent[len(b.recent)-maxWindow:])
		b.recent = b.recent[:maxWindow]
	}
	n := len(b.recent)
	maxPeriod := min(tokenCycleMaxPeriod, n/tokenCycleMinRepeats)
	for period := tokenCycleMinPeriod; period <= maxPeriod; period++ {
		if b.recent[n-1] != b.recent[n-1-period] {
			continue
		}
		base := n - period*tokenCycleMinRepeats
		match := true
		for i := period*(tokenCycleMinRepeats-1) - 1; i >= 0; i-- {
			if b.recent[base+i] != b.recent[base+i+period] {
				match = false
				break
			}
		}
		if match {
			b.broken = true
			b.period = period
			return true
		}
	}
	return false
}

// nativeReasoningTokenDetails populates CompletionTokensDetails.ReasoningTokens
// for native planners when ReasoningContent was extracted from the completion.
// The count is tokenized with the model's own tokenizer and clamped to
// [1, completionTokens] so completion_tokens always remains >= reasoning_tokens.
func nativeReasoningTokenDetails(tok *tokenizer.Tokenizer, reasoning string, completionTokens int) *UsageCompletionTokenDetails {
	if reasoning == "" || completionTokens <= 0 {
		return nil
	}
	n := 0
	if tok != nil {
		if ids, err := tok.Encode(reasoning); err == nil {
			n = len(ids)
		}
	}
	if n <= 0 {
		n = 1
	}
	if n > completionTokens {
		n = completionTokens
	}
	return &UsageCompletionTokenDetails{ReasoningTokens: n}
}
