package agent

import "github.com/anthony-chaudhary/fak/internal/tokenizer"

const defaultInheritedTemperature = 1.0

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

const (
	cycleBreakerMinSpan    = 16
	cycleBreakerMinRepeats = 4
	cycleBreakerMaxSpan    = 512
	cycleBreakerMaxHistory = cycleBreakerMinRepeats * cycleBreakerMaxSpan
)

// rollingTokenCycleBreaker detects intra-turn degenerate token loops where a
// span of at least 16 tokens repeats at least 4 times consecutively at the tail
// of the generated sequence.
type rollingTokenCycleBreaker struct {
	history []int
}

// Observe appends token to the intra-turn history and reports whether a
// consecutive repeating cycle (>=16 tokens repeated >=4x) has formed.
func (cb *rollingTokenCycleBreaker) Observe(token int) bool {
	cb.history = append(cb.history, token)
	if len(cb.history) > 2*cycleBreakerMaxHistory {
		copy(cb.history, cb.history[len(cb.history)-cycleBreakerMaxHistory:])
		cb.history = cb.history[:cycleBreakerMaxHistory]
	}
	n := len(cb.history)
	maxL := n / cycleBreakerMinRepeats
	if maxL > cycleBreakerMaxSpan {
		maxL = cycleBreakerMaxSpan
	}
	for L := cycleBreakerMinSpan; L <= maxL; L++ {
		if cb.history[n-1] != cb.history[n-1-L] ||
			cb.history[n-1-L] != cb.history[n-1-2*L] ||
			cb.history[n-1-2*L] != cb.history[n-1-3*L] {
			continue
		}
		match := true
		base := n - 4*L
		for i := 0; i < 3*L; i++ {
			if cb.history[base+i] != cb.history[base+L+i] {
				match = false
				break
			}
		}
		if match {
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
