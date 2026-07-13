package quality

import (
	"fmt"
	"strings"
)

// StopTruncationEOS is the sentinel token the stop-truncation oracle treats as a
// hard stop / end-of-sequence marker: a faithful engine emits NOTHING after its
// first occurrence.
const StopTruncationEOS = "<eos>"

// StopTruncation is the stop-semantics differential oracle (#4528): it verifies
// the engine honored the case's termination contract against the reference —
// the truncation cap (Params.MaxTokens), the hard stop token / EOS sentinel,
// the case's stop strings (Rubric.Forbidden entries, which should have halted
// generation before they reached the assembled text), and, when no earlier stop
// fired, termination at the same step as the reference. A decode that keeps
// talking past its stop is a stop-semantics defect even when every emitted
// token individually matches greedy truth, so this gate is separate from (and
// complementary to) greedy-token-diff.
type StopTruncation struct{}

func (StopTruncation) Name() string { return "stop-truncation" }
func (StopTruncation) Kind() string { return "differential" }

func init() { Register(StopTruncation{}) }

func (StopTruncation) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "stop-truncation", Kind: "differential", Pass: true}

	// Rules 1–2: indexed termination violations. Over-generation past the first
	// EOS sentinel fails at the first offending token; over-length past the
	// truncation cap fails at index MaxTokens. When both offend, the earlier
	// index is the first divergence — localization always names the first step
	// at which the engine should already have been silent.
	overEOS := -1
	if j := stopTruncFirstEOS(eng.Tokens); j >= 0 && len(eng.Tokens) > j+1 {
		overEOS = j + 1
	}
	overCap := -1
	if limit := c.Params.MaxTokens; limit > 0 && len(eng.Tokens) > limit {
		overCap = limit
	}
	switch {
	case overEOS >= 0 && (overCap < 0 || overEOS <= overCap):
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: overEOS, Reference: tokenAt(ref.Tokens, overEOS), Engine: eng.Tokens[overEOS]}
		v.Detail = fmt.Sprintf("engine emitted %q at token %d after the hard stop %q at token %d",
			eng.Tokens[overEOS], overEOS, StopTruncationEOS, overEOS-1)
		return v
	case overCap >= 0:
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: overCap, Reference: tokenAt(ref.Tokens, overCap), Engine: eng.Tokens[overCap]}
		v.Detail = fmt.Sprintf("engine emitted %d tokens, exceeding the truncation cap max_tokens=%d",
			len(eng.Tokens), c.Params.MaxTokens)
		return v
	}

	// Rule 3: stop strings. Each Rubric.Forbidden entry is a hard stop string —
	// a faithful engine halts before one reaches the assembled text, so its
	// presence means the stop was ignored.
	text := strings.ToLower(eng.Text)
	for _, s := range c.Rubric.Forbidden {
		if s != "" && strings.Contains(text, strings.ToLower(s)) {
			v.Pass = false
			v.Detail = fmt.Sprintf("stop string %q present in engine text; generation should have halted before emitting it", s)
			return v
		}
	}

	// Rule 4: consistency with the reference. When no earlier stop fired, the
	// engine must terminate at the same step k the reference did. An engine that
	// stopped short is excused only by a stop of its own: its trace ends on the
	// EOS sentinel, or it ran exactly to the truncation cap.
	if len(eng.Tokens) != len(ref.Tokens) {
		if len(eng.Tokens) < len(ref.Tokens) && stopTruncEarlierStopFired(eng.Tokens, c.Params.MaxTokens) {
			v.Detail = fmt.Sprintf("engine terminated early at token %d on a valid stop (reference ran to %d)",
				len(eng.Tokens), len(ref.Tokens))
			return v
		}
		n := len(ref.Tokens)
		if len(eng.Tokens) < n {
			n = len(eng.Tokens)
		}
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("termination step diverged: reference stopped at %d, engine at %d with no earlier stop fired",
			len(ref.Tokens), len(eng.Tokens))
		return v
	}

	v.Detail = fmt.Sprintf("stop semantics honored: %d tokens within max_tokens=%d, nothing after %s, no stop string emitted",
		len(eng.Tokens), c.Params.MaxTokens, StopTruncationEOS)
	return v
}

// stopTruncFirstEOS returns the index of the first EOS sentinel in toks, or -1.
func stopTruncFirstEOS(toks []string) int {
	for i, t := range toks {
		if t == StopTruncationEOS {
			return i
		}
	}
	return -1
}

// stopTruncEarlierStopFired reports whether a shorter-than-reference engine trace
// is excused by an earlier stop of its own: it ended on the EOS sentinel, or it
// ran exactly to the truncation cap.
func stopTruncEarlierStopFired(toks []string, maxTokens int) bool {
	if n := len(toks); n > 0 && toks[n-1] == StopTruncationEOS {
		return true
	}
	return maxTokens > 0 && len(toks) == maxTokens
}
