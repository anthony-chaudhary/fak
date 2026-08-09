package grammar

import (
	"errors"
	"fmt"
	"math"
)

// Matcher is the mutable token-position state for a compiled call grammar.
// It deliberately owns only generated token IDs; the compiled CallMask is
// immutable and may be shared by every speculative fork.
type Matcher struct {
	mask    *CallMask
	history []int
}

// NewMatcher starts a matcher at the grammar's initial state.
func NewMatcher(mask *CallMask) (*Matcher, error) {
	if mask == nil {
		return nil, errors.New("grammar: matcher needs a compiled mask")
	}
	return &Matcher{mask: mask}, nil
}

// Fork returns an independent matcher at exactly the same grammar position.
// Advancing the fork can never advance its parent (or another fork).
func (m *Matcher) Fork() *Matcher {
	if m == nil {
		return nil
	}
	return &Matcher{mask: m.mask, history: append([]int(nil), m.history...)}
}

// History returns a defensive copy of the accepted token IDs.
func (m *Matcher) History() []int {
	if m == nil {
		return nil
	}
	return append([]int(nil), m.history...)
}

// AdvanceByAccepted commits exactly the target-verified tokens to this
// matcher. The operation is transactional: if any token violates the grammar,
// no token in the batch is committed. Draft guesses and rejected suffixes must
// be advanced only on Forks and omitted from this call.
func (m *Matcher) AdvanceByAccepted(tokens []int) error {
	if m == nil || m.mask == nil {
		return errors.New("grammar: nil matcher")
	}
	candidate := append([]int(nil), m.history...)
	for i, token := range tokens {
		if token < 0 {
			return fmt.Errorf("grammar: accepted token %d has negative id %d", i, token)
		}
		logits := make([]float32, token+1)
		m.mask.MaskLogits(candidate, logits)
		if math.IsInf(float64(logits[token]), -1) {
			return fmt.Errorf("grammar: accepted token %d (id %d) violates the grammar", i, token)
		}
		candidate = append(candidate, token)
	}
	m.history = candidate
	return nil
}
