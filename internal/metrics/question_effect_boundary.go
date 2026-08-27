package metrics

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/snapshot"
)

// QuestionReceiptCause is the bounded, content-free reason a question receipt
// was refused at the answer-to-effect boundary.
type QuestionReceiptCause string

const (
	QuestionReceiptOmitted               QuestionReceiptCause = "receipt omitted"
	QuestionReceiptShapeInvalid          QuestionReceiptCause = "receipt shape invalid"
	QuestionReceiptOrStateInvalid        QuestionReceiptCause = "receipt or current state invalid"
	QuestionReceiptQuestionChanged       QuestionReceiptCause = "question_changed"
	QuestionReceiptEvidenceChanged       QuestionReceiptCause = "evidence_changed"
	QuestionReceiptGoverningInputChanged QuestionReceiptCause = "governing_input_changed"
	QuestionReceiptAuthorityChanged      QuestionReceiptCause = "authority_changed"
)

// QuestionReceiptRefusal identifies one refusal counter. It contains only the
// receipt refusal class and its bounded cause; question and answer contents
// never enter the aggregation key.
type QuestionReceiptRefusal struct {
	Refusal snapshot.ReceiptRefusal
	Cause   QuestionReceiptCause
}

// QuestionEffectBoundary is the common answer-to-effect gate for interactive
// and headless consumers. It is safe for concurrent use.
type QuestionEffectBoundary struct {
	mu       sync.Mutex
	refusals map[QuestionReceiptRefusal]uint64
}

// Apply parses and verifies receipt against the governing state at the point of
// effect. effect is invoked only for a valid, unchanged receipt. Callers may
// capture an answer in effect; the boundary neither accepts nor retains it.
func (b *QuestionEffectBoundary) Apply(receipt []byte, current snapshot.QuestionState, effect func()) snapshot.ReceiptCheck {
	parsed, check := snapshot.ParseQuestionReceipt(receipt)
	if !check.Accepted {
		b.count(check)
		return check
	}

	check = snapshot.VerifyQuestionReceipt(parsed, current)
	if !check.Accepted {
		b.count(check)
		return check
	}
	effect()
	return check
}

// RefusalCounts returns a detached snapshot of the content-free counters.
func (b *QuestionEffectBoundary) RefusalCounts() map[QuestionReceiptRefusal]uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()

	counts := make(map[QuestionReceiptRefusal]uint64, len(b.refusals))
	for refusal, count := range b.refusals {
		counts[refusal] = count
	}
	return counts
}

func (b *QuestionEffectBoundary) count(check snapshot.ReceiptCheck) {
	key := QuestionReceiptRefusal{Refusal: check.Refusal, Cause: QuestionReceiptCause(check.Cause)}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.refusals == nil {
		b.refusals = make(map[QuestionReceiptRefusal]uint64)
	}
	b.refusals[key]++
}
