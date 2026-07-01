// This file is the missing link the managed-context epic calls for (#1596): when a
// context->memory promotion candidate comes back AMBIGUOUS from the two existing gates
// — memq.NormConsent normalizing to ConsentUnknown (#1595), or ctxmmu.GateDisposition
// refusing as OutcomeRefusedInsufficientCorroboration (#1598) — the caller today has
// exactly two silent options: drop the fact (losing something the user may actually
// want remembered) or promote it anyway (the false-positive-durable-belief failure
// CONTEXT-IS-NOT-MEMORY.md warns is "strictly worse than absence"). Neither is
// acceptable for an ambiguous case: ambiguous means the gate genuinely could not decide,
// not that the answer is "no."
//
// PromotionClarification closes that gap by routing an ambiguous candidate through
// selfquery's EXISTING clarification broker (clarification.go, #1615) instead of
// inventing a second clarification-request shape: it returns a ClarificationQuestion
// whose Choices are exactly the two live options ("promote as explicit-consent durable"
// or "drop it") and whose DefaultChoice is the fail-closed one (drop) so an unattended
// caller that ignores the question still lands on the safe side. This is a pure
// function — no I/O, no model call, deterministic on its inputs — the same posture as
// GateDisposition and GateEphemeral it wraps.
package selfquery

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/memq"
)

// ClarificationAmbiguousPromotion is the reason class for a context->memory promotion
// candidate whose consent or disposition verdict came back ambiguous rather than a
// clean allow/refuse — the closed-vocabulary counterpart to ClarificationMissingContext
// /ClarificationLowConfidence/ClarificationStaleContext, added for #1596.
const ClarificationAmbiguousPromotion ClarificationReason = "ambiguous_promotion"

// Promotion choice values a caller resolves a PromotionClarification question to.
// These are the only two live outcomes for an ambiguous candidate — mirrors
// ctxmmu.Reclassification's closed shape at the answer end rather than a free string.
const (
	// PromotionChoicePromoteExplicit resolves the ambiguity by promoting the candidate
	// with ConsentExplicit — the caller (user) affirmatively said to keep it.
	PromotionChoicePromoteExplicit = "promote_explicit_consent"
	// PromotionChoiceDrop resolves the ambiguity by dropping the candidate — the
	// fail-closed default when nobody answers the question.
	PromotionChoiceDrop = "drop"
)

// PromotionCandidate is the ambiguous context->memory promotion the caller wants a
// clarification for: the raw text under consideration, its source span (memq's own
// SourceSpan — the same safe, extractive pointer PromotionRecord.SourceSpan carries, so
// this candidate can be explained/audited with the vocabulary #1595 already
// established), and the verdict(s) that came back ambiguous.
type PromotionCandidate struct {
	// CellID is the address of the candidate cell, if one already exists (memq.Cell.ID)
	// — empty when the candidate has not been written yet and is still being decided.
	CellID string
	// Text is the raw candidate fact, verbatim — never sealed bytes (same posture as
	// memq.Cell.Descriptor / ctxmmu.Observation.Text).
	Text string
	// SourceSpan is where the candidate came from in the live turn — reused verbatim
	// from memq so a resulting promotion (if the caller chooses to promote) can mint a
	// memq.PromotionRecord without re-deriving this data.
	SourceSpan memq.SourceSpan
	// Consent is the raw consent signal recorded for this candidate, BEFORE
	// normalization (may be empty, may be a recognized or unrecognized string) — the
	// same input memq.NormConsent takes. Only meaningful when Disposition is the zero
	// value; a caller with a real ctxmmu.DispositionOutcome should populate Disposition
	// instead, since that verdict is a stronger signal than a bare consent string.
	Consent string
	// Disposition is the ctxmmu.GateDisposition verdict for this candidate, when the
	// caller already ran that gate (e.g. a consolidation step trying to generalize a
	// single observation into a standing trait). The zero value (Kind ==
	// OutcomeRefusedUnsupported) is a valid "no disposition verdict was computed"
	// input — IsAmbiguous branches on Consent alone in that case.
	Disposition ctxmmu.DispositionOutcome
	// HasDisposition reports whether Disposition was actually computed by the caller
	// (distinguishes "no verdict was run" from "the verdict's Kind happens to be the
	// zero value, OutcomeRefusedUnsupported").
	HasDisposition bool
}

// IsAmbiguous reports whether a PromotionCandidate is the ambiguous case #1596 exists
// for, as opposed to a candidate whose gate(s) already resolved cleanly:
//
//   - Disposition (when HasDisposition): OutcomeRefusedInsufficientCorroboration is
//     ambiguous (evidence was offered, just not enough — a human can settle it in one
//     answer). OutcomeMinted is NOT ambiguous (already allowed). Plain
//     OutcomeRefusedUnsupported (zero evidence at all) is NOT the case this issue
//     targets either — that is a clean refuse, not an unresolved judgment call; asking
//     about EVERY unsupported one-off remark would defeat the point of a budgeted
//     clarification broker (clarification.go's own MaxQuestions/MaxBudgetTokens gate).
//   - Consent (when no Disposition was supplied): ConsentUnknown is ambiguous — no
//     consent signal was recorded at all, which is exactly the "system can't tell"
//     case. ConsentExplicit/ConsentInferred already resolved cleanly.
func (p PromotionCandidate) IsAmbiguous() bool {
	if p.HasDisposition {
		return p.Disposition.Kind == ctxmmu.OutcomeRefusedInsufficientCorroboration
	}
	return memq.NormConsent(p.Consent) == memq.ConsentUnknown
}

// PromotionClarification is the pure decision function #1596 asks for: given an
// ambiguous PromotionCandidate, it returns the ClarificationQuestion that asks the
// user/caller to explicitly resolve it, reusing selfquery's existing clarification
// broker shape unchanged (ClarificationQuestion/ClarificationChoice from
// clarification.go) rather than inventing a second request type. ok is false when the
// candidate is not actually ambiguous (IsAmbiguous() == false) — a clean allow or a
// clean zero-evidence refuse never becomes a query, so this function cannot be used to
// route EVERY refusal into a question (that would just move the silent-drop failure
// into the clarification budget instead of removing it).
//
// The returned question's DefaultChoice is always PromotionChoiceDrop: an unattended
// caller that surfaces the question but never gets an answer back still fails closed,
// exactly like every other gate in this package (classifyDurability's `default: turn`,
// GateDisposition's default refuse, GateEphemeral's default refuse).
func PromotionClarification(cand PromotionCandidate) (ClarificationQuestion, bool) {
	if !cand.IsAmbiguous() {
		return ClarificationQuestion{}, false
	}
	key := cand.CellID
	if strings.TrimSpace(key) == "" {
		key = "promotion:" + memq.Digest([]byte(strings.TrimSpace(cand.Text)))
	}
	q := ClarificationQuestion{
		Key:      key,
		Question: promotionClarificationText(cand),
		Reason:   ClarificationAmbiguousPromotion,
		Choices: []ClarificationChoice{
			{Value: PromotionChoicePromoteExplicit, Label: "Promote as explicit-consent durable memory"},
			{Value: PromotionChoiceDrop, Label: "Drop (do not remember this)"},
		},
		DefaultChoice: PromotionChoiceDrop,
		SourceRef:     promotionSourceRef(cand.SourceSpan),
	}
	q.BudgetTokens = estimateQuestionBudget(q)
	return q, true
}

// promotionClarificationText renders the fixed-template question body — string
// formatting over the candidate's own fields, never a model summarization call, same
// posture as clarificationText/PromotionLedger.Explain's Narrative elsewhere in the
// repo.
func promotionClarificationText(cand PromotionCandidate) string {
	subject := strings.TrimSpace(cand.Text)
	if subject == "" {
		subject = "this candidate fact"
	}
	if cand.HasDisposition && cand.Disposition.Kind == ctxmmu.OutcomeRefusedInsufficientCorroboration {
		return fmt.Sprintf("Not enough corroboration to remember %q as a standing preference — promote it anyway, or drop it?", subject)
	}
	return fmt.Sprintf("No consent signal was recorded for remembering %q — promote it as durable memory, or drop it?", subject)
}

// promotionSourceRef renders memq.SourceSpan into the same free-text SourceRef shape
// ClarificationQuestion already carries (clarificationQuestion sets it from
// ctxplan.AssumptionAssessment.SourceRef) — reusing the field rather than adding a new
// one for a second source-provenance shape.
func promotionSourceRef(span memq.SourceSpan) string {
	if span.Role == "" && span.Descriptor == "" && span.Step == 0 {
		return ""
	}
	if span.Descriptor != "" {
		return fmt.Sprintf("step %d (role=%s): %s", span.Step, span.Role, span.Descriptor)
	}
	return fmt.Sprintf("step %d (role=%s)", span.Step, span.Role)
}
