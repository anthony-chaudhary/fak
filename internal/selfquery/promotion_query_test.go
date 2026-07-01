package selfquery

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/memq"
)

// TestAmbiguousConsentProducesPromotionClarification is the #1596 fixture: a
// candidate whose consent signal normalizes to memq.ConsentUnknown must not be
// silently dropped or silently promoted — it must produce an actual queryable
// ClarificationQuestion, reusing selfquery's existing broker shape unchanged.
func TestAmbiguousConsentProducesPromotionClarification(t *testing.T) {
	cand := PromotionCandidate{
		CellID:  "cell:7",
		Text:    "I think I prefer dark mode",
		Consent: "", // no consent signal recorded at all -> NormConsent == ConsentUnknown
		SourceSpan: memq.SourceSpan{
			Step: 3, Role: "user", Descriptor: "user: I think I prefer dark mode",
		},
	}
	if !cand.IsAmbiguous() {
		t.Fatal("a candidate with no recorded consent signal must be ambiguous")
	}
	q, ok := PromotionClarification(cand)
	if !ok {
		t.Fatal("PromotionClarification should return ok=true for an ambiguous consent candidate")
	}
	if q.Key != "cell:7" {
		t.Errorf("Key = %q, want the candidate's CellID", q.Key)
	}
	if q.Reason != ClarificationAmbiguousPromotion {
		t.Errorf("Reason = %q, want %q", q.Reason, ClarificationAmbiguousPromotion)
	}
	if q.DefaultChoice != PromotionChoiceDrop {
		t.Errorf("DefaultChoice = %q, want %q (fail closed when unanswered)", q.DefaultChoice, PromotionChoiceDrop)
	}
	if len(q.Choices) != 2 {
		t.Fatalf("Choices = %+v, want exactly the two live resolutions", q.Choices)
	}
	var sawPromote, sawDrop bool
	for _, c := range q.Choices {
		if c.Value == PromotionChoicePromoteExplicit {
			sawPromote = true
		}
		if c.Value == PromotionChoiceDrop {
			sawDrop = true
		}
	}
	if !sawPromote || !sawDrop {
		t.Fatalf("Choices = %+v, want both promote-explicit and drop", q.Choices)
	}
	if q.BudgetTokens <= 0 {
		t.Error("BudgetTokens must be populated, same as every other clarification question")
	}
	if !strings.Contains(q.SourceRef, "step 3") || !strings.Contains(q.SourceRef, "user") {
		t.Errorf("SourceRef = %q, want it to carry the candidate's source span", q.SourceRef)
	}
	if !strings.Contains(q.Question, "dark mode") {
		t.Errorf("Question = %q, want it to name the candidate text", q.Question)
	}
}

// TestAmbiguousDispositionProducesPromotionClarification pins the #1598 integration:
// a disposition verdict of OutcomeRefusedInsufficientCorroboration — evidence was
// offered but fell short, a genuinely unresolved judgment call — must also route
// through the clarification broker, distinct in wording from the bare-consent case.
func TestAmbiguousDispositionProducesPromotionClarification(t *testing.T) {
	obs := ctxmmu.Observation{Text: "I am tired today", Source: "user"}
	outcome := ctxmmu.GateDisposition(obs, ctxmmu.Disposition{Trait: "user prefers short answers"}, ctxmmu.Evidence{
		Kind:          ctxmmu.EvidenceCorroboration,
		Corroborating: []ctxmmu.Observation{{Text: "I am wiped out"}},
	})
	if outcome.Kind != ctxmmu.OutcomeRefusedInsufficientCorroboration {
		t.Fatalf("test setup: outcome.Kind = %v, want OutcomeRefusedInsufficientCorroboration", outcome.Kind)
	}
	cand := PromotionCandidate{
		Text:           "user prefers short answers",
		Disposition:    outcome,
		HasDisposition: true,
	}
	if !cand.IsAmbiguous() {
		t.Fatal("insufficient corroboration must be ambiguous, not a clean refuse")
	}
	q, ok := PromotionClarification(cand)
	if !ok {
		t.Fatal("PromotionClarification should return ok=true for insufficient-corroboration candidates")
	}
	if q.Reason != ClarificationAmbiguousPromotion {
		t.Errorf("Reason = %q, want %q", q.Reason, ClarificationAmbiguousPromotion)
	}
	if !strings.Contains(q.Question, "corroboration") {
		t.Errorf("Question = %q, want it to mention corroboration for the disposition case", q.Question)
	}
	// The candidate has no CellID, so the key must fall back to a deterministic digest
	// of the text rather than being empty (a caller must be able to key on this).
	if strings.TrimSpace(q.Key) == "" {
		t.Error("Key must never be empty even when CellID is unset")
	}
}

// TestCleanVerdictsDoNotProduceAPromotionClarification is the budget-discipline half
// of #1596: a candidate that already resolved cleanly — minted, explicit consent, or a
// bare zero-evidence refusal with no corroboration at all — must NOT be turned into a
// question. Only a genuinely unresolved verdict should spend clarification budget;
// otherwise every one-off refused remark would flood the broker.
func TestCleanVerdictsDoNotProduceAPromotionClarification(t *testing.T) {
	minted := ctxmmu.GateDisposition(
		ctxmmu.Observation{Text: "I always want brief answers"},
		ctxmmu.Disposition{Trait: "user prefers short answers"},
		ctxmmu.Evidence{Kind: ctxmmu.EvidenceUserConfirmed, Confirmed: "yes, always keep it brief"},
	)
	cases := []struct {
		name string
		cand PromotionCandidate
	}{
		{"explicit_consent", PromotionCandidate{Text: "I prefer concise answers", Consent: memq.ConsentExplicit}},
		{"inferred_consent", PromotionCandidate{Text: "tier: gold", Consent: memq.ConsentInferred}},
		{"minted_disposition", PromotionCandidate{Text: "user prefers short answers", Disposition: minted, HasDisposition: true}},
		{
			"unsupported_disposition",
			PromotionCandidate{
				Text: "user prefers short answers",
				Disposition: ctxmmu.GateDisposition(
					ctxmmu.Observation{Text: "I am tired today"},
					ctxmmu.Disposition{Trait: "user prefers short answers"},
					ctxmmu.Evidence{},
				),
				HasDisposition: true,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.cand.IsAmbiguous() {
				t.Fatalf("%s: candidate must not be ambiguous", tc.name)
			}
			if _, ok := PromotionClarification(tc.cand); ok {
				t.Fatalf("%s: PromotionClarification should return ok=false for a clean verdict", tc.name)
			}
		})
	}
}

// TestPromotionClarificationDefaultChoiceAlwaysDrop pins the fail-closed contract:
// regardless of which ambiguous path produced the question, an unanswered
// clarification must resolve to dropping the candidate, never to promoting it.
func TestPromotionClarificationDefaultChoiceAlwaysDrop(t *testing.T) {
	byConsent, ok := PromotionClarification(PromotionCandidate{Text: "maybe remember this", Consent: "bogus"})
	if !ok {
		t.Fatal("unrecognized consent string normalizes to ConsentUnknown and must be ambiguous")
	}
	byDisposition, ok := PromotionClarification(PromotionCandidate{
		Text: "user prefers short answers",
		Disposition: ctxmmu.DispositionOutcome{
			Kind: ctxmmu.OutcomeRefusedInsufficientCorroboration,
		},
		HasDisposition: true,
	})
	if !ok {
		t.Fatal("insufficient-corroboration candidate must be ambiguous")
	}
	for _, q := range []ClarificationQuestion{byConsent, byDisposition} {
		if q.DefaultChoice != PromotionChoiceDrop {
			t.Errorf("DefaultChoice = %q, want %q", q.DefaultChoice, PromotionChoiceDrop)
		}
	}
}
