package selfquery

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/memq"
)

// TestShouldAskThresholdsByStakesAndReversibility pins the three confidence bars the
// policy table asks #1580 for: irreversible/high-stakes decisions need the strictest
// bar, costly/medium ones the middle bar, and cheap/low ones the loosest bar.
func TestShouldAskThresholdsByStakesAndReversibility(t *testing.T) {
	cases := []struct {
		name          string
		confidence    float64
		stakes        Stakes
		reversibility Reversibility
		wantAsk       bool
		wantThreshold float64
	}{
		{"irreversible_below_bar_asks", 0.80, StakesLow, Irreversible, true, 0.90},
		{"irreversible_at_bar_assumes", 0.95, StakesLow, Irreversible, false, 0.90},
		{"high_stakes_below_bar_asks", 0.80, StakesHigh, ReversibleCheap, true, 0.90},
		{"costly_below_bar_asks", 0.50, StakesLow, ReversibleCostly, true, 0.65},
		{"costly_at_bar_assumes", 0.70, StakesLow, ReversibleCostly, false, 0.65},
		{"medium_stakes_below_bar_asks", 0.50, StakesMedium, ReversibleCheap, true, 0.65},
		{"cheap_low_stakes_low_confidence_still_assumes", 0.40, StakesLow, ReversibleCheap, false, 0.35},
		{"cheap_low_stakes_very_low_confidence_asks", 0.10, StakesLow, ReversibleCheap, true, 0.35},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ShouldAsk(AskInput{Confidence: tc.confidence, Stakes: tc.stakes, Reversibility: tc.reversibility})
			if got.ShouldAsk != tc.wantAsk {
				t.Errorf("ShouldAsk = %v, want %v (verdict=%+v)", got.ShouldAsk, tc.wantAsk, got)
			}
			if got.Threshold != tc.wantThreshold {
				t.Errorf("Threshold = %v, want %v", got.Threshold, tc.wantThreshold)
			}
			if got.Reason == "" {
				t.Error("Reason must never be empty")
			}
		})
	}
}

// TestShouldAskMoreSevereAxisWins pins that the higher of the two axes governs: a
// mild stakes/reversibility combination on one axis cannot pull the bar down when the
// other axis is severe.
func TestShouldAskMoreSevereAxisWins(t *testing.T) {
	highStakesCheapRev := ShouldAsk(AskInput{Confidence: 0.85, Stakes: StakesHigh, Reversibility: ReversibleCheap})
	if !highStakesCheapRev.ShouldAsk {
		t.Errorf("high stakes must demand the strict bar even with cheap reversibility: %+v", highStakesCheapRev)
	}
	irreversibleLowStakes := ShouldAsk(AskInput{Confidence: 0.85, Stakes: StakesLow, Reversibility: Irreversible})
	if !irreversibleLowStakes.ShouldAsk {
		t.Errorf("irreversible must demand the strict bar even with low stakes: %+v", irreversibleLowStakes)
	}
}

// TestShouldAskClampsConfidence pins that out-of-range confidence never produces a
// verdict computed from a value outside [0,1] (same defensive posture as
// ctxplan.normalizeAssumptionConfidence's implicit clamp).
func TestShouldAskClampsConfidence(t *testing.T) {
	tooHigh := ShouldAsk(AskInput{Confidence: 5, Stakes: StakesLow, Reversibility: ReversibleCheap})
	if tooHigh.ShouldAsk {
		t.Errorf("confidence above 1 must clamp to 1 and clear any bar: %+v", tooHigh)
	}
	tooLow := ShouldAsk(AskInput{Confidence: -5, Stakes: StakesHigh, Reversibility: Irreversible})
	if !tooLow.ShouldAsk {
		t.Errorf("confidence below 0 must clamp to 0 and never clear any bar: %+v", tooLow)
	}
}

// TestShouldAskAgreesWithCtxplanDefaultPolicy pins the "the two policies are meant to
// agree, not drift" contract stated in ask_policy.go's doc comment: the medium-stakes
// / costly-reversibility bar (0.65) matches ctxplan.DefaultAssumptionPolicy's own
// MinConfidence exactly, so ctxplan's existing thresholds are already a conforming
// instance of this general policy rather than a competing one.
func TestShouldAskAgreesWithCtxplanDefaultPolicy(t *testing.T) {
	got := ShouldAsk(AskInput{Confidence: 0, Stakes: StakesMedium, Reversibility: ReversibleCostly})
	want := ctxplan.DefaultAssumptionPolicy().MinConfidence
	if got.Threshold != want {
		t.Errorf("selfquery.ShouldAsk medium/costly threshold = %v, want ctxplan.DefaultAssumptionPolicy().MinConfidence = %v (policies must not drift apart)", got.Threshold, want)
	}
}

// TestPromotionCandidateIsAmbiguousConformsToShouldAsk pins #1580's "ground the
// general policy against a real conforming example" requirement using #1596's own
// PromotionCandidate.IsAmbiguous: an ambiguous promotion candidate is exactly the
// "confidence unknown" case, which under this policy's Costly/Medium-stakes bar
// (a promoted memory persists and would need to be actively discovered and undone to
// correct) must resolve to ShouldAsk == true — matching what PromotionClarification
// already does by routing it through the clarification broker instead of silently
// promoting or dropping it.
func TestPromotionCandidateIsAmbiguousConformsToShouldAsk(t *testing.T) {
	cand := PromotionCandidate{
		Text:    "I think I prefer dark mode",
		Consent: "", // NormConsent -> ConsentUnknown -> ambiguous
	}
	if !cand.IsAmbiguous() {
		t.Fatal("test setup: candidate must be ambiguous")
	}
	// An ambiguous promotion candidate carries no confidence signal at all -- the
	// same "unknown" case ctxplan.AssumptionUnknown maps to confidence 0.
	verdict := ShouldAsk(AskInput{Confidence: 0, Stakes: StakesMedium, Reversibility: ReversibleCostly})
	if !verdict.ShouldAsk {
		t.Fatalf("an ambiguous promotion candidate must resolve to ShouldAsk=true under the general policy, got %+v", verdict)
	}
	// And the concrete decider agrees: it actually produces a clarification question
	// rather than silently promoting or dropping.
	if _, ok := PromotionClarification(cand); !ok {
		t.Fatal("PromotionClarification must produce a question for an ambiguous candidate, consistent with ShouldAsk=true")
	}
}

// TestCleanPromotionConformsToShouldAsk pins the other half of the conformance check:
// a candidate that resolved cleanly (explicit consent, full confidence) must NOT ask
// under the general policy either, matching PromotionClarification's ok=false.
func TestCleanPromotionConformsToShouldAsk(t *testing.T) {
	cand := PromotionCandidate{Text: "I prefer concise answers", Consent: memq.ConsentExplicit}
	if cand.IsAmbiguous() {
		t.Fatal("test setup: candidate must not be ambiguous")
	}
	verdict := ShouldAsk(AskInput{Confidence: 1, Stakes: StakesMedium, Reversibility: ReversibleCostly})
	if verdict.ShouldAsk {
		t.Fatalf("a clean, fully-confident promotion must resolve to ShouldAsk=false, got %+v", verdict)
	}
	if _, ok := PromotionClarification(cand); ok {
		t.Fatal("PromotionClarification must not produce a question for a clean candidate, consistent with ShouldAsk=false")
	}
}

// TestInsufficientCorroborationConformsToShouldAsk exercises the ctxmmu disposition
// path (the third existing ask-vs-assume decision point named in ask_policy.go's doc
// comment): insufficient corroboration is a genuinely unresolved judgment call, which
// under Irreversible reversibility (a minted disposition, once acted on as a standing
// trait, is hard to walk back) must resolve to ShouldAsk == true.
func TestInsufficientCorroborationConformsToShouldAsk(t *testing.T) {
	outcome := ctxmmu.GateDisposition(
		ctxmmu.Observation{Text: "I am tired today", Source: "user"},
		ctxmmu.Disposition{Trait: "user prefers short answers"},
		ctxmmu.Evidence{
			Kind:          ctxmmu.EvidenceCorroboration,
			Corroborating: []ctxmmu.Observation{{Text: "I am wiped out"}},
		},
	)
	if outcome.Kind != ctxmmu.OutcomeRefusedInsufficientCorroboration {
		t.Fatalf("test setup: outcome.Kind = %v, want OutcomeRefusedInsufficientCorroboration", outcome.Kind)
	}
	verdict := ShouldAsk(AskInput{Confidence: 0.3, Stakes: StakesLow, Reversibility: Irreversible})
	if !verdict.ShouldAsk {
		t.Fatalf("insufficient corroboration for a durable trait must resolve to ShouldAsk=true, got %+v", verdict)
	}
}
