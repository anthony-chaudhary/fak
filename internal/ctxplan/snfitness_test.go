package ctxplan

import "testing"

// span is a tiny fixture helper: Bytes=40 ⇒ TokenCost ceil(40/4)=10, so a Budget of 10 fits
// exactly one span and the planner's mis-ranking is observable as a single keep/elide flip.
func snSpan(id, desc string, step int) Span {
	return Span{ID: id, Descriptor: desc, Step: step, Bytes: 40, Durability: DurabilityTurn}
}

// TestWitnessedSNFitnessLearnedBeatsLexical is acceptance #1 on a controlled session: the forecast
// TRAINED on attention-reward (LearnFromAttention) scores a HIGHER witnessed-S/N fitness than the
// lexical-overlap baseline. The baseline forecast predicts the idle spans (lexical guess) and
// elides the one the model actually attends to; after learning promotes the faulted-but-attended
// span into the intents, the candidate keeps it resident and the witnessed S/N rises. The strict
// gain is exactly the keep-direction the rsiloop KEEP gate (shipgate.Evaluate) fires on.
func TestWitnessedSNFitnessLearnedBeatsLexical(t *testing.T) {
	// span:a is the one the model attends to (mass 0.9); span:b / span:c idle (mass 0).
	spans := []Span{
		snSpan("span:c", "gamma grape", 1),
		snSpan("span:b", "beta banana", 2),
		snSpan("span:a", "alpha apple", 3),
	}
	budget := Budget{Tokens: 10} // fits exactly one span
	attribution := Attribution{"span:a": 0.9, "span:b": 0.0, "span:c": 0.0}
	faults := []string{"span:a"} // the baseline elided span:a; the turn demand-paged it back

	// Lexical baseline: intents predict the idle spans, NOT the attended one.
	baseline := Forecast{Intents: []string{"beta", "gamma"}, Horizon: 1}
	session := []Turn{{Spans: spans, Budget: budget, Attribution: attribution, Faults: faults}}

	// Train on the witnessed outcome of the baseline's own plan (the loop #858 closed).
	basePlan := PlanCells(spans, baseline, budget, nil)
	maxStep := 3
	lf, lw := LearnFromAttention(baseline, baseline.Weights, basePlan, spans, attribution, faults, DefaultHitThreshold, maxStep)
	lf.Weights = lw // fold the tuned weights into the candidate forecast
	learned := lf

	baseFit := WitnessedSNFitness(baseline, session)
	learnedFit := WitnessedSNFitness(learned, session)

	if !(learnedFit > baseFit) {
		t.Fatalf("trained forecast did not beat lexical baseline: learned=%.4f base=%.4f", learnedFit, baseFit)
	}
	// Sanity on the magnitudes: the baseline keeps an idle span (ratio 0, plus a fault discount),
	// the learned forecast keeps the attended span (≈0.9·10=9 of 10 tokens signal, no fault).
	if baseFit != 0 {
		t.Fatalf("baseline kept an idle span; want fitness 0, got %.4f", baseFit)
	}
	if learnedFit < 0.8 {
		t.Fatalf("learned forecast kept the attended span; want fitness >= 0.8, got %.4f", learnedFit)
	}
}

// TestWitnessedSNFitnessFaultDiscountBites proves the fitness is NON-GAMEABLE by dropping needed
// spans: two sessions identical except one marks the elided span as faulted (demand-paged back).
// The faulted one scores STRICTLY lower, because the under-resident fault discount cancels the
// share of the high Ratio that was bought by starving the window — so "raise S/N" cannot be
// gamed by eliding a span the turn then needs (it just moves cost onto the fault axis). This is
// the property that lets the rsiloop KEEP gate fire only on a REAL S/N gain.
func TestWitnessedSNFitnessFaultDiscountBites(t *testing.T) {
	spans := []Span{
		snSpan("span:a", "alpha apple", 2), // attended, kept resident
		snSpan("span:d", "delta", 1),       // low relevance ⇒ elided
	}
	budget := Budget{Tokens: 10} // fits one span; the forecast keeps span:a
	attribution := Attribution{"span:a": 0.9}
	f := Forecast{Intents: []string{"alpha"}, Horizon: 1}

	noFault := WitnessedSNFitness(f, []Turn{{Spans: spans, Budget: budget, Attribution: attribution}})
	withFault := WitnessedSNFitness(f, []Turn{{Spans: spans, Budget: budget, Attribution: attribution, Faults: []string{"span:d"}}})

	if !(withFault < noFault) {
		t.Fatalf("fault discount did not bite: withFault=%.4f noFault=%.4f", withFault, noFault)
	}
	// Concretely: ratio 9/10=0.9; the fault moves 10 tokens onto the fault axis ⇒ FaultRatio
	// 10/20=0.5 ⇒ fitness 0.9·(1-0.5)=0.45 vs 0.9 with no fault.
	if noFault < 0.85 {
		t.Fatalf("no-fault fitness should be ~0.9, got %.4f", noFault)
	}
	if withFault > 0.5 {
		t.Fatalf("faulted fitness should be discounted to ~0.45, got %.4f", withFault)
	}
}

// TestWitnessedSNFitnessRangeAndEmpty pins the bounds and the empty-window convention: every
// fitness is in [0,1], and an empty session returns 1.0 (no turns, no noise — the same
// fail-to-best posture SignalNoise.Ratio takes on an empty resident view).
func TestWitnessedSNFitnessRangeAndEmpty(t *testing.T) {
	if got := WitnessedSNFitness(Forecast{Intents: []string{"x"}}, nil); got != 1.0 {
		t.Fatalf("empty session: want 1.0, got %.4f", got)
	}
	spans := []Span{snSpan("span:a", "alpha apple", 1)}
	got := WitnessedSNFitness(
		Forecast{Intents: []string{"alpha"}, Horizon: 1},
		[]Turn{{Spans: spans, Budget: Budget{Tokens: 10}, Attribution: Attribution{"span:a": 0.5}}},
	)
	if got < 0 || got > 1 {
		t.Fatalf("fitness out of [0,1]: %.4f", got)
	}
}

// TestWitnessedSNFitnessDeterministic proves the replay is a GATE: the same (forecast, session)
// yields a byte-identical fitness, so a re-run that differs is a real regression, not noise — the
// determinism a non-forgeable RSI keep-bit (and its journal trend) rests on.
func TestWitnessedSNFitnessDeterministic(t *testing.T) {
	spans := []Span{
		snSpan("span:a", "alpha apple", 3),
		snSpan("span:b", "beta banana", 2),
		snSpan("span:c", "gamma grape", 1),
	}
	session := []Turn{{
		Spans:       spans,
		Budget:      Budget{Tokens: 20},
		Attribution: Attribution{"span:a": 0.9, "span:b": 0.1, "span:c": 0.0},
		Faults:      []string{"span:c"},
	}}
	f := Forecast{Intents: []string{"alpha", "beta"}, Horizon: 1}
	first := WitnessedSNFitness(f, session)
	for i := 0; i < 5; i++ {
		if got := WitnessedSNFitness(f, session); got != first {
			t.Fatalf("non-deterministic fitness: run %d got %.17g, want %.17g", i, got, first)
		}
	}
}
