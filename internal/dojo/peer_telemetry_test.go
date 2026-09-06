package dojo

import (
	"testing"
)

func TestDojo_PeerContextCalibrationAndConversionFunnel(t *testing.T) {
	// 1. Clean run: Level0=100, Level1=40, Level2=10, AvoidedToolTokens=5000, PeerQueryTokens=2000
	cleanLedger := PeerSearchTelemetryLedger{
		Level0Queries:     100,
		Level1Queries:     40,
		Level2Queries:     10,
		AvoidedToolTokens: 5000,
		PeerQueryTokens:   2000,
		TaintLeaks:        0,
		Recorded:          true,
	}

	// Verify computed conversion rates and token savings
	wantCR01 := 0.40      // 40 / 100
	wantCR12 := 0.25      // 10 / 40
	wantNetTokens := 3000 // 5000 - 2000
	wantRatio := 0.60     // 3000 / 5000

	if cr01 := cleanLedger.ConversionRate0to1(); cr01 != wantCR01 {
		t.Errorf("ConversionRate0to1 = %f, want %f", cr01, wantCR01)
	}
	if cr12 := cleanLedger.ConversionRate1to2(); cr12 != wantCR12 {
		t.Errorf("ConversionRate1to2 = %f, want %f", cr12, wantCR12)
	}
	if net := cleanLedger.NetTokensSaved(); net != wantNetTokens {
		t.Errorf("NetTokensSaved = %d, want %d", net, wantNetTokens)
	}
	if ratio := cleanLedger.TokensSavedRatio(); ratio != wantRatio {
		t.Errorf("TokensSavedRatio = %f, want %f", ratio, wantRatio)
	}

	cleanInputs := PeerSearchEpisodes(cleanLedger)
	if len(cleanInputs) != 4 {
		t.Fatalf("expected 4 episodes, got %d", len(cleanInputs))
	}

	if !cleanInputs[0].Outcome.Measured || cleanInputs[0].Outcome.Realized != wantCR01 {
		t.Errorf("episode 0 (cr_0_to_1) realized = %f, want %f", cleanInputs[0].Outcome.Realized, wantCR01)
	}
	if !cleanInputs[1].Outcome.Measured || cleanInputs[1].Outcome.Realized != wantCR12 {
		t.Errorf("episode 1 (cr_1_to_2) realized = %f, want %f", cleanInputs[1].Outcome.Realized, wantCR12)
	}
	if !cleanInputs[2].Outcome.Measured || cleanInputs[2].Outcome.Realized != wantRatio {
		t.Errorf("episode 2 (tokens_saved_ratio) realized = %f, want %f", cleanInputs[2].Outcome.Realized, wantRatio)
	}
	if !cleanInputs[3].Outcome.Measured || cleanInputs[3].Outcome.Realized != 0.0 {
		t.Errorf("episode 3 (taint_leak_rate) realized = %f, want 0.0", cleanInputs[3].Outcome.Realized)
	}

	var cleanEpisodes []Episode
	for _, in := range cleanInputs {
		cleanEpisodes = append(cleanEpisodes, Score("peer-search-clean", in.Prediction, in.Outcome, DefaultCalibBand()))
	}
	cleanFold := FoldCalibrable(cleanEpisodes)
	if cleanFold.FloorBreachErr != 0.0 {
		t.Errorf("cleanFold.FloorBreachErr = %f, want 0.0 (floor holds)", cleanFold.FloorBreachErr)
	}

	// 2. Feeds a ledger with TaintLeaks > 0 and verifies FoldCalibrable or episode penalizes the breach.
	leakLedger := PeerSearchTelemetryLedger{
		Level0Queries:     100,
		Level1Queries:     40,
		Level2Queries:     10,
		AvoidedToolTokens: 5000,
		PeerQueryTokens:   2000,
		TaintLeaks:        2,
		Recorded:          true,
	}

	leakInputs := PeerSearchEpisodes(leakLedger)
	if len(leakInputs) != 4 {
		t.Fatalf("expected 4 episodes, got %d", len(leakInputs))
	}

	// Verify token savings outcome is penalized to failure (0.0)
	if leakInputs[2].Outcome.Realized > 0.0 {
		t.Errorf("expected token savings outcome penalized when taint leaks > 0, got %f", leakInputs[2].Outcome.Realized)
	}

	var leakEpisodes []Episode
	for _, in := range leakInputs {
		leakEpisodes = append(leakEpisodes, Score("peer-search-leak", in.Prediction, in.Outcome, DefaultCalibBand()))
	}

	// Verify episode penalizes the breach (taint floor episode has FloorRespectErr > 0 and OVER_CLAIM)
	taintEp := leakEpisodes[3]
	if err := FloorRespectErr(taintEp); err <= 0.0 {
		t.Errorf("expected FloorRespectErr > 0 for taint breach, got %f", err)
	}
	if taintEp.Verdict != VerdictOverClaim {
		t.Errorf("expected VerdictOverClaim for breached floor, got %s", taintEp.Verdict)
	}

	// Verify FoldCalibrable penalizes the breach
	leakFold := FoldCalibrable(leakEpisodes)
	if leakFold.FloorBreachErr <= 0.0 {
		t.Errorf("expected leakFold.FloorBreachErr > 0, got %f", leakFold.FloorBreachErr)
	}
	if leakFold.Value <= cleanFold.Value {
		t.Errorf("expected leakFold.Value (%f) > cleanFold.Value (%f)", leakFold.Value, cleanFold.Value)
	}
}

func TestPeerSearchEpisodes_Unrecorded(t *testing.T) {
	unrec := PeerSearchTelemetryLedger{Recorded: false}
	inputs := PeerSearchEpisodes(unrec)
	if len(inputs) != 4 {
		t.Fatalf("expected 4 inputs, got %d", len(inputs))
	}
	for i, in := range inputs {
		if in.Outcome.Measured {
			t.Errorf("input %d expected unmeasured, got measured", i)
		}
	}
}

func TestPeerSearchEpisodes_ClampingAndZeroQueries(t *testing.T) {
	// Zero queries
	zeroLedger := PeerSearchTelemetryLedger{
		Level0Queries:     0,
		Level1Queries:     0,
		Level2Queries:     0,
		AvoidedToolTokens: 0,
		Recorded:          true,
	}
	inputs := PeerSearchEpisodes(zeroLedger)
	if inputs[0].Outcome.Measured {
		t.Errorf("expected CR_0_to_1 unmeasured when Level0Queries=0")
	}
	if inputs[1].Outcome.Measured {
		t.Errorf("expected CR_1_to_2 unmeasured when Level1Queries=0")
	}
	if inputs[2].Outcome.Measured {
		t.Errorf("expected tokens_saved_ratio unmeasured when AvoidedToolTokens=0")
	}

	// Clamping: Level1 > Level0, Level2 > Level1
	overLedger := PeerSearchTelemetryLedger{
		Level0Queries:     10,
		Level1Queries:     50,
		Level2Queries:     100,
		AvoidedToolTokens: 1000,
		PeerQueryTokens:   100,
		Recorded:          true,
	}
	if cr01 := overLedger.ConversionRate0to1(); cr01 != 1.0 {
		t.Errorf("ConversionRate0to1 clamped = %f, want 1.0", cr01)
	}
	if cr12 := overLedger.ConversionRate1to2(); cr12 != 1.0 {
		t.Errorf("ConversionRate1to2 clamped = %f, want 1.0", cr12)
	}
}
