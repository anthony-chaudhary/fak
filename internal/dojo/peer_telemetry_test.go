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
	if len(cleanInputs) != 5 {
		t.Fatalf("expected 5 episodes, got %d", len(cleanInputs))
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
	if cleanInputs[3].Outcome.Sample != cleanLedger.Level0Queries {
		t.Errorf("episode 3 (taint_leak_rate) sample = %d, want %d", cleanInputs[3].Outcome.Sample, cleanLedger.Level0Queries)
	}
	if !cleanInputs[4].Outcome.Measured || cleanInputs[4].Outcome.Realized != 0.0 {
		t.Errorf("episode 4 (taint_leaks) realized = %f, want 0.0", cleanInputs[4].Outcome.Realized)
	}
	if cleanInputs[4].Outcome.Sample != cleanLedger.Level0Queries {
		t.Errorf("episode 4 (taint_leaks) sample = %d, want %d", cleanInputs[4].Outcome.Sample, cleanLedger.Level0Queries)
	}
	if cleanInputs[4].Prediction.Unit != "count" {
		t.Errorf("episode 4 (taint_leaks) unit = %q, want 'count'", cleanInputs[4].Prediction.Unit)
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
	if len(leakInputs) != 5 {
		t.Fatalf("expected 5 episodes, got %d", len(leakInputs))
	}

	// Verify token savings outcome is penalized to failure (0.0)
	if leakInputs[2].Outcome.Realized > 0.0 {
		t.Errorf("expected token savings outcome penalized when taint leaks > 0, got %f", leakInputs[2].Outcome.Realized)
	}

	// Verify sample size consistency for taint floors (both use Level0Queries denominator, not TaintLeaks)
	if leakInputs[3].Outcome.Sample != leakLedger.Level0Queries {
		t.Errorf("episode 3 (taint_leak_rate) sample = %d, want %d (Level0Queries)", leakInputs[3].Outcome.Sample, leakLedger.Level0Queries)
	}
	if leakInputs[4].Outcome.Sample != leakLedger.Level0Queries {
		t.Errorf("episode 4 (taint_leaks) sample = %d, want %d (Level0Queries)", leakInputs[4].Outcome.Sample, leakLedger.Level0Queries)
	}
	if leakInputs[4].Outcome.Realized != float64(leakLedger.TaintLeaks) {
		t.Errorf("episode 4 (taint_leaks) realized = %f, want %f", leakInputs[4].Outcome.Realized, float64(leakLedger.TaintLeaks))
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

	taintLeaksEp := leakEpisodes[4]
	if err := FloorRespectErr(taintLeaksEp); err <= 0.0 {
		t.Errorf("expected FloorRespectErr > 0 for taint_leaks breach, got %f", err)
	}
	if taintLeaksEp.Verdict != VerdictOverClaim {
		t.Errorf("expected VerdictOverClaim for breached taint_leaks floor, got %s", taintLeaksEp.Verdict)
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
	if len(inputs) != 5 {
		t.Fatalf("expected 5 inputs, got %d", len(inputs))
	}
	for i, in := range inputs {
		if in.Outcome.Measured {
			t.Errorf("input %d expected unmeasured, got measured", i)
		}
	}
	if inputs[4].Outcome.Sample != 0 {
		t.Errorf("input 4 (taint_leaks) sample = %d, want 0", inputs[4].Outcome.Sample)
	}
	if inputs[4].Outcome.Source != "peer search telemetry not recorded — taint_leaks is UNMEASURED" {
		t.Errorf("input 4 (taint_leaks) source = %q, want unmeasured message", inputs[4].Outcome.Source)
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

func TestPeerSearchEpisodes_AllRegisteredClaimsCovered(t *testing.T) {
	// Find all registered claims for lever "peer-search"
	registeredMetrics := make(map[string]Claim)
	for k, c := range Registry {
		if k.Lever == "peer-search" {
			registeredMetrics[k.Metric] = c
		}
	}
	for k, c := range registered {
		if k.Lever == "peer-search" {
			registeredMetrics[k.Metric] = c
		}
	}

	if len(registeredMetrics) == 0 {
		t.Fatal("no registered claims found for lever 'peer-search'")
	}

	// Verify that taint_leaks is specifically among the registered claims
	taintLeaksClaim, ok := registeredMetrics["taint_leaks"]
	if !ok {
		t.Fatal("claim 'taint_leaks' not found in registered claims for 'peer-search'")
	}
	if !taintLeaksClaim.IntentionalFloor {
		t.Error("expected taint_leaks to be registered as an intentional floor")
	}
	if !taintLeaksClaim.LowerIsBetter {
		t.Error("expected taint_leaks to have LowerIsBetter=true")
	}
	if taintLeaksClaim.Claimed != 0.0 {
		t.Errorf("expected taint_leaks Claimed=0.0, got %f", taintLeaksClaim.Claimed)
	}

	// Generate episodes across varied ledgers (clean, leak, unrecorded)
	ledgers := []PeerSearchTelemetryLedger{
		{
			Level0Queries:     50,
			Level1Queries:     20,
			Level2Queries:     5,
			AvoidedToolTokens: 1000,
			PeerQueryTokens:   200,
			TaintLeaks:        0,
			Recorded:          true,
		},
		{
			Level0Queries:     50,
			Level1Queries:     20,
			Level2Queries:     5,
			AvoidedToolTokens: 1000,
			PeerQueryTokens:   200,
			TaintLeaks:        3,
			Recorded:          true,
		},
		{
			Recorded: false,
		},
	}

	for _, led := range ledgers {
		episodes := PeerSearchEpisodes(led)
		emittedMetrics := make(map[string]ScoredInput)
		for _, ep := range episodes {
			if ep.Prediction.Lever != "peer-search" {
				t.Errorf("episode lever = %q, want 'peer-search'", ep.Prediction.Lever)
			}
			emittedMetrics[ep.Prediction.Metric] = ep
		}

		// Verify every registered claim has an evaluated episode in PeerSearchEpisodes
		for metric := range registeredMetrics {
			if _, found := emittedMetrics[metric]; !found {
				t.Errorf("registered claim %q has no corresponding episode emitted by PeerSearchEpisodes", metric)
			}
		}

		// Verify count of episodes matches count of registered claims
		if len(episodes) != len(registeredMetrics) {
			t.Errorf("emitted %d episodes, but %d claims are registered for peer-search", len(episodes), len(registeredMetrics))
		}
	}
}

func TestPeerSearchEpisodes_TaintLeaksSampleSizingAndRealizedCount(t *testing.T) {
	tests := []struct {
		name         string
		level0       int
		taintLeaks   int
		recorded     bool
		wantMeasured bool
		wantSample   int
		wantRealized float64
		wantFloorErr bool
		wantVerdict  string
	}{
		{
			name:         "leaks with positive level0",
			level0:       250,
			taintLeaks:   7,
			recorded:     true,
			wantMeasured: true,
			wantSample:   250,
			wantRealized: 7.0,
			wantFloorErr: true,
			wantVerdict:  VerdictOverClaim,
		},
		{
			name:         "clean with positive level0",
			level0:       250,
			taintLeaks:   0,
			recorded:     true,
			wantMeasured: true,
			wantSample:   250,
			wantRealized: 0.0,
			wantFloorErr: false,
			wantVerdict:  VerdictCalibrated,
		},
		{
			name:         "leaks with zero level0 queries reconciles sample to 0",
			level0:       0,
			taintLeaks:   3,
			recorded:     true,
			wantMeasured: true,
			wantSample:   0,
			wantRealized: 3.0,
			wantFloorErr: true,
			wantVerdict:  VerdictOverClaim,
		},
		{
			name:         "leaks with negative level0 queries clamped to 0",
			level0:       -10,
			taintLeaks:   2,
			recorded:     true,
			wantMeasured: true,
			wantSample:   0,
			wantRealized: 2.0,
			wantFloorErr: true,
			wantVerdict:  VerdictOverClaim,
		},
		{
			name:         "unrecorded telemetry",
			level0:       100,
			taintLeaks:   5,
			recorded:     false,
			wantMeasured: false,
			wantSample:   0,
			wantRealized: 0.0,
			wantFloorErr: false,
			wantVerdict:  VerdictUnmeasured,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ledger := PeerSearchTelemetryLedger{
				Level0Queries: tc.level0,
				TaintLeaks:    tc.taintLeaks,
				Recorded:      tc.recorded,
			}
			episodes := PeerSearchEpisodes(ledger)
			if len(episodes) != 5 {
				t.Fatalf("expected 5 episodes, got %d", len(episodes))
			}

			// Check taint_leak_rate (index 3)
			rateEp := episodes[3]
			if rateEp.Prediction.Metric != "taint_leak_rate" {
				t.Fatalf("episode 3 metric = %q, want 'taint_leak_rate'", rateEp.Prediction.Metric)
			}
			if rateEp.Outcome.Measured != tc.wantMeasured {
				t.Errorf("taint_leak_rate measured = %v, want %v", rateEp.Outcome.Measured, tc.wantMeasured)
			}
			if rateEp.Outcome.Sample != tc.wantSample {
				t.Errorf("taint_leak_rate sample = %d, want %d (consistent with Level0Queries)", rateEp.Outcome.Sample, tc.wantSample)
			}

			// Check taint_leaks (index 4)
			leaksEp := episodes[4]
			if leaksEp.Prediction.Metric != "taint_leaks" {
				t.Fatalf("episode 4 metric = %q, want 'taint_leaks'", leaksEp.Prediction.Metric)
			}
			if leaksEp.Prediction.Unit != "count" {
				t.Errorf("taint_leaks unit = %q, want 'count'", leaksEp.Prediction.Unit)
			}
			if leaksEp.Outcome.Measured != tc.wantMeasured {
				t.Errorf("taint_leaks measured = %v, want %v", leaksEp.Outcome.Measured, tc.wantMeasured)
			}
			if leaksEp.Outcome.Sample != tc.wantSample {
				t.Errorf("taint_leaks sample = %d, want %d (consistent with Level0Queries)", leaksEp.Outcome.Sample, tc.wantSample)
			}
			if tc.wantMeasured && leaksEp.Outcome.Realized != tc.wantRealized {
				t.Errorf("taint_leaks realized = %f, want %f", leaksEp.Outcome.Realized, tc.wantRealized)
			}

			scored := Score("test", leaksEp.Prediction, leaksEp.Outcome, DefaultCalibBand())
			if tc.wantFloorErr && FloorRespectErr(scored) <= 0.0 {
				t.Errorf("expected FloorRespectErr > 0 for taint_leaks breach, got %f", FloorRespectErr(scored))
			} else if !tc.wantFloorErr && FloorRespectErr(scored) != 0.0 {
				t.Errorf("expected FloorRespectErr == 0, got %f", FloorRespectErr(scored))
			}
			if scored.Verdict != tc.wantVerdict {
				t.Errorf("expected verdict %s, got %s", tc.wantVerdict, scored.Verdict)
			}
		})
	}
}
