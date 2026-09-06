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
	if len(cleanInputs) != 9 {
		t.Fatalf("expected 9 episodes, got %d", len(cleanInputs))
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
	if len(leakInputs) != 9 {
		t.Fatalf("expected 9 episodes, got %d", len(leakInputs))
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
	if len(inputs) != 9 {
		t.Fatalf("expected 9 inputs, got %d", len(inputs))
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
			if len(episodes) != 9 {
				t.Fatalf("expected 9 episodes, got %d", len(episodes))
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

// TestPeerSearchParameterTuningLevers validates tuning levers for excerpt budget,
// timeline lookback window, query-to-progress latency, and tool deduplication (#11931).
func TestPeerSearchParameterTuningLevers(t *testing.T) {
	// 1. Calibration and claim registration verification
	cases := []struct {
		metric        string
		wantClaimed   float64
		wantUnit      string
		lowerIsBetter bool
	}{
		{"excerpt_budget_tokens", 512.0, "tokens", false},
		{"timeline_window_seconds", 1800.0, "seconds", false},
		{"query_to_progress_latency_ms", 250.0, "ms", true},
		{"tool_dedup_ratio", 0.30, "fraction", false},
	}

	for _, tc := range cases {
		c, ok := Registry.Lookup("peer-search", tc.metric)
		if !ok {
			t.Fatalf("claim %q not found for lever 'peer-search'", tc.metric)
		}
		if c.Claimed != tc.wantClaimed {
			t.Errorf("%s Claimed = %f, want %f", tc.metric, c.Claimed, tc.wantClaimed)
		}
		if c.IntentionalFloor {
			t.Errorf("%s should not be an intentional floor", tc.metric)
		}
		if c.LowerIsBetter != tc.lowerIsBetter {
			t.Errorf("%s LowerIsBetter = %v, want %v", tc.metric, c.LowerIsBetter, tc.lowerIsBetter)
		}

		pred := Registry.MustPredict("peer-search", tc.metric, tc.wantUnit)
		if pred.Claimed != tc.wantClaimed {
			t.Errorf("pred.%s Claimed = %f, want %f", tc.metric, pred.Claimed, tc.wantClaimed)
		}
		if pred.Unit != tc.wantUnit {
			t.Errorf("pred.%s Unit = %q, want %q", tc.metric, pred.Unit, tc.wantUnit)
		}
		if pred.LowerIsBetter != tc.lowerIsBetter {
			t.Errorf("pred.%s LowerIsBetter = %v, want %v", tc.metric, pred.LowerIsBetter, tc.lowerIsBetter)
		}
	}

	// 2. Parameter adjustment and defaults
	emptyLedger := PeerSearchTelemetryLedger{}
	if eb := emptyLedger.EffectiveExcerptBudgetTokens(); eb != DefaultExcerptBudgetTokens {
		t.Errorf("empty ledger EffectiveExcerptBudgetTokens = %d, want %d", eb, DefaultExcerptBudgetTokens)
	}
	if tw := emptyLedger.EffectiveTimelineWindowSeconds(); tw != DefaultTimelineWindowSeconds {
		t.Errorf("empty ledger EffectiveTimelineWindowSeconds = %f, want %f", tw, DefaultTimelineWindowSeconds)
	}
	if lat := emptyLedger.EffectiveQueryToProgressLatencyMs(); lat != DefaultQueryToProgressLatencyMs {
		t.Errorf("empty ledger EffectiveQueryToProgressLatencyMs = %f, want %f", lat, DefaultQueryToProgressLatencyMs)
	}
	if dr := emptyLedger.EffectiveToolDedupRatio(); dr != 0.0 {
		t.Errorf("empty ledger EffectiveToolDedupRatio = %f, want 0.0", dr)
	}

	// Fluent chaining and parameter adjustments
	tuned := emptyLedger.
		WithExcerptBudget(1024).
		WithTimelineWindow(3600.0).
		WithQueryToProgressLatency(180.0).
		WithToolDedupRatio(0.45)

	if tuned.ExcerptBudgetTokens != 1024 || tuned.EffectiveExcerptBudgetTokens() != 1024 {
		t.Errorf("WithExcerptBudget failed: got %d", tuned.ExcerptBudgetTokens)
	}
	if tuned.TimelineWindowSeconds != 3600.0 || tuned.EffectiveTimelineWindowSeconds() != 3600.0 {
		t.Errorf("WithTimelineWindow failed: got %f", tuned.TimelineWindowSeconds)
	}
	if tuned.QueryToProgressLatencyMs != 180.0 || tuned.EffectiveQueryToProgressLatencyMs() != 180.0 {
		t.Errorf("WithQueryToProgressLatency failed: got %f", tuned.QueryToProgressLatencyMs)
	}
	if tuned.ToolDedupRatio != 0.45 || tuned.EffectiveToolDedupRatio() != 0.45 {
		t.Errorf("WithToolDedupRatio failed: got %f", tuned.ToolDedupRatio)
	}

	// Direct TuneParameters call
	reTuned := emptyLedger.TuneParameters(256, 900.0)
	if reTuned.ExcerptBudgetTokens != 256 || reTuned.TimelineWindowSeconds != 900.0 {
		t.Errorf("TuneParameters failed: got budget=%d, window=%f", reTuned.ExcerptBudgetTokens, reTuned.TimelineWindowSeconds)
	}

	// Clamping checks for ToolDedupRatio
	clampedLow := PeerSearchTelemetryLedger{ToolDedupRatio: -0.2}.EffectiveToolDedupRatio()
	if clampedLow != 0.0 {
		t.Errorf("EffectiveToolDedupRatio low clamp = %f, want 0.0", clampedLow)
	}
	clampedHigh := PeerSearchTelemetryLedger{ToolDedupRatio: 1.5}.EffectiveToolDedupRatio()
	if clampedHigh != 1.0 {
		t.Errorf("EffectiveToolDedupRatio high clamp = %f, want 1.0", clampedHigh)
	}

	// 3. Ledger calculation, episode evaluation, and scoring
	activeLedger := PeerSearchTelemetryLedger{
		Level0Queries:            100,
		Level1Queries:            40,
		Level2Queries:            10,
		AvoidedToolTokens:        5000,
		PeerQueryTokens:          2000,
		TaintLeaks:               0,
		ExcerptBudgetTokens:      512,
		TimelineWindowSeconds:    1800.0,
		QueryToProgressLatencyMs: 250.0,
		ToolDedupRatio:           0.30,
		Recorded:                 true,
	}

	episodes := activeLedger.Episodes()
	if len(episodes) != 9 {
		t.Fatalf("expected 9 episodes from activeLedger, got %d", len(episodes))
	}

	// Episode 5: excerpt_budget_tokens
	epBudget := episodes[5]
	if epBudget.Prediction.Metric != "excerpt_budget_tokens" {
		t.Errorf("episode 5 metric = %q, want 'excerpt_budget_tokens'", epBudget.Prediction.Metric)
	}
	if !epBudget.Outcome.Measured || epBudget.Outcome.Realized != 512.0 {
		t.Errorf("episode 5 realized = %f (measured=%v), want 512.0", epBudget.Outcome.Realized, epBudget.Outcome.Measured)
	}

	// Episode 6: timeline_window_seconds
	epWindow := episodes[6]
	if epWindow.Prediction.Metric != "timeline_window_seconds" {
		t.Errorf("episode 6 metric = %q, want 'timeline_window_seconds'", epWindow.Prediction.Metric)
	}
	if !epWindow.Outcome.Measured || epWindow.Outcome.Realized != 1800.0 {
		t.Errorf("episode 6 realized = %f (measured=%v), want 1800.0", epWindow.Outcome.Realized, epWindow.Outcome.Measured)
	}

	// Episode 7: query_to_progress_latency_ms
	epLatency := episodes[7]
	if epLatency.Prediction.Metric != "query_to_progress_latency_ms" {
		t.Errorf("episode 7 metric = %q, want 'query_to_progress_latency_ms'", epLatency.Prediction.Metric)
	}
	if !epLatency.Outcome.Measured || epLatency.Outcome.Realized != 250.0 {
		t.Errorf("episode 7 realized = %f (measured=%v), want 250.0", epLatency.Outcome.Realized, epLatency.Outcome.Measured)
	}
	if !epLatency.Prediction.LowerIsBetter {
		t.Errorf("episode 7 LowerIsBetter = false, want true")
	}

	// Episode 8: tool_dedup_ratio
	epDedup := episodes[8]
	if epDedup.Prediction.Metric != "tool_dedup_ratio" {
		t.Errorf("episode 8 metric = %q, want 'tool_dedup_ratio'", epDedup.Prediction.Metric)
	}
	if !epDedup.Outcome.Measured || epDedup.Outcome.Realized != 0.30 {
		t.Errorf("episode 8 realized = %f (measured=%v), want 0.30", epDedup.Outcome.Realized, epDedup.Outcome.Measured)
	}

	// Verify scoring of perfectly calibrated baseline
	for i := 5; i <= 8; i++ {
		scored := Score("test-calibrated", episodes[i].Prediction, episodes[i].Outcome, DefaultCalibBand())
		if scored.Verdict != VerdictCalibrated {
			t.Errorf("metric %s scored verdict = %s, want %s", episodes[i].Prediction.Metric, scored.Verdict, VerdictCalibrated)
		}
	}

	// 4. Latency LowerIsBetter polarity check
	fastLedger := activeLedger.WithQueryToProgressLatency(150.0)
	fastEp := Score("fast", fastLedger.Episodes()[7].Prediction, fastLedger.Episodes()[7].Outcome, DefaultCalibBand())
	if fastEp.Verdict != VerdictUnderClaim {
		t.Errorf("faster latency scored verdict = %s, want %s (under-claim / win)", fastEp.Verdict, VerdictUnderClaim)
	}

	slowLedger := activeLedger.WithQueryToProgressLatency(400.0)
	slowEp := Score("slow", slowLedger.Episodes()[7].Prediction, slowLedger.Episodes()[7].Outcome, DefaultCalibBand())
	if slowEp.Verdict != VerdictOverClaim {
		t.Errorf("slower latency scored verdict = %s, want %s (over-claim / regression)", slowEp.Verdict, VerdictOverClaim)
	}

	// 5. Taint breach penalty on tool deduplication ratio
	taintBreachLedger := activeLedger
	taintBreachLedger.TaintLeaks = 2
	taintBreachEpisodes := taintBreachLedger.Episodes()
	if taintBreachEpisodes[8].Outcome.Realized != 0.0 {
		t.Errorf("expected tool_dedup_ratio to be penalized to 0.0 on taint leak, got %f", taintBreachEpisodes[8].Outcome.Realized)
	}

	// 6. Unrecorded telemetry leaves tuning metrics unmeasured
	unrecorded := activeLedger
	unrecorded.Recorded = false
	unrecEpisodes := unrecorded.Episodes()
	for i := 5; i <= 8; i++ {
		if unrecEpisodes[i].Outcome.Measured {
			t.Errorf("metric %s was unexpectedly measured on unrecorded ledger", unrecEpisodes[i].Prediction.Metric)
		}
	}
}
