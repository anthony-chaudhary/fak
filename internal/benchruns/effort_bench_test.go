package benchruns

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestEffortBench(t *testing.T) {
	cfg := DefaultEffortBenchConfig()
	receipt, err := RunEffortBenchmark(cfg)
	if err != nil {
		t.Fatalf("RunEffortBenchmark failed: %v", err)
	}

	t.Run("RegimeDynamicIntraModelMetrics", func(t *testing.T) {
		dyn, ok := receipt.Regimes[string(RegimeDynamicIntraModel)]
		if !ok {
			t.Fatalf("missing regime %s in receipt", RegimeDynamicIntraModel)
		}

		// 1. Verify RegimeDynamicIntraModel achieves <= 1.5s median TTFA on tool turns
		if dyn.MedianTTFASeconds > 1.5 {
			t.Errorf("RegimeDynamicIntraModel median TTFA = %.4fs, want <= 1.5s", dyn.MedianTTFASeconds)
		}
		if dyn.MedianTTFASeconds <= 0 {
			t.Errorf("RegimeDynamicIntraModel median TTFA = %.4fs, want > 0s", dyn.MedianTTFASeconds)
		}

		// 2. Verify RegimeDynamicIntraModel retains >= 95% prompt cache prefix hits
		if dyn.CacheHitRatePct < 95.0 {
			t.Errorf("RegimeDynamicIntraModel cache hit rate pct = %.2f%%, want >= 95.0%%", dyn.CacheHitRatePct)
		}
		if dyn.CacheHitRate < 0.95 {
			t.Errorf("RegimeDynamicIntraModel cache hit rate ratio = %.4f, want >= 0.95", dyn.CacheHitRate)
		}

		// 3. Verify task resolution rate
		if dyn.TaskResolutionRate != 1.0 {
			t.Errorf("RegimeDynamicIntraModel task resolution rate = %.2f, want 1.0 (100%%)", dyn.TaskResolutionRate)
		}

		// 4. Verify tool turns count
		if dyn.ToolTurnsCount == 0 {
			t.Errorf("RegimeDynamicIntraModel tool turns count = 0, want > 0")
		}

		// Comparison against RegimeStaticHigh
		stat, ok := receipt.Regimes[string(RegimeStaticHigh)]
		if !ok {
			t.Fatalf("missing regime %s in receipt", RegimeStaticHigh)
		}

		// Static high thinking burns full thinking on tool turns, leading to much higher TTFA (> 1.5s)
		if stat.MedianTTFASeconds <= 1.5 {
			t.Errorf("RegimeStaticHigh median TTFA = %.4fs, expected > 1.5s due to static thinking on tool turns", stat.MedianTTFASeconds)
		}
		if dyn.MedianTTFASeconds >= stat.MedianTTFASeconds {
			t.Errorf("RegimeDynamicIntraModel TTFA (%.4fs) should be faster than RegimeStaticHigh TTFA (%.4fs)",
				dyn.MedianTTFASeconds, stat.MedianTTFASeconds)
		}

		if receipt.Comparison.DynamicVsStatic.TTFASpeedupX <= 1.0 {
			t.Errorf("TTFASpeedupX = %.2f, want > 1.0", receipt.Comparison.DynamicVsStatic.TTFASpeedupX)
		}
		if receipt.Comparison.DynamicVsStatic.ReasoningTokenReductionPct <= 0 {
			t.Errorf("ReasoningTokenReductionPct = %.2f, want > 0", receipt.Comparison.DynamicVsStatic.ReasoningTokenReductionPct)
		}
		if receipt.Comparison.DynamicVsStatic.CostReductionPct <= 0 {
			t.Errorf("CostReductionPct = %.2f, want > 0", receipt.Comparison.DynamicVsStatic.CostReductionPct)
		}
		if receipt.Comparison.DynamicVsStatic.WallClockSpeedupX <= 1.0 {
			t.Errorf("WallClockSpeedupX = %.2f, want > 1.0", receipt.Comparison.DynamicVsStatic.WallClockSpeedupX)
		}

		// Comparison against RegimeCrossModelBouncing
		cross, ok := receipt.Regimes[string(RegimeCrossModelBouncing)]
		if !ok {
			t.Fatalf("missing regime %s in receipt", RegimeCrossModelBouncing)
		}

		// Cross-model bouncing destroys provider prompt cache prefix
		if cross.CacheHitRatePct >= 95.0 {
			t.Errorf("RegimeCrossModelBouncing cache hit rate = %.2f%%, expected < 95.0%% due to model bouncing", cross.CacheHitRatePct)
		}
		if receipt.Comparison.DynamicVsCrossModel.CacheHitRateAdvantagePct <= 0 {
			t.Errorf("CacheHitRateAdvantagePct = %.2f, want > 0", receipt.Comparison.DynamicVsCrossModel.CacheHitRateAdvantagePct)
		}
		// Small model in cross-model bouncing has lower task resolution rate
		if cross.TaskResolutionRate >= dyn.TaskResolutionRate {
			t.Errorf("RegimeCrossModelBouncing resolution rate (%.2f) expected to be lower than DynamicIntraModel (%.2f)",
				cross.TaskResolutionRate, dyn.TaskResolutionRate)
		}
	})

	t.Run("ReceiptSchemaAndJSONFormatting", func(t *testing.T) {
		// Verify schema adherence
		if receipt.Schema != EffortBenchmarkSchema {
			t.Errorf("receipt schema = %q, want %q", receipt.Schema, EffortBenchmarkSchema)
		}
		if receipt.Suite == "" {
			t.Errorf("receipt suite is empty")
		}
		if receipt.Timestamp == "" {
			t.Errorf("receipt timestamp is empty")
		}
		if receipt.TasksCount != len(cfg.Tasks) {
			t.Errorf("receipt tasks count = %d, want %d", receipt.TasksCount, len(cfg.Tasks))
		}

		// Verify JSON formatting
		rawJSON, err := receipt.JSON()
		if err != nil {
			t.Fatalf("receipt.JSON() error: %v", err)
		}

		// Unmarshal into generic map to verify JSON structure and required keys
		var rawMap map[string]any
		if err := json.Unmarshal(rawJSON, &rawMap); err != nil {
			t.Fatalf("failed to unmarshal receipt JSON: %v", err)
		}

		if rawMap["schema"] != EffortBenchmarkSchema {
			t.Errorf("JSON schema = %v, want %s", rawMap["schema"], EffortBenchmarkSchema)
		}

		regimesMap, ok := rawMap["regimes"].(map[string]any)
		if !ok {
			t.Fatalf("JSON missing 'regimes' object")
		}

		for _, reg := range AllExecutionRegimes {
			regObj, exists := regimesMap[string(reg)].(map[string]any)
			if !exists {
				t.Errorf("JSON regimes missing entry for %s", reg)
				continue
			}
			// Verify presence of all required metric fields
			requiredFields := []string{
				"median_ttfa_seconds",
				"reasoning_tokens_spent",
				"cache_hit_rate_pct",
				"wall_clock_seconds",
				"simulated_cost_usd",
				"task_resolution_rate",
			}
			for _, field := range requiredFields {
				if _, present := regObj[field]; !present {
					t.Errorf("regime %s missing required JSON field %q", reg, field)
				}
			}
		}

		// Test round-trip via ParseEffortBenchmarkReceipt
		parsed, err := ParseEffortBenchmarkReceipt(rawJSON)
		if err != nil {
			t.Fatalf("ParseEffortBenchmarkReceipt failed: %v", err)
		}

		if parsed.Schema != receipt.Schema {
			t.Errorf("parsed schema = %q, want %q", parsed.Schema, receipt.Schema)
		}
		if parsed.TasksCount != receipt.TasksCount {
			t.Errorf("parsed tasks count = %d, want %d", parsed.TasksCount, receipt.TasksCount)
		}

		parsedDyn := parsed.Regimes[string(RegimeDynamicIntraModel)]
		origDyn := receipt.Regimes[string(RegimeDynamicIntraModel)]
		if parsedDyn.MedianTTFASeconds != origDyn.MedianTTFASeconds {
			t.Errorf("parsed median TTFA = %v, want %v", parsedDyn.MedianTTFASeconds, origDyn.MedianTTFASeconds)
		}
		if parsedDyn.CacheHitRatePct != origDyn.CacheHitRatePct {
			t.Errorf("parsed cache hit rate pct = %v, want %v", parsedDyn.CacheHitRatePct, origDyn.CacheHitRatePct)
		}

		// Test parse error on invalid schema
		badSchemaJSON := bytes.Replace(rawJSON, []byte(EffortBenchmarkSchema), []byte("invalid-schema/0"), 1)
		if _, err := ParseEffortBenchmarkReceipt(badSchemaJSON); err == nil {
			t.Errorf("ParseEffortBenchmarkReceipt should fail on invalid schema")
		}

		// Test parse error on invalid JSON
		if _, err := ParseEffortBenchmarkReceipt([]byte("not-json")); err == nil {
			t.Errorf("ParseEffortBenchmarkReceipt should fail on invalid JSON")
		}

		// Test text and markdown renderers
		textReport := RenderEffortBenchmarkReceipt(receipt)
		t.Logf("\n%s", textReport)
		if !strings.Contains(textReport, "Intra-Model Effort Modulation Benchmark") ||
			!strings.Contains(textReport, string(RegimeDynamicIntraModel)) ||
			!strings.Contains(textReport, "Dynamic Intra-Model achieves") {
			t.Errorf("RenderEffortBenchmarkReceipt missing expected content:\n%s", textReport)
		}

		mdReport := receipt.RenderMarkdown()
		if !strings.Contains(mdReport, "## Intra-Model Effort Modulation Benchmark Receipt") ||
			!strings.Contains(mdReport, "| Regime | Median TTFA |") {
			t.Errorf("RenderMarkdown missing expected content:\n%s", mdReport)
		}
	})

	t.Run("DeterministicExecution", func(t *testing.T) {
		// Running twice with identical configuration must produce byte-for-byte identical output
		receipt1, err := RunEffortBenchmark(cfg)
		if err != nil {
			t.Fatalf("first run failed: %v", err)
		}
		raw1, err := receipt1.JSON()
		if err != nil {
			t.Fatalf("marshal run 1 failed: %v", err)
		}

		receipt2, err := RunEffortBenchmark(cfg)
		if err != nil {
			t.Fatalf("second run failed: %v", err)
		}
		raw2, err := receipt2.JSON()
		if err != nil {
			t.Fatalf("marshal run 2 failed: %v", err)
		}

		if !bytes.Equal(raw1, raw2) {
			t.Errorf("runs are not byte-identical!\nRun 1:\n%s\nRun 2:\n%s", string(raw1), string(raw2))
		}
	})

	t.Run("EdgeCases", func(t *testing.T) {
		// Empty tasks config falls back to default tasks
		emptyCfg := EffortBenchConfig{
			Timestamp: "2026-09-04T12:00:00Z",
		}
		rec, err := RunEffortBenchmark(emptyCfg)
		if err != nil {
			t.Fatalf("RunEffortBenchmark with empty config failed: %v", err)
		}
		if rec.TasksCount == 0 {
			t.Errorf("expected default tasks count > 0")
		}

		// Task with only planning turns
		planOnlyTask := BenchmarkTask{
			ID:         "task-plan-only",
			Title:      "Architecture plan only",
			BaseTokens: 2000,
			Turns: []TurnSpec{
				{Index: 1, Kind: TurnKindPlan, DeltaPromptTokens: 100},
				{Index: 2, Kind: TurnKindPlan, DeltaPromptTokens: 100},
			},
		}
		planCfg := DefaultEffortBenchConfig()
		planCfg.Tasks = []BenchmarkTask{planOnlyTask}
		recPlan, err := RunEffortBenchmark(planCfg)
		if err != nil {
			t.Fatalf("RunEffortBenchmark with plan-only task failed: %v", err)
		}
		dynPlan := recPlan.Regimes[string(RegimeDynamicIntraModel)]
		if dynPlan.ToolTurnsCount != 0 {
			t.Errorf("tool turns count = %d, want 0", dynPlan.ToolTurnsCount)
		}
		if dynPlan.MedianTTFASeconds != 0 {
			t.Errorf("median TTFA on zero tool turns = %.2f, want 0", dynPlan.MedianTTFASeconds)
		}
	})
}
