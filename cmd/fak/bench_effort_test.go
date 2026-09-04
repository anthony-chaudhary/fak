package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBenchEffort(t *testing.T) {
	t.Run("MockJSONReportAndProofAssertions", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		exitCode := runBenchEffort(&stdout, &stderr, []string{"--mock", "--json"})
		if exitCode != 0 {
			t.Fatalf("expected exit code 0, got %d. Stderr: %s", exitCode, stderr.String())
		}

		var receipt EffortBenchmarkReceipt
		if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
			t.Fatalf("failed to decode JSON receipt: %v\nOutput: %s", err, stdout.String())
		}

		// 1. Validate receipt schema and metadata
		if receipt.Schema != "fak-effort-benchmark/1" {
			t.Errorf("expected schema 'fak-effort-benchmark/1', got %q", receipt.Schema)
		}
		if receipt.Model != "gemini-3.8-flash" {
			t.Errorf("expected default model 'gemini-3.8-flash', got %q", receipt.Model)
		}
		if !receipt.Mock {
			t.Errorf("expected mock=true in receipt")
		}
		if receipt.Turns != 10 {
			t.Errorf("expected 10 turns, got %d", receipt.Turns)
		}

		// 2. Validate all 3 regimes exist
		dyn, okDyn := receipt.Regimes["dynamic_intra_model"]
		if !okDyn || dyn == nil {
			t.Fatalf("missing 'dynamic_intra_model' regime in receipt")
		}
		stat, okStat := receipt.Regimes["static_high_reasoning"]
		if !okStat || stat == nil {
			t.Fatalf("missing 'static_high_reasoning' regime in receipt")
		}
		cross, okCross := receipt.Regimes["cross_model_bouncing"]
		if !okCross || cross == nil {
			t.Fatalf("missing 'cross_model_bouncing' regime in receipt")
		}

		// 3. Assert Dynamic effort TTFA on tool turns <= 1.5s
		if dyn.TTFAToolTurns.MedianS > 1.5 {
			t.Errorf("assertion failed: dynamic effort tool turn TTFA median (%f s) must be <= 1.5s", dyn.TTFAToolTurns.MedianS)
		}
		if dyn.TTFAToolTurns.MedianMS > 1500.0 {
			t.Errorf("assertion failed: dynamic effort tool turn TTFA median (%f ms) must be <= 1500ms", dyn.TTFAToolTurns.MedianMS)
		}

		// 4. Assert Dynamic effort cache hit rate >= 95%
		if dyn.CacheHitRatePct < 95.0 {
			t.Errorf("assertion failed: dynamic effort cache hit rate (%.2f%%) must be >= 95%%", dyn.CacheHitRatePct)
		}

		// 5. Assert Cross-model cache hit rate is low (< 50% or 0%)
		if cross.CacheHitRatePct >= 50.0 {
			t.Errorf("assertion failed: cross-model cache hit rate (%.2f%%) must be < 50%%", cross.CacheHitRatePct)
		}

		// 6. Assert Static high thinking token burn is significantly higher than dynamic
		if stat.ReasoningTokens.Total <= dyn.ReasoningTokens.Total*2 {
			t.Errorf("assertion failed: static high reasoning tokens (%d) must be significantly higher than dynamic (%d)",
				stat.ReasoningTokens.Total, dyn.ReasoningTokens.Total)
		}
		if stat.ReasoningTokens.ToolTurns <= 0 {
			t.Errorf("expected static high reasoning to burn tokens on tool turns, got %d", stat.ReasoningTokens.ToolTurns)
		}
		if dyn.ReasoningTokens.ToolTurns != 0 {
			t.Errorf("expected dynamic effort to clamp reasoning tokens on tool turns to 0, got %d", dyn.ReasoningTokens.ToolTurns)
		}

		// 7. Verify comparative proof flags and verdict
		if !receipt.Comparison.ProofTTFAUnder1500ms {
			t.Errorf("expected comparison.ProofTTFAUnder1500ms to be true")
		}
		if !receipt.Comparison.ProofCachePreservedAbove95Pct {
			t.Errorf("expected comparison.ProofCachePreservedAbove95Pct to be true")
		}
		if !strings.HasPrefix(receipt.Comparison.Verdict, "PASS") {
			t.Errorf("expected PASS verdict, got: %s", receipt.Comparison.Verdict)
		}
	})

	t.Run("TextReportFormatting", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		exitCode := runBenchEffort(&stdout, &stderr, []string{"--mock"})
		if exitCode != 0 {
			t.Fatalf("expected exit code 0, got %d. Stderr: %s", exitCode, stderr.String())
		}

		out := stdout.String()
		expectedPhrases := []string{
			"FAK INTRA-MODEL EFFORT MODULATION BENCHMARK REPORT",
			"Static High",
			"Cross-Model Bounce",
			"Dynamic Intra (fak)",
			"TTFA (Tool Turns, Median)",
			"Reasoning Tokens (Total)",
			"Prefix Cache Hit Rate",
			"VERDICT: PASS",
		}
		for _, phrase := range expectedPhrases {
			if !strings.Contains(out, phrase) {
				t.Errorf("text report missing phrase %q\nFull output:\n%s", phrase, out)
			}
		}
	})

	t.Run("OutputFileGeneration", func(t *testing.T) {
		tempDir := t.TempDir()
		outFile := filepath.Join(tempDir, "effort_report.json")

		var stdout, stderr bytes.Buffer
		exitCode := runBenchEffort(&stdout, &stderr, []string{"--mock", "--out", outFile})
		if exitCode != 0 {
			t.Fatalf("expected exit code 0, got %d. Stderr: %s", exitCode, stderr.String())
		}

		data, err := os.ReadFile(outFile)
		if err != nil {
			t.Fatalf("failed to read output file: %v", err)
		}

		var receipt EffortBenchmarkReceipt
		if err := json.Unmarshal(data, &receipt); err != nil {
			t.Fatalf("output file does not contain valid JSON receipt: %v", err)
		}
		if receipt.Schema != "fak-effort-benchmark/1" {
			t.Errorf("expected schema 'fak-effort-benchmark/1', got %q", receipt.Schema)
		}
	})

	t.Run("CustomModelAndTurnScaling", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		exitCode := runBenchEffort(&stdout, &stderr, []string{
			"--model", "gemini-2.5-flash",
			"--turns", "6",
			"--mock",
			"--json",
		})
		if exitCode != 0 {
			t.Fatalf("expected exit code 0, got %d. Stderr: %s", exitCode, stderr.String())
		}

		var receipt EffortBenchmarkReceipt
		if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
			t.Fatalf("failed to decode JSON receipt: %v", err)
		}
		if receipt.Model != "gemini-2.5-flash" {
			t.Errorf("expected model 'gemini-2.5-flash', got %q", receipt.Model)
		}
		if receipt.Turns != 6 {
			t.Errorf("expected 6 turns, got %d", receipt.Turns)
		}
		dyn := receipt.Regimes["dynamic_intra_model"]
		if len(dyn.Turns) != 6 {
			t.Errorf("expected 6 turn records in dynamic regime, got %d", len(dyn.Turns))
		}
	})

	t.Run("HelpFlag", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		exitCode := runBenchEffort(&stdout, &stderr, []string{"--help"})
		if exitCode != 0 {
			t.Errorf("expected exit code 0 for --help, got %d", exitCode)
		}
		if !strings.Contains(stdout.String(), "Usage: fak bench effort") {
			t.Errorf("expected help text in stdout, got:\n%s", stdout.String())
		}
	})

	t.Run("InvalidArguments", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		exitCode := runBenchEffort(&stdout, &stderr, []string{"--turns", "0", "--mock"})
		if exitCode != 1 {
			t.Errorf("expected exit code 1 for --turns 0, got %d", exitCode)
		}
		if !strings.Contains(stderr.String(), "--turns must be >= 1") {
			t.Errorf("expected error message for --turns 0, got: %s", stderr.String())
		}
	})

	t.Run("LiveModeWithoutAPIKeyFailsGracefully", func(t *testing.T) {
		// Ensure GEMINI_API_KEY is unset for this test
		origKey := os.Getenv("GEMINI_API_KEY")
		_ = os.Unsetenv("GEMINI_API_KEY")
		defer func() {
			if origKey != "" {
				_ = os.Setenv("GEMINI_API_KEY", origKey)
			}
		}()

		var stdout, stderr bytes.Buffer
		exitCode := runBenchEffort(&stdout, &stderr, []string{}) // live mode without --mock
		if exitCode != 1 {
			t.Errorf("expected exit code 1 when live mode lacks API key, got %d", exitCode)
		}
		if !strings.Contains(stderr.String(), "requires GEMINI_API_KEY") {
			t.Errorf("expected API key error message, got: %s", stderr.String())
		}
	})
}
