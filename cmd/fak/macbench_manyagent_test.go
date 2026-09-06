package main

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestMacBenchManyAgent_Validation(t *testing.T) {
	tests := []struct {
		name    string
		opts    ManyAgentOptions
		wantErr string
	}{
		{
			name: "valid options",
			opts: ManyAgentOptions{
				Concurrency: 4,
				Model:       "Qwen3.8-27B",
				Horizon:     20,
				Cache:       true,
				Output:      "summary",
			},
			wantErr: "",
		},
		{
			name: "concurrency zero",
			opts: ManyAgentOptions{
				Concurrency: 0,
				Model:       "Qwen3.8-27B",
				Horizon:     20,
			},
			wantErr: "--concurrency must be positive",
		},
		{
			name: "concurrency negative",
			opts: ManyAgentOptions{
				Concurrency: -3,
				Model:       "Qwen3.8-27B",
				Horizon:     20,
			},
			wantErr: "--concurrency must be positive",
		},
		{
			name: "horizon zero",
			opts: ManyAgentOptions{
				Concurrency: 4,
				Model:       "Qwen3.8-27B",
				Horizon:     0,
			},
			wantErr: "--horizon must be positive",
		},
		{
			name: "empty model",
			opts: ManyAgentOptions{
				Concurrency: 4,
				Model:       "   ",
				Horizon:     20,
			},
			wantErr: "--model must not be empty",
		},
		{
			name: "invalid output format",
			opts: ManyAgentOptions{
				Concurrency: 4,
				Model:       "Qwen3.8-27B",
				Horizon:     20,
				Output:      "xml",
			},
			wantErr: "invalid --output",
		},
		{
			name: "prefix tokens negative",
			opts: ManyAgentOptions{
				Concurrency:        4,
				Model:              "Qwen3.8-27B",
				Horizon:            20,
				SharedPrefixTokens: -1,
			},
			wantErr: "--prefix-tokens must be non-negative",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateManyAgentOptions(tc.opts)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got %v", tc.wantErr, err)
				}
			}
		})
	}
}

func TestMacBenchManyAgent_PrefixSharingComputation(t *testing.T) {
	optsOn := ManyAgentOptions{
		Concurrency:        4,
		Model:              "Qwen3.8-27B",
		Horizon:            20,
		Cache:              true,
		SharedPrefixTokens: DefaultSharedPrefixTokens,
	}
	repOn, err := RunManyAgentSpine(optsOn)
	if err != nil {
		t.Fatalf("RunManyAgentSpine failed: %v", err)
	}

	if repOn.PrefixEvalCount != 1 {
		t.Errorf("PrefixEvalCount with cache=true: got %d, want 1", repOn.PrefixEvalCount)
	}
	if repOn.ReusedTokens == 0 {
		t.Errorf("ReusedTokens with cache=true should be > 0, got %d", repOn.ReusedTokens)
	}
	if repOn.PromptTokens <= repOn.ReusedTokens {
		t.Errorf("PromptTokens (%d) must be strictly greater than ReusedTokens (%d)", repOn.PromptTokens, repOn.ReusedTokens)
	}
	if repOn.ReuseRatio < 0.85 {
		t.Errorf("expected ReuseRatio >= 0.85 over 20 turns, got %.4f", repOn.ReuseRatio)
	}
	if !repOn.Verified {
		t.Errorf("expected Verified == true with cache on")
	}

	// Now test with Cache: false
	optsOff := ManyAgentOptions{
		Concurrency:        4,
		Model:              "Qwen3.8-27B",
		Horizon:            20,
		Cache:              false,
		SharedPrefixTokens: DefaultSharedPrefixTokens,
	}
	repOff, err := RunManyAgentSpine(optsOff)
	if err != nil {
		t.Fatalf("RunManyAgentSpine failed: %v", err)
	}

	if repOff.PrefixEvalCount != 4*20 {
		t.Errorf("PrefixEvalCount with cache=false: got %d, want %d", repOff.PrefixEvalCount, 4*20)
	}
	if repOff.ReusedTokens != 0 {
		t.Errorf("ReusedTokens with cache=false: got %d, want 0", repOff.ReusedTokens)
	}
	if repOff.ReuseRatio != 0.0 {
		t.Errorf("ReuseRatio with cache=false: got %.4f, want 0", repOff.ReuseRatio)
	}
	if repOff.PromptTokens != repOn.PromptTokens {
		t.Errorf("PromptTokens presented must match regardless of cache: on=%d, off=%d", repOn.PromptTokens, repOff.PromptTokens)
	}
	if repOff.Verified {
		t.Errorf("expected Verified == false with cache off")
	}
}

func TestMacBenchManyAgent_MetricsCalculation(t *testing.T) {
	optsOn := ManyAgentOptions{
		Concurrency:        4,
		Model:              "Qwen3.8-27B",
		Horizon:            20,
		Cache:              true,
		SharedPrefixTokens: DefaultSharedPrefixTokens,
	}
	repOn, err := RunManyAgentSpine(optsOn)
	if err != nil {
		t.Fatalf("RunManyAgentSpine on failed: %v", err)
	}

	optsOff := ManyAgentOptions{
		Concurrency:        4,
		Model:              "Qwen3.8-27B",
		Horizon:            20,
		Cache:              false,
		SharedPrefixTokens: DefaultSharedPrefixTokens,
	}
	repOff, err := RunManyAgentSpine(optsOff)
	if err != nil {
		t.Fatalf("RunManyAgentSpine off failed: %v", err)
	}

	// 1. Peak memory with cache=true must be significantly lower than cache=false
	// For K=4, 3 copies of 4096-token prefix (3 * 256 MB = 768 MB) are saved under 16 full-attention layers (3:1 GDN cadence).
	savedMB := repOff.PeakMemoryMB - repOn.PeakMemoryMB
	if savedMB < 750.0 || savedMB > 780.0 {
		t.Errorf("expected ~768 MB saved by caching 4096 prefix across 4 agents under 16 full-attention layers, got %.1f MB", savedMB)
	}

	// 2. Peak memory must be strictly lower with cache=true and AgentsPerGB must not regress (equal or higher after 2-decimal rounding)
	if repOn.PeakMemoryMB >= repOff.PeakMemoryMB || repOn.AgentsPerGB < repOff.AgentsPerGB {
		t.Errorf("expected PeakMemoryMB on (%.1f) < off (%.1f) and AgentsPerGB on (%.2f) >= off (%.2f)",
			repOn.PeakMemoryMB, repOff.PeakMemoryMB, repOn.AgentsPerGB, repOff.AgentsPerGB)
	}

	// 3. TTFT must be order-of-magnitude faster with cache=true (< 25 ms vs > 500 ms)
	if repOn.P50TTFTMS >= 25.0 {
		t.Errorf("expected P50TTFTMS on to be fast (< 25ms), got %.1f ms", repOn.P50TTFTMS)
	}
	if repOff.P50TTFTMS < 500.0 {
		t.Errorf("expected P50TTFTMS off to be slow (> 500ms), got %.1f ms", repOff.P50TTFTMS)
	}
	if repOn.P95TTFTMS >= 25.0 {
		t.Errorf("expected P95TTFTMS on (< 25ms), got %.1f ms", repOn.P95TTFTMS)
	}
}

func TestMacBenchManyAgent_FlatTTFTUnderConcurrency(t *testing.T) {
	// With fak caching on, shared prefix is evaluated once, holding TTFT flat as K grows.
	concurrencies := []int{1, 2, 4, 8}
	var baselineP50 float64

	for i, k := range concurrencies {
		opts := ManyAgentOptions{
			Concurrency:        k,
			Model:              "Qwen3.8-27B",
			Horizon:            20,
			Cache:              true,
			SharedPrefixTokens: DefaultSharedPrefixTokens,
		}
		rep, err := RunManyAgentSpine(opts)
		if err != nil {
			t.Fatalf("RunManyAgentSpine for K=%d failed: %v", k, err)
		}

		if rep.PrefixEvalCount != 1 {
			t.Errorf("K=%d: PrefixEvalCount = %d, want 1", k, rep.PrefixEvalCount)
		}
		if !rep.TTFTFlat {
			t.Errorf("K=%d: TTFTFlat = false", k)
		}
		if !rep.Verified {
			t.Errorf("K=%d: Verified = false", k)
		}

		if i == 0 {
			baselineP50 = rep.P50TTFTMS
		} else {
			// Check that p50 TTFT remains flat (within 2.0 ms of baseline)
			diff := rep.P50TTFTMS - baselineP50
			if diff < -2.0 || diff > 2.0 {
				t.Errorf("K=%d: P50 TTFT drifted from baseline %.1f to %.1f (diff %.1f)",
					k, baselineP50, rep.P50TTFTMS, diff)
			}
		}
	}
}

func TestMacBenchManyAgent_CLI(t *testing.T) {
	t.Run("default summary output", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runMacBenchManyAgent(&stdout, &stderr, []string{})
		if code != 0 {
			t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		for _, want := range []string{
			"model=Qwen3.8-27B",
			"concurrency=4",
			"horizon=20",
			"cache=true",
			"prefix        : 4096 tokens",
			"prompt_tokens :",
			"reused_tokens :",
			"agents_per_gb :",
			"p50_ttft_ms   :",
			"p95_ttft_ms   :",
			"peak_memory_mb:",
			"prefix_evals  : 1",
			"verification  : PASS",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("output missing %q:\n%s", want, out)
			}
		}
	})

	t.Run("json output via --output json", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runMacBenchManyAgent(&stdout, &stderr, []string{"--output", "json"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
		}
		var rep ManyAgentReport
		if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
			t.Fatalf("failed to parse JSON output: %v\nOutput: %s", err, stdout.String())
		}
		if rep.Schema != ManyAgentSchema {
			t.Errorf("schema = %q, want %q", rep.Schema, ManyAgentSchema)
		}
		if rep.Provenance != "MODELED" {
			t.Errorf("provenance = %q, want %q", rep.Provenance, "MODELED")
		}
		if rep.IsPhysicalSilicon {
			t.Errorf("is_physical_silicon = true, want false")
		}
		if len(rep.UnmodeledEffects) < 3 {
			t.Errorf("unmodeled_effects len = %d, want >= 3", len(rep.UnmodeledEffects))
		}
		if rep.Model != "Qwen3.8-27B" || rep.Concurrency != 4 || rep.Horizon != 20 || !rep.Cache {
			t.Errorf("unexpected parameters in report: %+v", rep)
		}
		if rep.PrefixEvalCount != 1 || !rep.TTFTFlat || !rep.Verified {
			t.Errorf("verification failed in report: %+v", rep)
		}
		if rep.ReusedTokens == 0 || rep.PromptTokens == 0 || rep.AgentsPerGB <= 0 || rep.P50TTFTMS <= 0 {
			t.Errorf("metrics incomplete in report: %+v", rep)
		}
	})

	t.Run("json output via --json flag", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runMacBenchManyAgent(&stdout, &stderr, []string{"--json"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
		}
		var rep ManyAgentReport
		if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
			t.Fatalf("failed to parse JSON output: %v\nOutput: %s", err, stdout.String())
		}
		if rep.Schema != ManyAgentSchema {
			t.Errorf("schema = %q, want %q", rep.Schema, ManyAgentSchema)
		}
		if rep.Provenance != "MODELED" {
			t.Errorf("provenance = %q, want %q", rep.Provenance, "MODELED")
		}
		if rep.IsPhysicalSilicon {
			t.Errorf("is_physical_silicon = true, want false")
		}
		if len(rep.UnmodeledEffects) < 3 {
			t.Errorf("unmodeled_effects len = %d, want >= 3", len(rep.UnmodeledEffects))
		}
	})

	t.Run("cache off flag", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runMacBenchManyAgent(&stdout, &stderr, []string{"--cache=false", "--concurrency", "2", "--horizon", "5"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		if !strings.Contains(out, "reused_tokens : 0 (0.0% reuse)") {
			t.Errorf("expected 0 reused tokens, got:\n%s", out)
		}
		if !strings.Contains(out, "verification  : FAIL") {
			t.Errorf("expected verification FAIL with cache=false, got:\n%s", out)
		}
	})

	t.Run("invalid flags", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runMacBenchManyAgent(&stdout, &stderr, []string{"--concurrency", "0"})
		if code != 2 {
			t.Errorf("expected code 2 for --concurrency 0, got %d", code)
		}
		if !strings.Contains(stderr.String(), "--concurrency must be positive") {
			t.Errorf("stderr missing error message: %s", stderr.String())
		}
	})

	t.Run("unexpected positional argument", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runMacBenchManyAgent(&stdout, &stderr, []string{"extra_arg"})
		if code != 2 {
			t.Errorf("expected code 2 for unexpected arg, got %d", code)
		}
	})

	t.Run("compare llama flag summary", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runMacBenchManyAgent(&stdout, &stderr, []string{"--compare-llama", "-c", "4", "--horizon", "20"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		if !strings.Contains(out, "fak macbench many-agent head-to-head") {
			t.Errorf("output missing comparison header: %s", out)
		}
		if !strings.Contains(out, "[MODELED PROJECTION]") {
			t.Errorf("output missing [MODELED PROJECTION] banner: %s", out)
		}
		if !strings.Contains(out, "verification          : PROJECTED (MODELED") {
			t.Errorf("expected verification PROJECTED (MODELED, got: %s", out)
		}
		if !strings.Contains(out, "wall-clock speedup projected") {
			t.Errorf("expected speedup projected statement, got: %s", out)
		}
		if strings.Contains(out, "TRUE") || strings.Contains(out, "achieved") {
			t.Errorf("unwanted un-sanitized claim in output: %s", out)
		}
	})

	t.Run("compare llama flag json", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runMacBenchManyAgent(&stdout, &stderr, []string{"--compare-llama", "-c", "4", "--horizon", "20", "--json"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
		}
		var rep ManyAgentComparisonReport
		if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
			t.Fatalf("failed to parse comparison JSON: %v\nOutput: %s", err, stdout.String())
		}
		if rep.Schema != ManyAgentComparisonSchema {
			t.Errorf("schema = %q, want %q", rep.Schema, ManyAgentComparisonSchema)
		}
		if rep.Provenance != "MODELED" {
			t.Errorf("provenance = %q, want %q", rep.Provenance, "MODELED")
		}
		if rep.IsPhysicalSilicon {
			t.Errorf("is_physical_silicon = true, want false")
		}
		if len(rep.UnmodeledEffects) < 3 {
			t.Errorf("unmodeled_effects len = %d, want >= 3", len(rep.UnmodeledEffects))
		}
		if rep.Modeled4xProjected {
			t.Errorf("modeled_4x_projected = true, want false")
		}
		if rep.True4xAchieved {
			t.Errorf("true_4x_achieved = true, want false")
		}
		if rep.SpeedupRatio < 1.5 {
			t.Errorf("speedup ratio = %.2f, want >= 1.5", rep.SpeedupRatio)
		}
		if rep.MemorySavedMB < 700.0 {
			t.Errorf("memory saved = %.1f MB, want > 700 MB", rep.MemorySavedMB)
		}
		if !rep.Verified {
			t.Errorf("expected Verified == true")
		}
	})
}

func TestMacBenchManyAgent_RunManyAgentComparison_True4x(t *testing.T) {
	opts := ManyAgentOptions{
		Concurrency:        4,
		Model:              "Qwen3.8-27B",
		Horizon:            20,
		SharedPrefixTokens: DefaultSharedPrefixTokens,
	}
	rep, err := RunManyAgentComparison(opts)
	if err != nil {
		t.Fatalf("RunManyAgentComparison: %v", err)
	}

	// Witness: llamaTotalWallMS matches physical serial execution without quadratic queue wait double-counting
	// and without unevidenced 600s slot contention.
	// Expected llamaTotalWallMS = ~732201.5 ms (~732.2 s).
	if rep.LlamaCPP.TotalWallMS > 800000.0 {
		t.Errorf("LlamaCPP.TotalWallMS = %.1f ms, want < 800000.0 ms (unphysical queue wait/slot penalty inflated)", rep.LlamaCPP.TotalWallMS)
	}
	if math.Abs(rep.LlamaCPP.TotalWallMS-732201.5) > 10.0 {
		t.Errorf("LlamaCPP.TotalWallMS = %.1f ms, want ~732201.5 ms", rep.LlamaCPP.TotalWallMS)
	}
	if rep.SpeedupRatio < 1.80 || rep.SpeedupRatio > 1.95 {
		t.Errorf("SpeedupRatio = %.2f, want ~1.86x", rep.SpeedupRatio)
	}
	if rep.True4xAchieved {
		t.Errorf("True4xAchieved = true, want false at honest 1.86x speedup")
	}
	if rep.Modeled4xProjected {
		t.Errorf("Modeled4xProjected = true, want false at honest 1.86x speedup")
	}
	if rep.Provenance != "MODELED" {
		t.Errorf("Provenance = %q, want %q", rep.Provenance, "MODELED")
	}
	if rep.IsPhysicalSilicon {
		t.Errorf("IsPhysicalSilicon = true, want false")
	}
	if len(rep.UnmodeledEffects) < 3 {
		t.Errorf("UnmodeledEffects len = %d, want >= 3", len(rep.UnmodeledEffects))
	}
	if rep.MemorySavedMB < 700.0 {
		t.Errorf("MemorySavedMB = %.1f, want > 700.0 MB", rep.MemorySavedMB)
	}
	if rep.TTFTSpeedupP50 < 1.5 || rep.TTFTSpeedupP50 > 2.0 {
		t.Errorf("TTFTSpeedupP50 = %.2f, want ~1.71x (matched distribution)", rep.TTFTSpeedupP50)
	}
	if rep.Turn1ColdTTFTLlamaMS/rep.Turn1ColdTTFTFakMS < 1000.0 {
		t.Errorf("Turn 1 cold TTFT speedup = %.1f, want > 1000.0x", rep.Turn1ColdTTFTLlamaMS/rep.Turn1ColdTTFTFakMS)
	}
	if !rep.Verified {
		t.Errorf("Verified = false, want true")
	}
	if rep.FakNative.PrefixEvalCount != 1 {
		t.Errorf("FakNative prefix eval = %d, want 1", rep.FakNative.PrefixEvalCount)
	}
	if rep.LlamaCPP.PrefixEvalCount != 4 {
		t.Errorf("LlamaCPP prefix eval = %d, want 4", rep.LlamaCPP.PrefixEvalCount)
	}
}

func TestMacBenchManyAgent_TwoBitQuantModelSpec(t *testing.T) {
	spec := resolveManyAgentModelSpec("Qwen3.8-27B-UD-Q2_K_XL")
	if spec.WeightMB > 10000.0 || spec.WeightMB < 9000.0 {
		t.Errorf("WeightMB for 2-bit quant = %.1f, want ~9830 MB", spec.WeightMB)
	}
	if spec.BasePrefillTokPerSec <= 65.0 {
		t.Errorf("BasePrefillTokPerSec for 2-bit quant = %.1f, want > 65.0", spec.BasePrefillTokPerSec)
	}
}

func TestMacBenchManyAgent_20kContext_Qwen38_UD_Q2_K_XL(t *testing.T) {
	opts := ManyAgentOptions{
		Model:              "Qwen3.8-27B-UD-Q2_K_XL",
		SharedPrefixTokens: 20000,
		Concurrency:        4,
		Horizon:            20,
		Cache:              true,
	}

	rep, err := RunManyAgentSpine(opts)
	if err != nil {
		t.Fatalf("RunManyAgentSpine failed: %v", err)
	}
	if rep.SharedPrefixTokens != 20000 {
		t.Errorf("rep.SharedPrefixTokens = %d, want 20000", rep.SharedPrefixTokens)
	}
	if !rep.Verified {
		t.Errorf("rep.Verified = false, want true")
	}
	if rep.ReuseRatio <= 0.90 {
		t.Errorf("rep.ReuseRatio = %.4f, want > 0.90", rep.ReuseRatio)
	}

	compRep, err := RunManyAgentComparison(opts)
	if err != nil {
		t.Fatalf("RunManyAgentComparison failed: %v", err)
	}
	// At 20k context, honest physical speedup is ~2.51x (previously artificially inflated to 6.06x
	// by double-counting 1411.8s queue contention and adding 600s slot contention).
	if compRep.LlamaCPP.TotalWallMS > 1500000.0 {
		t.Errorf("compRep.LlamaCPP.TotalWallMS = %.1f ms, want < 1500000.0 ms", compRep.LlamaCPP.TotalWallMS)
	}
	if math.Abs(compRep.LlamaCPP.TotalWallMS-1421316.5) > 10.0 {
		t.Errorf("compRep.LlamaCPP.TotalWallMS = %.1f ms, want ~1421316.5 ms", compRep.LlamaCPP.TotalWallMS)
	}
	if compRep.SpeedupRatio < 2.4 || compRep.SpeedupRatio > 2.6 {
		t.Errorf("compRep.SpeedupRatio = %.2f, want ~2.51x", compRep.SpeedupRatio)
	}
	if compRep.True4xAchieved {
		t.Errorf("compRep.True4xAchieved = true, want false at honest 2.51x speedup")
	}
	if compRep.Modeled4xProjected {
		t.Errorf("compRep.Modeled4xProjected = true, want false at honest 2.51x speedup")
	}
	if compRep.Provenance != "MODELED" {
		t.Errorf("compRep.Provenance = %q, want %q", compRep.Provenance, "MODELED")
	}
	if compRep.IsPhysicalSilicon {
		t.Errorf("compRep.IsPhysicalSilicon = true, want false")
	}
	if len(compRep.UnmodeledEffects) < 3 {
		t.Errorf("compRep.UnmodeledEffects len = %d, want >= 3", len(compRep.UnmodeledEffects))
	}
	if !compRep.Verified {
		t.Errorf("compRep.Verified = false, want true")
	}

	var stdout, stderr bytes.Buffer
	argv := []string{"--prefix-tokens", "20000", "--model", "Qwen3.8-27B-UD-Q2_K_XL", "--compare-llama", "--json"}
	code := runMacBenchManyAgent(&stdout, &stderr, argv)
	if code != 0 {
		t.Fatalf("runMacBenchManyAgent returned code %d, stderr: %s", code, stderr.String())
	}

	var cliRep ManyAgentComparisonReport
	if err := json.Unmarshal(stdout.Bytes(), &cliRep); err != nil {
		t.Fatalf("failed to parse CLI JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if cliRep.Schema != ManyAgentComparisonSchema {
		t.Errorf("cliRep.Schema = %q, want %q", cliRep.Schema, ManyAgentComparisonSchema)
	}
	if cliRep.Provenance != "MODELED" {
		t.Errorf("cliRep.Provenance = %q, want %q", cliRep.Provenance, "MODELED")
	}
	if cliRep.IsPhysicalSilicon {
		t.Errorf("cliRep.IsPhysicalSilicon = true, want false")
	}
	if len(cliRep.UnmodeledEffects) < 3 {
		t.Errorf("cliRep.UnmodeledEffects len = %d, want >= 3", len(cliRep.UnmodeledEffects))
	}
	if cliRep.SharedPrefixTokens != 20000 {
		t.Errorf("cliRep.SharedPrefixTokens = %d, want 20000", cliRep.SharedPrefixTokens)
	}
	if cliRep.Model != "Qwen3.8-27B-UD-Q2_K_XL" {
		t.Errorf("cliRep.Model = %q, want %q", cliRep.Model, "Qwen3.8-27B-UD-Q2_K_XL")
	}
	if cliRep.SpeedupRatio < 2.4 || cliRep.SpeedupRatio > 2.6 {
		t.Errorf("cliRep.SpeedupRatio = %.2f, want ~2.51x", cliRep.SpeedupRatio)
	}
	if cliRep.True4xAchieved {
		t.Errorf("cliRep.True4xAchieved = true, want false at honest 2.51x speedup")
	}
	if cliRep.Modeled4xProjected {
		t.Errorf("cliRep.Modeled4xProjected = true, want false at honest 2.51x speedup")
	}
	if !cliRep.Verified {
		t.Errorf("cliRep.Verified = false, want true")
	}
}

func TestManyAgent_ExplicitZeroPrefixTokens(t *testing.T) {
	opts := ManyAgentOptions{
		Concurrency:        4,
		Model:              "Qwen3.8-27B",
		Horizon:            20,
		Cache:              true,
		SharedPrefixTokens: 0,
	}

	rep, err := RunManyAgentSpine(opts)
	if err != nil {
		t.Fatalf("RunManyAgentSpine failed: %v", err)
	}
	if rep.SharedPrefixTokens != 0 {
		t.Errorf("rep.SharedPrefixTokens = %d, want 0 (did silent fallback to %d occur?)",
			rep.SharedPrefixTokens, DefaultSharedPrefixTokens)
	}

	compRep, err := RunManyAgentComparison(opts)
	if err != nil {
		t.Fatalf("RunManyAgentComparison failed: %v", err)
	}
	if compRep.SharedPrefixTokens != 0 {
		t.Errorf("compRep.SharedPrefixTokens = %d, want 0", compRep.SharedPrefixTokens)
	}

	// Verify via CLI flag --prefix-tokens 0
	var stdout, stderr bytes.Buffer
	code := runMacBenchManyAgent(&stdout, &stderr, []string{"--prefix-tokens", "0", "--json"})
	if code != 0 {
		t.Fatalf("runMacBenchManyAgent --prefix-tokens 0 returned code %d, stderr: %s", code, stderr.String())
	}
	var cliRep ManyAgentReport
	if err := json.Unmarshal(stdout.Bytes(), &cliRep); err != nil {
		t.Fatalf("failed to parse CLI JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if cliRep.SharedPrefixTokens != 0 {
		t.Errorf("cliRep.SharedPrefixTokens = %d, want 0", cliRep.SharedPrefixTokens)
	}

	// Verify summary format prints 0 tokens
	stdout.Reset()
	stderr.Reset()
	code = runMacBenchManyAgent(&stdout, &stderr, []string{"-p", "0"})
	if code != 0 {
		t.Fatalf("runMacBenchManyAgent -p 0 returned code %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "prefix        : 0 tokens") {
		t.Errorf("summary output missing 'prefix        : 0 tokens':\n%s", stdout.String())
	}
}

func TestManyAgentComparison(t *testing.T) {
	opts := ManyAgentOptions{
		Concurrency:        4,
		Model:              "Qwen3.8-27B",
		Horizon:            20,
		SharedPrefixTokens: DefaultSharedPrefixTokens,
	}
	rep, err := RunManyAgentComparison(opts)
	if err != nil {
		t.Fatalf("RunManyAgentComparison: %v", err)
	}

	t.Run("SymmetricTTFTDistribution", func(t *testing.T) {
		// Verify both arms evaluate TTFT over all K * H turns (80 turns for K=4, H=20).
		// Fak-native has 1 cold prefill and 79 warm cache hits -> p50 is warm (~12.3 ms).
		if rep.FakNative.P50TTFTMS >= 25.0 || rep.FakNative.P50TTFTMS <= 0 {
			t.Errorf("FakNative.P50TTFTMS = %.1f ms, want warm (< 25 ms, > 0)", rep.FakNative.P50TTFTMS)
		}
		// Llama.cpp has 4 cold turns and 76 steady-state turns -> p50 falls in steady-state delta (~21 ms).
		// If unaligned (sampling only Turn 1), p50 would be > 100,000 ms.
		if rep.LlamaCPP.P50TTFTMS > 100.0 || rep.LlamaCPP.P50TTFTMS <= 0 {
			t.Errorf("LlamaCPP.P50TTFTMS = %.1f ms, want steady-state (< 100 ms, > 0); was it evaluated over symmetric turns?", rep.LlamaCPP.P50TTFTMS)
		}
		// Overall P50 speedup reflects matched K*H distribution (~1.71x).
		if rep.TTFTSpeedupP50 < 1.5 || rep.TTFTSpeedupP50 > 2.0 {
			t.Errorf("TTFTSpeedupP50 = %.2f, want ~1.71x (matched distribution)", rep.TTFTSpeedupP50)
		}
	})

	t.Run("ColdAndSteadyStateFields", func(t *testing.T) {
		// 1. Verify populated cold and steady-state TTFT fields
		if rep.Turn1ColdTTFTFakMS <= 0 {
			t.Errorf("Turn1ColdTTFTFakMS = %.1f, want > 0", rep.Turn1ColdTTFTFakMS)
		}
		if rep.Turn1ColdTTFTLlamaMS <= 0 {
			t.Errorf("Turn1ColdTTFTLlamaMS = %.1f, want > 0", rep.Turn1ColdTTFTLlamaMS)
		}
		if rep.SteadyStateTTFTFakMS <= 0 {
			t.Errorf("SteadyStateTTFTFakMS = %.1f, want > 0", rep.SteadyStateTTFTFakMS)
		}
		if rep.SteadyStateTTFTLlamaMS <= 0 {
			t.Errorf("SteadyStateTTFTLlamaMS = %.1f, want > 0", rep.SteadyStateTTFTLlamaMS)
		}

		// 2. Physical invariants:
		// Turn 1 cold prefill for llama.cpp is heavily serialized across slots (> 100s)
		if rep.Turn1ColdTTFTLlamaMS < 100000.0 {
			t.Errorf("Turn1ColdTTFTLlamaMS = %.1f ms, want > 100,000 ms", rep.Turn1ColdTTFTLlamaMS)
		}
		// Fak-native Turn 1 cold median benefits from prefix reuse for agents 2..K (< 50ms)
		if rep.Turn1ColdTTFTFakMS > 50.0 {
			t.Errorf("Turn1ColdTTFTFakMS = %.1f ms, want < 50 ms", rep.Turn1ColdTTFTFakMS)
		}
		// Steady state for llama.cpp operates at delta prefill with contention (~21ms)
		if rep.SteadyStateTTFTLlamaMS < 20.0 || rep.SteadyStateTTFTLlamaMS > 25.0 {
			t.Errorf("SteadyStateTTFTLlamaMS = %.1f ms, want ~21 ms", rep.SteadyStateTTFTLlamaMS)
		}
		// Steady state for fak-native operates at delta prefill with RadixAttention hit (~12ms)
		if rep.SteadyStateTTFTFakMS < 11.0 || rep.SteadyStateTTFTFakMS > 15.0 {
			t.Errorf("SteadyStateTTFTFakMS = %.1f ms, want ~12 ms", rep.SteadyStateTTFTFakMS)
		}

		// 3. Valid speedup calculations on matched distributions
		// Steady-state speedup is matched (~1.68x)
		steadySpeedup := rep.SteadyStateTTFTLlamaMS / rep.SteadyStateTTFTFakMS
		if steadySpeedup < 1.5 || steadySpeedup > 2.0 {
			t.Errorf("steadySpeedup = %.2f, want ~1.68x", steadySpeedup)
		}
		// Cold Turn 1 speedup remains massive (> 1000x)
		coldSpeedup := rep.Turn1ColdTTFTLlamaMS / rep.Turn1ColdTTFTFakMS
		if coldSpeedup < 1000.0 {
			t.Errorf("coldSpeedup = %.1f, want > 1000.0x", coldSpeedup)
		}
	})

	t.Run("JSONSerialization", func(t *testing.T) {
		// Verify JSON marshaling/unmarshaling includes and preserves new fields
		data, err := json.Marshal(rep)
		if err != nil {
			t.Fatalf("json.Marshal failed: %v", err)
		}
		jsonStr := string(data)
		for _, key := range []string{
			`"turn1_cold_ttft_fak_ms"`,
			`"turn1_cold_ttft_llama_ms"`,
			`"steady_state_ttft_fak_ms"`,
			`"steady_state_ttft_llama_ms"`,
		} {
			if !strings.Contains(jsonStr, key) {
				t.Errorf("JSON missing key %q: %s", key, jsonStr)
			}
		}
		var unmarshaled ManyAgentComparisonReport
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("json.Unmarshal failed: %v", err)
		}
		if unmarshaled.Turn1ColdTTFTFakMS != rep.Turn1ColdTTFTFakMS ||
			unmarshaled.Turn1ColdTTFTLlamaMS != rep.Turn1ColdTTFTLlamaMS ||
			unmarshaled.SteadyStateTTFTFakMS != rep.SteadyStateTTFTFakMS ||
			unmarshaled.SteadyStateTTFTLlamaMS != rep.SteadyStateTTFTLlamaMS {
			t.Errorf("unmarshaled fields mismatch: %+v vs %+v", unmarshaled, rep)
		}
	})

	t.Run("Horizon1EdgeCase", func(t *testing.T) {
		// Verify edge case: horizon = 1 (only Turn 1, no steady state)
		optsH1 := ManyAgentOptions{
			Concurrency:        2,
			Model:              "Qwen3.8-27B",
			Horizon:            1,
			SharedPrefixTokens: DefaultSharedPrefixTokens,
		}
		repH1, err := RunManyAgentComparison(optsH1)
		if err != nil {
			t.Fatalf("RunManyAgentComparison (H=1) failed: %v", err)
		}
		if repH1.Turn1ColdTTFTFakMS <= 0 || repH1.Turn1ColdTTFTLlamaMS <= 0 {
			t.Errorf("H=1 cold TTFT fields must be positive: fak=%.1f, llama=%.1f",
				repH1.Turn1ColdTTFTFakMS, repH1.Turn1ColdTTFTLlamaMS)
		}
		if repH1.SteadyStateTTFTFakMS != 0 || repH1.SteadyStateTTFTLlamaMS != 0 {
			t.Errorf("H=1 steady-state TTFT fields must be 0: fak=%.1f, llama=%.1f",
				repH1.SteadyStateTTFTFakMS, repH1.SteadyStateTTFTLlamaMS)
		}
	})

	t.Run("CLISummaryFormatting", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runMacBenchManyAgent(&stdout, &stderr, []string{"--compare-llama", "-c", "4", "--horizon", "20"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		for _, want := range []string{
			"p50_ttft_ms",
			"turn1_cold_ttft_ms",
			"steady_ttft_ms",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("CLI summary missing %q:\n%s", want, out)
			}
		}
	})
}

func TestMacBenchManyAgent_SymmetricTTFTDistribution(t *testing.T) {
	TestManyAgentComparison(t)
}

func TestManyAgent_Qwen38HybridKVCacheDiscount(t *testing.T) {
	spec := resolveManyAgentModelSpec("Qwen3.8-27B")

	// 1. Verify model architecture parameters for Qwen3.8-27B:
	// 64 total layers, 16 full-attention layers, 48 recurrent layers (3:1 linear-to-full attention cadence).
	if spec.Layers != 64 {
		t.Errorf("Layers = %d, want 64", spec.Layers)
	}
	if spec.FullAttnLayers != 16 {
		t.Errorf("FullAttnLayers = %d, want 16", spec.FullAttnLayers)
	}
	if spec.RecurrentLayers != 48 {
		t.Errorf("RecurrentLayers = %d, want 48", spec.RecurrentLayers)
	}
	if spec.EffectiveKVLayers() != 16 {
		t.Errorf("EffectiveKVLayers = %d, want 16", spec.EffectiveKVLayers())
	}
	if spec.HybridRatio() != 0.25 {
		t.Errorf("HybridRatio = %.4f, want 0.25 (3:1 GDN ratio)", spec.HybridRatio())
	}
	if spec.RecurrentStateBytes != DefaultQwen38RecurrentStateBytes {
		t.Errorf("RecurrentStateBytes = %d, want %d", spec.RecurrentStateBytes, DefaultQwen38RecurrentStateBytes)
	}

	// 2. Verify token-indexed KV cache bytes per token:
	// 2 * 16 layers * 8 heads * 128 head_dim * 2 bytes = 65,536 bytes/token.
	// Uniform 64-layer calculation would be 262,144 bytes/token (4x higher).
	wantBytesPerToken := uint64(2 * 16 * 8 * 128 * 2) // 65,536 bytes
	if spec.KVBytesPerToken() != wantBytesPerToken {
		t.Errorf("KVBytesPerToken = %d, want %d", spec.KVBytesPerToken(), wantBytesPerToken)
	}
	uniformBytesPerToken := uint64(2 * spec.Layers * spec.KVHeads * spec.HeadDim * spec.BytesPerElement)
	if uniformBytesPerToken != 4*wantBytesPerToken {
		t.Errorf("uniformBytesPerToken = %d, want 4x %d", uniformBytesPerToken, wantBytesPerToken)
	}

	// 3. Analytical memory calculation at large context (20k context, e.g. 60,000 total tokens across 3 sessions):
	// Token-indexed KV cache with 16 layers: 60,000 * 65,536 = 3,932,160,000 bytes (~3.93 GB).
	// Overestimated uniform 64 layers: 60,000 * 262,144 = 15,728,640,000 bytes (~15.7 GB).
	const tokens60k = uint64(60000)
	kvBytes60k := spec.KVBytes(tokens60k)
	wantKVBytes60k := uint64(3932160000)
	if kvBytes60k != wantKVBytes60k {
		t.Errorf("KVBytes(60000) = %d, want %d (~3.93 GB)", kvBytes60k, wantKVBytes60k)
	}
	uniformKVBytes60k := tokens60k * uniformBytesPerToken
	wantUniformKVBytes60k := uint64(15728640000)
	if uniformKVBytes60k != wantUniformKVBytes60k {
		t.Errorf("uniform KV bytes = %d, want %d (~15.7 GB)", uniformKVBytes60k, wantUniformKVBytes60k)
	}
	if float64(uniformKVBytes60k)/float64(kvBytes60k) != 4.0 {
		t.Errorf("expected 4.0x ratio between uniform and hybrid KV bytes, got %.2f", float64(uniformKVBytes60k)/float64(kvBytes60k))
	}

	// 4. Verify simulation memory with RunManyAgentSpine reflects the 16 full-attention layers:
	// For K=4, prefix=4096:
	// Prefix caching saves 3 copies of 4096 tokens = 3 * 4096 * 65,536 = 805,306,368 bytes = 768.0 MB.
	optsOn := ManyAgentOptions{
		Concurrency:        4,
		Model:              "Qwen3.8-27B",
		Horizon:            20,
		Cache:              true,
		SharedPrefixTokens: 4096,
	}
	repOn, err := RunManyAgentSpine(optsOn)
	if err != nil {
		t.Fatalf("RunManyAgentSpine on failed: %v", err)
	}
	optsOff := optsOn
	optsOff.Cache = false
	repOff, err := RunManyAgentSpine(optsOff)
	if err != nil {
		t.Fatalf("RunManyAgentSpine off failed: %v", err)
	}
	savedMB := repOff.PeakMemoryMB - repOn.PeakMemoryMB
	const expectedSavedMB = 768.0
	if math.Abs(savedMB-expectedSavedMB) > 1.0 {
		t.Errorf("savedMB = %.1f, want %.1f MB (4x discount from 3072 MB)", savedMB, expectedSavedMB)
	}
}

func TestMacBenchManyAgent_Qwen38HybridKVCacheDiscount(t *testing.T) {
	TestManyAgent_Qwen38HybridKVCacheDiscount(t)
}
