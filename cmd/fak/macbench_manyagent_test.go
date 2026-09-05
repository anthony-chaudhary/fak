package main

import (
	"bytes"
	"encoding/json"
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
		Concurrency: 4,
		Model:       "Qwen3.8-27B",
		Horizon:     20,
		Cache:       true,
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
		Concurrency: 4,
		Model:       "Qwen3.8-27B",
		Horizon:     20,
		Cache:       false,
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
		Concurrency: 4,
		Model:       "Qwen3.8-27B",
		Horizon:     20,
		Cache:       true,
	}
	repOn, err := RunManyAgentSpine(optsOn)
	if err != nil {
		t.Fatalf("RunManyAgentSpine on failed: %v", err)
	}

	optsOff := ManyAgentOptions{
		Concurrency: 4,
		Model:       "Qwen3.8-27B",
		Horizon:     20,
		Cache:       false,
	}
	repOff, err := RunManyAgentSpine(optsOff)
	if err != nil {
		t.Fatalf("RunManyAgentSpine off failed: %v", err)
	}

	// 1. Peak memory with cache=true must be significantly lower than cache=false
	// For K=4, 3 copies of 4096-token prefix (3 * 1024 MB = 3072 MB) are saved.
	savedMB := repOff.PeakMemoryMB - repOn.PeakMemoryMB
	if savedMB < 3000.0 || savedMB > 3150.0 {
		t.Errorf("expected ~3072 MB saved by caching 4096 prefix across 4 agents, got %.1f MB", savedMB)
	}

	// 2. Agents per GB must be higher with cache=true
	if repOn.AgentsPerGB <= repOff.AgentsPerGB {
		t.Errorf("expected AgentsPerGB on (%.2f) > off (%.2f)", repOn.AgentsPerGB, repOff.AgentsPerGB)
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
			Concurrency: k,
			Model:       "Qwen3.8-27B",
			Horizon:     20,
			Cache:       true,
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
		if !strings.Contains(out, "verification          : PASS") {
			t.Errorf("expected verification PASS, got: %s", out)
		}
		if !strings.Contains(out, "TRUE") || !strings.Contains(out, ">= 4.0x achieved") {
			t.Errorf("expected TRUE 4x achieved statement, got: %s", out)
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
		if !rep.True4xAchieved || rep.SpeedupRatio < 4.0 {
			t.Errorf("speedup ratio = %.2f, True4xAchieved = %v, want >= 4.0", rep.SpeedupRatio, rep.True4xAchieved)
		}
		if rep.MemorySavedMB < 3000.0 {
			t.Errorf("memory saved = %.1f MB, want > 3000 MB", rep.MemorySavedMB)
		}
		if !rep.Verified {
			t.Errorf("expected Verified == true")
		}
	})
}

func TestMacBenchManyAgent_RunManyAgentComparison_True4x(t *testing.T) {
	opts := ManyAgentOptions{
		Concurrency: 4,
		Model:       "Qwen3.8-27B",
		Horizon:     20,
	}
	rep, err := RunManyAgentComparison(opts)
	if err != nil {
		t.Fatalf("RunManyAgentComparison: %v", err)
	}

	if rep.SpeedupRatio < 4.0 {
		t.Errorf("SpeedupRatio = %.2f, want >= 4.00 (True 4x)", rep.SpeedupRatio)
	}
	if !rep.True4xAchieved {
		t.Errorf("True4xAchieved = false, want true")
	}
	if rep.MemorySavedMB < 3000.0 {
		t.Errorf("MemorySavedMB = %.1f, want > 3000.0 MB", rep.MemorySavedMB)
	}
	if rep.TTFTSpeedupP50 < 1000.0 {
		t.Errorf("TTFTSpeedupP50 = %.1f, want > 1000.0x", rep.TTFTSpeedupP50)
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
