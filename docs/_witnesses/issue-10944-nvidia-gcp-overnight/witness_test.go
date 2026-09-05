package issue10944witness_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

var pinnedFiles = map[string]string{
	"fak_native_qwen38_a100_bench_raw.json": "32558710b7673d63465c52d0442d60c886a74b6b77631f4aaf8d826ead689c38",
	"qwen38_fanout_concurrency_raw.json":    "63a6bf4a50466ad8741826a2496d6a9e7fd7939109873ee71f000d4658f14384",
	"vllm_fp8_a100_bench_raw.json":          "c6b5178aaaf7df9aa2823e076207a0fa0e44147cc112d8e5af27b06784245b1d",
	"vllm_bf16_tp2_a100_bench_raw.json":     "d92742184c933b5a74a00240f59038bde5645aba72632fbeeead03e1c65adadd",
	"gcp_h100_hopper_bench_raw.json":        "490ea3611de9c6fddf1877d2329cd68c5da83e7bbcec9268d84f4191fa59bfb6",
	"gcp_h100_paired_report.json":           "4e7ca8d04c58db5cc168ffe652a9e2ea7ff97f38aced9af08a29c6684e5610ae",
	"qwen38_a100_bench_v2_raw.json":         "32ae037c91039f7079faa30ab250b2695678328264868cb0f0293c32ef1ab065",
	"fold-report.json":                      "a722dd98225391d882c67c2ae23189ad95065ed85871af1411a60964e4b85fa6",
}

func readAndVerify(t *testing.T, filename string) []byte {
	t.Helper()
	wantHash, ok := pinnedFiles[filename]
	if !ok {
		t.Fatalf("unpinned file %s", filename)
	}
	content, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("failed to read %s: %v", filename, err)
	}
	sum := sha256.Sum256(content)
	gotHash := hex.EncodeToString(sum[:])
	if gotHash != wantHash {
		t.Fatalf("hash mismatch for %s: got %s, want %s", filename, gotHash, wantHash)
	}
	return content
}

func TestPinnedFilesIntegrity(t *testing.T) {
	for filename := range pinnedFiles {
		readAndVerify(t, filename)
	}
}

func TestNativeBenchmarkInvariants(t *testing.T) {
	data := readAndVerify(t, "fak_native_qwen38_a100_bench_raw.json")

	var root struct {
		Schema   string `json:"schema"`
		Hardware struct {
			GPU string `json:"gpu"`
		} `json:"hardware"`
		Service struct {
			Model   string `json:"model"`
			Backend string `json:"backend"`
		} `json:"service"`
		ContextScalingSweep []struct {
			Success            bool    `json:"success"`
			TargetPromptTokens int     `json:"target_prompt_tokens"`
			PromptTokens       int     `json:"prompt_tokens"`
			CompletionTokens   int     `json:"completion_tokens"`
			TTFTMS             float64 `json:"ttft_ms"`
			DecodeTokensPerSec float64 `json:"decode_tokens_per_second"`
			Error              *string `json:"error"`
		} `json:"context_scaling_sweep"`
		PrefixCachingAblation struct {
			ColdRun struct {
				TTFTMS float64 `json:"ttft_ms"`
			} `json:"cold_run"`
			WarmRun1 struct {
				TTFTMS float64 `json:"ttft_ms"`
			} `json:"warm_run_1"`
			WarmRun2 struct {
				TTFTMS float64 `json:"ttft_ms"`
			} `json:"warm_run_2"`
			UniqueNonceRun struct {
				TTFTMS float64 `json:"ttft_ms"`
			} `json:"unique_nonce_run"`
			Metrics struct {
				TTFTColdMS            float64 `json:"ttft_cold_ms"`
				TTFTWarmMS            float64 `json:"ttft_warm_ms"`
				TTFTUniqueMS          float64 `json:"ttft_unique_ms"`
				PrefixSpeedupVsCold   float64 `json:"prefix_speedup_vs_cold"`
				PrefixSpeedupVsUnique float64 `json:"prefix_speedup_vs_unique"`
			} `json:"metrics"`
		} `json:"prefix_caching_ablation"`
	}

	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatalf("failed to parse native bench JSON: %v", err)
	}

	if root.Schema != "fak.qwen38-native-a100-benchmark/1" {
		t.Fatalf("unexpected schema: %s", root.Schema)
	}
	if root.Hardware.GPU != "NVIDIA A100-SXM4-40GB" {
		t.Fatalf("unexpected GPU: %s", root.Hardware.GPU)
	}
	if root.Service.Model != "Qwen3.8-27B-Q4_K_M" {
		t.Fatalf("unexpected model: %s", root.Service.Model)
	}

	// Verify context scaling passes for 256, 1024, 2048, 4096
	successes := 0
	oomCount := 0
	for _, row := range root.ContextScalingSweep {
		if row.TargetPromptTokens <= 4096 {
			if !row.Success {
				t.Fatalf("expected success for target tokens %d", row.TargetPromptTokens)
			}
			if row.DecodeTokensPerSec <= 0 {
				t.Fatalf("decode tokens per sec must be positive, got %f", row.DecodeTokensPerSec)
			}
			successes++
		} else if row.TargetPromptTokens == 8192 {
			if row.Success {
				t.Fatalf("expected OOM fail-closed at 8192 on 40GB A100")
			}
			oomCount++
		}
	}
	if successes != 12 { // 4 context lengths * 3 reps
		t.Fatalf("expected 12 successful runs, got %d", successes)
	}
	if oomCount != 3 { // 3 reps for 8192
		t.Fatalf("expected 3 OOM reps, got %d", oomCount)
	}

	// Verify prefix caching speedup
	ab := root.PrefixCachingAblation
	if ab.Metrics.PrefixSpeedupVsCold < 1.5 {
		t.Fatalf("prefix speedup vs cold expected >= 1.5, got %f", ab.Metrics.PrefixSpeedupVsCold)
	}
	if ab.Metrics.PrefixSpeedupVsUnique < 2.0 {
		t.Fatalf("prefix speedup vs unique expected >= 2.0, got %f", ab.Metrics.PrefixSpeedupVsUnique)
	}
	// Best warm run vs cold run speedup
	warm1Speedup := ab.ColdRun.TTFTMS / ab.WarmRun1.TTFTMS
	if warm1Speedup < 4.0 {
		t.Fatalf("warm run 1 speedup vs cold expected >= 4.0, got %f", warm1Speedup)
	}
	if ab.WarmRun1.TTFTMS >= ab.ColdRun.TTFTMS {
		t.Fatalf("warm TTFT (%f) must be lower than cold TTFT (%f)", ab.WarmRun1.TTFTMS, ab.ColdRun.TTFTMS)
	}
}

func TestFanoutConcurrencyInvariants(t *testing.T) {
	data := readAndVerify(t, "qwen38_fanout_concurrency_raw.json")

	var root struct {
		Schema        string `json:"schema"`
		Configuration struct {
			Model              string `json:"model"`
			SharedPrefixTokens int    `json:"shared_prefix_tokens"`
			ConcurrencyGrid    []int  `json:"concurrency_grid"`
		} `json:"configuration"`
		GridResults map[string]struct {
			Concurrency                     int     `json:"concurrency"`
			TotalWallMS                     float64 `json:"total_wall_ms"`
			EffectiveSubagentsPerSec        float64 `json:"effective_subagents_per_sec"`
			SuccessCount                    int     `json:"success_count"`
			ErrorCount                      int     `json:"error_count"`
			EffectiveThroughputTokensPerSec float64 `json:"effective_throughput_tokens_per_sec"`
		} `json:"grid_results"`
	}

	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatalf("failed to parse fanout concurrency JSON: %v", err)
	}

	if root.Schema != "fak.qwen38-fanout-concurrency-raw/1" {
		t.Fatalf("unexpected schema: %s", root.Schema)
	}
	if root.Configuration.Model != "Qwen3.8-27B-Q4_K_M" {
		t.Fatalf("unexpected model: %s", root.Configuration.Model)
	}

	expectedGrid := []int{1, 5, 10, 25, 50}
	totalSubagents := 0
	for _, n := range expectedGrid {
		key := string(rune('0' + n))
		if n >= 10 {
			key = filepath.Base(filepath.Join("", string(rune('0'+n/10))+string(rune('0'+n%10))))
		}
		// Look up key from map
		res, ok := root.GridResults[key]
		if !ok {
			// Try direct format
			for _, r := range root.GridResults {
				if r.Concurrency == n {
					res = r
					ok = true
					break
				}
			}
		}
		if !ok {
			t.Fatalf("missing concurrency grid entry for N=%d", n)
		}
		if res.ErrorCount != 0 {
			t.Fatalf("expected zero errors for N=%d, got %d", n, res.ErrorCount)
		}
		if res.SuccessCount != n {
			t.Fatalf("expected %d successes for N=%d, got %d", n, n, res.SuccessCount)
		}
		if res.EffectiveThroughputTokensPerSec < 900.0 {
			t.Fatalf("effective throughput for N=%d must be >= 900 tok/s, got %f", n, res.EffectiveThroughputTokensPerSec)
		}
		if res.EffectiveSubagentsPerSec < 0.6 {
			t.Fatalf("effective subagents/s for N=%d must be >= 0.6, got %f", n, res.EffectiveSubagentsPerSec)
		}
		totalSubagents += res.SuccessCount
	}

	if totalSubagents != 91 {
		t.Fatalf("expected 91 total subagent requests, got %d", totalSubagents)
	}
}

func TestReferenceRuntimesInvariants(t *testing.T) {
	fp8Data := readAndVerify(t, "vllm_fp8_a100_bench_raw.json")
	var fp8Root struct {
		Schema             string `json:"schema"`
		Model              string `json:"model"`
		TensorParallelSize int    `json:"tensor_parallel_size"`
		Dtype              string `json:"dtype"`
		Summary            struct {
			Repeated map[string]struct {
				AvgTTFTMS float64 `json:"avg_ttft_ms"`
				MinTTFTMS float64 `json:"min_ttft_ms"`
			} `json:"repeated"`
			Unique map[string]struct {
				AvgTTFTMS float64 `json:"avg_ttft_ms"`
				MinTTFTMS float64 `json:"min_ttft_ms"`
			} `json:"unique"`
		} `json:"summary"`
	}
	if err := json.Unmarshal(fp8Data, &fp8Root); err != nil {
		t.Fatalf("failed to parse FP8 bench JSON: %v", err)
	}
	if fp8Root.Schema != "fak.qwen38-fp8-perf-raw/1" {
		t.Fatalf("unexpected schema: %s", fp8Root.Schema)
	}
	if fp8Root.TensorParallelSize != 1 {
		t.Fatalf("expected TP=1 for FP8, got %d", fp8Root.TensorParallelSize)
	}

	bf16Data := readAndVerify(t, "vllm_bf16_tp2_a100_bench_raw.json")
	var bf16Root struct {
		Schema             string `json:"schema"`
		Model              string `json:"model"`
		TensorParallelSize int    `json:"tensor_parallel_size"`
		Dtype              string `json:"dtype"`
		Summary            struct {
			Repeated map[string]struct {
				AvgTTFTMS float64 `json:"avg_ttft_ms"`
				MinTTFTMS float64 `json:"min_ttft_ms"`
			} `json:"repeated"`
			Unique map[string]struct {
				AvgTTFTMS float64 `json:"avg_ttft_ms"`
				MinTTFTMS float64 `json:"min_ttft_ms"`
			} `json:"unique"`
		} `json:"summary"`
	}
	if err := json.Unmarshal(bf16Data, &bf16Root); err != nil {
		t.Fatalf("failed to parse BF16 bench JSON: %v", err)
	}
	if bf16Root.Schema != "fak.qwen38-bf16-tp2-perf-raw/1" {
		t.Fatalf("unexpected schema: %s", bf16Root.Schema)
	}
	if bf16Root.TensorParallelSize != 2 {
		t.Fatalf("expected TP=2 for BF16, got %d", bf16Root.TensorParallelSize)
	}

	// Compare 8k cold TTFT: BF16 TP2 should be roughly 2x faster than single-GPU FP8
	fp8Cold8k := fp8Root.Summary.Unique["8192"].AvgTTFTMS
	bf16Cold8k := bf16Root.Summary.Unique["8192"].AvgTTFTMS
	if fp8Cold8k <= 0 || bf16Cold8k <= 0 {
		t.Fatalf("invalid 8k TTFT: fp8=%f, bf16=%f", fp8Cold8k, bf16Cold8k)
	}
	ratio := fp8Cold8k / bf16Cold8k
	if ratio < 1.8 {
		t.Fatalf("expected BF16 TP2 cold 8k prefill to be >= 1.8x faster than FP8, got ratio %f", ratio)
	}

	// Prefix speedup at 8k on warm min TTFT
	fp8WarmMin8k := fp8Root.Summary.Repeated["8192"].MinTTFTMS
	fp8Speedup := fp8Cold8k / fp8WarmMin8k
	if fp8Speedup < 8.0 {
		t.Fatalf("expected FP8 8k prefix speedup >= 8.0x, got %f", fp8Speedup)
	}
}

func TestHopperH100BenchmarkInvariants(t *testing.T) {
	data := readAndVerify(t, "gcp_h100_hopper_bench_raw.json")
	var root struct {
		Schema  string `json:"schema"`
		GPUs    string `json:"gpus"`
		Model   string `json:"model"`
		Arch    string `json:"arch"`
		OK      bool   `json:"ok"`
		Engines map[string]struct {
			Engine           string  `json:"engine"`
			OK               bool    `json:"ok"`
			Precision        string  `json:"precision"`
			PrefillTokPerSec float64 `json:"prefill_tok_per_sec"`
			DecodeTokPerSec  float64 `json:"decode_tok_per_sec"`
		} `json:"engines"`
	}
	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatalf("failed to parse H100 bench JSON: %v", err)
	}
	if root.Schema != "fak.gcp-vm-bench.v2" {
		t.Fatalf("unexpected schema: %s", root.Schema)
	}
	if root.Arch != "sm_90" {
		t.Fatalf("expected sm_90, got %s", root.Arch)
	}
	if !root.OK {
		t.Fatalf("expected overall bench OK")
	}

	// Verify all 4 engines completed cleanly
	for _, engineKey := range []string{"llama", "fak-cuda", "fak-cuda-q8", "fak-cuda-tf32"} {
		e, ok := root.Engines[engineKey]
		if !ok || !e.OK {
			t.Fatalf("engine %s failed or missing", engineKey)
		}
		if e.DecodeTokPerSec <= 0 || e.PrefillTokPerSec <= 0 {
			t.Fatalf("engine %s produced non-positive throughput", engineKey)
		}
	}

	// Verify Q8 decode speedup over F32 on Hopper
	f32Decode := root.Engines["fak-cuda"].DecodeTokPerSec
	q8Decode := root.Engines["fak-cuda-q8"].DecodeTokPerSec
	if q8Decode <= f32Decode {
		t.Fatalf("expected Q8 decode (%f) to exceed F32 (%f)", q8Decode, f32Decode)
	}
	ratio := q8Decode / f32Decode
	if ratio < 1.15 {
		t.Fatalf("expected >= 15%% speedup on Q8 decode over F32, got ratio %f", ratio)
	}

	// Verify paired report
	pairedData := readAndVerify(t, "gcp_h100_paired_report.json")
	var pairedRoot struct {
		Schema        string `json:"schema"`
		TunedBaseline string `json:"tuned_baseline"`
		Arms          map[string]struct {
			TreatmentArm   string  `json:"treatment_arm"`
			BaselineArm    string  `json:"baseline_arm"`
			ApplesToApples bool    `json:"apples_to_apples"`
			SpeedupDecode  float64 `json:"speedup_decode"`
		} `json:"arms"`
	}
	if err := json.Unmarshal(pairedData, &pairedRoot); err != nil {
		t.Fatalf("failed to parse paired report: %v", err)
	}
	if pairedRoot.Schema != "fak.armbench.paired-report/1" {
		t.Fatalf("unexpected schema: %s", pairedRoot.Schema)
	}
	if pairedRoot.TunedBaseline != "llama" {
		t.Fatalf("expected baseline llama, got %s", pairedRoot.TunedBaseline)
	}
	if !pairedRoot.Arms["fak-cuda-q8"].ApplesToApples {
		t.Fatalf("fak-cuda-q8 must be apples-to-apples")
	}
}

func TestFreshQwen38A100V2Invariants(t *testing.T) {
	data := readAndVerify(t, "qwen38_a100_bench_v2_raw.json")
	var root struct {
		Schema   string `json:"schema"`
		Hardware struct {
			GPU string `json:"gpu"`
		} `json:"hardware"`
		Service struct {
			Model   string `json:"model"`
			Backend string `json:"backend"`
		} `json:"service"`
		ContextScalingSweep []struct {
			Success            bool    `json:"success"`
			TargetTokens       int     `json:"target_tokens"`
			PromptTokens       int     `json:"prompt_tokens"`
			CompletionTokens   int     `json:"completion_tokens"`
			DecodeTokensPerSec float64 `json:"decode_tokens_per_second"`
			LatencyMS          float64 `json:"latency_ms"`
		} `json:"context_scaling_sweep"`
		PrefixCachingAblation struct {
			ColdMS          float64 `json:"cold_ms"`
			Warm1MS         float64 `json:"warm1_ms"`
			Warm2MS         float64 `json:"warm2_ms"`
			UniqueMS        float64 `json:"unique_ms"`
			SpeedupVsCold   float64 `json:"speedup_vs_cold"`
			SpeedupVsUnique float64 `json:"speedup_vs_unique"`
		} `json:"prefix_caching_ablation"`
		ConcurrencyGrid map[string]struct {
			Concurrency                     int     `json:"concurrency"`
			SuccessCount                    int     `json:"success_count"`
			ErrorCount                      int     `json:"error_count"`
			TotalWallMS                     float64 `json:"total_wall_ms"`
			EffectiveSubagentsPerSec        float64 `json:"effective_subagents_per_sec"`
			EffectiveThroughputTokensPerSec float64 `json:"effective_throughput_tokens_per_sec"`
			P50LatencyMS                    float64 `json:"p50_latency_ms"`
		} `json:"concurrency_grid"`
	}

	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatalf("failed to parse Qwen3.8 A100 V2 bench JSON: %v", err)
	}

	if root.Schema != "fak.qwen38-a100-bench-v2/1" {
		t.Fatalf("unexpected schema: %s", root.Schema)
	}
	if root.Hardware.GPU != "1x NVIDIA A100-SXM4-40GB" {
		t.Fatalf("unexpected GPU: %s", root.Hardware.GPU)
	}
	if root.Service.Model != "Qwen3.8-27B-Q4_K_M" {
		t.Fatalf("unexpected model: %s", root.Service.Model)
	}

	// Verify all 12 context scaling runs succeeded
	if len(root.ContextScalingSweep) != 12 {
		t.Fatalf("expected 12 context scaling runs, got %d", len(root.ContextScalingSweep))
	}
	for _, run := range root.ContextScalingSweep {
		if !run.Success {
			t.Fatalf("expected all context scaling runs to succeed, failed at target %d", run.TargetTokens)
		}
		if run.DecodeTokensPerSec <= 0 {
			t.Fatalf("decode tokens per sec must be positive, got %f", run.DecodeTokensPerSec)
		}
	}

	// Verify prefix caching speedup vs unique dynamic prompt
	if root.PrefixCachingAblation.SpeedupVsUnique < 1.8 {
		t.Fatalf("expected prefix caching speedup vs unique >= 1.8x, got %f", root.PrefixCachingAblation.SpeedupVsUnique)
	}

	// Verify all 66 subagents in concurrency grid succeeded with 0 errors
	expectedGrid := []int{1, 5, 10, 20, 30}
	totalSubagents := 0
	for _, n := range expectedGrid {
		gridEntry, ok := root.ConcurrencyGrid[string(rune('0'+n))]
		if !ok {
			for _, g := range root.ConcurrencyGrid {
				if g.Concurrency == n {
					gridEntry = g
					ok = true
					break
				}
			}
		}
		if !ok {
			t.Fatalf("missing concurrency grid entry for N=%d", n)
		}
		if gridEntry.ErrorCount != 0 {
			t.Fatalf("expected 0 errors for N=%d, got %d", n, gridEntry.ErrorCount)
		}
		if gridEntry.SuccessCount != n {
			t.Fatalf("expected %d successes for N=%d, got %d", n, n, gridEntry.SuccessCount)
		}
		totalSubagents += gridEntry.SuccessCount
	}
	if totalSubagents != 66 {
		t.Fatalf("expected 66 total subagents across grid, got %d", totalSubagents)
	}
}

func TestFoldReportIntegrity(t *testing.T) {
	data := readAndVerify(t, "fold-report.json")
	var report struct {
		Schema  string `json:"schema"`
		Verdict string `json:"verdict"`
		Summary struct {
			NativeServing struct {
				Node                     string  `json:"node"`
				PrefixReuseSpeedupVsCold float64 `json:"prefix_reuse_speedup_vs_cold"`
				Fanout50AgentSuccessRate float64 `json:"fanout_50_agent_success_rate"`
			} `json:"native_serving"`
		} `json:"summary"`
		Artifacts []struct {
			Filename string `json:"filename"`
			SHA256   string `json:"sha256"`
		} `json:"artifacts"`
	}

	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("failed to parse fold report: %v", err)
	}
	if report.Schema != "fak.nvidia-gcp-overnight-witness/1" {
		t.Fatalf("unexpected schema: %s", report.Schema)
	}
	if report.Verdict != "KEEP_NATIVE_ACCELERATION_CONFIRMED" {
		t.Fatalf("unexpected verdict: %s", report.Verdict)
	}
	if report.Summary.NativeServing.Fanout50AgentSuccessRate != 1.0 {
		t.Fatalf("expected 1.0 success rate, got %f", report.Summary.NativeServing.Fanout50AgentSuccessRate)
	}
	if len(report.Artifacts) != 7 {
		t.Fatalf("expected 7 artifacts in fold report, got %d", len(report.Artifacts))
	}
}
