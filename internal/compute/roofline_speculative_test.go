package compute

import (
	"math"
	"testing"
)

func TestBaselineHardwareProfiles(t *testing.T) {
	profiles := []struct {
		name                   string
		expectedName           string
		minBandwidthGBs        float64
		minComputeTFLOPs       float64
		minMemoryCapacityBytes int64
	}{
		{"l4", "NVIDIA L4", 300.0, 240.0, 24 * (1 << 30)},
		{"L4", "NVIDIA L4", 300.0, 240.0, 24 * (1 << 30)},
		{"nvidia_l4", "NVIDIA L4", 300.0, 240.0, 24 * (1 << 30)},
		{"a100", "NVIDIA A100-SXM4-80GB", 2000.0, 300.0, 80 * (1 << 30)},
		{"A100", "NVIDIA A100-SXM4-80GB", 2000.0, 300.0, 80 * (1 << 30)},
		{"a100-80gb", "NVIDIA A100-SXM4-80GB", 2000.0, 300.0, 80 * (1 << 30)},
		{"m3_max", "Apple M3 Max", 400.0, 30.0, 64 * (1 << 30)},
		{"M3-Max", "Apple M3 Max", 400.0, 30.0, 64 * (1 << 30)},
		{"m3max", "Apple M3 Max", 400.0, 30.0, 64 * (1 << 30)},
		{"h100", "NVIDIA H100-SXM5-80GB", 3300.0, 900.0, 80 * (1 << 30)},
		{"H100", "NVIDIA H100-SXM5-80GB", 3300.0, 900.0, 80 * (1 << 30)},
	}

	for _, tc := range profiles {
		hw := BaselineHardwareProfile(tc.name)
		if hw.Name != tc.expectedName {
			t.Errorf("BaselineHardwareProfile(%q).Name = %q, want %q", tc.name, hw.Name, tc.expectedName)
		}
		if hw.PeakMemoryBandwidthGBs < tc.minBandwidthGBs {
			t.Errorf("BaselineHardwareProfile(%q).PeakMemoryBandwidthGBs = %v, want >= %v", tc.name, hw.PeakMemoryBandwidthGBs, tc.minBandwidthGBs)
		}
		if hw.PeakComputeTFLOPs < tc.minComputeTFLOPs {
			t.Errorf("BaselineHardwareProfile(%q).PeakComputeTFLOPs = %v, want >= %v", tc.name, hw.PeakComputeTFLOPs, tc.minComputeTFLOPs)
		}
		if hw.MemoryCapacityBytes < tc.minMemoryCapacityBytes {
			t.Errorf("BaselineHardwareProfile(%q).MemoryCapacityBytes = %v, want >= %v", tc.name, hw.MemoryCapacityBytes, tc.minMemoryCapacityBytes)
		}
		if hw.OverheadLatencySec <= 0 {
			t.Errorf("BaselineHardwareProfile(%q).OverheadLatencySec must be positive, got %v", tc.name, hw.OverheadLatencySec)
		}
	}

	// Unknown profile fallback
	unknown := BaselineHardwareProfile("unknown_chip")
	if unknown.Name != "unknown_chip" || unknown.PeakMemoryBandwidthGBs != 0 {
		t.Errorf("expected empty unknown profile, got %+v", unknown)
	}
}

func TestBaselineModelCostProfiles(t *testing.T) {
	models := []struct {
		name         string
		expectedName string
		minParams    int64
		layers       int
		hidden       int
	}{
		{"qwen_7b", "Qwen2.5-7B", 7_000_000_000, 28, 3584},
		{"Qwen-7B", "Qwen2.5-7B", 7_000_000_000, 28, 3584},
		{"qwen_0_5b", "Qwen2.5-0.5B", 400_000_000, 24, 896},
		{"Qwen-0.5B", "Qwen2.5-0.5B", 400_000_000, 24, 896},
		{"qwen_72b", "Qwen2.5-72B", 70_000_000_000, 80, 8192},
		{"Qwen-72B", "Qwen2.5-72B", 70_000_000_000, 80, 8192},
	}

	for _, tc := range models {
		m := BaselineModelCostProfile(tc.name)
		if m.Name != tc.expectedName {
			t.Errorf("BaselineModelCostProfile(%q).Name = %q, want %q", tc.name, m.Name, tc.expectedName)
		}
		if m.ActiveParams < tc.minParams {
			t.Errorf("BaselineModelCostProfile(%q).ActiveParams = %v, want >= %v", tc.name, m.ActiveParams, tc.minParams)
		}
		if m.NLayers != tc.layers {
			t.Errorf("BaselineModelCostProfile(%q).NLayers = %v, want %v", tc.name, m.NLayers, tc.layers)
		}
		if m.HiddenDim != tc.hidden {
			t.Errorf("BaselineModelCostProfile(%q).HiddenDim = %v, want %v", tc.name, m.HiddenDim, tc.hidden)
		}
		if m.BytesPerWeight <= 0 || m.BytesPerKVElem <= 0 {
			t.Errorf("BaselineModelCostProfile(%q) precision bytes must be positive", tc.name)
		}
	}
}

func TestCalculateStepLatencyMemoryBound(t *testing.T) {
	hw := BaselineHardwareProfile("l4")
	model := BaselineModelCostProfile("qwen_7b")

	// Single token generation at batchSize=1, seqLen=2048
	step := CalculateStepLatency(hw, model, 1, 2048, 1)

	if step.Regime != "memory_bound" {
		t.Fatalf("expected memory_bound regime at batch=1, got %q", step.Regime)
	}
	if step.TMemorySec <= step.TArithmeticSec {
		t.Fatalf("memory time %v must exceed arithmetic time %v at batch=1", step.TMemorySec, step.TArithmeticSec)
	}
	if step.ArithmeticIntensity > 5.0 {
		t.Fatalf("arithmetic intensity at batch=1 should be low (memory bound), got %v", step.ArithmeticIntensity)
	}
	if step.TTotalSec <= step.TOverheadSec {
		t.Fatalf("total latency %v must exceed overhead %v", step.TTotalSec, step.TOverheadSec)
	}

	// Verify that verifying K=5 tokens on batch=1 is STILL memory bound and takes nearly identical time
	verifyStep := CalculateStepLatency(hw, model, 1, 2048, 5)
	if verifyStep.Regime != "memory_bound" {
		t.Fatalf("expected verifyStep at batch=1 to remain memory_bound, got %q", verifyStep.Regime)
	}
	// Weight reading dominates; latency difference between K=1 and K=5 at batch=1 should be negligible (< 2%)
	ratio := math.Abs(verifyStep.TTotalSec-step.TTotalSec) / step.TTotalSec
	if ratio > 0.02 {
		t.Fatalf("verifying K=5 tree tokens should cost ~1x autoregressive step when memory bound; diff ratio = %v", ratio)
	}
}

func TestCalculateStepLatencyComputeBoundCrossover(t *testing.T) {
	// Apple M3 Max: lower peak compute (32.8 TFLOPs) vs memory bandwidth (320 GB/s)
	hw := BaselineHardwareProfile("m3_max")
	model := BaselineModelCostProfile("qwen_7b")

	// Batch=1 is memory bound
	stepB1 := CalculateStepLatency(hw, model, 1, 512, 1)
	if stepB1.Regime != "memory_bound" {
		t.Fatalf("batch=1 should be memory_bound on M3 Max, got %q", stepB1.Regime)
	}

	// High batch size with multi-token verification saturates compute capacity
	stepB64 := CalculateStepLatency(hw, model, 64, 512, 8)
	if stepB64.Regime != "compute_bound" {
		t.Fatalf("batch=64, tokens=8 should be compute_bound on M3 Max, got %q", stepB64.Regime)
	}
	if stepB64.TArithmeticSec <= stepB64.TMemorySec {
		t.Fatalf("expected arithmetic time %v > memory time %v in compute bound regime", stepB64.TArithmeticSec, stepB64.TMemorySec)
	}
	if stepB64.ArithmeticIntensity <= stepB1.ArithmeticIntensity {
		t.Fatalf("arithmetic intensity at batch=64 (%v) should exceed batch=1 (%v)", stepB64.ArithmeticIntensity, stepB1.ArithmeticIntensity)
	}
}

func TestProjectSpeculativeSpeedupBatch1MemoryBound(t *testing.T) {
	hw := BaselineHardwareProfile("l4")
	target := BaselineModelCostProfile("qwen_7b")
	draft := BaselineModelCostProfile("qwen_0_5b")

	acceptance := SpecAcceptanceProfile{
		Method:    "speculative",
		MeanAlpha: 0.8,
	}

	// Batch=1, seqLen=2048, draftLength=5
	res := ProjectSpeculativeSpeedup(hw, target, draft, acceptance, 1, 2048, 5, false)

	if !res.Viable {
		t.Fatalf("expected speculative decoding to be viable at batch=1 on L4, got viable=%v (speedup=%v)", res.Viable, res.Speedup)
	}
	if res.Speedup <= 1.5 {
		t.Fatalf("expected substantial speedup (> 1.5x) at batch=1 with alpha=0.8, got %v", res.Speedup)
	}
	if res.SpeculativeTPOTSec >= res.AutoregressiveTPOTSec {
		t.Fatalf("speculative TPOT (%v) must be strictly lower than autoregressive TPOT (%v)", res.SpeculativeTPOTSec, res.AutoregressiveTPOTSec)
	}
	if res.ExpectedAcceptedTokens <= 1.0 {
		t.Fatalf("expected accepted tokens should be > 1.0, got %v", res.ExpectedAcceptedTokens)
	}
	if res.OptimalK <= 0 {
		t.Fatalf("expected optimal K > 0, got %v", res.OptimalK)
	}
	if res.OptimalSpeedup < res.Speedup {
		t.Fatalf("optimal speedup %v should be >= current speedup %v", res.OptimalSpeedup, res.Speedup)
	}
}

func TestProjectSpeculativeSpeedupComputeBoundCrossover(t *testing.T) {
	hw := BaselineHardwareProfile("m3_max")
	target := BaselineModelCostProfile("qwen_7b")
	draft := BaselineModelCostProfile("qwen_0_5b")

	// Lower acceptance rate simulating complex code/math reasoning
	acceptance := SpecAcceptanceProfile{
		Method:    "speculative",
		MeanAlpha: 0.4,
	}

	// At batch=1, even with alpha=0.4, draft model is cheap enough to maintain speedup > 1.0
	resB1 := ProjectSpeculativeSpeedup(hw, target, draft, acceptance, 1, 1024, 6, false)
	if resB1.Speedup <= 1.0 {
		t.Fatalf("expected viable speedup at batch=1, got %v", resB1.Speedup)
	}

	// At batch=64, verification compute saturates M3 Max capacity and draft overhead causes speedup < 1.0
	resB64 := ProjectSpeculativeSpeedup(hw, target, draft, acceptance, 64, 1024, 6, false)
	if resB64.Speedup >= 1.0 {
		t.Fatalf("expected speedup crossover to < 1.0 at batch=64 compute-saturated regime, got speedup = %v", resB64.Speedup)
	}
	if resB64.Viable {
		t.Fatalf("expected Viable = false when speedup < 1.0, got %v", resB64.Viable)
	}
}

func TestTreeStructuredSpeculation(t *testing.T) {
	hw := BaselineHardwareProfile("a100")
	target := BaselineModelCostProfile("qwen_7b")
	draft := BaselineModelCostProfile("qwen_0_5b")

	// Tree yield = 3.5 accepted tokens per step with 1 draft forward pass
	acceptance := SpecAcceptanceProfile{
		Method:    "tree",
		TreeYield: 3.5,
	}

	resTree := ProjectSpeculativeSpeedup(hw, target, draft, acceptance, 1, 2048, 8, true)

	if !resTree.Viable {
		t.Fatalf("tree speculative decoding must be viable on A100 at batch=1")
	}
	if resTree.ExpectedAcceptedTokens != 3.5 {
		t.Fatalf("expected TreeYield 3.5, got %v", resTree.ExpectedAcceptedTokens)
	}
	if resTree.Speedup < 2.0 {
		t.Fatalf("expected tree speedup >= 2.0x with yield 3.5, got %v", resTree.Speedup)
	}
}

func TestNonParametricDraftModel(t *testing.T) {
	hw := BaselineHardwareProfile("l4")
	target := BaselineModelCostProfile("qwen_7b")
	// Non-parametric heuristic draft (prompt lookup / n-gram): ActiveParams = 0
	draft := ModelCostProfile{
		Name:         "prompt_lookup",
		ActiveParams: 0,
	}

	acceptance := SpecAcceptanceProfile{
		MeanAlpha: 0.75,
	}

	res := ProjectSpeculativeSpeedup(hw, target, draft, acceptance, 1, 1024, 4, false)
	if !res.Viable {
		t.Fatalf("prompt lookup speculative decoding must be viable")
	}
	// With 0 draft compute, speedup is essentially E[N_acc]
	if math.Abs(res.Speedup-res.ExpectedAcceptedTokens) > 0.05 {
		t.Fatalf("with 0 draft latency, speedup (%v) should approximate expected tokens (%v)", res.Speedup, res.ExpectedAcceptedTokens)
	}
}

func TestPositionalAlpha(t *testing.T) {
	hw := BaselineHardwareProfile("l4")
	target := BaselineModelCostProfile("qwen_7b")
	draft := BaselineModelCostProfile("qwen_0_5b")

	// Monotonically decaying acceptance probabilities
	acceptance := SpecAcceptanceProfile{
		PositionalAlpha: []float64{0.9, 0.8, 0.7, 0.6, 0.5},
	}

	res := ProjectSpeculativeSpeedup(hw, target, draft, acceptance, 1, 2048, 5, false)

	// E[N_acc] = 1 + 0.9 + 0.9*0.8 + 0.9*0.8*0.7 + 0.9*0.8*0.7*0.6 + 0.9*0.8*0.7*0.6*0.5
	// = 1 + 0.9 + 0.72 + 0.504 + 0.3024 + 0.1512 = 3.5776
	wantExpected := 1.0 + 0.9 + 0.72 + 0.504 + 0.3024 + 0.1512
	if math.Abs(res.ExpectedAcceptedTokens-wantExpected) > 1e-6 {
		t.Fatalf("expected accepted tokens = %v, got %v", wantExpected, res.ExpectedAcceptedTokens)
	}
}

func TestFindOptimalDraftLength(t *testing.T) {
	hw := BaselineHardwareProfile("l4")
	target := BaselineModelCostProfile("qwen_7b")
	draft := BaselineModelCostProfile("qwen_0_5b")

	acceptance := SpecAcceptanceProfile{
		MeanAlpha: 0.8,
	}

	optK, optSpeedup := FindOptimalDraftLength(hw, target, draft, acceptance, 1, 2048, 16)
	if optK < 3 || optK > 10 {
		t.Fatalf("optimal K for alpha=0.8 should be in [3, 10], got %v", optK)
	}
	if optSpeedup <= 1.5 {
		t.Fatalf("optimal speedup should be > 1.5, got %v", optSpeedup)
	}

	// Defensive bounds testing
	kDef, _ := FindOptimalDraftLength(hw, target, draft, acceptance, -1, -5, 0)
	if kDef <= 0 {
		t.Fatalf("FindOptimalDraftLength must handle invalid bounds gracefully, got k=%v", kDef)
	}
}

func TestCalculateStepLatencyDefensiveBounds(t *testing.T) {
	hw := BaselineHardwareProfile("l4")
	model := BaselineModelCostProfile("qwen_7b")

	// Non-positive batch size
	b0 := CalculateStepLatency(hw, model, 0, 1024, 1)
	if b0.TArithmeticSec != 0 || b0.TMemorySec != 0 {
		t.Fatalf("batch=0 must yield 0 compute and memory times, got %+v", b0)
	}

	// Non-positive tokens per step
	t0 := CalculateStepLatency(hw, model, 1, 1024, 0)
	if t0.TArithmeticSec != 0 || t0.TMemorySec != 0 {
		t.Fatalf("tokens=0 must yield 0 compute and memory times, got %+v", t0)
	}

	// Negative context length clamped to 0
	negCtx := CalculateStepLatency(hw, model, 1, -100, 1)
	zeroCtx := CalculateStepLatency(hw, model, 1, 0, 1)
	if math.Abs(negCtx.TTotalSec-zeroCtx.TTotalSec) > 1e-9 {
		t.Fatalf("negative context length should clamp to 0; got %v vs %v", negCtx.TTotalSec, zeroCtx.TTotalSec)
	}
}
