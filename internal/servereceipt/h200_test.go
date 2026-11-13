package servereceipt

import (
	"strings"
	"testing"
)

// TestH200KimiNVFP4Receipt pins the H200 (Hopper, SM 9.0) verdicts for the
// NVFP4 checkpoint: the 8-GPU rig is INFEASIBLE on weights alone (the printed
// expert-offload escape hatch costs ~17.5 s/forward at 50 GB/s host bandwidth,
// so it is rejected, not shipped), the 16-GPU rig fits with headroom at util
// 0.90, and the non-native-FP4 warning fires on every H200 plan (Hopper rides
// the Marlin W4A16 dequant fallback — no native FP4 tensor cores).
func TestH200KimiNVFP4Receipt(t *testing.T) {
	facts := loadKimiOffloadFacts(t)
	gpu := testGPU(t, "h200-141gb")

	t.Run("tp8-infeasible", func(t *testing.T) {
		receipt, err := Plan(facts, gpu, Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
		if err != nil {
			t.Fatalf("Plan(tp8): %v", err)
		}
		if got := receipt.Arithmetic.WeightsPerGPUBytes; got != 200_640_705_704 {
			t.Fatalf("tp8 weights/gpu = %d, want 200640705704", got)
		}
		if got := receipt.Arithmetic.ExpertUnionBytes; got != 874_857_142_784 {
			t.Fatalf("tp8 expert union = %d, want 874857142784", got)
		}
		if receipt.Verdict != VerdictInfeasible {
			t.Fatalf("tp8 verdict = %s, want INFEASIBLE", receipt.Verdict)
		}
		t.Logf("tp8: weights/GPU %d B (%.2f GiB) vs budget %.2f GiB; expert union %d B = %.2f s/forward @ %.0f GB/s host bw",
			receipt.Arithmetic.WeightsPerGPUBytes, float64(receipt.Arithmetic.WeightsPerGPUBytes)/float64(GibBytes),
			float64(receipt.Arithmetic.UtilBudgetBytes)/float64(GibBytes),
			receipt.Arithmetic.ExpertUnionBytes, receipt.Arithmetic.ExpertSecondsPerForward, receipt.Arithmetic.HostBandwidthGBs)
	})

	t.Run("tp16-fits", func(t *testing.T) {
		receipt, err := Plan(facts, gpu, Config{TP: 16, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2, MaxModelLen: 1_048_576})
		if err != nil {
			t.Fatalf("Plan(tp16): %v", err)
		}
		if got := receipt.Arithmetic.WeightsPerGPUBytes; got != 100_320_352_852 {
			t.Fatalf("tp16 weights/gpu = %d, want 100320352852", got)
		}
		if receipt.Verdict != VerdictOK {
			t.Fatalf("tp16 verdict = %s, want OK", receipt.Verdict)
		}
		if got := receipt.Arithmetic.PhysicalTokens; got != 1_908_454 {
			t.Fatalf("tp16 physical tokens = %d, want 1908454", got)
		}
		if got := receipt.Arithmetic.HeadroomTokens; got != 859_878 {
			t.Fatalf("tp16 headroom = %d, want 859878", got)
		}
		if len(receipt.Warnings) != 1 || !strings.Contains(receipt.Warnings[0], "NVFP4 checkpoints require Blackwell SM≥10.0") {
			t.Fatalf("tp16 warnings = %v, want exactly the non-native-FP4 warning", receipt.Warnings)
		}
		t.Logf("tp16: weights/GPU %d B (%.2f GiB); pool %.2f GiB; kv/token %d B; physical tokens %d; headroom over max_model_len 1048576: %d",
			receipt.Arithmetic.WeightsPerGPUBytes, float64(receipt.Arithmetic.WeightsPerGPUBytes)/float64(GibBytes),
			float64(receipt.Arithmetic.UsablePerGPUBytes)/float64(GibBytes),
			receipt.Arithmetic.KVBytesPerTokenPerGPU, receipt.Arithmetic.PhysicalTokens, receipt.Arithmetic.HeadroomTokens)
	})

	t.Run("tp8pp3-overkill", func(t *testing.T) {
		receipt, err := Plan(facts, gpu, Config{TP: 8, PP: 3, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
		if err != nil {
			t.Fatalf("Plan(tp8pp3): %v", err)
		}
		if receipt.Verdict != VerdictOK {
			t.Fatalf("tp8pp3 verdict = %s, want OK", receipt.Verdict)
		}
		if got := receipt.Arithmetic.WeightsPerGPUBytes; got != 66_880_235_235 {
			t.Fatalf("tp8pp3 weights/gpu = %d, want 66880235235", got)
		}
		if got := receipt.Arithmetic.PhysicalTokens; got != 3_821_761 {
			t.Fatalf("tp8pp3 physical tokens = %d, want 3821761", got)
		}
		if len(receipt.Warnings) != 1 || !strings.Contains(receipt.Warnings[0], "NVFP4 checkpoints require Blackwell SM≥10.0") {
			t.Fatalf("tp8pp3 warnings = %v, want exactly the non-native-FP4 warning", receipt.Warnings)
		}
		t.Logf("tp8pp3: weights/GPU %d B (%.2f GiB); pool %.2f GiB; kv/token %d B; physical tokens %d",
			receipt.Arithmetic.WeightsPerGPUBytes, float64(receipt.Arithmetic.WeightsPerGPUBytes)/float64(GibBytes),
			float64(receipt.Arithmetic.UsablePerGPUBytes)/float64(GibBytes),
			receipt.Arithmetic.KVBytesPerTokenPerGPU, receipt.Arithmetic.PhysicalTokens)
	})
}
