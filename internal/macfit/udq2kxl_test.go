package macfit

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// TestQualifyUDQ2KXL36GBAt20KContext asserts that on a 36GB Mac with 20k context,
// the model is admitted with >60% headroom, zero swap risk, and profile seam "36GB_BASELINE".
func TestQualifyUDQ2KXL36GBAt20KContext(t *testing.T) {
	cases := []struct {
		name        string
		memoryBytes uint64
	}{
		{name: "36 GiB binary", memoryBytes: 36 * GiB},
		{name: "36 GB decimal", memoryBytes: 36 * 1_000_000_000},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q, err := QualifyUDQ2KXL(tc.memoryBytes, 20000, 1)
			if err != nil {
				t.Fatalf("QualifyUDQ2KXL: unexpected error: %v", err)
			}

			if !q.Admitted {
				t.Fatalf("expected model to be admitted on 36GB Mac at 20k context; refusal: %s", q.RefusalReason)
			}

			if q.HeadroomRatio <= 0.60 {
				t.Errorf("headroom ratio = %.4f (%.2f%%), want > 0.60 (> 60%%)", q.HeadroomRatio, q.HeadroomRatio*100)
			}

			if q.SwapRisk != SwapRiskZero {
				t.Errorf("swap risk = %q, want %q", q.SwapRisk, SwapRiskZero)
			}
			if !q.ZeroSwapRisk {
				t.Errorf("zero swap risk = false, want true")
			}

			if q.ProfileSeam != ProfileSeam36GBBaseline {
				t.Errorf("profile seam = %q, want %q", q.ProfileSeam, ProfileSeam36GBBaseline)
			}

			// Full attention KV cache for 20,000 tokens:
			// 2 (K,V) * 16 layers * 4 heads * 128 dim * 2 bytes = 32,768 bytes/token.
			// 20,000 * 32,768 = 655,360,000 bytes (~625 MiB).
			const wantKVBytesPerToken uint64 = 32768
			const wantKVCacheBytes uint64 = 655360000
			if q.FullAttnKVBytesPerToken != wantKVBytesPerToken {
				t.Errorf("KV bytes per token = %d, want %d", q.FullAttnKVBytesPerToken, wantKVBytesPerToken)
			}
			if q.FullAttnKVCacheBytes != wantKVCacheBytes {
				t.Errorf("full-attention KV cache = %d, want %d", q.FullAttnKVCacheBytes, wantKVCacheBytes)
			}

			if q.ResidentWeightBytes != ResidentWeightBytes {
				t.Errorf("resident weight bytes = %d, want %d", q.ResidentWeightBytes, ResidentWeightBytes)
			}
			if q.RecurrentStateBytes != RecurrentStatePerAgentBytes {
				t.Errorf("recurrent state bytes = %d, want %d", q.RecurrentStateBytes, RecurrentStatePerAgentBytes)
			}
			if q.StagingScratchBytes != StagingScratchBytes {
				t.Errorf("staging scratch bytes = %d, want %d", q.StagingScratchBytes, StagingScratchBytes)
			}

			// Peak memory fits strictly within wired ceiling
			if q.PeakMemoryBytes > q.WiredCeilingBytes {
				t.Errorf("peak memory %d exceeds wired ceiling %d", q.PeakMemoryBytes, q.WiredCeilingBytes)
			}
		})
	}
}

// TestQualifyUDQ2KXLMultiAgent24Concurrency asserts 24 agents with 4096 shared preamble
// + 1024 private tail fit within 27 GB wired ceiling on 36GB Mac.
func TestQualifyUDQ2KXLMultiAgent24Concurrency(t *testing.T) {
	const sharedPreamble = uint64(4096)
	const privateTail = uint64(1024)
	const concurrency = uint64(24)

	cases := []struct {
		name        string
		memoryBytes uint64
	}{
		{name: "36 GiB binary", memoryBytes: 36 * GiB},
		{name: "36 GB decimal", memoryBytes: 36 * 1_000_000_000},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q, err := QualifyUDQ2KXLMultiAgent(tc.memoryBytes, sharedPreamble, privateTail, concurrency)
			if err != nil {
				t.Fatalf("QualifyUDQ2KXLMultiAgent: unexpected error: %v", err)
			}

			if !q.Admitted {
				t.Fatalf("expected 24 agents to fit within wired ceiling; refusal: %s", q.RefusalReason)
			}

			// Assert within 27 GB wired ceiling
			const wiredCeiling27GBDecimal = uint64(27 * 1_000_000_000)
			if q.PeakMemoryBytes > q.WiredCeilingBytes {
				t.Errorf("peak memory %d bytes exceeds wired ceiling %d bytes", q.PeakMemoryBytes, q.WiredCeilingBytes)
			}
			if q.PeakMemoryBytes > wiredCeiling27GBDecimal {
				t.Errorf("peak memory %d bytes exceeds 27 GB (%d bytes)", q.PeakMemoryBytes, wiredCeiling27GBDecimal)
			}

			if !q.ZeroSwapRisk {
				t.Errorf("zero swap risk = false, want true")
			}
			if q.SwapRisk != SwapRiskZero {
				t.Errorf("swap risk = %q, want %q", q.SwapRisk, SwapRiskZero)
			}

			// Validate KV cache sizing:
			// Shared preamble: 4096 * 32,768 = 134,217,728 bytes (~128 MiB)
			// Private tail: 24 * 1024 * 32,768 = 805,306,368 bytes (~768 MiB)
			// Total KV: 134,217,728 + 805,306,368 = 939,524,096 bytes (~896 MiB)
			const wantTotalKV = uint64(939524096)
			if q.FullAttnKVCacheBytes != wantTotalKV {
				t.Errorf("total KV cache = %d, want %d", q.FullAttnKVCacheBytes, wantTotalKV)
			}

			// Validate recurrent state sizing:
			// 24 agents * 3,355,443 bytes = 80,530,632 bytes (~76.8 MiB)
			const wantRecurrent = uint64(24 * 3355443)
			if q.RecurrentStateBytes != wantRecurrent {
				t.Errorf("recurrent state = %d, want %d", q.RecurrentStateBytes, wantRecurrent)
			}

			if q.Concurrency != concurrency {
				t.Errorf("concurrency = %d, want %d", q.Concurrency, concurrency)
			}
			if q.SharedPrefixTokens != sharedPreamble {
				t.Errorf("shared prefix = %d, want %d", q.SharedPrefixTokens, sharedPreamble)
			}
			if q.PrivateTailTokens != privateTail {
				t.Errorf("private tail = %d, want %d", q.PrivateTailTokens, privateTail)
			}
		})
	}
}

// TestQualifyUDQ2KXLCapabilityScaling tests that 64GB Mac Studio selects
// "CAPABILITY_SCALED_LARGER" with an expanded context budget.
func TestQualifyUDQ2KXLCapabilityScaling(t *testing.T) {
	q36, err := QualifyUDQ2KXL(36*GiB, 20000, 1)
	if err != nil {
		t.Fatalf("QualifyUDQ2KXL(36GiB): %v", err)
	}

	q64, err := QualifyUDQ2KXL(64*GiB, 20000, 1)
	if err != nil {
		t.Fatalf("QualifyUDQ2KXL(64GiB): %v", err)
	}

	if !q64.Admitted {
		t.Fatalf("expected 64GB Mac Studio to be admitted; refusal: %s", q64.RefusalReason)
	}

	if q64.ProfileSeam != ProfileSeamCapabilityScaledLarger {
		t.Errorf("64GB profile seam = %q, want %q", q64.ProfileSeam, ProfileSeamCapabilityScaledLarger)
	}

	if q64.ContextBudgetTokens <= q36.ContextBudgetTokens {
		t.Errorf("64GB context budget (%d) must exceed 36GB context budget (%d)", q64.ContextBudgetTokens, q36.ContextBudgetTokens)
	}

	if q64.MaxContextTokens <= q36.MaxContextTokens {
		t.Errorf("64GB max context tokens (%d) must exceed 36GB max context tokens (%d)", q64.MaxContextTokens, q36.MaxContextTokens)
	}

	if q64.WiredCeilingBytes <= q36.WiredCeilingBytes {
		t.Errorf("64GB wired ceiling (%d) must exceed 36GB wired ceiling (%d)", q64.WiredCeilingBytes, q36.WiredCeilingBytes)
	}

	if q64.HeadroomRatio <= q36.HeadroomRatio {
		t.Errorf("64GB headroom ratio (%.4f) must exceed 36GB headroom ratio (%.4f)", q64.HeadroomRatio, q36.HeadroomRatio)
	}

	// Also verify 128GB Mac Studio capability scaling
	q128, err := QualifyUDQ2KXL(128*GiB, 20000, 1)
	if err != nil {
		t.Fatalf("QualifyUDQ2KXL(128GiB): %v", err)
	}
	if q128.ProfileSeam != ProfileSeamCapabilityScaledLarger {
		t.Errorf("128GB profile seam = %q, want %q", q128.ProfileSeam, ProfileSeamCapabilityScaledLarger)
	}
	if q128.MaxContextTokens <= q64.MaxContextTokens {
		t.Errorf("128GB max context tokens (%d) must exceed 64GB (%d)", q128.MaxContextTokens, q64.MaxContextTokens)
	}
}

// TestQualifyUDQ2KXLOverbudgetRefusal tests over-budget refusal when memory
// is severely constrained (e.g. 8GB), reporting the limiting component.
func TestQualifyUDQ2KXLOverbudgetRefusal(t *testing.T) {
	// 8GB Mac (8 GiB or 8 GB decimal) cannot hold 9.1GB resident weights within 6GB wired ceiling
	q, err := QualifyUDQ2KXL(8*GiB, 20000, 1)
	if err != nil {
		t.Fatalf("QualifyUDQ2KXL(8GiB): unexpected error: %v", err)
	}

	if q.Admitted {
		t.Fatalf("expected 8GB Mac to be refused, but was admitted")
	}

	if q.ProfileSeam != ProfileSeamSub36GBConstrained {
		t.Errorf("8GB profile seam = %q, want %q", q.ProfileSeam, ProfileSeamSub36GBConstrained)
	}

	if q.LimitingComponent != LimitingComponentWeights {
		t.Errorf("limiting component = %q, want %q", q.LimitingComponent, LimitingComponentWeights)
	}

	if q.RefusalReason == "" {
		t.Errorf("expected non-empty refusal reason")
	}
	if !strings.Contains(q.RefusalReason, "resident weights") {
		t.Errorf("refusal reason %q should mention resident weights", q.RefusalReason)
	}

	if q.ZeroSwapRisk {
		t.Errorf("zero swap risk = true, want false on refusal")
	}
	if q.SwapRisk != SwapRiskCritical && q.SwapRisk != SwapRiskHigh {
		t.Errorf("swap risk = %q, want CRITICAL or HIGH", q.SwapRisk)
	}

	// Also test KV cache overflow on 36GB Mac with an impossible context length (e.g. 2,000,000 tokens)
	qKVOverflow, err := QualifyUDQ2KXL(36*GiB, 2000000, 1)
	if err != nil {
		t.Fatalf("QualifyUDQ2KXL(36GiB, 2M tok): %v", err)
	}
	if qKVOverflow.Admitted {
		t.Fatalf("expected 2M tokens to exceed 36GB Mac wired ceiling, but was admitted")
	}
	if qKVOverflow.LimitingComponent != LimitingComponentKVCache {
		t.Errorf("limiting component = %q, want %q", qKVOverflow.LimitingComponent, LimitingComponentKVCache)
	}
	if !strings.Contains(qKVOverflow.RefusalReason, "KV cache") {
		t.Errorf("refusal reason %q should mention KV cache", qKVOverflow.RefusalReason)
	}

	// Memory = 0 error handling
	_, err = QualifyUDQ2KXL(0, 20000, 1)
	if err == nil {
		t.Errorf("expected error when memoryBytes == 0")
	}
}

// TestQualifyUDQ2KXLExactArtifactConstants asserts exact SHA-256 and byte size match #11961 specification.
func TestQualifyUDQ2KXLExactArtifactConstants(t *testing.T) {
	const wantFileName = "Qwen3.8-27B-UD-Q2_K_XL.gguf"
	const wantBytes uint64 = 9828981664
	const wantSHA256 = "fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0"
	const wantWeights uint64 = 9126805504
	const wantFullAttnLayers uint64 = 16
	const wantRecurrentLayers uint64 = 48
	const wantTotalLayers uint64 = 64
	const wantKVHeads uint64 = 4
	const wantHeadDim uint64 = 128
	const wantRecurrentStatePerAgent uint64 = 3355443

	if ArtifactFileName != wantFileName {
		t.Errorf("ArtifactFileName = %q, want %q", ArtifactFileName, wantFileName)
	}
	if ArtifactBytes != wantBytes {
		t.Errorf("ArtifactBytes = %d, want %d", ArtifactBytes, wantBytes)
	}
	if ArtifactLFS_SHA256 != wantSHA256 {
		t.Errorf("ArtifactLFS_SHA256 = %q, want %q", ArtifactLFS_SHA256, wantSHA256)
	}
	if ArtifactLFSSHA256 != wantSHA256 {
		t.Errorf("ArtifactLFSSHA256 = %q, want %q", ArtifactLFSSHA256, wantSHA256)
	}
	if ResidentWeightBytes != wantWeights {
		t.Errorf("ResidentWeightBytes = %d, want %d", ResidentWeightBytes, wantWeights)
	}
	if FullAttnLayers != wantFullAttnLayers {
		t.Errorf("FullAttnLayers = %d, want %d", FullAttnLayers, wantFullAttnLayers)
	}
	if RecurrentLayers != wantRecurrentLayers {
		t.Errorf("RecurrentLayers = %d, want %d", RecurrentLayers, wantRecurrentLayers)
	}
	if TotalLayers != wantTotalLayers {
		t.Errorf("TotalLayers = %d, want %d", TotalLayers, wantTotalLayers)
	}
	if KVHeads != wantKVHeads {
		t.Errorf("KVHeads = %d, want %d", KVHeads, wantKVHeads)
	}
	if HeadDim != wantHeadDim {
		t.Errorf("HeadDim = %d, want %d", HeadDim, wantHeadDim)
	}
	if RecurrentStatePerAgentBytes != wantRecurrentStatePerAgent {
		t.Errorf("RecurrentStatePerAgentBytes = %d, want %d", RecurrentStatePerAgentBytes, wantRecurrentStatePerAgent)
	}

	// Architectural hybrid sanity check
	if FullAttnLayers+RecurrentLayers != TotalLayers {
		t.Errorf("FullAttnLayers (%d) + RecurrentLayers (%d) != TotalLayers (%d)",
			FullAttnLayers, RecurrentLayers, TotalLayers)
	}
}

// TestUDQ2KXLSerialization tests JSON serialization and CLI formatting.
func TestUDQ2KXLSerialization(t *testing.T) {
	q, err := QualifyUDQ2KXL(36*GiB, 20000, 1)
	if err != nil {
		t.Fatalf("QualifyUDQ2KXL: %v", err)
	}

	// JSON roundtrip
	jsonStr, err := q.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}
	if !strings.Contains(jsonStr, `"schema": "fak-udq2kxl-qualification/1"`) {
		t.Errorf("JSON missing schema: %s", jsonStr)
	}

	var parsed UDQ2KXLQualification
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if parsed.ArtifactFileName != ArtifactFileName || !parsed.Admitted {
		t.Errorf("unmarshaled qualification mismatch: %+v", parsed)
	}

	// CLI formatting
	var buf bytes.Buffer
	if err := q.FormatCLI(&buf); err != nil {
		t.Fatalf("FormatCLI: %v", err)
	}
	output := buf.String()
	if !strings.Contains(output, "Qwen3.8-27B-UD-Q2_K_XL") || !strings.Contains(output, "ADMITTED") {
		t.Errorf("FormatCLI output missing key sections:\n%s", output)
	}

	// String() output
	summary := q.String()
	if !strings.Contains(summary, "ADMITTED") || !strings.Contains(summary, "36GB_BASELINE") {
		t.Errorf("String() summary mismatch: %s", summary)
	}
}

// TestQualifyUDQ2KXLOverflowSafety tests that arithmetic overflow in context tokens
// or concurrency is caught and safely returns an error rather than false admission.
func TestQualifyUDQ2KXLOverflowSafety(t *testing.T) {
	// 1. Extreme context tokens overflow (2^49)
	const hugeContext = uint64(1) << 49
	_, err := QualifyUDQ2KXL(36*GiB, hugeContext, 1)
	if err == nil {
		t.Errorf("expected overflow error for hugeContext %d", hugeContext)
	}

	// 2. Extreme concurrency overflow
	const hugeConcurrency = ^uint64(0)
	_, err = QualifyUDQ2KXL(36*GiB, 20000, hugeConcurrency)
	if err == nil {
		t.Errorf("expected overflow error for hugeConcurrency")
	}

	// 3. Multi-agent overflow
	_, err = QualifyUDQ2KXLMultiAgent(36*GiB, hugeContext, 1024, 24)
	if err == nil {
		t.Errorf("expected overflow error for multi-agent huge preamble")
	}
}
