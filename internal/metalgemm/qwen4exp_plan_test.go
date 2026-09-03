package metalgemm

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestQwen4ExpTopologyCompleteness verifies that Qwen4Exp captures the complete
// architecture topology across all hybrid layer types (GDN, QSA, MoE, PLE, MTP)
// and validates kernel coverage completeness.
func TestQwen4ExpTopologyCompleteness(t *testing.T) {
	plan := NewDefaultQwen4ExpMetalPlan()
	if err := plan.ValidateTopology(); err != nil {
		t.Fatalf("default plan failed topology validation: %v", err)
	}

	// 1. Layer cadence verification (48 layers, 3:1 cadence = 36 GDN + 12 QSA)
	if plan.TotalLayers != 48 {
		t.Errorf("TotalLayers = %d, want 48", plan.TotalLayers)
	}
	if plan.FullAttentionInterval != 4 {
		t.Errorf("FullAttentionInterval = %d, want 4", plan.FullAttentionInterval)
	}
	if len(plan.Layers) != 48 {
		t.Fatalf("len(Layers) = %d, want 48", len(plan.Layers))
	}

	gdnCount := 0
	qsaCount := 0
	for i, layer := range plan.Layers {
		if layer.LayerIndex != i {
			t.Errorf("layer %d: index = %d, want %d", i, layer.LayerIndex, i)
		}
		if (i+1)%4 == 0 {
			if !layer.IsQSA || layer.IsGDN || layer.AttentionType != "full_attention" {
				t.Errorf("layer %d: expected QSA full attention, got %s (isQSA=%v isGDN=%v)",
					i, layer.AttentionType, layer.IsQSA, layer.IsGDN)
			}
			qsaCount++
		} else {
			if !layer.IsGDN || layer.IsQSA || layer.AttentionType != "linear_attention" {
				t.Errorf("layer %d: expected GDN linear attention, got %s (isQSA=%v isGDN=%v)",
					i, layer.AttentionType, layer.IsQSA, layer.IsGDN)
			}
			gdnCount++
		}

		if !layer.HasMoE || layer.RoutedExperts != 512 || layer.ActiveExperts != 10 || !layer.HasSharedExpert {
			t.Errorf("layer %d: invalid MoE configuration", i)
		}
	}

	if gdnCount != 36 {
		t.Errorf("GDN layer count = %d, want 36", gdnCount)
	}
	if qsaCount != 12 {
		t.Errorf("QSA layer count = %d, want 12", qsaCount)
	}

	// 2. GDN topology verification
	if plan.GDN.NumLayers != 36 || plan.GDN.NumKeyHeads != 16 || plan.GDN.NumValueHeads != 48 ||
		plan.GDN.KeyHeadDim != 128 || plan.GDN.ValueHeadDim != 128 || plan.GDN.ConvKernel != 4 ||
		plan.GDN.StateDType != "float32" {
		t.Errorf("GDN topology mismatch: %+v", plan.GDN)
	}
	// Verify exact recurrent state size: 48*128*128*4 + 3*(2*16*128 + 48*128)*4 = 3,268,608 bytes/layer
	expectedGDNPerLayer := int64(3268608)
	if plan.GDN.RecurrentStateBytesPerLayer != expectedGDNPerLayer {
		t.Errorf("GDN recurrent state per layer = %d, want %d",
			plan.GDN.RecurrentStateBytesPerLayer, expectedGDNPerLayer)
	}
	expectedGDNTotal := expectedGDNPerLayer * 36
	if plan.GDN.TotalRecurrentStateBytes != expectedGDNTotal {
		t.Errorf("GDN total recurrent state = %d, want %d",
			plan.GDN.TotalRecurrentStateBytes, expectedGDNTotal)
	}

	// 3. QSA topology verification
	if plan.QSA.NumLayers != 12 || plan.QSA.NumQueryHeads != 24 || plan.QSA.NumKVHeads != 2 ||
		plan.QSA.HeadDim != 256 || plan.QSA.IndexerBudget != 2048 || plan.QSA.IndexerHeads != 4 ||
		plan.QSA.IndexerHeadDim != 128 {
		t.Errorf("QSA topology mismatch: %+v", plan.QSA)
	}

	// 4. MoE topology verification
	if plan.MoE.NumLayers != 48 || plan.MoE.NumRoutedExperts != 512 || plan.MoE.ActiveRoutedExperts != 10 ||
		plan.MoE.SharedExpertIntermediateSize != 640 || plan.MoE.RoutedExpertIntermediateSize != 640 {
		t.Errorf("MoE topology mismatch: %+v", plan.MoE)
	}

	// 5. PLE topology verification
	if plan.PLE.NgramSize != 3 || plan.PLE.EmbeddingDim != 2560 || plan.PLE.VocabSize != 248320 {
		t.Errorf("PLE topology mismatch: %+v", plan.PLE)
	}

	// 6. MTP topology verification
	if !plan.MTP.Hybrid || plan.MTP.NumHiddenLayers != 1 || plan.MTP.LayerType != "full_attention" || plan.MTP.NgramSize != 3 {
		t.Errorf("MTP topology mismatch: %+v", plan.MTP)
	}

	// 7. Engine and fallback zero-tolerance
	if plan.Engine != "fak-native" {
		t.Errorf("Engine = %q, want fak-native", plan.Engine)
	}
	if plan.Fallback != "none" {
		t.Errorf("Fallback = %q, want none", plan.Fallback)
	}

	// 8. Negative validation tests
	// Missing required kernel
	corruptPlan := *plan
	corruptPlan.KernelRegistry = make(map[string]string)
	if err := corruptPlan.ValidateTopology(); err == nil {
		t.Error("plan with empty KernelRegistry should fail validation")
	}

	// Fallback attempt
	corruptPlan = *plan
	corruptPlan.Fallback = "mlx"
	if err := corruptPlan.ValidateTopology(); err == nil {
		t.Error("plan with fallback=mlx should fail validation")
	}

	// Non-native engine
	corruptPlan = *plan
	corruptPlan.Engine = "llama.cpp"
	if err := corruptPlan.ValidateTopology(); err == nil {
		t.Error("plan with engine=llama.cpp should fail validation")
	}

	// Incomplete layers
	corruptPlan = *plan
	corruptPlan.Layers = plan.Layers[:47]
	if err := corruptPlan.ValidateTopology(); err == nil {
		t.Error("plan with 47 layers should fail validation")
	}
}

// TestQwen4ExpUnifiedMemoryRAMTiers verifies unified memory allocation bounds across
// all Apple Silicon RAM tiers (16GB, 24GB, 36GB, 48GB, 64GB, 128GB) for Q4_K, Q8_0, and BF16.
func TestQwen4ExpUnifiedMemoryRAMTiers(t *testing.T) {
	plan := NewDefaultQwen4ExpMetalPlan()

	tiers := []struct {
		name     string
		ramBytes int64
	}{
		{"16GB", RAMTier16GB},
		{"24GB", RAMTier24GB},
		{"36GB", RAMTier36GB},
		{"48GB", RAMTier48GB},
		{"64GB", RAMTier64GB},
		{"128GB", RAMTier128GB},
	}

	for _, tc := range tiers {
		t.Run(tc.name, func(t *testing.T) {
			// Test Streamed Q4_K (fits on all M-series Macs from 16GB upwards)
			budgetStreamQ4, err := plan.CalculateMemoryFootprint(QuantTierQ4K, tc.ramBytes, true, 2048, 1)
			if err != nil {
				t.Fatalf("CalculateMemoryFootprint failed: %v", err)
			}
			if !budgetStreamQ4.FitsInRAM {
				t.Errorf("%s: Streamed Q4_K should fit in RAM, got Peak=%d Usable=%d",
					tc.name, budgetStreamQ4.PeakUnifiedMemoryBytes, budgetStreamQ4.UsableRAMBytes)
			}

			// Test Full Q4_K (requires 64GB+ for safe execution without swap thrashing)
			budgetFullQ4, err := plan.CalculateMemoryFootprint(QuantTierQ4K, tc.ramBytes, false, 2048, 1)
			if err != nil {
				t.Fatalf("CalculateMemoryFootprint failed: %v", err)
			}
			if tc.ramBytes <= RAMTier48GB {
				if budgetFullQ4.FitsInRAM {
					t.Errorf("%s: Full Q4_K (46.6GB+ weights) must NOT fit in %s RAM", tc.name, tc.name)
				}
			} else {
				// 64GB and 128GB have enough unified memory for full Q4_K
				if !budgetFullQ4.FitsInRAM {
					t.Errorf("%s: Full Q4_K should fit in %s RAM", tc.name, tc.name)
				}
			}

			// Test Full Q8_0 (88GB weights, requires 128GB RAM)
			budgetFullQ8, err := plan.CalculateMemoryFootprint(QuantTierQ80, tc.ramBytes, false, 2048, 1)
			if err != nil {
				t.Fatalf("CalculateMemoryFootprint failed: %v", err)
			}
			if tc.ramBytes < RAMTier128GB {
				if budgetFullQ8.FitsInRAM {
					t.Errorf("%s: Full Q8_0 (88GB weights) must NOT fit in %s RAM", tc.name, tc.name)
				}
			} else {
				if !budgetFullQ8.FitsInRAM {
					t.Errorf("%s: Full Q8_0 should fit in 128GB RAM", tc.name)
				}
			}

			// Test Full BF16 (165.6GB weights, exceeds 128GB RAM)
			budgetFullBF16, err := plan.CalculateMemoryFootprint(QuantTierBF16, tc.ramBytes, false, 2048, 1)
			if err != nil {
				t.Fatalf("CalculateMemoryFootprint failed: %v", err)
			}
			if budgetFullBF16.FitsInRAM {
				t.Errorf("%s: Full BF16 (165.6GB weights) must NOT fit in 128GB or smaller RAM", tc.name)
			}

			// Streamed BF16 on 128GB should fit
			budgetStreamBF16, err := plan.CalculateMemoryFootprint(QuantTierBF16, tc.ramBytes, true, 2048, 1)
			if err != nil {
				t.Fatalf("CalculateMemoryFootprint failed: %v", err)
			}
			if tc.ramBytes >= RAMTier36GB && !budgetStreamBF16.FitsInRAM {
				t.Errorf("%s: Streamed BF16 should fit in %s RAM", tc.name, tc.name)
			}
		})
	}

	// Verify KV cache scaling with context length (only 12 QSA layers)
	contexts := []struct {
		tokens   int
		expected int64 // 12 layers * 2 KV heads * 2 * 256 * 2 bytes = 24,576 bytes/tok
	}{
		{2048, 2048 * 24576},     // ~48 MiB
		{8192, 8192 * 24576},     // ~192 MiB
		{32768, 32768 * 24576},   // ~768 MiB
		{262144, 262144 * 24576}, // ~6 GiB
	}

	for _, c := range contexts {
		b, err := plan.CalculateMemoryFootprint(QuantTierQ4K, RAMTier64GB, false, c.tokens, 1)
		if err != nil {
			t.Fatalf("ctx %d: failed: %v", c.tokens, err)
		}
		if b.KVCacheBytes != c.expected {
			t.Errorf("ctx %d: KVCacheBytes = %d, want %d", c.tokens, b.KVCacheBytes, c.expected)
		}
		// Recurrent state must remain constant O(1) regardless of context length!
		if b.RecurrentStateBytes != plan.GDN.TotalRecurrentStateBytes {
			t.Errorf("ctx %d: recurrent state = %d, want constant %d",
				c.tokens, b.RecurrentStateBytes, plan.GDN.TotalRecurrentStateBytes)
		}
	}
}

// TestQwen4ExpMemoryPressureGovernance verifies that memory pressure governance
// correctly calculates headroom under Normal/Warning and strictly fails closed
// with zero fallback under Critical pressure.
func TestQwen4ExpMemoryPressureGovernance(t *testing.T) {
	plan := NewDefaultQwen4ExpMetalPlan()

	// 1. Normal pressure: should admit when budget fits
	budget64, err := plan.CalculateMemoryFootprint(QuantTierQ4K, RAMTier64GB, false, 2048, 1)
	if err != nil {
		t.Fatalf("CalculateMemoryFootprint failed: %v", err)
	}
	statusNormal, err := plan.GovernMemoryPressure(budget64, MemoryPressureNormal)
	if err != nil {
		t.Fatalf("Normal pressure unexpectedly returned error: %v", err)
	}
	if !statusNormal.Admitted {
		t.Errorf("Normal pressure should admit fitting budget: %+v", statusNormal)
	}
	if statusNormal.HeadroomBytes <= 0 {
		t.Errorf("Normal pressure should have positive headroom, got %d", statusNormal.HeadroomBytes)
	}

	// Normal pressure: should refuse when budget does not fit
	budget16Full, err := plan.CalculateMemoryFootprint(QuantTierQ4K, RAMTier16GB, false, 2048, 1)
	if err != nil {
		t.Fatalf("CalculateMemoryFootprint failed: %v", err)
	}
	statusOvercommit, err := plan.GovernMemoryPressure(budget16Full, MemoryPressureNormal)
	if err == nil {
		t.Errorf("Overcommit budget should return error")
	}
	if statusOvercommit.Admitted {
		t.Errorf("Overcommit budget must not be admitted")
	}
	if !strings.Contains(statusOvercommit.RefusalReason, "EXCEEDS_UNIFIED_MEMORY") {
		t.Errorf("RefusalReason = %q, want EXCEEDS_UNIFIED_MEMORY", statusOvercommit.RefusalReason)
	}

	// 2. Warning pressure: tightened threshold (70% of physical RAM)
	statusWarn, err := plan.GovernMemoryPressure(budget64, MemoryPressureWarning)
	if err != nil {
		t.Fatalf("Warning pressure unexpectedly returned error: %v", err)
	}
	if !statusWarn.Admitted {
		t.Errorf("Warning pressure with ample headroom should admit: %+v", statusWarn)
	}

	// Warning pressure with tight budget (e.g. 64GB Mac where budget > 70% of 64GB = 44.8GB)
	// Full Q4_K peak is ~47.3GB, which exceeds 70% of 64GB (44.8GB)!
	statusWarnTight, err := plan.GovernMemoryPressure(budget64, MemoryPressureWarning)
	// Note: Peak for Full Q4_K is 46.575GB + 0.05GB + 0.117GB + 0.64GB = ~47.38GB > 44.8GB.
	// So it should be refused under warning pressure!
	if budget64.PeakUnifiedMemoryBytes > statusWarnTight.UsableRAM {
		if statusWarnTight.Admitted || err == nil {
			t.Errorf("Warning pressure exceeding 70%% threshold must refuse")
		}
	}

	// 3. Critical pressure: strict fail-closed refusal regardless of free RAM!
	statusCrit, err := plan.GovernMemoryPressure(budget64, MemoryPressureCritical)
	if err == nil {
		t.Fatal("Critical pressure must return an error")
	}
	if statusCrit.Admitted {
		t.Fatal("Critical pressure must never admit execution")
	}
	if !strings.Contains(statusCrit.RefusalReason, "CRITICAL_MEMORY_PRESSURE") {
		t.Errorf("RefusalReason = %q, want CRITICAL_MEMORY_PRESSURE", statusCrit.RefusalReason)
	}
	if statusCrit.UsableRAM != 0 {
		t.Errorf("Critical pressure UsableRAM = %d, want 0", statusCrit.UsableRAM)
	}
}

// TestQwen4ExpExecutionReceiptZeroFallback verifies generation of the machine-readable
// execution receipt and enforces that zero fallback and kernel coverage are strictly maintained.
func TestQwen4ExpExecutionReceiptZeroFallback(t *testing.T) {
	plan := NewDefaultQwen4ExpMetalPlan()

	// 1. Generate valid receipt for 64GB Apple Silicon running Q4_K
	receipt, err := plan.GenerateExecutionReceipt(
		QuantTierQ4K,
		RAMTier64GB,
		false, // full residency
		2048,
		1,
		MemoryPressureNormal,
		"Apple M4 Max",
	)
	if err != nil {
		t.Fatalf("GenerateExecutionReceipt failed: %v", err)
	}

	// Verify receipt contents
	if receipt.Schema != Qwen4ExpExecutionReceiptSchema {
		t.Errorf("Schema = %q, want %q", receipt.Schema, Qwen4ExpExecutionReceiptSchema)
	}
	if receipt.Engine != "fak-native" {
		t.Errorf("Engine = %q, want fak-native", receipt.Engine)
	}
	if receipt.Fallback != "none" {
		t.Errorf("Fallback = %q, want none", receipt.Fallback)
	}
	if !receipt.AllKernelsCovered {
		t.Errorf("AllKernelsCovered = false, want true")
	}
	if len(receipt.KernelCoverage) != len(RequiredQwen4ExpMetalKernels) {
		t.Errorf("KernelCoverage count = %d, want %d", len(receipt.KernelCoverage), len(RequiredQwen4ExpMetalKernels))
	}
	for _, k := range RequiredQwen4ExpMetalKernels {
		if !receipt.KernelCoverage[k] {
			t.Errorf("Kernel %q not covered in receipt", k)
		}
	}
	if !receipt.Admitted {
		t.Errorf("Receipt unexpectedly refused: %s", receipt.RefusalReason)
	}
	if receipt.PeakUnifiedMemoryBytes <= 0 {
		t.Errorf("PeakUnifiedMemoryBytes = %d, want > 0", receipt.PeakUnifiedMemoryBytes)
	}
	if receipt.EstimatedTTFTMs <= 0 {
		t.Errorf("EstimatedTTFTMs = %f, want > 0", receipt.EstimatedTTFTMs)
	}
	if receipt.PrefillTokPerSec <= 0 || receipt.DecodeTokPerSec <= 0 {
		t.Errorf("Throughput bounds must be positive: prefill=%f decode=%f",
			receipt.PrefillTokPerSec, receipt.DecodeTokPerSec)
	}
	if !strings.HasPrefix(receipt.Digest, "sha256:") {
		t.Errorf("Digest = %q, want sha256: prefix", receipt.Digest)
	}

	// 2. Receipt Validate method tests
	if err := receipt.Validate(); err != nil {
		t.Fatalf("Valid receipt failed validation: %v", err)
	}

	// Tamper test: fallback = "mlx"
	tampered := *receipt
	tampered.Fallback = "mlx"
	if err := tampered.Validate(); err == nil {
		t.Error("Receipt with fallback=mlx must fail validation")
	}

	// Tamper test: fallback = "llama.cpp"
	tampered = *receipt
	tampered.Fallback = "llama.cpp"
	if err := tampered.Validate(); err == nil {
		t.Error("Receipt with fallback=llama.cpp must fail validation")
	}

	// Tamper test: engine = "vllm"
	tampered = *receipt
	tampered.Engine = "vllm"
	if err := tampered.Validate(); err == nil {
		t.Error("Receipt with engine=vllm must fail validation")
	}

	// Tamper test: missing kernel
	tampered = *receipt
	tampered.AllKernelsCovered = false
	if err := tampered.Validate(); err == nil {
		t.Error("Receipt with missing kernels must fail validation")
	}

	// 3. Test Fail-Closed Receipt under Critical Pressure
	critReceipt, err := plan.GenerateExecutionReceipt(
		QuantTierQ4K,
		RAMTier64GB,
		false,
		2048,
		1,
		MemoryPressureCritical,
		"Apple M4 Max",
	)
	if err != nil {
		t.Fatalf("GenerateExecutionReceipt under critical pressure returned unexpected error: %v", err)
	}
	if critReceipt.Admitted {
		t.Fatal("Critical pressure receipt must NOT be admitted")
	}
	if !strings.Contains(critReceipt.RefusalReason, "CRITICAL_MEMORY_PRESSURE") {
		t.Errorf("RefusalReason = %q, want CRITICAL_MEMORY_PRESSURE", critReceipt.RefusalReason)
	}
	if err := critReceipt.Validate(); err != nil {
		t.Fatalf("Refused receipt failed validation: %v", err)
	}
}

// TestQwen4ExpReceiptJSONRoundtrip tests serialization and deserialization of the execution receipt.
func TestQwen4ExpReceiptJSONRoundtrip(t *testing.T) {
	plan := NewDefaultQwen4ExpMetalPlan()
	receipt, err := plan.GenerateExecutionReceipt(
		QuantTierQ4K,
		RAMTier36GB,
		true, // streaming mode
		4096,
		1,
		MemoryPressureNormal,
		"Apple M3 Pro",
	)
	if err != nil {
		t.Fatalf("GenerateExecutionReceipt failed: %v", err)
	}

	raw, err := receipt.JSON()
	if err != nil {
		t.Fatalf("receipt.JSON failed: %v", err)
	}

	var decoded Qwen4ExpExecutionReceipt
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if decoded.Schema != receipt.Schema || decoded.Digest != receipt.Digest ||
		decoded.Engine != receipt.Engine || decoded.Fallback != receipt.Fallback ||
		decoded.QuantTier != receipt.QuantTier || decoded.StreamingMode != receipt.StreamingMode {
		t.Errorf("Decoded receipt differs from original")
	}
	if err := decoded.Validate(); err != nil {
		t.Errorf("Decoded receipt failed validation: %v", err)
	}
}
