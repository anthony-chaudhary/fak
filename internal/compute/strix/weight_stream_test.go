// Package strix implements the AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151)
// non-temporal weight streaming pipeline, RDNA 3.5 V# buffer descriptor synthesis,
// and 32MB MALL (Memory Attached Last-Level) Infinity Cache eviction protection.
package strix

import (
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"testing"
)

// TestWeightStream_35BForwardPassNonTemporalBypass verifies Acceptance Criteria 1, 2, 3, 4:
// A 35B model forward pass with SLC=1 bypass maintains >=95% MALL residency (100%),
// incurs zero MALL evictions, saves >= 32.0 GB/s of DRAM bandwidth, and achieves >= 220.0 GB/s throughput.
func TestWeightStream_35BForwardPassNonTemporalBypass(t *testing.T) {
	cfg := DefaultWeightStreamConfig()
	pipeline := NewWeightStreamPipeline(cfg)
	defer pipeline.Close()

	// 35B INT4 weights (~17.5 GB), 40 layers, 32MB pinned KV prefix (8,192 tokens FP16)
	weightBytes := Default35BWeightBytes
	numLayers := 40
	pinnedKVBytes := MALLCapacityBytes

	telem, err := pipeline.SimulateForwardPass(weightBytes, numLayers, pinnedKVBytes, true)
	if err != nil {
		t.Fatalf("SimulateForwardPass failed: %v", err)
	}

	// 1. Zero MALL Evictions (Acceptance Criterion 2)
	if telem.MALLEvictions != 0 {
		t.Errorf("expected 0 MALL evictions under non-temporal bypass, got %d", telem.MALLEvictions)
	}
	if pipeline.MALL().EvictionCount() != 0 {
		t.Errorf("expected 0 cache model evictions, got %d", pipeline.MALL().EvictionCount())
	}
	if pipeline.MALL().PinnedKVEvictionCount() != 0 {
		t.Errorf("expected 0 pinned KV evictions, got %d", pipeline.MALL().PinnedKVEvictionCount())
	}

	// 2. MALL Residency >= 95% (Acceptance Criterion 4: 100% maintained)
	if telem.MALLResidencyPct < 95.0 {
		t.Errorf("expected MALL residency >= 95.0%%, got %.2f%%", telem.MALLResidencyPct)
	}
	if math.Abs(telem.MALLResidencyPct-100.0) > 0.001 {
		t.Errorf("expected exactly 100.0%% MALL residency, got %.2f%%", telem.MALLResidencyPct)
	}

	// 3. DRAM Bandwidth Saved >= 32.0 GB/s (Acceptance Criterion 4)
	if telem.DRAMBandwidthSavedGBs < 32.0 {
		t.Errorf("expected DRAM bandwidth saved >= 32.0 GB/s, got %.2f GB/s", telem.DRAMBandwidthSavedGBs)
	}

	// 4. Weight Streaming Throughput >= 220.0 GB/s
	if telem.ThroughputGBs < 220.0 {
		t.Errorf("expected throughput >= 220.0 GB/s, got %.2f GB/s", telem.ThroughputGBs)
	}

	// 5. Zero DRAM KV Re-fetch Bytes
	if telem.DRAMKVBytesRead != 0 {
		t.Errorf("expected 0 DRAM KV re-fetch bytes, got %d", telem.DRAMKVBytesRead)
	}

	// 6. Mathematical Eviction Ratio calculation
	expectedRatio := float64(weightBytes) / float64(MALLCapacityBytes)
	if math.Abs(telem.EvictionRatio-expectedRatio) > 0.1 {
		t.Errorf("expected eviction ratio ~%.1f, got %.1f", expectedRatio, telem.EvictionRatio)
	}

	// 7. Bypassed Descriptors count equals numLayers
	if telem.BypassedDescriptors != uint64(numLayers) {
		t.Errorf("expected %d bypassed descriptors, got %d", numLayers, telem.BypassedDescriptors)
	}

	t.Logf("[SW-VERIFIED] 35B Forward Pass Non-Temporal Bypass Verified:")
	t.Logf("  Weight Footprint: %.2f GB across %d layers", float64(weightBytes)/(1024*1024*1024), numLayers)
	t.Logf("  MALL Evictions: %d", telem.MALLEvictions)
	t.Logf("  MALL Residency: %.2f%%", telem.MALLResidencyPct)
	t.Logf("  DRAM Bandwidth Saved: %.2f GB/s", telem.DRAMBandwidthSavedGBs)
	t.Logf("  Sustained Throughput: %.2f GB/s", telem.ThroughputGBs)
	t.Logf("  Theoretical Eviction Ratio Avoided: %.1fx", telem.EvictionRatio)
}

// TestWeightStream_BufferDescriptorSynthesis tests 128-bit RDNA 3.5 V# buffer resource descriptor
// generation, virtual address validation, stride injection, and SLC/GLC/DLC cache bitmasks.
func TestWeightStream_BufferDescriptorSynthesis(t *testing.T) {
	cfg := DefaultWeightStreamConfig()
	pipeline := NewWeightStreamPipeline(cfg)
	defer pipeline.Close()

	// 1. Non-temporal bypass descriptor (SLC=1, GLC=0, DLC=0)
	baseAddr := uint64(0x00007FFF10002000) // Valid 48-bit address, 16-byte aligned
	sizeBytes := uint32(1048576)           // 1 MiB

	desc, err := pipeline.SynthesizeDescriptor(baseAddr, sizeBytes, FlagsStreamingBypass)
	if err != nil {
		t.Fatalf("SynthesizeDescriptor failed: %v", err)
	}

	// Verify Base Address reconstruction
	if desc.BaseAddress() != baseAddr {
		t.Errorf("BaseAddress mismatch: expected 0x%X, got 0x%X", baseAddr, desc.BaseAddress())
	}
	if desc.Size() != sizeBytes {
		t.Errorf("Size mismatch: expected %d, got %d", sizeBytes, desc.Size())
	}
	if desc.Stride() != uint16(BurstStrideBytes) {
		t.Errorf("Stride mismatch: expected %d, got %d", BurstStrideBytes, desc.Stride())
	}

	// Verify Cache Bitflags in Word3
	if !desc.IsSLCSet() {
		t.Errorf("expected SLC bit (bit 22) to be set in Word3: 0x%08X", desc.Word3)
	}
	if desc.IsGLCSet() {
		t.Errorf("expected GLC bit (bit 12) to be cleared in Word3: 0x%08X", desc.Word3)
	}
	if desc.IsDLCSet() {
		t.Errorf("expected DLC bit (bit 13) to be cleared in Word3: 0x%08X", desc.Word3)
	}
	if desc.ResourceType() != 0x8 {
		t.Errorf("expected resource type 0x8 in Word3 bits [31:28], got 0x%X", desc.ResourceType())
	}

	// Verify exact dword bitfields
	expectedWord0 := uint32(baseAddr & 0xFFFFFFFF)
	expectedWord1 := uint32((baseAddr>>32)&0xFFFF) | (uint32(BurstStrideBytes) << 16)
	expectedWord2 := sizeBytes
	expectedWord3 := (uint32(0x8) << 28) | (1 << 22) // Resource type 0x8 | SLC bit 22

	if desc.Word0 != expectedWord0 {
		t.Errorf("Word0 mismatch: expected 0x%08X, got 0x%08X", expectedWord0, desc.Word0)
	}
	if desc.Word1 != expectedWord1 {
		t.Errorf("Word1 mismatch: expected 0x%08X, got 0x%08X", expectedWord1, desc.Word1)
	}
	if desc.Word2 != expectedWord2 {
		t.Errorf("Word2 mismatch: expected 0x%08X, got 0x%08X", expectedWord2, desc.Word2)
	}
	if desc.Word3 != expectedWord3 {
		t.Errorf("Word3 mismatch: expected 0x%08X, got 0x%08X", expectedWord3, desc.Word3)
	}

	// 2. Temporal pinned descriptor (SLC=0, GLC=0, DLC=0)
	descTemporal, err := pipeline.SynthesizeDescriptor(baseAddr, sizeBytes, FlagsTemporalPinned)
	if err != nil {
		t.Fatalf("SynthesizeDescriptor for temporal failed: %v", err)
	}
	if descTemporal.IsSLCSet() {
		t.Errorf("expected SLC bit to be 0 for temporal pinned descriptor: 0x%08X", descTemporal.Word3)
	}
	if (descTemporal.Word3 & (1 << 22)) != 0 {
		t.Errorf("expected SLC bit 22 to be 0 in Word3: 0x%08X", descTemporal.Word3)
	}

	// 3. Address Space Limits: 48-bit Virtual Address boundary
	invalidAddr := uint64(0x0001000000000000) // Exceeds 48-bit address limit
	_, err = pipeline.SynthesizeDescriptor(invalidAddr, sizeBytes, FlagsStreamingBypass)
	if err != ErrInvalidBaseAddress {
		t.Errorf("expected ErrInvalidBaseAddress for 49-bit address, got %v", err)
	}

	// 4. Alignment Requirement: 16-byte alignment
	unalignedAddr := uint64(0x00007FFF10002004) // Not 16-byte aligned
	_, err = pipeline.SynthesizeDescriptor(unalignedAddr, sizeBytes, FlagsStreamingBypass)
	if err != ErrUnalignedBaseAddress {
		t.Errorf("expected ErrUnalignedBaseAddress for unaligned address, got %v", err)
	}
}

// TestWeightStream_TensorClassification tests mutual exclusion between non-temporal
// weight streaming (SLC=1) and temporal KV cache / tree attention mask pinning (SLC=0).
func TestWeightStream_TensorClassification(t *testing.T) {
	cfg := DefaultWeightStreamConfig()
	pipeline := NewWeightStreamPipeline(cfg)
	defer pipeline.Close()

	tests := []struct {
		name         string
		attr         TensorMemoryAttributes
		expectedSLC  int
		expectedGLC  int
		expectedDLC  int
		wantBypass   bool
		wantTemporal bool
		wantPolicy   string
	}{
		{
			name: "Model Weight QKV Matrix",
			attr: TensorMemoryAttributes{
				Name:         "transformer.layer.0.attention.qkv_proj",
				SizeBytes:    512 * 1024 * 1024,
				Kind:         TensorKindWeight,
				NonRecurrent: true,
				ReadOnce:     true,
			},
			expectedSLC:  1,
			expectedGLC:  0,
			expectedDLC:  0,
			wantBypass:   true,
			wantTemporal: false,
			wantPolicy:   "NON_TEMPORAL_BYPASS",
		},
		{
			name: "Model Weight MLP Gate/Up Matrix",
			attr: TensorMemoryAttributes{
				Name:         "transformer.layer.0.mlp.gate_up_proj",
				SizeBytes:    1024 * 1024 * 1024,
				Kind:         TensorKindWeight,
				NonRecurrent: true,
				ReadOnce:     true,
			},
			expectedSLC:  1,
			expectedGLC:  0,
			expectedDLC:  0,
			wantBypass:   true,
			wantTemporal: false,
			wantPolicy:   "NON_TEMPORAL_BYPASS",
		},
		{
			name: "Heuristic Weight (ReadOnce >= 10MB)",
			attr: TensorMemoryAttributes{
				Name:         "unspecified_large_weight_buffer",
				SizeBytes:    15 * 1024 * 1024,
				Kind:         "",
				NonRecurrent: true,
				ReadOnce:     true,
			},
			expectedSLC:  1,
			expectedGLC:  0,
			expectedDLC:  0,
			wantBypass:   true,
			wantTemporal: false,
			wantPolicy:   "NON_TEMPORAL_BYPASS",
		},
		{
			name: "KV Cache Block (32MB Root Prefix)",
			attr: TensorMemoryAttributes{
				Name:         "kv_cache.root_prefix",
				SizeBytes:    32 * 1024 * 1024,
				Kind:         TensorKindKVCache,
				NonRecurrent: false,
				ReadOnce:     false,
			},
			expectedSLC:  0,
			expectedGLC:  0,
			expectedDLC:  0,
			wantBypass:   false,
			wantTemporal: true,
			wantPolicy:   "TEMPORAL_PINNED",
		},
		{
			name: "Speculative Draft Tree Attention Mask",
			attr: TensorMemoryAttributes{
				Name:         "tree_attention_mask.draft64",
				SizeBytes:    8192,
				Kind:         TensorKindTreeMask,
				NonRecurrent: false,
				ReadOnce:     false,
			},
			expectedSLC:  0,
			expectedGLC:  0,
			expectedDLC:  0,
			wantBypass:   false,
			wantTemporal: true,
			wantPolicy:   "TEMPORAL_PINNED",
		},
		{
			name: "Transient Intermediate Activation",
			attr: TensorMemoryAttributes{
				Name:         "transformer.layer.0.post_attention_layernorm",
				SizeBytes:    16 * 1024 * 1024,
				Kind:         TensorKindActivation,
				NonRecurrent: true,
				ReadOnce:     true,
			},
			expectedSLC:  0,
			expectedGLC:  0,
			expectedDLC:  0,
			wantBypass:   false,
			wantTemporal: true,
			wantPolicy:   "TEMPORAL_ACTIVATION",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			flags := pipeline.ClassifyTensor(tc.attr)

			if flags.SLC != tc.expectedSLC {
				t.Errorf("SLC mismatch: expected %d, got %d", tc.expectedSLC, flags.SLC)
			}
			if flags.GLC != tc.expectedGLC {
				t.Errorf("GLC mismatch: expected %d, got %d", tc.expectedGLC, flags.GLC)
			}
			if flags.DLC != tc.expectedDLC {
				t.Errorf("DLC mismatch: expected %d, got %d", tc.expectedDLC, flags.DLC)
			}
			if flags.IsBypass() != tc.wantBypass {
				t.Errorf("IsBypass mismatch: expected %v, got %v", tc.wantBypass, flags.IsBypass())
			}
			if flags.IsTemporal() != tc.wantTemporal {
				t.Errorf("IsTemporal mismatch: expected %v, got %v", tc.wantTemporal, flags.IsTemporal())
			}
			if flags.PolicyName != tc.wantPolicy {
				t.Errorf("PolicyName mismatch: expected %s, got %s", tc.wantPolicy, flags.PolicyName)
			}

			// Invariant verification: Strict Mutual Exclusion
			if flags.IsBypass() && flags.IsTemporal() {
				t.Fatalf("INVARIANT VIOLATION: tensor flagged as both Bypass and Temporal: %+v", flags)
			}
			if tc.attr.Kind == TensorKindWeight && flags.SLC != 1 {
				t.Fatalf("INVARIANT VIOLATION: weight tensor allocated with SLC=%d (expected 1)", flags.SLC)
			}
			if tc.attr.Kind == TensorKindKVCache && flags.SLC != 0 {
				t.Fatalf("INVARIANT VIOLATION: KV cache tensor allocated with SLC=%d (expected 0)", flags.SLC)
			}
		})
	}
}

// TestWeightStream_TemporalThrashingBaseline tests Ablation Arm 3:
// When weights are loaded with default unmanaged temporal caching (SLC=0),
// 17.5 GB of weights completely evicts the 32MB MALL cache (521.5x cache purges),
// causing prompt prefix residency to drop to 0% and triggering massive DRAM KV re-fetches.
func TestWeightStream_TemporalThrashingBaseline(t *testing.T) {
	cfg := DefaultWeightStreamConfig()
	pipeline := NewWeightStreamPipeline(cfg)
	defer pipeline.Close()

	weightBytes := Default35BWeightBytes
	numLayers := 40
	pinnedKVBytes := MALLCapacityBytes

	telem, err := pipeline.SimulateForwardPass(weightBytes, numLayers, pinnedKVBytes, false)
	if err != nil {
		t.Fatalf("SimulateForwardPass failed: %v", err)
	}

	// 1. Massive MALL Evictions
	if telem.MALLEvictions == 0 {
		t.Errorf("expected massive MALL evictions in unmanaged baseline, got 0")
	}
	if pipeline.MALL().EvictionCount() == 0 {
		t.Errorf("expected non-zero cache model evictions, got 0")
	}

	// 2. Prompt Prefix residency drops to 0% (obliterated in layer 1)
	if telem.MALLResidencyPct > 5.0 {
		t.Errorf("expected prompt prefix residency to drop near 0%%, got %.2f%%", telem.MALLResidencyPct)
	}
	if pipeline.MALL().PinnedKVResidencyPct() > 5.0 {
		t.Errorf("expected cache model residency near 0%%, got %.2f%%", pipeline.MALL().PinnedKVResidencyPct())
	}

	// 3. Zero DRAM Bandwidth Saved
	if telem.DRAMBandwidthSavedGBs != 0.0 {
		t.Errorf("expected 0.0 GB/s DRAM bandwidth saved in unmanaged baseline, got %.2f", telem.DRAMBandwidthSavedGBs)
	}

	// 4. DRAM KV Re-fetch Bytes equals 40 layers * 32MB = 1.28 GB
	expectedDRAMKVBytes := pinnedKVBytes * int64(numLayers)
	if telem.DRAMKVBytesRead != expectedDRAMKVBytes {
		t.Errorf("expected %d DRAM KV re-fetch bytes, got %d", expectedDRAMKVBytes, telem.DRAMKVBytesRead)
	}

	// 5. Eviction ratio >= 500.0 (mathematical 521.5x)
	if telem.EvictionRatio < 500.0 {
		t.Errorf("expected eviction ratio >= 500.0x, got %.1fx", telem.EvictionRatio)
	}

	// 6. Degraded Throughput (154.2 GB/s due to memory bus thrashing)
	if telem.ThroughputGBs >= 200.0 {
		t.Errorf("expected degraded throughput (< 200.0 GB/s), got %.2f GB/s", telem.ThroughputGBs)
	}

	t.Logf("[SW-VERIFIED] Temporal Thrashing Baseline (Ablation Arm 3) Witnessed:")
	t.Logf("  MALL Evictions: %d", telem.MALLEvictions)
	t.Logf("  Prompt Prefix Residency: %.2f%% (evicted)", telem.MALLResidencyPct)
	t.Logf("  DRAM KV Re-fetch Footprint: %.2f GB", float64(telem.DRAMKVBytesRead)/(1024*1024*1024))
	t.Logf("  Degraded Bus Throughput: %.2f GB/s", telem.ThroughputGBs)
}

// TestWeightStream_SelectiveMLPBypass tests Ablation Arm 2:
// Bypassing only large MLP weights (12 GB) while leaving attention projection weights
// (5.5 GB) unmanaged with default SLC=0 still purges the 32MB MALL cache 163.9 times,
// proving why full non-temporal bypass across all weight matrices (Arm 1) is mandatory.
func TestWeightStream_SelectiveMLPBypass(t *testing.T) {
	cfg := DefaultWeightStreamConfig()
	pipeline := NewWeightStreamPipeline(cfg)
	defer pipeline.Close()

	totalWeightBytes := Default35BWeightBytes
	mlpWeightBytes := int64(12 * 1024 * 1024 * 1024)
	attnWeightBytes := int64(5500 * 1024 * 1024)
	numLayers := 40
	pinnedKVBytes := MALLCapacityBytes

	telem, err := pipeline.SimulateSelectiveMLPPass(totalWeightBytes, mlpWeightBytes, attnWeightBytes, numLayers, pinnedKVBytes)
	if err != nil {
		t.Fatalf("SimulateSelectiveMLPPass failed: %v", err)
	}

	// 1. Both bypassed and temporal descriptors generated
	if telem.BypassedDescriptors == 0 {
		t.Errorf("expected bypassed descriptors for MLP weights, got 0")
	}
	if telem.TemporalDescriptors == 0 {
		t.Errorf("expected temporal descriptors for attention weights, got 0")
	}

	// 2. MALL Evictions still occur due to 5.5 GB of unmanaged attention weights
	if telem.MALLEvictions == 0 {
		t.Errorf("expected MALL evictions from unmanaged attention weights, got 0")
	}

	// 3. Prompt Prefix residency still drops to ~0%
	if telem.MALLResidencyPct > 5.0 {
		t.Errorf("expected prompt prefix residency to drop near 0%% due to 5.5GB attention weights, got %.2f%%", telem.MALLResidencyPct)
	}

	// 4. DRAM KV Re-fetches still occur
	if telem.DRAMKVBytesRead == 0 {
		t.Errorf("expected non-zero DRAM KV re-fetches, got 0")
	}

	t.Logf("[SW-VERIFIED] Selective MLP Bypass (Ablation Arm 2) Witnessed:")
	t.Logf("  MLP Bypassed Bytes: %.2f GB", float64(mlpWeightBytes)/(1024*1024*1024))
	t.Logf("  Unmanaged Attention Bytes: %.2f GB", float64(attnWeightBytes)/(1024*1024*1024))
	t.Logf("  Attention Eviction Ratio: %.1fx", telem.EvictionRatio)
	t.Logf("  Resulting MALL Residency: %.2f%% (demonstrates Arm 1 necessity)", telem.MALLResidencyPct)
}

// TestWeightStream_StrideAlignment tests burst stride alignment validation and descriptor encoding.
func TestWeightStream_StrideAlignment(t *testing.T) {
	// 1. Valid stride configurations (multiples of 128 bytes)
	for _, stride := range []int{128, 256, 512, 1024} {
		cfg := WeightStreamConfig{
			BurstStrideBytes:           stride,
			MemoryChannels:             8,
			PrefetchLookahead:          1,
			WeightStreamThresholdBytes: 10 * 1024 * 1024,
			TargetBandwidthGBs:         220.0,
			MaxQueueDepth:              8,
		}
		if err := cfg.Validate(); err != nil {
			t.Errorf("expected stride %d to be valid, got error: %v", stride, err)
		}

		pipeline := NewWeightStreamPipeline(cfg)
		desc, err := pipeline.SynthesizeDescriptor(0x10000000, 65536, FlagsStreamingBypass)
		if err != nil {
			t.Errorf("descriptor synthesis failed for stride %d: %v", stride, err)
		}
		if desc.Stride() != uint16(stride) {
			t.Errorf("expected stride %d in descriptor, got %d", stride, desc.Stride())
		}
		pipeline.Close()
	}

	// 2. Invalid unaligned strides
	for _, invalidStride := range []int{0, -1, 64, 100, 200, 250} {
		cfg := WeightStreamConfig{
			BurstStrideBytes:           invalidStride,
			MemoryChannels:             8,
			PrefetchLookahead:          1,
			WeightStreamThresholdBytes: 10 * 1024 * 1024,
		}
		if err := cfg.Validate(); err == nil {
			t.Errorf("expected invalid stride %d to fail validation", invalidStride)
		}
	}
}

// TestWeightStream_MaintenanceJSON tests JSON serialization for the /system/maintenance endpoint.
func TestWeightStream_MaintenanceJSON(t *testing.T) {
	cfg := DefaultWeightStreamConfig()
	pipeline := NewWeightStreamPipeline(cfg)
	defer pipeline.Close()

	// Run forward pass to populate telemetry
	_, err := pipeline.SimulateForwardPass(Default35BWeightBytes, 40, MALLCapacityBytes, true)
	if err != nil {
		t.Fatalf("SimulateForwardPass failed: %v", err)
	}

	jsonBytes, err := pipeline.ExportMaintenanceJSON()
	if err != nil {
		t.Fatalf("ExportMaintenanceJSON failed: %v", err)
	}

	if len(jsonBytes) == 0 {
		t.Fatalf("exported JSON is empty")
	}

	// Verify unmarshaling into map and checking required fields
	var data map[string]interface{}
	if err := json.Unmarshal(jsonBytes, &data); err != nil {
		t.Fatalf("failed to unmarshal exported JSON: %v\nJSON:\n%s", err, string(jsonBytes))
	}

	requiredKeys := []string{
		"mall_evictions",
		"mall_residency_pct",
		"dram_bandwidth_saved_gbs",
		"throughput_gbs",
		"weight_bytes_streamed",
		"pinned_kv_bytes",
		"layers_processed",
		"bypassed_descriptors",
		"temporal_descriptors",
		"eviction_ratio",
		"dram_kv_bytes_read",
		"timestamp",
	}

	for _, key := range requiredKeys {
		if _, exists := data[key]; !exists {
			t.Errorf("missing required telemetry key in JSON: %s", key)
		}
	}

	// Verify unmarshaling into WeightStreamTelemetry struct
	var telem WeightStreamTelemetry
	if err := json.Unmarshal(jsonBytes, &telem); err != nil {
		t.Fatalf("failed to unmarshal into WeightStreamTelemetry: %v", err)
	}

	if telem.MALLEvictions != 0 {
		t.Errorf("JSON field mall_evictions mismatch: expected 0, got %d", telem.MALLEvictions)
	}
	if telem.MALLResidencyPct < 95.0 {
		t.Errorf("JSON field mall_residency_pct mismatch: expected >= 95.0, got %.2f", telem.MALLResidencyPct)
	}
	if telem.DRAMBandwidthSavedGBs < 32.0 {
		t.Errorf("JSON field dram_bandwidth_saved_gbs mismatch: expected >= 32.0, got %.2f", telem.DRAMBandwidthSavedGBs)
	}

	t.Logf("Maintenance JSON telemetry successfully verified:\n%s", string(jsonBytes))
}

// TestWeightStream_ConcurrentRace verifies thread-safety and race condition freedom under -race.
func TestWeightStream_ConcurrentRace(t *testing.T) {
	cfg := DefaultWeightStreamConfig()
	pipeline := NewWeightStreamPipeline(cfg)
	defer pipeline.Close()

	// Pre-populate MALL with pinned KV
	pipeline.MALL().PinKVRange(0x10000000, MALLCapacityBytes)

	var wg sync.WaitGroup
	workers := 16
	iterations := 100

	for w := 0; w < workers; w++ {
		wg.Add(1)
		workerID := w
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				// 1. Classify tensors concurrently
				attr := TensorMemoryAttributes{
					Name:         fmt.Sprintf("layer.%d.weight", (workerID*iterations+i)%40),
					SizeBytes:    int64((i + 1) * 1024 * 1024),
					Kind:         TensorKindWeight,
					NonRecurrent: true,
					ReadOnce:     true,
				}
				flags := pipeline.ClassifyTensor(attr)

				// 2. Synthesize buffer descriptors concurrently
				baseAddr := uint64(0x200000000000) + uint64(workerID*iterations+i)*uint64(BurstStrideBytes)
				_, err := pipeline.SynthesizeDescriptor(baseAddr, uint32(attr.SizeBytes), flags)
				if err != nil {
					t.Errorf("concurrent SynthesizeDescriptor failed: %v", err)
					return
				}

				// 3. Access MALL cache model concurrently
				pipeline.MALL().Access(baseAddr, 128, flags, false)

				// 4. Query telemetry concurrently
				_ = pipeline.Telemetry()
				_ = pipeline.MALL().PinnedKVResidencyPct()
				_ = pipeline.MALL().EvictionCount()

				// 5. Export JSON periodically
				if i%25 == 0 {
					_, _ = pipeline.ExportMaintenanceJSON()
				}
			}
		}()
	}

	wg.Wait()
}
