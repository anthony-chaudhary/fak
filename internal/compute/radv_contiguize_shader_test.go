package compute

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"strings"
	"testing"
)

// TestRADVContiguizeShader is the main test suite covering the RADV compute shader abstraction
// for pre-attention f16 KV contiguization on AMD Strix Halo (#11903).
//
// In accordance with #12120 and #12124, this selector represents a truthful host-side
// contract verification (CPU reference emulation, geometry, GLSL source, and entropy restoration)
// rather than a Vulkan physical device dispatch. It emits exactly one typed
// fak.strix.subkernel-parity/v1 event with device_observed=false and oracle_kind=host_contract,
// which is strictly barred from physical-device parity credit.
func TestRADVContiguizeShader(t *testing.T) {
	var totalCases int

	t.Run("PushConstantsEncoding", func(t *testing.T) {
		totalCases += testRADVPushConstantsEncoding(t)
	})
	t.Run("WorkgroupGeometryAndDispatch", func(t *testing.T) {
		totalCases += testRADVWorkgroupGeometryAndDispatch(t)
	})
	t.Run("InVRAMScratchAllocation", func(t *testing.T) {
		totalCases += testRADVScratchAllocation(t)
	})
	t.Run("MathematicalParityWithCPURef", func(t *testing.T) {
		totalCases += testRADVMathematicalParity(t)
	})
	t.Run("ChannelEntropyRestoration", func(t *testing.T) {
		totalCases += testRADVChannelEntropyRestoration(t)
	})
	t.Run("PipelineDescriptorAndGLSL", func(t *testing.T) {
		totalCases += testRADVPipelineDescriptorAndGLSL(t)
	})

	emitRADVContiguizeHostContractEvent(t, totalCases, !t.Failed())
}

func testRADVPushConstantsEncoding(t *testing.T) int {
	testCases := []struct {
		nPos    int
		nKV     int
		headDim int
	}{
		{32768, 8, 128},
		{65536, 16, 128},
		{131072, 8, 128},
		{4096, 4, 64},
		{1024, 2, 80},
		{17, 3, 19},
	}

	for _, tc := range testCases {
		pc, err := NewRADVContiguizePushConstants(tc.nPos, tc.nKV, tc.headDim)
		if err != nil {
			t.Fatalf("NewRADVContiguizePushConstants(%d, %d, %d) failed: %v", tc.nPos, tc.nKV, tc.headDim, err)
		}

		if pc.Size() != RADVPushConstantBytes {
			t.Errorf("expected pc.Size() == %d, got %d", RADVPushConstantBytes, pc.Size())
		}

		encoded := pc.Encode()
		if len(encoded) != RADVPushConstantBytes {
			t.Fatalf("expected encoded length %d, got %d", RADVPushConstantBytes, len(encoded))
		}

		decoded, err := DecodeRADVContiguizePushConstants(encoded)
		if err != nil {
			t.Fatalf("DecodeRADVContiguizePushConstants failed: %v", err)
		}

		if decoded.NPos != pc.NPos {
			t.Errorf("decoded.NPos = %d, want %d", decoded.NPos, pc.NPos)
		}
		if decoded.NKV != pc.NKV {
			t.Errorf("decoded.NKV = %d, want %d", decoded.NKV, pc.NKV)
		}
		if decoded.HeadDim != pc.HeadDim {
			t.Errorf("decoded.HeadDim = %d, want %d", decoded.HeadDim, pc.HeadDim)
		}
		if decoded.StrideToken != pc.StrideToken {
			t.Errorf("decoded.StrideToken = %d, want %d", decoded.StrideToken, pc.StrideToken)
		}
		if decoded.StrideHeadContig != pc.StrideHeadContig {
			t.Errorf("decoded.StrideHeadContig = %d, want %d", decoded.StrideHeadContig, pc.StrideHeadContig)
		}
	}

	// Truncated buffer check
	var shortBuf [16]byte
	var badPC RADVContiguizePushConstants
	if err := badPC.Decode(shortBuf[:]); err == nil {
		t.Errorf("expected error decoding short buffer (< 20 bytes), got nil")
	}

	// Invalid dimensions
	if _, err := NewRADVContiguizePushConstants(0, 8, 128); err == nil {
		t.Errorf("expected error for nPos=0, got nil")
	}
	if _, err := NewRADVContiguizePushConstants(32768, -1, 128); err == nil {
		t.Errorf("expected error for negative nKV, got nil")
	}

	// Inconsistent strides check
	inconsistentPC := RADVContiguizePushConstants{
		NPos:             32768,
		NKV:              8,
		HeadDim:          128,
		StrideToken:      1000, // Should be 8 * 128 = 1024
		StrideHeadContig: 32768 * 128,
	}
	if err := inconsistentPC.Validate(); err == nil {
		t.Errorf("expected validation error for inconsistent strideToken, got nil")
	}
	return len(testCases)
}

func testRADVWorkgroupGeometryAndDispatch(t *testing.T) int {
	geom64 := NewWorkgroupGeometryWave64()
	if geom64.TotalThreads() != 64 {
		t.Errorf("Wave64 total threads = %d, want 64", geom64.TotalThreads())
	}
	if geom64.WaveMode != RADVWave64 {
		t.Errorf("Wave64 wave mode = %s, want %s", geom64.WaveMode, RADVWave64)
	}

	geom32 := NewWorkgroupGeometryWave32()
	if geom32.TotalThreads() != 32 {
		t.Errorf("Wave32 total threads = %d, want 32", geom32.TotalThreads())
	}
	if geom32.WaveMode != RADVWave32 {
		t.Errorf("Wave32 wave mode = %s, want %s", geom32.WaveMode, RADVWave32)
	}

	// Dispatch tests
	// 1. Exact division: headDim=128, Wave64 (LocalSizeX=64) -> GridX = 2
	plan64, err := PlanRADVContiguizeDispatch(32768, 8, 128, geom64)
	if err != nil {
		t.Fatalf("PlanRADVContiguizeDispatch failed: %v", err)
	}
	if plan64.Dimensions.GridX != 2 {
		t.Errorf("plan64.Dimensions.GridX = %d, want 2", plan64.Dimensions.GridX)
	}
	if plan64.Dimensions.GridY != 32768 {
		t.Errorf("plan64.Dimensions.GridY = %d, want 32768", plan64.Dimensions.GridY)
	}
	if plan64.Dimensions.GridZ != 8 {
		t.Errorf("plan64.Dimensions.GridZ = %d, want 8", plan64.Dimensions.GridZ)
	}
	expectedWorkgroups := uint64(2 * 32768 * 8)
	if plan64.TotalWorkgroups != expectedWorkgroups {
		t.Errorf("plan64.TotalWorkgroups = %d, want %d", plan64.TotalWorkgroups, expectedWorkgroups)
	}
	if plan64.TotalThreads != expectedWorkgroups*64 {
		t.Errorf("plan64.TotalThreads = %d, want %d", plan64.TotalThreads, expectedWorkgroups*64)
	}
	if plan64.ThreadEfficiency != 1.0 {
		t.Errorf("plan64.ThreadEfficiency = %f, want 1.0", plan64.ThreadEfficiency)
	}

	// 2. Exact division: headDim=128, Wave32 (LocalSizeX=32) -> GridX = 4
	plan32, err := PlanRADVContiguizeDispatch(32768, 8, 128, geom32)
	if err != nil {
		t.Fatalf("PlanRADVContiguizeDispatch failed: %v", err)
	}
	if plan32.Dimensions.GridX != 4 {
		t.Errorf("plan32.Dimensions.GridX = %d, want 4", plan32.Dimensions.GridX)
	}
	if plan32.TotalWorkgroups != uint64(4*32768*8) {
		t.Errorf("plan32.TotalWorkgroups = %d, want %d", plan32.TotalWorkgroups, 4*32768*8)
	}

	// 3. Non-multiple dimension: headDim=80, Wave64 -> GridX = ceil(80/64) = 2
	planOdd, err := PlanRADVContiguizeDispatch(1024, 4, 80, geom64)
	if err != nil {
		t.Fatalf("PlanRADVContiguizeDispatch failed: %v", err)
	}
	if planOdd.Dimensions.GridX != 2 {
		t.Errorf("planOdd.Dimensions.GridX = %d, want 2", planOdd.Dimensions.GridX)
	}
	// Active elements: 1024 * 4 * 80 = 327680
	// Total threads: (2 * 1024 * 4) * 64 = 524288
	// Efficiency: 327680 / 524288 = 0.625
	expectedEff := float64(327680) / float64(524288)
	if planOdd.ThreadEfficiency != expectedEff {
		t.Errorf("planOdd.ThreadEfficiency = %f, want %f", planOdd.ThreadEfficiency, expectedEff)
	}

	// 4. Arbitrary small geometry: nPos=17, nKV=3, headDim=19 with Wave64
	planArbitrary, err := PlanRADVContiguizeDispatch(17, 3, 19, geom64)
	if err != nil {
		t.Fatalf("PlanRADVContiguizeDispatch failed: %v", err)
	}
	if planArbitrary.Dimensions.GridX != 1 {
		t.Errorf("planArbitrary.Dimensions.GridX = %d, want 1", planArbitrary.Dimensions.GridX)
	}
	if planArbitrary.Dimensions.GridY != 17 {
		t.Errorf("planArbitrary.Dimensions.GridY = %d, want 17", planArbitrary.Dimensions.GridY)
	}
	if planArbitrary.Dimensions.GridZ != 3 {
		t.Errorf("planArbitrary.Dimensions.GridZ = %d, want 3", planArbitrary.Dimensions.GridZ)
	}
	return 4
}

func testRADVScratchAllocation(t *testing.T) int {
	// 1. Standard large APU case: nPos=32768, nKV=8, headDim=128
	// Raw per buffer: 8 * 32768 * 128 * 2 = 67,108,864 bytes (64 MB).
	// Total raw bytes: 2 * 67,108,864 = 134,217,728 bytes (128 MB).
	// Since 67,108,864 % 256 == 0, aligned == raw.
	alloc, err := ComputeRADVScratchAllocation(32768, 8, 128)
	if err != nil {
		t.Fatalf("ComputeRADVScratchAllocation failed: %v", err)
	}

	expectedPerBuf := int64(8 * 32768 * 128 * 2)
	expectedTotal := 2 * expectedPerBuf

	if alloc.PerBufferRawBytes != expectedPerBuf {
		t.Errorf("PerBufferRawBytes = %d, want %d", alloc.PerBufferRawBytes, expectedPerBuf)
	}
	if alloc.PerBufferAlignedBytes != expectedPerBuf {
		t.Errorf("PerBufferAlignedBytes = %d, want %d", alloc.PerBufferAlignedBytes, expectedPerBuf)
	}
	if alloc.TotalRawBytes != expectedTotal {
		t.Errorf("TotalRawBytes = %d, want %d", alloc.TotalRawBytes, expectedTotal)
	}
	if alloc.TotalAlignedBytes != expectedTotal {
		t.Errorf("TotalAlignedBytes = %d, want %d", alloc.TotalAlignedBytes, expectedTotal)
	}
	if alloc.KOffset != 0 {
		t.Errorf("KOffset = %d, want 0", alloc.KOffset)
	}
	if alloc.VOffset != alloc.PerBufferAlignedBytes {
		t.Errorf("VOffset = %d, want %d", alloc.VOffset, alloc.PerBufferAlignedBytes)
	}
	if alloc.VOffset%RADVScratchAlignmentBytes != 0 {
		t.Errorf("VOffset %d not aligned to %d bytes", alloc.VOffset, RADVScratchAlignmentBytes)
	}

	// 2. Unaligned case: nPos=3, nKV=1, headDim=17
	// Raw per buffer: 1 * 3 * 17 * 2 = 102 bytes.
	// Aligned to 256: 256 bytes.
	// Total raw: 204 bytes.
	// Total aligned: 2 * 256 = 512 bytes.
	unalignedAlloc, err := ComputeRADVScratchAllocation(3, 1, 17)
	if err != nil {
		t.Fatalf("ComputeRADVScratchAllocation failed: %v", err)
	}
	if unalignedAlloc.PerBufferRawBytes != 102 {
		t.Errorf("PerBufferRawBytes = %d, want 102", unalignedAlloc.PerBufferRawBytes)
	}
	if unalignedAlloc.PerBufferAlignedBytes != 256 {
		t.Errorf("PerBufferAlignedBytes = %d, want 256", unalignedAlloc.PerBufferAlignedBytes)
	}
	if unalignedAlloc.TotalRawBytes != 204 {
		t.Errorf("TotalRawBytes = %d, want 204", unalignedAlloc.TotalRawBytes)
	}
	if unalignedAlloc.TotalAlignedBytes != 512 {
		t.Errorf("TotalAlignedBytes = %d, want 512", unalignedAlloc.TotalAlignedBytes)
	}
	if unalignedAlloc.VOffset != 256 {
		t.Errorf("VOffset = %d, want 256", unalignedAlloc.VOffset)
	}
	if unalignedAlloc.VOffset%RADVScratchAlignmentBytes != 0 {
		t.Errorf("VOffset not 256-byte aligned")
	}

	// 3. Invalid parameters
	if _, err := ComputeRADVScratchAllocation(0, 8, 128); err == nil {
		t.Errorf("expected error for nPos=0, got nil")
	}
	if _, err := ComputeRADVScratchAllocation(32768, 0, 128); err == nil {
		t.Errorf("expected error for nKV=0, got nil")
	}
	if _, err := ComputeRADVScratchAllocation(32768, 8, 0); err == nil {
		t.Errorf("expected error for headDim=0, got nil")
	}
	return 2
}

func testRADVMathematicalParity(t *testing.T) int {
	pipeline64, err := NewRADVContiguizePipelineDescriptor(RADVTargetArchGfx1151, RADVWave64, RADVDefaultInterleaveBytes)
	if err != nil {
		t.Fatalf("NewRADVContiguizePipelineDescriptor(Wave64) failed: %v", err)
	}
	emulator64 := NewRADVContiguizeShaderEmulator(pipeline64)

	pipeline32, err := NewRADVContiguizePipelineDescriptor(RADVTargetArchGfx1151, RADVWave32, RADVDefaultInterleaveBytes)
	if err != nil {
		t.Fatalf("NewRADVContiguizePipelineDescriptor(Wave32) failed: %v", err)
	}
	emulator32 := NewRADVContiguizeShaderEmulator(pipeline32)

	testCases := []struct {
		name    string
		nPos    int
		nKV     int
		headDim int
	}{
		{"SmallShape", 4, 2, 16},
		{"PrimeShape", 7, 3, 19},
		{"MediumShape", 128, 4, 64},
		{"OddHeadDimWave64", 32, 2, 80},
		{"LargeContext", 1024, 8, 128},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			totalElems := tc.nPos * tc.nKV * tc.headDim
			srcK := make([]uint16, totalElems)
			srcV := make([]uint16, totalElems)

			rng := rand.New(rand.NewSource(int64(tc.nPos*1000 + tc.nKV*100 + tc.headDim)))
			for i := range srcK {
				srcK[i] = uint16(rng.Uint32() & 0xFFFF)
				srcV[i] = uint16(rng.Uint32() & 0xFFFF)
			}

			// 1. CPU reference contiguization
			refK, err := ContiguizeF16KVCache(srcK, nil, tc.nPos, tc.nKV, tc.headDim)
			if err != nil {
				t.Fatalf("ContiguizeF16KVCache(K) failed: %v", err)
			}
			refV, err := ContiguizeF16KVCache(srcV, nil, tc.nPos, tc.nKV, tc.headDim)
			if err != nil {
				t.Fatalf("ContiguizeF16KVCache(V) failed: %v", err)
			}

			// 2. Modeled shader dispatch with Wave64
			pc, err := NewRADVContiguizePushConstants(tc.nPos, tc.nKV, tc.headDim)
			if err != nil {
				t.Fatalf("NewRADVContiguizePushConstants failed: %v", err)
			}
			shaderOutK64, err := emulator64.Execute(srcK, pc, pipeline64.Geometry)
			if err != nil {
				t.Fatalf("emulator64.Execute(K) failed: %v", err)
			}
			if len(shaderOutK64) != len(refK) {
				t.Fatalf("length mismatch: got %d, want %d", len(shaderOutK64), len(refK))
			}
			for i := range refK {
				if shaderOutK64[i] != refK[i] {
					t.Fatalf("Wave64 K parity mismatch at elem %d: got 0x%04X, want 0x%04X", i, shaderOutK64[i], refK[i])
				}
			}

			// 3. Modeled shader dispatch with Wave32
			shaderOutK32, err := emulator32.Execute(srcK, pc, pipeline32.Geometry)
			if err != nil {
				t.Fatalf("emulator32.Execute(K) failed: %v", err)
			}
			for i := range refK {
				if shaderOutK32[i] != refK[i] {
					t.Fatalf("Wave32 K parity mismatch at elem %d: got 0x%04X, want 0x%04X", i, shaderOutK32[i], refK[i])
				}
			}

			// 4. Test ExecuteKV running both K and V
			emulK, emulV, err := emulator64.ExecuteKV(srcK, srcV, tc.nPos, tc.nKV, tc.headDim, pipeline64.Geometry)
			if err != nil {
				t.Fatalf("emulator64.ExecuteKV failed: %v", err)
			}
			for i := range refK {
				if emulK[i] != refK[i] {
					t.Fatalf("ExecuteKV K mismatch at elem %d: got 0x%04X, want 0x%04X", i, emulK[i], refK[i])
				}
				if emulV[i] != refV[i] {
					t.Fatalf("ExecuteKV V mismatch at elem %d: got 0x%04X, want 0x%04X", i, emulV[i], refV[i])
				}
			}
		})
	}
	return len(testCases)
}

func testRADVChannelEntropyRestoration(t *testing.T) int {
	testContexts := []int{32768, 65536, 131072}
	interleaveOptions := []int{RADVInterleaveBytes128, RADVInterleaveBytes64}

	for _, interleave := range interleaveOptions {
		for _, nPos := range testContexts {
			// 1. Strided layout simulation: verify channel camping
			stridedRep := SimulateStrixHaloChannelDistribution(nPos, 8, 128, false, interleave)

			if stridedRep.ActiveChannels > 2 {
				t.Errorf("interleave=%dB, nPos=%d: expected strided active channels <= 2, got %d (counts: %v)",
					interleave, nPos, stridedRep.ActiveChannels, stridedRep.ChannelCounts)
			}
			if stridedRep.Entropy >= 0.25 {
				t.Errorf("interleave=%dB, nPos=%d: expected strided entropy < 0.25, got %.4f (raw: %.4f)",
					interleave, nPos, stridedRep.Entropy, stridedRep.RawEntropy)
			}
			if stridedRep.IsContiguized {
				t.Errorf("expected IsContiguized = false")
			}

			// 2. Contiguized layout simulation: verify uniform spread across all 16 channels
			contigRep := SimulateStrixHaloChannelDistribution(nPos, 8, 128, true, interleave)

			if contigRep.ActiveChannels != RADVChannelCountStrixHalo {
				t.Errorf("interleave=%dB, nPos=%d: expected contiguized active channels == %d, got %d",
					interleave, nPos, RADVChannelCountStrixHalo, contigRep.ActiveChannels)
			}
			if contigRep.Entropy <= 0.95 {
				t.Errorf("interleave=%dB, nPos=%d: expected contiguized entropy > 0.95, got %.4f (raw: %.4f)",
					interleave, nPos, contigRep.Entropy, contigRep.RawEntropy)
			}
			if !contigRep.IsContiguized {
				t.Errorf("expected IsContiguized = true")
			}

			// Verify channel counts are perfectly balanced across all 16 channels
			expectedPerChannel := contigRep.ChannelCounts[0]
			for c := 1; c < RADVChannelCountStrixHalo; c++ {
				if contigRep.ChannelCounts[c] != expectedPerChannel {
					t.Errorf("interleave=%dB, nPos=%d: channel %d count %d != channel 0 count %d",
						interleave, nPos, c, contigRep.ChannelCounts[c], expectedPerChannel)
				}
			}

			// Entropy restoration gain (> 3.5x improvement)
			entropyGain := contigRep.Entropy / stridedRep.Entropy
			if entropyGain < 3.5 {
				t.Errorf("interleave=%dB, nPos=%d: expected entropy restoration gain >= 3.5x, got %.2fx",
					interleave, nPos, entropyGain)
			}
		}
	}
	return len(interleaveOptions) * len(testContexts)
}

func testRADVPipelineDescriptorAndGLSL(t *testing.T) int {
	// Valid descriptor for gfx1151
	pipe, err := NewRADVContiguizePipelineDescriptor(RADVTargetArchGfx1151, RADVWave64, RADVDefaultInterleaveBytes)
	if err != nil {
		t.Fatalf("NewRADVContiguizePipelineDescriptor failed: %v", err)
	}

	if pipe.ShaderStage != RADVShaderStageCompute {
		t.Errorf("ShaderStage = %q, want %q", pipe.ShaderStage, RADVShaderStageCompute)
	}
	if pipe.EntryPoint != RADVDefaultEntryPoint {
		t.Errorf("EntryPoint = %q, want %q", pipe.EntryPoint, RADVDefaultEntryPoint)
	}
	if pipe.PushConstantSize != RADVPushConstantBytes {
		t.Errorf("PushConstantSize = %d, want %d", pipe.PushConstantSize, RADVPushConstantBytes)
	}
	if len(pipe.Bindings) != 2 {
		t.Fatalf("expected 2 bindings, got %d", len(pipe.Bindings))
	}
	if pipe.Bindings[0].Access != "readonly" || pipe.Bindings[1].Access != "writeonly" {
		t.Errorf("unexpected binding accesses: %v", pipe.Bindings)
	}

	// GLSL source checks
	glsl := pipe.GLSLSource
	requiredTokens := []string{
		"#version 450",
		"GL_EXT_shader_explicit_arithmetic_types_float16",
		"layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in",
		"layout(push_constant) uniform RADVContiguizePushConstants",
		"uint nPos;",
		"uint nKV;",
		"uint headDim;",
		"uint strideToken;",
		"uint strideHeadContig;",
		"float16_t srcData[];",
		"float16_t dstData[];",
		"gl_GlobalInvocationID",
	}

	for _, token := range requiredTokens {
		if !strings.Contains(glsl, token) {
			t.Errorf("GLSL missing required token %q", token)
		}
	}

	// Non-APU architecture rejection
	invalidArchs := []string{"gfx1100", "gfx1030", "sm_90", "cuda", "cpu"}
	for _, arch := range invalidArchs {
		if _, err := NewRADVContiguizePipelineDescriptor(arch, RADVWave64, RADVDefaultInterleaveBytes); err == nil {
			t.Errorf("expected error for non-APU arch %q, got nil", arch)
		}
	}
	return 1
}

// Standalone tests for specific -run matching
func TestRADVContiguizeShader_PushConstants(t *testing.T) {
	_ = testRADVPushConstantsEncoding(t)
}

func TestRADVContiguizeShader_DispatchDimensions(t *testing.T) {
	_ = testRADVWorkgroupGeometryAndDispatch(t)
}

func TestRADVContiguizeShader_ScratchAllocation(t *testing.T) {
	_ = testRADVScratchAllocation(t)
}

func TestRADVContiguizeShader_Parity(t *testing.T) {
	_ = testRADVMathematicalParity(t)
}

func TestRADVContiguizeShader_ChannelEntropy(t *testing.T) {
	_ = testRADVChannelEntropyRestoration(t)
}

func TestRADVContiguizeShader_PipelineDescriptor(t *testing.T) {
	_ = testRADVPipelineDescriptorAndGLSL(t)
}

// RADVContiguizeParityEvent defines the truthful host-contract event schema
// emitted by TestRADVContiguizeShader for the f16_kv_contiguize subkernel selector (#12120, #12124).
// Because host emulation is non-device, device_observed is explicitly false
// and cannot be promoted to physical-device parity credit.
type RADVContiguizeParityEvent struct {
	Schema         string                       `json:"schema"`
	Selector       string                       `json:"selector"`
	TestName       string                       `json:"test_name"`
	Test           string                       `json:"test,omitempty"`
	OracleKind     string                       `json:"oracle_kind"`
	CaseCount      int                          `json:"case_count"`
	DeviceObserved bool                         `json:"device_observed"`
	Engine         string                       `json:"engine"`
	Passed         bool                         `json:"passed"`
	Observed       RADVContiguizeParityObserved `json:"observed"`
	Bounds         RADVContiguizeParityBounds   `json:"bounds"`
	Detail         string                       `json:"detail,omitempty"`
}

// PhysicalParityCredit reports whether this event earns physical-device parity credit.
// Host contracts and unobserved events never earn physical-device parity credit.
func (e RADVContiguizeParityEvent) PhysicalParityCredit() bool {
	if e.OracleKind == "host_contract" {
		return false
	}
	if !e.DeviceObserved {
		return false
	}
	return e.Passed
}

type RADVContiguizeParityObserved struct {
	ContractHolds *bool  `json:"contract_holds,omitempty"`
	ContractName  string `json:"contract_name,omitempty"`
}

type RADVContiguizeParityBounds struct {
	Comparison       string `json:"comparison,omitempty"`
	ContractExpected *bool  `json:"contract_expected,omitempty"`
}

func emitRADVContiguizeHostContractEvent(t *testing.T, casesEvaluated int, passed bool) RADVContiguizeParityEvent {
	t.Helper()
	holds := passed
	expected := true
	detail := "host-side bit-exact emulation, geometry, GLSL source, and entropy restoration verified; non-device host contract"
	if !passed {
		detail = "host contract verification failed; non-device host contract"
	}
	ev := RADVContiguizeParityEvent{
		Schema:         "fak.strix.subkernel-parity/v1",
		Selector:       "f16_kv_contiguize",
		TestName:       "TestRADVContiguizeShader",
		Test:           "TestRADVContiguizeShader",
		OracleKind:     "host_contract",
		CaseCount:      casesEvaluated,
		DeviceObserved: false,
		Engine:         "fak-native/host",
		Passed:         passed,
		Observed: RADVContiguizeParityObserved{
			ContractHolds: &holds,
			ContractName:  "radv_contiguize_host_contract",
		},
		Bounds: RADVContiguizeParityBounds{
			Comparison:       "==",
			ContractExpected: &expected,
		},
		Detail: detail,
	}

	raw, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("failed to marshal contiguize host contract event: %v", err)
	}
	t.Log(string(raw))
	return ev
}

func TestRADVContiguizeHostContractEvent(t *testing.T) {
	// 1. Verify successful host contract event emission
	ev := emitRADVContiguizeHostContractEvent(t, 24, true)
	if ev.Schema != "fak.strix.subkernel-parity/v1" {
		t.Errorf("expected schema 'fak.strix.subkernel-parity/v1', got %q", ev.Schema)
	}
	if ev.Selector != "f16_kv_contiguize" {
		t.Errorf("expected selector 'f16_kv_contiguize', got %q", ev.Selector)
	}
	if ev.TestName != "TestRADVContiguizeShader" {
		t.Errorf("expected test_name 'TestRADVContiguizeShader', got %q", ev.TestName)
	}
	if ev.Test != "TestRADVContiguizeShader" {
		t.Errorf("expected test 'TestRADVContiguizeShader', got %q", ev.Test)
	}
	if ev.OracleKind != "host_contract" {
		t.Errorf("expected oracle_kind 'host_contract', got %q", ev.OracleKind)
	}
	if ev.DeviceObserved {
		t.Errorf("host contract MUST have device_observed=false, got true")
	}
	if ev.PhysicalParityCredit() {
		t.Errorf("host contract MUST NOT earn physical parity credit")
	}
	if ev.Engine != "fak-native/host" {
		t.Errorf("expected engine 'fak-native/host', got %q", ev.Engine)
	}
	if !ev.Passed {
		t.Errorf("expected passed=true")
	}
	if ev.CaseCount != 24 {
		t.Errorf("expected case_count=24, got %d", ev.CaseCount)
	}
	if ev.Observed.ContractHolds == nil || !*ev.Observed.ContractHolds {
		t.Errorf("expected observed.contract_holds=true")
	}
	if ev.Observed.ContractName != "radv_contiguize_host_contract" {
		t.Errorf("expected observed.contract_name 'radv_contiguize_host_contract', got %q", ev.Observed.ContractName)
	}
	if ev.Bounds.Comparison != "==" {
		t.Errorf("expected bounds.comparison '==', got %q", ev.Bounds.Comparison)
	}
	if ev.Bounds.ContractExpected == nil || !*ev.Bounds.ContractExpected {
		t.Errorf("expected bounds.contract_expected=true")
	}

	// 2. JSON serialization contract checks (no cosine conflation, explicit device_observed=false)
	raw, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	rawStr := string(raw)

	if strings.Contains(rawStr, "cosine_similarity") || strings.Contains(rawStr, `"cosine"`) {
		t.Errorf("host contract must not contain cosine (conflation risk): %s", rawStr)
	}
	if !strings.Contains(rawStr, `"device_observed":false`) {
		t.Errorf("host contract json missing explicit '\"device_observed\":false': %s", rawStr)
	}
	if !strings.Contains(rawStr, `"oracle_kind":"host_contract"`) {
		t.Errorf("host contract json missing '\"oracle_kind\":\"host_contract\"': %s", rawStr)
	}
	if !strings.Contains(rawStr, `"test_name":"TestRADVContiguizeShader"`) {
		t.Errorf("host contract json missing '\"test_name\":\"TestRADVContiguizeShader\"': %s", rawStr)
	}
	if !strings.Contains(rawStr, `"schema":"fak.strix.subkernel-parity/v1"`) {
		t.Errorf("host contract json missing '\"schema\":\"fak.strix.subkernel-parity/v1\"': %s", rawStr)
	}

	// 3. Negative failure mode: when passed=false, contract_holds is false and PhysicalParityCredit is false
	evFail := emitRADVContiguizeHostContractEvent(t, 24, false)
	if evFail.Passed {
		t.Errorf("expected passed=false for failing contract")
	}
	if evFail.Observed.ContractHolds == nil || *evFail.Observed.ContractHolds {
		t.Errorf("expected contract_holds=false for failing contract")
	}
	if evFail.PhysicalParityCredit() {
		t.Errorf("failing host contract must not earn physical parity credit")
	}
}

// Suppress unused import warnings if any
var _ = bytes.Equal
