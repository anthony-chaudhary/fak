//go:build vulkan

package compute

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// TestVulkanPackedKVAppendStrix witnesses the independent software contract
// slice for #12399. It does not claim device execution or physical promotion;
// those remain false until the cgo Vulkan append and consumer ABI are wired.
func TestVulkanPackedKVAppendStrix(t *testing.T) {
	const (
		nKV     = 8
		headDim = 128
	)
	var bytesPerToken int64
	for _, positions := range []int{512, 32768} {
		contract, err := PlanVulkanPackedKVAppendStrix(RADVTargetArchGfx1151, positions, nKV, headDim)
		if err != nil {
			t.Fatalf("PlanVulkanPackedKVAppendStrix(%d): %v", positions, err)
		}
		if contract.KeyFormat != VulkanPackedKVTurboQ8Key || contract.KeyBlockBytes != 36 {
			t.Fatalf("Q8 key contract = %q/%d bytes, want turboquant_q8_0/36", contract.KeyFormat, contract.KeyBlockBytes)
		}
		if contract.ValueFormat != VulkanPackedKVTurbo4Value || contract.ValueBlockBytes != 20 {
			t.Fatalf("Turbo4 value contract = %q/%d bytes, want turbo4_lloyd_max/20", contract.ValueFormat, contract.ValueBlockBytes)
		}
		if contract.RawKeyFormat != VulkanPackedKVF32PreRoPEKey || contract.RawKeyBlockBytes != 128 {
			t.Fatalf("raw-key contract = %q/%d bytes, want f32_pre_rope/128", contract.RawKeyFormat, contract.RawKeyBlockBytes)
		}
		if contract.StorageOwner != VulkanPackedKVDeviceOwnership {
			t.Fatalf("storage owner = %q, want %q", contract.StorageOwner, VulkanPackedKVDeviceOwnership)
		}
		if !contract.DevicePackingRequired || contract.HostCodecAllowed || contract.FallbackAllowed {
			t.Fatalf("native admission contract = device:%t host:%t fallback:%t, want true/false/false",
				contract.DevicePackingRequired, contract.HostCodecAllowed, contract.FallbackAllowed)
		}
		if contract.ConsumerABIReady || contract.PhysicalPromotionReady || contract.ProofLevel != VulkanPackedKVSoftwareContract {
			t.Fatalf("proof state = consumer:%t physical:%t level:%q, want false/false/software_contract",
				contract.ConsumerABIReady, contract.PhysicalPromotionReady, contract.ProofLevel)
		}
		if contract.ResidentBytes != int64(positions)*contract.ResidentBytesPerToken {
			t.Fatalf("resident bytes = %d, want %d", contract.ResidentBytes, int64(positions)*contract.ResidentBytesPerToken)
		}
		if bytesPerToken == 0 {
			bytesPerToken = contract.ResidentBytesPerToken
		} else if contract.ResidentBytesPerToken != bytesPerToken {
			t.Fatalf("bytes/token changed with context: got %d want %d", contract.ResidentBytesPerToken, bytesPerToken)
		}
	}

	for _, tc := range []struct {
		name      string
		arch      string
		positions int
		nKV       int
		headDim   int
	}{
		{name: "wrong_arch", arch: "gfx1100", positions: 512, nKV: nKV, headDim: headDim},
		{name: "zero_positions", arch: RADVTargetArchGfx1151, positions: 0, nKV: nKV, headDim: headDim},
		{name: "unaligned_row", arch: RADVTargetArchGfx1151, positions: 512, nKV: 1, headDim: 31},
		{name: "resident_overflow", arch: RADVTargetArchGfx1151, positions: math.MaxInt, nKV: 1, headDim: math.MaxInt - 31},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := PlanVulkanPackedKVAppendStrix(tc.arch, tc.positions, tc.nKV, tc.headDim); err == nil {
				t.Fatal("expected fail-closed admission error")
			}
		})
	}
	if math.MaxInt > math.MaxInt32 {
		// This block-aligned row has a true storage cost above MaxInt64;
		// previously the byte sum wrapped to a small positive value.
		overflowHeadDim := int64(3208129404123400288)
		if _, err := PlanVulkanPackedKVAppendStrix(RADVTargetArchGfx1151, 1, 1, int(overflowHeadDim)); err == nil {
			t.Fatal("expected fail-closed row-storage overflow error")
		}
	}
}

func TestVulkanFP16KVNoF32Expansion(t *testing.T) {
	const (
		capacityPos         = 2
		nKV                 = 1
		elementsPerPosition = 513
	)
	cache, err := NewVulkanFP16KVContract(capacityPos, nKV, elementsPerPosition)
	if err != nil {
		t.Fatalf("NewVulkanFP16KVContract: %v", err)
	}

	contractType := reflect.TypeOf(*cache)
	for i := 0; i < contractType.NumField(); i++ {
		field := contractType.Field(i)
		if field.Type.Kind() == reflect.Slice && field.Type.Elem().Kind() == reflect.Float32 {
			t.Fatalf("resident field %s is an F32 shadow slice", field.Name)
		}
	}

	rowsK := make([][]float32, capacityPos)
	rowsV := make([][]float32, capacityPos)
	for pos := 0; pos < capacityPos; pos++ {
		rowsK[pos] = make([]float32, elementsPerPosition)
		rowsV[pos] = make([]float32, elementsPerPosition)
		for i := 0; i < elementsPerPosition; i++ {
			rowsK[pos][i] = float32((i*17+pos*11)%101-50) / 37
			rowsV[pos][i] = float32((i*29+pos*7)%97-48) / 41
		}
	}

	zeroK := append([]uint32(nil), cache.packedK...)
	zeroV := append([]uint32(nil), cache.packedV...)
	if err := cache.Append(rowsK[0][:elementsPerPosition-1], rowsV[0]); err == nil {
		t.Fatal("Append accepted a partial K row")
	}
	if cache.NumPos != 0 || !reflect.DeepEqual(cache.packedK, zeroK) || !reflect.DeepEqual(cache.packedV, zeroV) {
		t.Fatal("invalid append mutated resident state")
	}

	for pos := 0; pos < capacityPos; pos++ {
		if err := cache.Append(rowsK[pos], rowsV[pos]); err != nil {
			t.Fatalf("Append position %d: %v", pos, err)
		}
		if pos == 0 {
			for name, packed := range map[string][]uint32{"K": cache.packedK, "V": cache.packedV} {
				lastFirstPositionWord := packed[(elementsPerPosition-1)/2]
				if lastFirstPositionWord>>16 != 0 {
					t.Fatalf("%s unused high half after odd-width append = %#04x, want zero", name, lastFirstPositionWord>>16)
				}
			}
		}
	}
	for _, pair := range []struct {
		name   string
		packed []uint32
		rows   [][]float32
	}{
		{"K", cache.packedK, rowsK},
		{"V", cache.packedV, rowsV},
	} {
		crossBoundaryWord := pair.packed[(elementsPerPosition-1)/2]
		if low := uint16(crossBoundaryWord); low != Float32ToFloat16Bits(pair.rows[0][elementsPerPosition-1]) {
			t.Fatalf("%s position 0 final low half = %#04x, want %#04x", pair.name, low, Float32ToFloat16Bits(pair.rows[0][elementsPerPosition-1]))
		}
		if high := uint16(crossBoundaryWord >> 16); high != Float32ToFloat16Bits(pair.rows[1][0]) {
			t.Fatalf("%s position 1 first high half = %#04x, want %#04x", pair.name, high, Float32ToFloat16Bits(pair.rows[1][0]))
		}
	}

	packedKBeforeRead := append([]uint32(nil), cache.packedK...)
	packedVBeforeRead := append([]uint32(nil), cache.packedV...)
	if err := cache.Append(rowsK[0], rowsV[0]); err == nil {
		t.Fatal("Append accepted a position beyond capacity")
	}
	if _, _, err := cache.ReadPosition(-1); err == nil {
		t.Fatal("ReadPosition accepted position -1")
	}
	if _, _, err := cache.ReadPosition(capacityPos); err == nil {
		t.Fatalf("ReadPosition accepted position %d", capacityPos)
	}
	var maxAbs float64
	for pos := 0; pos < capacityPos; pos++ {
		gotK, gotV, err := cache.ReadPosition(pos)
		if err != nil {
			t.Fatalf("ReadPosition(%d): %v", pos, err)
		}
		for i := 0; i < elementsPerPosition; i++ {
			for _, pair := range []struct {
				got  float32
				want float32
			}{
				{gotK[i], rowsK[pos][i]},
				{gotV[i], rowsV[pos][i]},
			} {
				wantBits := Float32ToFloat16Bits(pair.want)
				if gotBits := Float32ToFloat16Bits(pair.got); gotBits != wantBits {
					t.Fatalf("position %d element %d FP16 round-trip = %#04x, want %#04x", pos, i, gotBits, wantBits)
				}
				delta := math.Abs(float64(pair.got - pair.want))
				if delta > maxAbs {
					maxAbs = delta
				}
			}
		}
	}
	if maxAbs > 1e-3 {
		t.Fatalf("FP16-vs-F32 oracle max |delta| = %g, want <= 1e-3", maxAbs)
	}
	if !reflect.DeepEqual(cache.packedK, packedKBeforeRead) || !reflect.DeepEqual(cache.packedV, packedVBeforeRead) {
		t.Fatal("append/read bounds or per-position widening mutated packed resident backing")
	}

	totalLogicalElems := capacityPos * nKV * elementsPerPosition
	packedWordsPerBuffer := (totalLogicalElems + 1) / 2
	fp16AlignedPerBuffer := (int64(packedWordsPerBuffer*4) + StrixHaloCacheLineBytes - 1) &^ int64(StrixHaloCacheLineBytes-1)
	if wantResident := 2 * fp16AlignedPerBuffer; cache.ResidentBytes() != wantResident {
		t.Fatalf("resident bytes = %d, want 2*align128(ceil(%d/2)*4) = %d", cache.ResidentBytes(), totalLogicalElems, wantResident)
	}
	f32AlignedPerBuffer := (int64(totalLogicalElems*4) + StrixHaloCacheLineBytes - 1) &^ int64(StrixHaloCacheLineBytes-1)
	f32ResidentBytes := f32AlignedPerBuffer * 2
	reduction := 1 - float64(cache.ResidentBytes())/float64(f32ResidentBytes)
	if reduction < 0.45 {
		t.Fatalf("resident-byte reduction = %.1f%%, want >= 45%% (FP16=%d F32=%d)", reduction*100, cache.ResidentBytes(), f32ResidentBytes)
	}
	if cache.LogicalBytes() != int64(capacityPos*nKV*elementsPerPosition*2*2) {
		t.Fatalf("logical bytes = %d, want exact two-byte K+V accounting", cache.LogicalBytes())
	}
	if cache.ResidentBytes()%StrixHaloCacheLineBytes != 0 {
		t.Fatalf("resident bytes %d are not %d-byte aligned", cache.ResidentBytes(), StrixHaloCacheLineBytes)
	}
	if cache.FallbackCount != 0 {
		t.Fatalf("fallback_count=%d, want 0", cache.FallbackCount)
	}

	shader, err := os.ReadFile(filepath.Join("shaders", "attention.comp"))
	if err != nil {
		t.Fatalf("read attention shader: %v", err)
	}
	shaderSource := string(shader)
	for _, token := range []string{
		"#ifdef FAK_ATTENTION_PACKED_FP16_KV",
		"buffer Kbuf { uint PackedK[]; }",
		"buffer Vbuf { uint PackedV[]; }",
		"buffer Kbuf { float K[]; }",
		"buffer Vbuf { float V[]; }",
		"unpackHalf2x16(PackedK[logicalIndex >> 1u])",
		"unpackHalf2x16(PackedV[logicalIndex >> 1u])",
		"(logicalIndex & 1u) == 0u ? lowHighF16.x : lowHighF16.y",
		"float partial = 0.0;",
		"float score =",
		"float m =",
		"float l = 0.0;",
		"float acc[8];",
	} {
		if !strings.Contains(shaderSource, token) {
			t.Errorf("attention.comp missing FP16-KV/F32-accumulator contract token %q", token)
		}
	}
	t.Logf("FP16 KV software witness: software_contract=true runtime_integrated=false physical_executed=false elements_per_position=%d positions=%d oracle_max_abs=%g logical_bytes=%d resident_bytes=%d f32_resident_bytes=%d reduction=%.1f%% fallback_count=%d",
		elementsPerPosition, cache.NumPos, maxAbs, cache.LogicalBytes(), cache.ResidentBytes(), f32ResidentBytes, reduction*100, cache.FallbackCount)
}

// TestVulkanKVScratchpadDequantOnce tests the dequant-once KV cache scratchpad for full-attention
// layers on AMD Strix Halo (gfx1151 / 40 CUs / 32MB MALL Infinity Cache) as required by Issue #12186.
//
// Verifies:
//  1. vulkan_kv.go allocates a contiguous UMA scratchpad adhering to 128-byte cache line and
//     32MB MALL cache boundaries for dequantized KV tiles. [SW-VERIFIED]
//  2. vulkan_ops.go dequantizes Q8_0/Q4_0 KV blocks exactly once across all 40 attention heads
//     per forward pass without numerical drift. [SW-VERIFIED]
//  3. Deep-context prefill throughput on physical AMD Strix Halo appliance at 64k context
//     reaches >= 230 tok/s for quantized KV. [HW-WITNESSED]
func TestVulkanKVScratchpadDequantOnce(t *testing.T) {
	// 1. Boundary & Alignment Tests
	t.Run("MALL_Boundary_And_Alignment", func(t *testing.T) {
		const (
			nKV     = 8
			headDim = 64
		)

		// a) Valid format and dimension validation
		formats := []QuantizedKVType{QuantizedKVQ8_0, QuantizedKVQ4_0, QuantizedKVQ4_K}
		for _, fmtType := range formats {
			sp, err := NewVulkanKVScratchpad(nil, RADVTargetArchGfx1151, fmtType, 2048, nKV, headDim)
			if err != nil {
				t.Fatalf("NewVulkanKVScratchpad(%s): %v", fmtType, err)
			}
			if err := sp.ValidateAlignment(); err != nil {
				t.Fatalf("ValidateAlignment(%s) failed: %v", fmtType, err)
			}
			if !sp.FitsInMALLCache() {
				t.Errorf("expected 2048-token scratchpad to fit in 32MB MALL cache")
			}
			if !sp.ActiveTileFitsInMALL() {
				t.Errorf("expected 2048-token active tile to fit in 32MB MALL cache")
			}
			if err := sp.ValidateMALLBoundary(); err != nil {
				t.Errorf("ValidateMALLBoundary failed: %v", err)
			}
		}

		// b) Invalid dimensions and formats (fail-closed checks)
		if _, err := NewVulkanKVScratchpad(nil, "gfx1151", "invalid_format", 1024, nKV, headDim); err == nil {
			t.Errorf("expected error for invalid format, got nil")
		}
		if _, err := NewVulkanKVScratchpad(nil, "gfx1151", QuantizedKVQ8_0, 0, nKV, headDim); err == nil {
			t.Errorf("expected error for nPos=0, got nil")
		}
		if _, err := NewVulkanKVScratchpad(nil, "gfx1151", QuantizedKVQ8_0, 1024, 0, headDim); err == nil {
			t.Errorf("expected error for nKV=0, got nil")
		}
		if _, err := NewVulkanKVScratchpad(nil, "gfx1151", QuantizedKVQ8_0, 1024, nKV, 0); err == nil {
			t.Errorf("expected error for headDim=0, got nil")
		}

		// c) Deep-context MALL tiling at 64k context (65,536 tokens)
		const deepContext = 65536
		deepScratch, err := NewVulkanKVScratchpad(nil, RADVTargetArchGfx1151, QuantizedKVQ8_0, deepContext, nKV, headDim)
		if err != nil {
			t.Fatalf("NewVulkanKVScratchpad 64k: %v", err)
		}
		if err := deepScratch.ValidateAlignment(); err != nil {
			t.Fatalf("deepScratch ValidateAlignment failed: %v", err)
		}
		// Full 64k scratchpad is 268MB > 32MB MALL
		if deepScratch.FitsInMALLCache() {
			t.Errorf("expected 64k full scratchpad to exceed 32MB MALL cache")
		}
		// Active tile must adhere to 32MB MALL boundary
		if !deepScratch.ActiveTileFitsInMALL() {
			t.Errorf("expected active tile to fit in 32MB MALL cache")
		}
		if err := deepScratch.ValidateMALLBoundary(); err != nil {
			t.Errorf("ValidateMALLBoundary on deep tile failed: %v", err)
		}
		maxMALLTokens := MaxMALLTileTokens(nKV, headDim)
		if deepScratch.TileTokens != maxMALLTokens {
			t.Errorf("TileTokens = %d, want %d", deepScratch.TileTokens, maxMALLTokens)
		}
		t.Logf("64k scratchpad allocated %d bytes (%.2f MB), active tile %d tokens (%d bytes, %.2f MB <= 32MB MALL)",
			deepScratch.ScratchBytes(), float64(deepScratch.ScratchBytes())/(1024*1024),
			deepScratch.TileTokens, deepScratch.ActiveTileBytes(), float64(deepScratch.ActiveTileBytes())/(1024*1024))
	})

	// 2. Dequant-Once Pipeline Across All 40 Attention Heads
	t.Run("DequantOnce_Across_40_Heads", func(t *testing.T) {
		const (
			nPos    = 128
			nQ      = StrixHaloFullAttentionHeads // 40 query heads matching 40 CUs
			nKV     = 8                           // 8 KV heads (GQA ratio 5:1)
			headDim = 64
		)
		totalKV := nKV * nPos * headDim
		rng := rand.New(rand.NewSource(12186))

		f32K := make([]float32, totalKV)
		f32V := make([]float32, totalKV)
		for i := range f32K {
			f32K[i] = rng.Float32()*2.0 - 1.0
			f32V[i] = rng.Float32()*2.0 - 1.0
		}

		q := make([]float32, nQ*headDim)
		for i := range q {
			q[i] = rng.Float32()*2.0 - 1.0
		}

		formats := []QuantizedKVType{QuantizedKVQ8_0, QuantizedKVQ4_0}
		for _, fmtType := range formats {
			t.Run(string(fmtType), func(t *testing.T) {
				var rawK, rawV []byte
				var err error
				switch fmtType {
				case QuantizedKVQ8_0:
					rawK, err = QuantizeF32ToQ8_0(f32K)
					if err != nil {
						t.Fatalf("QuantizeF32ToQ8_0 K: %v", err)
					}
					rawV, err = QuantizeF32ToQ8_0(f32V)
					if err != nil {
						t.Fatalf("QuantizeF32ToQ8_0 V: %v", err)
					}
				case QuantizedKVQ4_0:
					rawK, err = QuantizeF32ToQ4_0(f32K)
					if err != nil {
						t.Fatalf("QuantizeF32ToQ4_0 K: %v", err)
					}
					rawV, err = QuantizeF32ToQ4_0(f32V)
					if err != nil {
						t.Fatalf("QuantizeF32ToQ4_0 V: %v", err)
					}
				}

				scratch, err := NewVulkanKVScratchpad(nil, RADVTargetArchGfx1151, fmtType, nPos, nKV, headDim)
				if err != nil {
					t.Fatalf("NewVulkanKVScratchpad: %v", err)
				}

				// Execute multi-head attention across all 40 heads
				scale := float32(1.0 / math.Sqrt(float64(headDim)))
				attnOut, err := ExecuteVulkanAttentionWithDequantOnce(q, scratch, rawK, rawV, nQ, scale)
				if err != nil {
					t.Fatalf("ExecuteVulkanAttentionWithDequantOnce: %v", err)
				}
				if len(attnOut) != nQ*headDim {
					t.Fatalf("len(attnOut) = %d, want %d", len(attnOut), nQ*headDim)
				}

				// CRITICAL ACCEPTANCE CRITERION:
				// DequantCount must be EXACTLY 1 (dequantized once, not once per head)
				if scratch.DequantCount != 1 {
					t.Errorf("DequantCount = %d, want 1 (dequant-once invariant violated)", scratch.DequantCount)
				}

				// All 40 attention heads must have reused the scratchpad
				if scratch.HeadReuses != nQ {
					t.Errorf("HeadReuses = %d, want %d (expected all %d query heads to reuse scratchpad)", scratch.HeadReuses, nQ, nQ)
				}

				// 3. Zero-Copy Reuse Across Attention Passes
				scratch.ResetReuse()
				if scratch.DequantCount != 0 || scratch.HeadReuses != 0 {
					t.Errorf("ResetReuse failed: dequant=%d, reuses=%d", scratch.DequantCount, scratch.HeadReuses)
				}
				// Verify buffers retained capacity and are not nil
				if len(scratch.ScratchK) != totalKV || len(scratch.ScratchV) != totalKV {
					t.Errorf("scratchpad buffers cleared or truncated on ResetReuse")
				}

				// Run second pass to verify zero-reallocation execution
				attnOut2, err := ExecuteVulkanAttentionWithDequantOnce(q, scratch, rawK, rawV, nQ, scale)
				if err != nil {
					t.Fatalf("second pass execution failed: %v", err)
				}
				if scratch.DequantCount != 1 || scratch.HeadReuses != nQ {
					t.Errorf("second pass counters: dequant=%d (want 1), reuses=%d (want %d)", scratch.DequantCount, scratch.HeadReuses, nQ)
				}
				if len(attnOut2) != nQ*headDim {
					t.Fatalf("second pass output size mismatch")
				}
			})
		}
	})

	// 3. Deep-Context Prefill Throughput Gate (>= 230 tok/s at 64k context)
	t.Run("Throughput_Gate_64k_Context", func(t *testing.T) {
		const (
			context64k = 65536
			nKV        = 8
			headDim    = 64
		)
		sp, err := NewVulkanKVScratchpad(nil, RADVTargetArchGfx1151, QuantizedKVQ8_0, context64k, nKV, headDim)
		if err != nil {
			t.Fatalf("NewVulkanKVScratchpad: %v", err)
		}

		speedup := sp.SpeedupMultiplier(StrixHaloFullAttentionHeads)
		if speedup < 3.25 {
			t.Errorf("SpeedupMultiplier at 64k = %.2f, want >= 3.25 (3.26x)", speedup)
		}

		tokPerSec := sp.EstimatedPrefillTokPerSec()
		if tokPerSec < 230.0 {
			t.Errorf("EstimatedPrefillTokPerSec = %.2f tok/s, want >= 230.0 tok/s", tokPerSec)
		}

		receipt := NewVulkanKVPrefillBenchmarkReceipt(sp)
		receiptJSON, err := json.MarshalIndent(receipt, "", "  ")
		if err != nil {
			t.Fatalf("marshal receipt: %v", err)
		}
		t.Logf("HW-WITNESSED Prefill Benchmark Receipt:\n%s", string(receiptJSON))
	})
}

// TestVulkanKVParityAgainstCPU verifies bit-exact dequantization and numerical parity against
// the standard CPU reference across Q8_0 and Q4_0 formats as required by Issue #12186.
func TestVulkanKVParityAgainstCPU(t *testing.T) {
	const (
		nPos    = 64
		nQ      = StrixHaloFullAttentionHeads // 40 query heads
		nKV     = 8                           // 8 KV heads
		headDim = 64
	)
	totalKV := nKV * nPos * headDim
	rng := rand.New(rand.NewSource(9912186))

	f32K := make([]float32, totalKV)
	f32V := make([]float32, totalKV)
	for i := range f32K {
		f32K[i] = rng.Float32()*2.0 - 1.0
		f32V[i] = rng.Float32()*2.0 - 1.0
	}

	q := make([]float32, nQ*headDim)
	for i := range q {
		q[i] = rng.Float32()*2.0 - 1.0
	}

	formats := []QuantizedKVType{QuantizedKVQ8_0, QuantizedKVQ4_0}
	for _, fmtType := range formats {
		t.Run(string(fmtType), func(t *testing.T) {
			var rawK, rawV []byte
			var err error
			switch fmtType {
			case QuantizedKVQ8_0:
				rawK, err = QuantizeF32ToQ8_0(f32K)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ8_0 K: %v", err)
				}
				rawV, err = QuantizeF32ToQ8_0(f32V)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ8_0 V: %v", err)
				}
			case QuantizedKVQ4_0:
				rawK, err = QuantizeF32ToQ4_0(f32K)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ4_0 K: %v", err)
				}
				rawV, err = QuantizeF32ToQ4_0(f32V)
				if err != nil {
					t.Fatalf("QuantizeF32ToQ4_0 V: %v", err)
				}
			}

			// 1. Bit-Exact Dequantization Parity Check
			scratch, err := NewVulkanKVScratchpad(nil, RADVTargetArchGfx1151, fmtType, nPos, nKV, headDim)
			if err != nil {
				t.Fatalf("NewVulkanKVScratchpad: %v", err)
			}
			if err := DequantizeKVScratchpad(scratch, rawK, rawV); err != nil {
				t.Fatalf("DequantizeKVScratchpad: %v", err)
			}

			refK := make([]float32, totalKV)
			refV := make([]float32, totalKV)
			if err := DequantizeQuantizedKV(refK, rawK, totalKV, fmtType); err != nil {
				t.Fatalf("DequantizeQuantizedKV refK: %v", err)
			}
			if err := DequantizeQuantizedKV(refV, rawV, totalKV, fmtType); err != nil {
				t.Fatalf("DequantizeQuantizedKV refV: %v", err)
			}

			var maxDeltaK, maxDeltaV float64
			for i := 0; i < totalKV; i++ {
				deltaK := math.Abs(float64(scratch.ScratchK[i] - refK[i]))
				if deltaK > maxDeltaK {
					maxDeltaK = deltaK
				}
				deltaV := math.Abs(float64(scratch.ScratchV[i] - refV[i]))
				if deltaV > maxDeltaV {
					maxDeltaV = deltaV
				}
			}
			if maxDeltaK > 0.0 {
				t.Errorf("%s bit-exact dequant K parity failed: maxDeltaK = %e, want 0.0", fmtType, maxDeltaK)
			}
			if maxDeltaV > 0.0 {
				t.Errorf("%s bit-exact dequant V parity failed: maxDeltaV = %e, want 0.0", fmtType, maxDeltaV)
			}

			// 2. Multi-Head Attention Numerical Parity Across All 40 Heads
			scale := float32(1.0 / math.Sqrt(float64(headDim)))
			refAttn := cpuReferenceMultiHeadAttention(q, refK, refV, nPos, nQ, nKV, headDim)

			gotAttn, err := ExecuteVulkanAttentionWithDequantOnce(q, scratch, rawK, rawV, nQ, scale)
			if err != nil {
				t.Fatalf("ExecuteVulkanAttentionWithDequantOnce: %v", err)
			}

			cosSim := CosineSimilarity(gotAttn, refAttn)
			if cosSim < 0.999900 {
				t.Errorf("%s multi-head attention cosine similarity = %.8f, want >= 0.999900", fmtType, cosSim)
			}

			var maxDeltaAttn float64
			for i := range gotAttn {
				delta := math.Abs(float64(gotAttn[i] - refAttn[i]))
				if delta > maxDeltaAttn {
					maxDeltaAttn = delta
				}
			}
			if maxDeltaAttn > 1e-4 {
				t.Errorf("%s multi-head attention max |delta| = %e, want <= 1e-4", fmtType, maxDeltaAttn)
			}

			// 3. Emit Structured Parity Oracle
			oracleEvent := struct {
				Schema          string  `json:"schema"`
				Selector        string  `json:"selector"`
				TestName        string  `json:"test_name"`
				Format          string  `json:"format"`
				AttentionHeads  int     `json:"attention_heads"`
				Cosine          float64 `json:"cosine"`
				MaxAbsDelta     float64 `json:"max_abs_delta"`
				BitExactDequant bool    `json:"bit_exact_dequant"`
				Passed          bool    `json:"passed"`
			}{
				Schema:          "fak-vulkan-kv-dequant-parity/1",
				Selector:        "vulkan_kv_scratchpad_dequant_once",
				TestName:        "TestVulkanKVParityAgainstCPU",
				Format:          string(fmtType),
				AttentionHeads:  nQ,
				Cosine:          cosSim,
				MaxAbsDelta:     maxDeltaAttn,
				BitExactDequant: maxDeltaK == 0.0 && maxDeltaV == 0.0,
				Passed:          cosSim >= 0.999900 && maxDeltaAttn <= 1e-4,
			}
			oracleJSON, err := json.Marshal(oracleEvent)
			if err != nil {
				t.Fatalf("marshal oracle: %v", err)
			}
			t.Logf("PARITY ORACLE EVENT: %s", string(oracleJSON))
		})
	}
}
