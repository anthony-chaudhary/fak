package compute

import (
	"os"
	"strconv"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41RoutedFusedMetadataDomain(t *testing.T) {
	t.Parallel()
	supported := func(d Dtype) bool { return d == Q2_K }
	weights := func(in int) []Tensor {
		return []Tensor{{Dtype: Q2_K, Shape: []int{1, in}}}
	}
	if _, _, err := validateV41RoutedFusedWeights("gate", weights(256), 1, supported); err != nil {
		t.Fatal(err)
	}
	if strconv.IntSize == 64 {
		// Both widths narrow to a valid native width of 256. Refuse the
		// original metadata before the native implementation can observe it.
		for _, width := range []uint64{1<<32 + 256, 1<<63 - 1<<32 + 256} {
			if _, _, err := validateV41RoutedFusedWeights("gate", weights(int(width)), 1, supported); err == nil {
				t.Errorf("accepted C.int-narrowing width %d", width)
			}
		}
	}
	// A shared row is valid; eight rows of this width exceed INT32_MAX.
	width := (maxInt32/8 + 256) &^ 255
	x := Tensor{Dtype: F32, Shape: []int{width}}
	if stride, err := validateV41RoutedFusedActivation(x, 8, width); err != nil || stride != 0 {
		t.Fatalf("shared activation: stride=%d err=%v", stride, err)
	}
	if strconv.IntSize == 64 {
		x.Shape[0] = int(uint64(width) * 8)
		if _, err := validateV41RoutedFusedActivation(x, 8, width); err == nil {
			t.Error("accepted per-pick activation beyond native count domain")
		}
	}
	for _, bad := range []struct{ picks, in int }{{0, 256}, {9, 256}, {1, 0}, {1, -256}} {
		x.Shape[0] = bad.in
		if _, err := validateV41RoutedFusedActivation(x, bad.picks, bad.in); err == nil {
			t.Errorf("accepted activation geometry picks=%d in=%d", bad.picks, bad.in)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41RoutedFusedGeometry(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		dtype  Dtype
		stride int
		want   v41RoutedFusedGeometry
	}{
		{"gate_shared", Q2_K, 0, v41RoutedFusedGeometry{21504, 2048, 24, 1024, 32}},
		{"gate_per_pick", Q2_K, 512, v41RoutedFusedGeometry{21504, 4096, 24, 1024, 32}},
		{"down", Q3_K, 512, v41RoutedFusedGeometry{28160, 4096, 24, 512, 32}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := checkedV41RoutedFusedGeometry(128, 512, 2, tc.stride, tc.dtype)
			if err != nil || got != tc.want {
				t.Fatalf("geometry=%+v err=%v, want %+v", got, err, tc.want)
			}
		})
	}
	for _, odd := range []struct{ rows, resident int }{{1, 112}, {3, 332}} {
		g, err := checkedV41RoutedFusedGeometry(odd.rows, 256, 1, 256, Q3_K)
		if err != nil || g.weightBytes != odd.resident {
			t.Fatalf("Q3_K rows=%d: resident bytes=%d err=%v, want %d", odd.rows, g.weightBytes, err, odd.resident)
		}
		for _, available := range []uint64{uint64(odd.resident - 2), uint64(odd.resident - 1), uint64(odd.resident), uint64(odd.resident + 128)} {
			if got, want := v41RoutedFusedExtentOK(available, g.weightBytes), available >= uint64(odd.resident); got != want {
				t.Errorf("Q3_K rows=%d available=%d admitted=%v, want %v", odd.rows, available, got, want)
			}
		}
	}
	for _, tc := range []struct {
		name                     string
		rows, in, picks, xStride int
		dtype                    Dtype
	}{
		{"no_picks", 1, 256, 0, 0, Q2_K},
		{"too_many_picks", 1, 256, 9, 0, Q2_K},
		{"output_count_overflow", maxInt32/8 + 1, 256, 8, 0, Q2_K},
		{"unaligned_input", 1, 257, 1, 0, Q2_K},
		{"invalid_stride", 1, 256, 2, 128, Q2_K},
		{"gate_input_count_overflow", 1, 1 << 28, 8, 1 << 28, Q2_K},
		{"down_input_count_overflow", 1, 1 << 28, 8, 1 << 28, Q3_K},
		{"down_shared_input", 1, 256, 2, 0, Q3_K},
		{"wrong_dtype", 1, 256, 1, 0, F32},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := checkedV41RoutedFusedGeometry(tc.rows, tc.in, tc.picks, tc.xStride, tc.dtype)
			if err == nil || got != (v41RoutedFusedGeometry{}) {
				t.Fatalf("invalid geometry returned %+v, %v", got, err)
			}
		})
	}
	limit := uint64(1<<32 - 1)
	if hostLimit := uint64(^uint(0) >> 1); limit > hostLimit {
		limit = hostLimit
	}
	lastAligned := int(limit &^ 3)
	if n, ok := checkedV41RoutedFusedWeightBytes(1, lastAligned); !ok || n != lastAligned {
		t.Errorf("last representable aligned extent returned %d, %v", n, ok)
	}
	if n, ok := checkedV41RoutedFusedWeightBytes(1, lastAligned+1); ok || n != 0 {
		t.Errorf("accepted round-up beyond native/host byte domain: %d, %v", n, ok)
	}
	for _, quant := range []struct {
		dtype Dtype
		bytes int
	}{{Q2_K, 84}, {Q3_K, 110}} {
		rows := int(limit / uint64(quant.bytes))
		if _, err := checkedV41RoutedFusedGeometry(rows, 256, 1, 256, quant.dtype); err != nil {
			t.Errorf("last representable %v extent refused: %v", quant.dtype, err)
		}
		if _, err := checkedV41RoutedFusedGeometry(rows+1, 256, 1, 256, quant.dtype); err == nil {
			t.Errorf("accepted oversized %v quantized extent", quant.dtype)
		}
	}
	maxHostInt := int(^uint(0) >> 1)
	if n, ok := checkedV41RoutedFusedBytes(maxHostInt, 4, ^uint64(0)); ok || n != 0 {
		t.Fatalf("host byte product wrapped: %d, %v", n, ok)
	}
}

// This is a source-only ordering witness, not C ABI or device execution.
// fak-test:runtime fast est=2ms lane=default
func TestV41RoutedFusedPreallocationSourceContract(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile("vulkan_v41_routed_fused.go")
	if err != nil {
		t.Fatal(err)
	}
	source := strings.ReplaceAll(string(raw), "\r\n", "\n")
	section := func(start, end string) string {
		t.Helper()
		i := strings.Index(source, start)
		if i < 0 {
			t.Fatalf("missing source section %q", start)
		}
		body := source[i:]
		if end != "" {
			j := strings.Index(body, end)
			if j < 0 {
				t.Fatalf("missing section end %q", end)
			}
			body = body[:j]
		}
		return body
	}
	ordered := func(body string, terms ...string) {
		t.Helper()
		pos := 0
		for _, term := range terms {
			i := strings.Index(body[pos:], term)
			if i < 0 {
				t.Fatalf("missing or misordered %q", term)
			}
			pos += i + len(term)
		}
	}
	gate := section("func (v *vulkanBackend) V41RoutedGateUpSwiGLUDeferred(", "func (v *vulkanBackend) V41RoutedDownSumDeferred(")
	down := section("func (v *vulkanBackend) V41RoutedDownSumDeferred(", "")
	ordered(gate, "validateV41RoutedFusedWeights(\"gate\"", "validateV41RoutedFusedWeights(\"up\"", "validateV41RoutedIDsTensor(",
		"validateV41RoutedFusedActivation(", "!finite32(limit)", "checkedV41RoutedFusedGeometry(rows, in, picks, xStride, Q2_K)",
		"if invalid != nil", "vulkanMu.Lock()", "defer vulkanMu.Unlock()", "C.fvk_have_v41_routed_fused()", "v.fusedVulkanSlotPtrs(gates)",
		"v.fusedVulkanSlotPtrs(ups)", "v.ownedVulkanBuf(x)", "v.ownedVulkanBuf(ids)", "if !xOK || !iOK", "C.fvk_submission_status()",
		"C.fvk_batch_foreign_active()", "!fusedVulkanSlotBytesOKLocked(gatePtrs, geometry.weightBytes)",
		"!fusedVulkanSlotBytesOKLocked(upPtrs, geometry.weightBytes)", "!fusedVulkanBufferBytesOKLocked(xb.ptr, geometry.inputBytes)",
		"!fusedVulkanBufferBytesOKLocked(ib.ptr, geometry.idsBytes)", "v.dallocForClass(geometry.outputBytes,",
		"v.dallocForClass(geometry.statusBytes,", "C.fvk_v41_routed_gate_up_swiglu_id_f32(", "C.int(in)", "C.int(xStride)")
	ordered(down, "validateV41RoutedFusedWeights(\"down\"", "validateV41RoutedIDsTensor(", "!V41BF16PublishSite(site)", "mid.Dtype != F32",
		"checkedV41RoutedFusedGeometry(rows, in, picks, in, Q3_K)", "if invalid != nil", "vulkanMu.Lock()", "defer vulkanMu.Unlock()",
		"C.fvk_v41_attention_bf16_state()", "C.fvk_have_v41_routed_fused()", "v.fusedVulkanSlotPtrs(downs)",
		"v.ownedVulkanBuf(mid)", "v.ownedVulkanBuf(ids)", "if !mOK || !iOK", "C.fvk_v41_attention_bf16_count_supported(C.int(rows))",
		"!fusedVulkanSlotBytesOKLocked(downPtrs, geometry.weightBytes)", "!fusedVulkanBufferBytesOKLocked(mb.ptr, geometry.inputBytes)",
		"!fusedVulkanBufferBytesOKLocked(ib.ptr, geometry.idsBytes)", "v.dallocForClass(geometry.outputBytes,",
		"v.dallocForClass(geometry.statusBytes,", "C.fvk_v41_routed_down_sum_f32(", "C.int(in)")
	helper := section("func fusedVulkanBufferBytesOKLocked(", "func fusedVulkanSlotBytesOKLocked(")
	ordered(helper, "ptr != nil && required > 0", "C.fvk_buffer_reservation(ptr, &info) != 0", "v41RoutedFusedExtentOK(uint64(info.buffer_bytes), required)")
	slots := section("func fusedVulkanSlotBytesOKLocked(", "func (v *vulkanBackend) V41RoutedGateUpSwiGLUDeferred(")
	ordered(slots, "for _, ptr := range ptrs", "!fusedVulkanBufferBytesOKLocked(ptr, required)", "return false")
	for _, body := range []string{gate, down, helper, slots} {
		for _, forbidden := range []string{"fvk_d2h", "fvk_h2d", "fvk_batch_flush", "v.Read(", "v.Host("} {
			if strings.Contains(body, forbidden) {
				t.Errorf("fused admission added host replay, transfer or flush: %q", forbidden)
			}
		}
	}
}
