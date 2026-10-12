//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
)

// Decode-shaped shared attention (64 heads x 512) logs output digests that are
// compared across shader builds: a relayout that only moves where each serial
// sum runs must leave every bit equal. Rows above 2048 cover the chunked path.
// FAK_V41_SHARED_ATTENTION_BENCH=<n> repeats each shape n times for timestamps.
// fak-test:runtime fast est=6s lane=default
func TestVulkanV41SharedAttentionDecodeShapeDigest(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41SharedAttention() {
		t.Skip("V4.1 shared attention pipeline unavailable")
	}
	repeats := 1
	if n, err := strconv.Atoi(os.Getenv("FAK_V41_SHARED_ATTENTION_BENCH")); err == nil && n > 1 {
		repeats = n
	}
	const heads, dim = 64, 512
	for _, rows := range []int{1, 130, 640, 2049, 3000} {
		for _, mode := range []V41SharedAttentionMode{V41SharedAttentionPlain, V41SharedAttentionCompressed} {
			hasSink := rows%2 == 0
			q := make([]float32, heads*dim)
			for i := range q {
				q[i] = decodeSplitValue(i, rows) * 0.25
			}
			kv := make([]float32, rows*dim)
			for i := range kv {
				kv[i] = decodeSplitValue(i, rows+7)
			}
			sink := make([]float32, heads)
			for i := range sink {
				sink[i] = decodeSplitValue(i, rows+13)
			}
			got, err := v41AttentionRun(v, q, kv, sink, rows, heads, dim, 0.044, mode, hasSink, repeats)
			if err != nil {
				t.Fatalf("rows=%d mode=%d: %v", rows, mode, err)
			}
			var oracleSink []float32
			if hasSink {
				oracleSink = sink
			}
			want := v41SharedAttentionFloat64Oracle(q, kv, oracleSink, heads, dim, 0.044)
			for i := range got {
				if d := math.Abs(float64(got[i] - want[i])); d > 1e-4+1e-4*math.Abs(float64(want[i])) {
					t.Fatalf("rows=%d mode=%d out[%d]=%g want %g", rows, mode, i, got[i], want[i])
				}
			}
			t.Logf("SHARED_ATTN_DIGEST rows%d_mode%d_sink%t %s", rows, mode, hasSink, decodeSplitDigest(got))
		}
	}
}

// Ten equal rows of MaxFloat32 weighted 1/10 each round up past MaxFloat32 at
// the tenth slot. The first fault in slot-major order is element 5, even though
// a higher lane-owned element overflows at the same slot.
// fak-test:runtime fast est=50ms lane=default
func TestVulkanV41SharedAttentionFirstValueFaultAcrossLanes(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41SharedAttention() {
		t.Skip("V4.1 shared attention pipeline unavailable")
	}
	const heads, dim, rows = 2, 512, 10
	kv := make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		kv[r*dim+5] = math.MaxFloat32
		kv[r*dim+300] = math.MaxFloat32
	}
	_, err := v41AttentionRun(v, make([]float32, heads*dim), kv, nil, rows, heads, dim, 1, V41SharedAttentionPlain, false, 1)
	var arithmetic *V41SharedAttentionArithmeticError
	if !errors.As(err, &arithmetic) || arithmetic.Stage != V41SharedAttentionStageValue || arithmetic.Head != 0 ||
		arithmetic.SelectedSlot != 9 || arithmetic.Element != 5 || arithmetic.ValueBits != math.Float32bits(float32(math.Inf(1))) {
		t.Fatalf("first value fault = %+v (err %v)", arithmetic, err)
	}
}

func v41AttentionRun(v *vulkanBackend, q, kv, sink []float32, rows, heads, dim int, scale float32, mode V41SharedAttentionMode, hasSink bool, repeats int) ([]float32, error) {
	upload := func(shape []int, values []float32) Tensor {
		return v.UploadClass(NewF32(Default(), shape, values), F32, MemoryActivation, "shared attention digest input")
	}
	if !hasSink {
		sink = []float32{0}
	}
	qt, kt, st := upload([]int{heads, dim}, q), upload([]int{rows, dim}, kv), upload([]int{len(sink)}, sink)
	defer v.Free(qt)
	defer v.Free(kt)
	defer v.Free(st)
	var got []float32
	for i := 0; i < repeats; i++ {
		v.BeginBatch()
		out, err := v.V41SharedAttention(qt, kt, st, rows, heads, dim, scale, mode, hasSink)
		if err != nil {
			v.FlushBatch()
			return nil, err
		}
		values := v.Read(out)
		v.Free(out)
		v.FlushBatch()
		if got != nil && decodeSplitDigest(got) != decodeSplitDigest(values) {
			return nil, fmt.Errorf("repeat %d changed output bits", i)
		}
		got = values
	}
	return got, nil
}
