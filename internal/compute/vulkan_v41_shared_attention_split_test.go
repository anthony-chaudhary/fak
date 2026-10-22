//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"math"
	"os"
	"strconv"
	"testing"
)

func v41SharedAttentionRequireSplit(t *testing.T, v *vulkanBackend) {
	t.Helper()
	v41SharedAttentionRequireFast(t, v)
	if !v.V41SharedAttentionSplitAvailable() {
		if os.Getenv("FAK_V41_SHARED_ATTENTION_SPLIT_REQUIRED") == "1" {
			t.Fatal("required split-k shared attention module unavailable")
		}
		t.Skip("split-k shared attention module unavailable")
	}
}

// The split-k class repeats its own bits at every decode shape and stays
// within F32 reordering distance of the serial oracle. Rows 9000 exceed the
// split envelope and take the run-to-run class.
// FAK_V41_SHARED_ATTENTION_BENCH=<n> repeats each split shape n times, and
// FAK_V41_SHARED_ATTENTION_BENCH_ROWS=<n> keeps one shape per process.
// fak-test:runtime fast est=8s lane=default
func TestVulkanV41SharedAttentionSplitRunToRunParity(t *testing.T) {
	v := vk(t)
	v41SharedAttentionRequireSplit(t, v)
	repeats := 2
	if n, err := strconv.Atoi(os.Getenv("FAK_V41_SHARED_ATTENTION_BENCH")); err == nil && n > repeats {
		repeats = n
	}
	const heads, dim = 64, 512
	for _, rows := range []int{32, 130, 640, 2049, 3000, 8192, 9000} {
		if !v41SharedAttentionBenchRows(rows) {
			continue
		}
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
			v41SharedAttentionUseClass(t, v, V41SharedAttentionBitExact)
			serial, err := v41AttentionRun(v, q, kv, sink, rows, heads, dim, 0.044, mode, hasSink, 1)
			if err != nil {
				t.Fatalf("serial rows=%d mode=%d: %v", rows, mode, err)
			}
			v41SharedAttentionUseClass(t, v, V41SharedAttentionSplitK)
			split, err := v41AttentionRun(v, q, kv, sink, rows, heads, dim, 0.044, mode, hasSink, repeats)
			if err != nil {
				t.Fatalf("split rows=%d mode=%d: %v", rows, mode, err)
			}
			v41SharedAttentionUseClass(t, v, V41SharedAttentionBitExact)
			again, err := v41AttentionRun(v, q, kv, sink, rows, heads, dim, 0.044, mode, hasSink, 1)
			if err != nil {
				t.Fatalf("serial again rows=%d mode=%d: %v", rows, mode, err)
			}
			if decodeSplitDigest(again) != decodeSplitDigest(serial) {
				t.Fatalf("rows=%d mode=%d: serial digest moved across a class switch", rows, mode)
			}
			var maxDelta float64
			for i := range split {
				d := math.Abs(float64(split[i] - serial[i]))
				if d > 1e-5+1e-5*math.Abs(float64(serial[i])) {
					t.Fatalf("rows=%d mode=%d out[%d] split=%g serial=%g", rows, mode, i, split[i], serial[i])
				}
				maxDelta = math.Max(maxDelta, d)
			}
			t.Logf("SHARED_ATTN_SPLIT_DIGEST rows%d_mode%d_sink%t split=%s serial=%s max_abs_delta=%.3g repeats=%d",
				rows, mode, hasSink, decodeSplitDigest(split), decodeSplitDigest(serial), maxDelta, repeats)
		}
	}
}

// A head whose split partials would be unsafe reruns the run-to-run body, so
// the split-k class reports the serial oracle's stage, head, slot and element
// (and value bits for value faults), and a large finite value that the split
// guard routes to that body still returns ordinary output.
// fak-test:runtime fast est=300ms lane=default
func TestVulkanV41SharedAttentionSplitFaultParity(t *testing.T) {
	v := vk(t)
	v41SharedAttentionRequireSplit(t, v)
	type faultCase struct {
		name             string
		heads, dim, rows int
		q, kv            []float32
		wantFault        bool
	}
	// The serial oracle overflows only on rounding: ten weights of 0.1f sum
	// past 1. Rows 10..31 pad to the split minimum with weight exactly zero.
	value := faultCase{name: "value-overflow-across-lanes", heads: 2, dim: 512, rows: 32, q: make([]float32, 2*512), kv: make([]float32, 32*512), wantFault: true}
	for h := 0; h < value.heads; h++ {
		value.q[h*512] = 1
	}
	for r := 0; r < value.rows; r++ {
		if r < 10 {
			value.kv[r*512+5] = math.MaxFloat32
			value.kv[r*512+300] = math.MaxFloat32
		} else {
			value.kv[r*512] = -1e30
		}
	}
	score := faultCase{name: "score-nan-slot", heads: 3, dim: 512, rows: 700, q: make([]float32, 3*512), kv: make([]float32, 700*512), wantFault: true}
	for i := range score.q {
		score.q[i] = decodeSplitValue(i, 3) * 0.25
	}
	for i := range score.kv {
		score.kv[i] = decodeSplitValue(i, 11)
	}
	score.kv[517*512+40] = float32(math.NaN())
	score.kv[611*512+2] = float32(math.NaN())
	large := faultCase{name: "large-finite-value-guard", heads: 65, dim: 512, rows: 96, q: make([]float32, 65*512), kv: make([]float32, 96*512)}
	for i := range large.q {
		large.q[i] = decodeSplitValue(i, 5) * 0.25
	}
	for i := range large.kv {
		large.kv[i] = decodeSplitValue(i, 17)
	}
	large.kv[50*512+7] = 1e38
	for _, c := range []faultCase{value, score, large} {
		var records [2]*V41SharedAttentionArithmeticError
		var outputs [2][]float32
		for arm, class := range []V41SharedAttentionClass{V41SharedAttentionBitExact, V41SharedAttentionSplitK} {
			v41SharedAttentionUseClass(t, v, class)
			out, err := v41AttentionRun(v, c.q, c.kv, nil, c.rows, c.heads, c.dim, 1, V41SharedAttentionPlain, false, 2)
			outputs[arm] = out
			if !c.wantFault {
				if err != nil {
					t.Fatalf("%s class %d: %v", c.name, class, err)
				}
				continue
			}
			if !errors.As(err, &records[arm]) {
				t.Fatalf("%s class %d: want arithmetic fault, got %v", c.name, class, err)
			}
		}
		if !c.wantFault {
			for i := range outputs[0] {
				s, f := float64(outputs[0][i]), float64(outputs[1][i])
				if math.Abs(f-s) > 1e-5+1e-5*math.Abs(s) {
					t.Fatalf("%s out[%d] split=%g serial=%g", c.name, i, f, s)
				}
			}
			continue
		}
		s, f := records[0], records[1]
		if s.Stage != f.Stage || s.Head != f.Head || s.SelectedSlot != f.SelectedSlot || s.Element != f.Element {
			t.Fatalf("%s: split fault %+v != serial %+v", c.name, *f, *s)
		}
		if s.Stage == V41SharedAttentionStageValue && s.ValueBits != f.ValueBits {
			t.Fatalf("%s: split value bits %#x != serial %#x", c.name, f.ValueBits, s.ValueBits)
		}
	}
}
