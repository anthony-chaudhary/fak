//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"math"
	"os"
	"strconv"
	"testing"
)

// FAK_V41_SHARED_ATTENTION_BENCH_ROWS=<n> keeps one shape so a process's
// cumulative timestamp profile is that shape's per-call device time.
func v41SharedAttentionBenchRows(rows int) bool {
	want, err := strconv.Atoi(os.Getenv("FAK_V41_SHARED_ATTENTION_BENCH_ROWS"))
	return err != nil || want == rows
}

func v41SharedAttentionUseClass(t *testing.T, v *vulkanBackend, class V41SharedAttentionClass) {
	t.Helper()
	if err := v.SetV41SharedAttentionClass(class); err != nil {
		t.Fatal(err)
	}
}

func v41SharedAttentionRequireFast(t *testing.T, v *vulkanBackend) {
	t.Helper()
	if !v.SupportsV41SharedAttention() {
		t.Skip("V4.1 shared attention pipeline unavailable")
	}
	if !v.V41SharedAttentionFastAvailable() {
		if os.Getenv("FAK_V41_SHARED_ATTENTION_FAST_REQUIRED") == "1" {
			t.Fatal("required run-to-run shared attention module unavailable")
		}
		t.Skip("run-to-run shared attention module unavailable (no Wave32 subgroup size control)")
	}
	prior := v.V41SharedAttentionClass()
	t.Cleanup(func() { _ = v.SetV41SharedAttentionClass(prior) })
}

// The run-to-run class repeats its own bits at every decode shape and stays
// within F32 reordering distance of the serial oracle, whose digests the
// class switch must not move. Rows 9000 cover the fast module's chunked path.
// FAK_V41_SHARED_ATTENTION_BENCH=<n> repeats each fast shape n times.
// fak-test:runtime fast est=8s lane=default
func TestVulkanV41SharedAttentionFastRunToRunParity(t *testing.T) {
	v := vk(t)
	v41SharedAttentionRequireFast(t, v)
	repeats := 2
	if n, err := strconv.Atoi(os.Getenv("FAK_V41_SHARED_ATTENTION_BENCH")); err == nil && n > repeats {
		repeats = n
	}
	const heads, dim = 64, 512
	for _, rows := range []int{1, 130, 640, 2049, 3000, 9000} {
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
			v41SharedAttentionUseClass(t, v, V41SharedAttentionRunToRun)
			fast, err := v41AttentionRun(v, q, kv, sink, rows, heads, dim, 0.044, mode, hasSink, repeats)
			if err != nil {
				t.Fatalf("fast rows=%d mode=%d: %v", rows, mode, err)
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
			for i := range fast {
				d := math.Abs(float64(fast[i] - serial[i]))
				if d > 1e-5+1e-5*math.Abs(float64(serial[i])) {
					t.Fatalf("rows=%d mode=%d out[%d] fast=%g serial=%g", rows, mode, i, fast[i], serial[i])
				}
				maxDelta = math.Max(maxDelta, d)
			}
			t.Logf("SHARED_ATTN_FAST_DIGEST rows%d_mode%d_sink%t fast=%s serial=%s max_abs_delta=%.3g repeats=%d",
				rows, mode, hasSink, decodeSplitDigest(fast), decodeSplitDigest(serial), maxDelta, repeats)
		}
	}
}

// Fault records are a closed contract: the run-to-run class reports the same
// stage, head, slot and element as the serial oracle for score and value faults.
// fak-test:runtime fast est=200ms lane=default
func TestVulkanV41SharedAttentionFastFaultParity(t *testing.T) {
	v := vk(t)
	v41SharedAttentionRequireFast(t, v)
	type faultCase struct {
		name             string
		heads, dim, rows int
		q, kv            []float32
	}
	value := faultCase{name: "value-overflow-across-lanes", heads: 2, dim: 512, rows: 10, q: make([]float32, 2*512), kv: make([]float32, 10*512)}
	for r := 0; r < value.rows; r++ {
		value.kv[r*512+5] = math.MaxFloat32
		value.kv[r*512+300] = math.MaxFloat32
	}
	score := faultCase{name: "score-nan-slot", heads: 3, dim: 512, rows: 700, q: make([]float32, 3*512), kv: make([]float32, 700*512)}
	for i := range score.q {
		score.q[i] = decodeSplitValue(i, 3) * 0.25
	}
	for i := range score.kv {
		score.kv[i] = decodeSplitValue(i, 11)
	}
	score.kv[517*512+40] = float32(math.NaN())
	score.kv[611*512+2] = float32(math.NaN())
	for _, c := range []faultCase{value, score} {
		var records [2]*V41SharedAttentionArithmeticError
		for arm, class := range []V41SharedAttentionClass{V41SharedAttentionBitExact, V41SharedAttentionRunToRun} {
			v41SharedAttentionUseClass(t, v, class)
			_, err := v41AttentionRun(v, c.q, c.kv, nil, c.rows, c.heads, c.dim, 1, V41SharedAttentionPlain, false, 1)
			if !errors.As(err, &records[arm]) {
				t.Fatalf("%s class %d: want arithmetic fault, got %v", c.name, class, err)
			}
		}
		s, f := records[0], records[1]
		if s.Stage != f.Stage || s.Head != f.Head || s.SelectedSlot != f.SelectedSlot || s.Element != f.Element {
			t.Fatalf("%s: fast fault %+v != serial %+v", c.name, *f, *s)
		}
		if s.Stage == V41SharedAttentionStageValue && s.ValueBits != f.ValueBits {
			t.Fatalf("%s: fast value bits %#x != serial %#x", c.name, f.ValueBits, s.ValueBits)
		}
	}
}
