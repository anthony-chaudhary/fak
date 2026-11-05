package compute

import (
	"fmt"
	"math"
	"testing"
)

// cpuref_test.go — pins the Q8_0 decode group-column walk (llama.cpp AMX-inspired, INSPIRE-ONLY;
// ggml/src/ggml-cpu/amx/mmq.cpp:1573@8df6229c0fd56f735f91ddbbd4e7308fce4a96ab, MIT). The closed
// contract: for every output row o, y[o] = Σ_b float32(exactIntBlockSum(o,b)) * dw[o,b] * dx[b],
// accumulated in block order b=0..nblk-1. Any traversal that keeps that per-output sequence is
// bit-identical; TestCpuRefGroupColumnWalk proves it against an independent first-principles
// oracle on fixed seeds, across group widths and geometries. No throughput claim — unmeasured.

// q8WalkOracle recomputes the Q8_0 decode result from first principles, independently of the
// kernel's lane structure: one sequential int32 accumulator per block (exact — |sum| ≤ blk·127²
// ≪ int32 max, so no grouping can differ), then the same left-to-right float fold in block
// order. It reads only the quantized codes/scales, never qdot8scalar or q8GroupColumnWalk.
func q8WalkOracle(out, in, blk int, wc []int8, ws []float32, qx []int8, dx []float32) []float32 {
	nblk := in / blk
	want := make([]float32, out)
	for o := 0; o < out; o++ {
		var acc float32
		for b := 0; b < nblk; b++ {
			var s32 int32
			base := b * blk
			for i := 0; i < blk; i++ {
				s32 += int32(wc[o*in+base+i]) * int32(qx[base+i])
			}
			acc += float32(s32) * ws[o*nblk+b] * dx[b]
		}
		want[o] = acc
	}
	return want
}

// assertBitEqual fails on the first differing bit pattern (index + both sides' bits and values);
// NaN vs ±0 vs sign-bit drift all trip it.
func assertBitEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len(got)=%d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: y[%d] bits=%08x (%v), want bits=%08x (%v)", label, i,
				math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
		}
	}
}

// fak-test:runtime fast est=1s
func TestCpuRefGroupColumnWalk(t *testing.T) {
	const blk = 32 // Q8_0 block; unchanged by this work
	geos := []struct{ out, in int }{{96, 512}, {97, 256}, {8, 64}, {1, 32}}
	groups := []int{1, 2, 3, amxDecodeGroupRows, 16, 97}
	be := &cpuBackend{}

	for _, g := range geos {
		out, in := g.out, g.in
		nblk := in / blk
		var s lcg = 0xA11CE
		w := randVec(&s, out*in)
		x := randVec(&s, in)
		wt := QuantizeQ8(be, []int{out, in}, w, blk)
		wc := wt.buf.(*hostBuf).i8
		ws := wt.Quant.Scale
		qx, dx := quantizeVecQ8(x, blk)
		want := q8WalkOracle(out, in, blk, wc, ws, qx, dx)

		// Production decode path: MatMul's Q8_0 case runs the group-column walk at the default
		// width; its output must equal the oracle bit-for-bit on the fixed seed.
		got := be.Read(be.MatMul(wt, NewF32(be, []int{in}, x)))
		assertBitEqual(t, fmt.Sprintf("MatMul out=%d in=%d", out, in), got, want)

		// Walk invariant: the reorder is a pure loop-nest swap, so EVERY group width — including
		// widths that do not divide out (tail group) and clamps above out — lands on the same
		// bits as the row-innermost oracle.
		for _, group := range groups {
			y := make([]float32, out)
			q8GroupColumnWalk(y, out, in, blk, wc, ws, qx, dx, group)
			assertBitEqual(t, fmt.Sprintf("walk out=%d in=%d group=%d", out, in, group), y, want)
		}

		// Refactor invariant: qdot8scalar (the row-innermost kernel, also the Vulkan parity
		// oracle) still equals the oracle per row after being factored through
		// qdot8BlockPartial.
		for o := 0; o < out; o++ {
			if v := qdot8scalar(wc[o*in:o*in+in], ws[o*nblk:o*nblk+nblk], qx, dx, blk); math.Float32bits(v) != math.Float32bits(want[o]) {
				t.Fatalf("qdot8scalar out=%d in=%d row %d: bits=%08x (%v), want bits=%08x (%v)",
					out, in, o, math.Float32bits(v), v, math.Float32bits(want[o]), want[o])
			}
		}
	}
}
