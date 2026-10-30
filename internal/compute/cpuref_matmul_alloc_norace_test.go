//go:build !race

package compute

import "testing"

// Allocation ceilings for sync.Pool reuse require a non-race build: the race
// runtime deliberately drops Pool.Put values, injecting fresh allocations.
// Numerical and ownership tests remain enabled in race builds.

// TestCPURefQ8MatMulReusesActivationScratch pins the Q8_0 activation quantization to a pooled
// buffer. The activation is wide relative to the output row, so a per-call quantization into fresh
// int8+codes slices would allocate about the activation's own bytes; reuse keeps the per-op bytes to
// the output envelope. The ceiling sits between the two, and a zero block is explicitly cleared so
// reuse stays bit-identical (covered by TestAllocCeilingCPURefMatMulReadsWeightInPlace parity).
// fak-test:runtime fast est=1s
func TestCPURefQ8MatMulReusesActivationScratch(t *testing.T) {
	const out, in, p = 64, 4096, 2
	be := &cpuBackend{}
	x := cpuRefBenchX(in)
	xb := cpuRefBenchX(p * in)
	w := QuantizeQ8(be, []int{out, in}, func() []float32 {
		f := make([]float32, out*in)
		for i := range f {
			f[i] = float32(i%9) - 4
		}
		return f
	}(), 32)

	// A fresh per-call activation alloc is ~in int8 + in/32 float32 scales (576 B here for in=512;
	// ~4.6 KB for in=4096). The ceiling is set at a quarter of that, well above the output envelope.
	scratch := int64(in) + 4*int64(in/32)
	ceiling := scratch / 4
	mm := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			be.MatMul(w, x)
		}
	})
	if per := mm.AllocedBytesPerOp(); per >= ceiling {
		t.Errorf("Q8_0 MatMul allocates %d B/op >= ceiling %d (activation scratch %d): quantizing fresh per call", per, ceiling, scratch)
	}
	bm := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			be.BatchedMatMul(w, xb, p)
		}
	})
	if per := bm.AllocedBytesPerOp(); per >= ceiling {
		t.Errorf("Q8_0 BatchedMatMul allocates %d B/op >= ceiling %d (activation scratch %d): quantizing fresh per call", per, ceiling, scratch)
	}
}
