//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"math/rand"
	"testing"
)

const ink2VMMAProbeSource = `#include <metal_stdlib>
using namespace metal;
kernel void probe_mma(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const float *c [[buffer(2)]],
                      device float *d [[buffer(3)]], uint g [[threadgroup_position_in_grid]]) {
    simdgroup_float8x8 A, B, C;
    simdgroup_load(A, a + 64 * g, 8);
    simdgroup_load(B, b + 64 * g, 8);
    simdgroup_load(C, c + 64 * g, 8);
    simdgroup_multiply_accumulate(C, A, B, C);
    simdgroup_store(C, d + 64 * g, 8);
}
kernel void probe_chain(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const float *c [[buffer(2)]],
                        device float *d [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint e [[thread_position_in_threadgroup]]) {
    device const float *A = a + 64 * g + 8 * (e / 8), *B = b + 64 * g + e % 8;
    float s = c[64 * g + e];
    for (int k = 0; k < 8; k++) s = fma(A[k], B[8 * k], s);
    d[64 * g + e] = s;
}
`

// The verify MMA kernels lean on an fp32 8x8x8 simdgroup MMA being, per
// element, the fma chain over k = 0..7 from the accumulator, bit for bit.
//
// fak-test:runtime integration est=1s lane=default
func TestInk2VMMAProbeOrder(t *testing.T) {
	d := ink2TestDevice(t)
	const groups = 4096
	rng := rand.New(rand.NewSource(7))
	r := func() Ink2Ref { return ink2VTestPut(t, d, ink2OpsTestRand(rng, groups*64, 1)) }
	a, b, c := r(), r(), r()
	mm, chain := ink2TestAlloc(t, d, groups*64), ink2TestAlloc(t, d, groups*64)
	// the MMA in the verify kernels' compile mode, the chain without fast math
	p, err := d.Pipeline(ink2VMMAProbeSource, "probe_mma")
	if err != nil {
		t.Fatal(err)
	}
	q, err := d.PipelinePrecise(ink2VMMAProbeSource, "probe_chain")
	if err != nil {
		t.Fatal(err)
	}
	args := func(out Ink2Ref) []Ink2Arg {
		return []Ink2Arg{a.Span(groups * 64), b.Span(groups * 64), c.Span(groups * 64), out.Span(groups * 64)}
	}
	if err := d.Dispatch(p, [3]int{groups, 1, 1}, [3]int{32, 1, 1}, args(mm)...); err != nil {
		t.Fatal(err)
	}
	if err := d.Dispatch(q, [3]int{groups, 1, 1}, [3]int{64, 1, 1}, args(chain)...); err != nil {
		t.Fatal(err)
	}
	m, s := ink2OpsTestRead(t, d, mm, groups*64), ink2OpsTestRead(t, d, chain, groups*64)
	for i := range m {
		if math.Float32bits(m[i]) != math.Float32bits(s[i]) {
			t.Fatalf("element %d: MMA %v (%#x), fma chain %v (%#x)", i, m[i], math.Float32bits(m[i]), s[i], math.Float32bits(s[i]))
		}
	}
}
