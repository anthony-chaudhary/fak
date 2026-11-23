//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// TestCUDASituMoEMatchesHostComposition holds the fused situ MoE to the host
// composition over per-weight MatMul outputs: same GEMVs, float64 situ, float32
// scales and in-order weighted sum. Only CUDA's tanh/exp may differ, in the last
// float64 ulp, so the bound is a few float32 ulps relative.
// fak-test:runtime fast est=3s lane=default
func TestCUDASituMoEMatchesHostComposition(t *testing.T) {
	cb := cudaOrSkip(t)
	r := rand.New(rand.NewSource(23))
	const picks, in, width, hidden = 3, 128, 96, 80
	const beta, linBeta = float32(3.5), float32(6)
	mat := func(rows, cols int, s float64) Tensor {
		f := make([]float32, rows*cols)
		for i := range f {
			f[i] = float32(r.NormFloat64() * s)
		}
		return cb.Upload(NewF32(cb, []int{rows, cols}, f), F32)
	}
	x := rowViewTestX(r, in)
	gu, dn := make([]Tensor, picks), make([]Tensor, picks)
	weight, gs, ds := make([]float32, picks), make([]float32, picks), make([]float32, picks)
	for p := range gu {
		gu[p], dn[p] = mat(2*width, in, 0.2), mat(hidden, width, 0.2)
		weight[p], gs[p], ds[p] = float32(r.Float64()), float32(0.5+r.Float64()), float32(0.5+r.Float64())
	}
	got, err := cb.SituMoE(gu, dn, x, weight, gs, ds, beta, linBeta)
	if err != nil {
		t.Fatal(err)
	}
	dx := cb.Upload(NewF32(cb, []int{in}, x), F32)
	want := make([]float32, hidden)
	for p := range gu {
		g := cb.Read(cb.MatMul(gu[p], dx))
		act := make([]float32, width)
		for j := range act {
			gv, uv := float64(g[j]*gs[p]), float64(g[width+j]*gs[p])
			a := float64(beta) * math.Tanh(gv/float64(beta)) * (1 / (1 + math.Exp(-gv)))
			uv = float64(linBeta) * math.Tanh(uv/float64(linBeta))
			act[j] = float32(a * uv)
		}
		da := cb.Upload(NewF32(cb, []int{width}, act), F32)
		d := cb.Read(cb.MatMul(dn[p], da))
		for j := range want {
			want[j] += weight[p] * (d[j] * ds[p])
		}
	}
	var worst float64
	for j := range want {
		diff := math.Abs(float64(got[j] - want[j]))
		if tol := 1e-6*math.Abs(float64(want[j])) + 1e-7; diff > tol {
			t.Fatalf("out[%d] = %v, host composition %v (diff %g)", j, got[j], want[j], diff)
		}
		worst = math.Max(worst, diff)
	}
	t.Logf("situ moe max abs diff vs host composition: %g", worst)
}
