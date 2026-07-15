//go:build cuda

package compute

import (
	"math"
	"strings"
	"testing"
)

func TestCUDAPartialRoPEQKMatchesRotateHalfReference(t *testing.T) {
	be := cudaGDNBackend(t)
	const (
		pos = 7
		nQ  = 2
		nK  = 1
		hd  = 8
		rd  = 6
	)
	qHost := []float32{.1, .2, .3, .4, .5, .6, .7, .8, -.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8}
	kHost := []float32{.9, .8, .7, .6, .5, .4, .3, .2}
	q := uploadCUDAGDN(t, be, []int{nQ * hd}, qHost, MemoryActivation, "partial-rope-q-test")
	k := uploadCUDAGDN(t, be, []int{nK * hd}, kHost, MemoryActivation, "partial-rope-k-test")
	be.ResetHostXfer()
	be.ResetH2DXfer()
	qOut, kOut := be.PartialRoPEQK(q, k, pos, nQ, nK, hd, rd, 10000)
	t.Cleanup(func() { be.Free(qOut); be.Free(kOut) })
	if got := be.HostXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes D2H", got)
	}
	if got := be.H2DXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes H2D", got)
	}

	ref := func(in []float32, heads int) []float32 {
		out := append([]float32(nil), in...)
		half := rd / 2
		for h := 0; h < heads; h++ {
			for j := 0; j < half; j++ {
				freq := math.Pow(10000, -2*float64(j)/rd)
				cs, sn := float32(math.Cos(pos*freq)), float32(math.Sin(pos*freq))
				a, b := in[h*hd+j], in[h*hd+j+half]
				out[h*hd+j], out[h*hd+j+half] = a*cs-b*sn, b*cs+a*sn
			}
		}
		return out
	}
	assertNear := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length %d != %d", label, len(got), len(want))
		}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 2e-5 {
				t.Fatalf("%s[%d]=%g want %g", label, i, got[i], want[i])
			}
		}
	}
	assertNear("q", be.Read(qOut), ref(qHost, nQ))
	assertNear("k", be.Read(kOut), ref(kHost, nK))
	assertNear("q input", be.Read(q), qHost)
	assertNear("k input", be.Read(k), kHost)
}

func TestCUDASigmoidMulInPlace(t *testing.T) {
	be := cudaGDNBackend(t)
	xHost := []float32{2, -3, 4, -5}
	gateHost := []float32{-2, 0, 2, 8}
	x := uploadCUDAGDN(t, be, []int{len(xHost)}, xHost, MemoryActivation, "sigmoid-x-test")
	gate := uploadCUDAGDN(t, be, []int{len(gateHost)}, gateHost, MemoryActivation, "sigmoid-gate-test")
	be.ResetHostXfer()
	be.ResetH2DXfer()
	be.SigmoidMulInPlace(x, gate)
	if got := be.HostXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes D2H", got)
	}
	if got := be.H2DXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes H2D", got)
	}
	got := be.Read(x)
	for i := range got {
		want := xHost[i] / (1 + float32(math.Exp(float64(-gateHost[i]))))
		if math.Abs(float64(got[i]-want)) > 2e-6 {
			t.Fatalf("x[%d]=%g want %g", i, got[i], want)
		}
	}

	short := uploadCUDAGDN(t, be, []int{1}, []float32{1}, MemoryActivation, "sigmoid-short-test")
	defer func() {
		r := recover()
		if r == nil || !strings.Contains(r.(string), "shape mismatch") {
			t.Fatalf("shape mismatch panic = %v", r)
		}
	}()
	be.SigmoidMulInPlace(x, short)
}

func TestCUDASplitQwen35QueryGate(t *testing.T) {
	be := cudaGDNBackend(t)
	qgHost := []float32{
		1, 2, 3, 4, 11, 12, 13, 14,
		5, 6, 7, 8, 15, 16, 17, 18,
	}
	qg := uploadCUDAGDN(t, be, []int{len(qgHost)}, qgHost, MemoryActivation, "qg-split-test")
	be.ResetHostXfer()
	be.ResetH2DXfer()
	q, gate := be.SplitQwen35QueryGate(qg, 2, 4)
	t.Cleanup(func() { be.Free(q); be.Free(gate) })
	if got := be.HostXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes D2H", got)
	}
	if got := be.H2DXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes H2D", got)
	}
	assert := func(label string, got, want []float32) {
		t.Helper()
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s[%d]=%g want %g", label, i, got[i], want[i])
			}
		}
	}
	assert("query", be.Read(q), []float32{1, 2, 3, 4, 5, 6, 7, 8})
	assert("gate", be.Read(gate), []float32{11, 12, 13, 14, 15, 16, 17, 18})
}
