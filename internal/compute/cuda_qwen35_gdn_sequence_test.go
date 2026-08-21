//go:build cuda

package compute

import (
	"math"
	"testing"
)

func TestCUDAQwen35GDNSequenceMatchesDecodeAndStaysResident(t *testing.T) {
	be := cudaGDNBackend(t)
	g := cudaGDNGeometry{hidden: 16, nK: 2, nV: 2, kHd: 4, vHd: 4, kernel: 4, eps: 1e-5}
	seqOps := newCUDAGDNOperands(t, be, g)
	refOps := seqOps
	refOps.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, make([]float32, (g.kernel-1)*g.convDim()), MemoryKVCache, "qwen35-gdn-reference-conv-state")
	refOps.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, make([]float32, g.nV*g.kHd*g.vHd), MemoryKVCache, "qwen35-gdn-reference-recurrent-state")

	// Both fixtures use the same deterministic host values; only mutable state
	// needs copying because the operation updates it in place.
	inputs := []float32{
		0.25, -0.5, 0.75, -1, 1.25, -1.5, 1.75, -2, 2.25, -2.5, 2.75, -3, 3.25, -3.5, 3.75, -4,
		-0.4, 0.2, 0.6, -0.8, 1, -1.2, 1.4, -1.6, 1.8, -2, 2.2, -2.4, 2.6, -2.8, 3, -3.2,
		0.1, 0.3, -0.5, 0.7, -0.9, 1.1, -1.3, 1.5, -1.7, 1.9, -2.1, 2.3, -2.5, 2.7, -2.9, 3.1,
	}
	panel := uploadCUDAGDN(t, be, []int{3, g.hidden}, inputs, MemoryActivation, "qwen35-gdn-sequence-input")

	be.ResetHostXfer()
	be.ResetH2DXfer()
	got, gotConv, gotRecurrent, err := be.Qwen35GDNSequence(
		panel, seqOps.inQKV, seqOps.inZ, seqOps.inB, seqOps.inA, seqOps.convW,
		seqOps.aLog, seqOps.dtBias, seqOps.norm, seqOps.outW, seqOps.convState,
		seqOps.recurrentState, g.nK, g.nV, g.kHd, g.vHd, g.kernel, 1e-5,
	)
	if err != nil {
		t.Fatalf("sequence: %v", err)
	}
	if gotConv.buf != seqOps.convState.buf || gotRecurrent.buf != seqOps.recurrentState.buf {
		t.Fatal("sequence replaced persistent state")
	}
	if be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
		t.Fatalf("resident sequence transferred bytes: d2h=%d h2d=%d", be.HostXferBytes(), be.H2DXferBytes())
	}

	gotHost := be.Read(got)
	gotConvHost := be.Read(seqOps.convState)
	gotRecurrentHost := be.Read(seqOps.recurrentState)
	want := make([]float32, 0, len(inputs))
	for token := 0; token < 3; token++ {
		x := uploadCUDAGDN(t, be, []int{g.hidden}, inputs[token*g.hidden:(token+1)*g.hidden], MemoryActivation, "qwen35-gdn-reference-input")
		out, _, _, err := be.Qwen35GDNDecode(
			x, refOps.inQKV, refOps.inZ, refOps.inB, refOps.inA, refOps.convW,
			refOps.aLog, refOps.dtBias, refOps.norm, refOps.outW, refOps.convState,
			refOps.recurrentState, g.nK, g.nV, g.kHd, g.vHd, g.kernel, 1e-5,
		)
		if err != nil {
			t.Fatalf("decode token %d: %v", token, err)
		}
		want = append(want, be.Read(out)...)
	}
	assertClose := func(name string, a, b []float32) {
		t.Helper()
		if len(a) != len(b) {
			t.Fatalf("%s len %d != %d", name, len(a), len(b))
		}
		var dot, aa, bb float64
		for i := range a {
			delta := math.Abs(float64(a[i] - b[i]))
			scale := math.Max(1, math.Max(math.Abs(float64(a[i])), math.Abs(float64(b[i]))))
			if delta > 2e-5*scale {
				t.Fatalf("%s[%d] got=%g want=%g delta=%g", name, i, a[i], b[i], delta)
			}
			dot += float64(a[i] * b[i])
			aa += float64(a[i] * a[i])
			bb += float64(b[i] * b[i])
		}
		cos := dot / math.Sqrt(aa*bb)
		if cos < 0.999999 {
			t.Fatalf("%s cosine=%g", name, cos)
		}
	}
	assertClose("output", gotHost, want)
	assertClose("conv state", gotConvHost, be.Read(refOps.convState))
	assertClose("recurrent state", gotRecurrentHost, be.Read(refOps.recurrentState))

	// Decode-after-prefill proves the persisted sequence state is usable by the
	// production one-token operation, not merely numerically plausible in place.
	nextHost := []float32{0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1, 0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8}
	nextSeq := uploadCUDAGDN(t, be, []int{g.hidden}, nextHost, MemoryActivation, "qwen35-gdn-next-input")
	nextRef := uploadCUDAGDN(t, be, []int{g.hidden}, nextHost, MemoryActivation, "qwen35-gdn-next-input")
	seqOut, _, _, err := be.Qwen35GDNDecode(nextSeq, seqOps.inQKV, seqOps.inZ, seqOps.inB, seqOps.inA, seqOps.convW, seqOps.aLog, seqOps.dtBias, seqOps.norm, seqOps.outW, seqOps.convState, seqOps.recurrentState, g.nK, g.nV, g.kHd, g.vHd, g.kernel, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	refOut, _, _, err := be.Qwen35GDNDecode(nextRef, refOps.inQKV, refOps.inZ, refOps.inB, refOps.inA, refOps.convW, refOps.aLog, refOps.dtBias, refOps.norm, refOps.outW, refOps.convState, refOps.recurrentState, g.nK, g.nV, g.kHd, g.vHd, g.kernel, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	assertClose("decode after prefill", be.Read(seqOut), be.Read(refOut))
}
