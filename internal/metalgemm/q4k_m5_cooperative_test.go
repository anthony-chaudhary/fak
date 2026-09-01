//go:build darwin && arm64 && cgo

package metalgemm

import (
	"slices"
	"testing"
)

func TestQ4KM5CooperativeSMEMCandidateMatchesIndependentCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()

	const out, in, prompt = 96, 512, 64
	raw := q4kTestRaw(out, in, 0x9937)
	x := make([]float32, prompt*in)
	for i := range x {
		x[i] = float32((i*37)%251-125) / 127
	}
	w := UploadQ4K(raw, out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	control := make([]float32, prompt*out)
	controlID := w.GEMMWithEventsMode(x, prompt, control, nil, Q4KGEMMModeScalar)
	if controlID != (Q4KGEMMIdentity{Requested: Q4KGEMMExecutedScalar, Executed: Q4KGEMMExecutedScalar}) {
		t.Fatalf("scalar identity=%+v", controlID)
	}

	candidate := make([]float32, prompt*out)
	candidateID := w.GEMMWithEventsMode(x, prompt, candidate, nil, Q4KGEMMModeM5CooperativeSMEM)
	wantCandidate := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedM5CooperativeSMEM, Executed: Q4KGEMMExecutedM5CooperativeSMEM}
	if candidateID != wantCandidate {
		t.Fatalf("candidate identity=%+v want %+v", candidateID, wantCandidate)
	}

	reference := make([]float32, prompt*out)
	for p := 0; p < prompt; p++ {
		copy(reference[p*out:(p+1)*out], q4kVectorizedReference(raw, out, in, x[p*in:(p+1)*in]))
	}
	for name, got := range map[string][]float32{"control": control, "candidate": candidate} {
		cosine, maxRel := q4kTestCosineMaxRel(reference, got)
		if cosine < 0.999999 || maxRel > 5e-3 {
			t.Fatalf("%s vs independent CPU oracle: cosine=%g maxRel=%g", name, cosine, maxRel)
		}
	}
	cosine, maxRel := q4kTestCosineMaxRel(control, candidate)
	if cosine < 0.999999 || maxRel > 5e-3 {
		t.Fatalf("candidate vs scalar fallback: cosine=%g maxRel=%g", cosine, maxRel)
	}

	untouched := make([]float32, prompt*out)
	for i := range untouched {
		untouched[i] = 9937
	}
	before := slices.Clone(untouched)
	unavailable := w.GEMMWithEventsMode(x, prompt, untouched, nil, Q4KGEMMModeM5CooperativeSMEMUnavailable)
	if unavailable != (Q4KGEMMIdentity{Requested: Q4KGEMMExecutedM5CooperativeSMEM, Executed: Q4KGEMMNotExecuted}) {
		t.Fatalf("unavailable identity=%+v", unavailable)
	}
	if !slices.Equal(untouched, before) {
		t.Fatal("unavailable candidate changed output instead of failing closed")
	}
}
