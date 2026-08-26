//go:build darwin && arm64 && cgo

package metalgemm

import "testing"

func TestMixedQ4KQ8Observation(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device available")
	}
	defer ResetQ4K()
	defer ResetQ8()

	const in = 256
	q4 := UploadQ4K(q4kTestRaw(64, in, 0x8973), 64, in)
	if q4 == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer q4.Release()
	q8ws := make([]*Q8Weight, 2)
	for i, out := range []int{64, 32} {
		codes := make([]int8, out*in)
		scales := make([]float32, out*(in/32))
		for j := range codes {
			codes[j] = int8(j%7) - 3
		}
		for j := range scales {
			scales[j] = 0.01
		}
		q8ws[i] = UploadQ8(codes, scales, out, in)
		if q8ws[i] == nil {
			t.Fatalf("UploadQ8[%d] returned nil", i)
		}
	}
	x := make([]float32, in)
	xq := make([]int8, in)
	xd := make([]float32, in/32)
	for i := range x {
		x[i] = float32(i%11-5) * 0.02
		xq[i] = int8(i%9) - 4
	}
	for i := range xd {
		xd[i] = 0.02
	}

	q4ControlObservation := NewExecutionObservation(ExecutionQ4KGEMV)
	q4Control := make([]float32, q4.Out)
	q4.GEMVWithEvents(x, q4Control, q4ControlObservation)
	q8ControlObservation := NewExecutionObservation(ExecutionQ8GEMVGroup)
	q8Control := GEMVGroupQ8WithEvents(q8ws, xq, xd, q8ControlObservation)

	candidateObservation := NewExecutionObservation(ExecutionMixedQ4KQ8QKV)
	q4Candidate, q8Candidate, err := GEMVGroupMixedQ4KQ8([]*Q4KWeight{q4}, q8ws, x, xq, xd, candidateObservation)
	if err != nil {
		t.Fatal(err)
	}
	for name, observation := range map[string]*ExecutionObservation{
		"q4 control": q4ControlObservation, "q8 control": q8ControlObservation,
		"candidate": candidateObservation,
	} {
		snapshot, err := observation.Snapshot()
		if err != nil || len(snapshot.Events) != 1 {
			t.Fatalf("%s events=%+v err=%v", name, snapshot.Events, err)
		}
		if !snapshot.Events[0].Committed || !snapshot.Events[0].CompletedWait || !snapshot.Events[0].HostReadback {
			t.Fatalf("%s lifecycle=%+v", name, snapshot.Events[0])
		}
	}
	candidateSnapshot, _ := candidateObservation.Snapshot()
	if candidateSnapshot.Events[0].Operation != ExecutionMixedQ4KQ8QKV {
		t.Fatalf("candidate operation=%q", candidateSnapshot.Events[0].Operation)
	}
	t.Logf("captured lifecycle: control_events=2 candidate_events=%d candidate=%+v", len(candidateSnapshot.Events), candidateSnapshot.Events[0])
	cosine, maxRel := q4kTestCosineMaxRel(q4Control, q4Candidate[0])
	if cosine < 0.9999 || maxRel > 0.02 {
		t.Fatalf("Q4_K parity cosine=%g maxRel=%g", cosine, maxRel)
	}
	t.Logf("captured Q4_K parity: outputs=%d cosine=%.6f max_rel=%.6f", len(q4Control), cosine, maxRel)
	for w := range q8Control {
		for i := range q8Control[w] {
			if q8Candidate[w][i] != q8Control[w][i] {
				t.Fatalf("q8[%d][%d]=%g want %g", w, i, q8Candidate[w][i], q8Control[w][i])
			}
		}
	}
	t.Logf("captured Q8 parity: groups=%d exact=true", len(q8Control))
}

func TestMixedQ4KQ8ObservationTypedFailure(t *testing.T) {
	if err := mixedQ4KQ8StatusError(1); err != nil {
		t.Fatalf("successful status error = %v", err)
	}
	if err := mixedQ4KQ8StatusError(0); IsMixedQ4KQ8PostSubmit(err) {
		t.Fatalf("preflight error classified post-submit: %v", err)
	}
	if err := mixedQ4KQ8StatusError(-1); !IsMixedQ4KQ8PostSubmit(err) {
		t.Fatalf("post-submit error = %T %v", err, err)
	}
}
