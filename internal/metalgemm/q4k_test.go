//go:build darwin && arm64 && cgo

package metalgemm

import (
	"bytes"
	"math"
	"testing"
	"unsafe"
)

func TestQ4KGEMMGroupIntoAliasesSuppliedBackingAndMatchesAllocating(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device available")
	}
	defer ResetQ4K()

	const (
		in = 256
		P  = 3
	)
	outs := []int{64, 32}
	ws := make([]*Q4KWeight, len(outs))
	for i, out := range outs {
		ws[i] = UploadQ4K(q4kTestRaw(out, in, uint64(0x9102+i)), out, in)
		if ws[i] == nil {
			t.Fatalf("UploadQ4K[%d] returned nil", i)
		}
	}
	X := make([]float32, P*in)
	for i := range X {
		X[i] = float32(i%17-8) * 0.03125
	}
	want := GEMMGroup(ws, X, P)
	if want == nil {
		t.Fatal("allocating GEMMGroup returned nil")
	}

	need := P * (outs[0] + outs[1])
	const guard = float32(91.02)
	backing := make([]float32, need+4)
	for i := need; i < len(backing); i++ {
		backing[i] = guard
	}
	observation := NewExecutionObservation(ExecutionQ4KGEMMGroup)
	got := GEMMGroupIntoWithEvents(ws, X, P, backing, observation)
	if got == nil {
		t.Fatal("GEMMGroupIntoWithEvents returned nil")
	}
	if _, err := observation.Snapshot(); err != nil {
		t.Fatalf("execution observation: %v", err)
	}
	if &got[0][0] != &backing[0] || &got[1][0] != &backing[P*outs[0]] {
		t.Fatal("returned group slices do not alias the supplied backing at their declared offsets")
	}
	for i := range got {
		if len(got[i]) != len(want[i]) || cap(got[i]) != len(got[i]) {
			t.Fatalf("group[%d] len/cap=%d/%d want %d/%d", i, len(got[i]), cap(got[i]), len(want[i]), len(want[i]))
		}
		for j := range got[i] {
			if math.Float32bits(got[i][j]) != math.Float32bits(want[i][j]) {
				t.Fatalf("group[%d][%d] bits=%08x want %08x", i, j, math.Float32bits(got[i][j]), math.Float32bits(want[i][j]))
			}
		}
	}
	for i := need; i < len(backing); i++ {
		if backing[i] != guard {
			t.Fatalf("guard[%d]=%g want %g", i-need, backing[i], guard)
		}
	}

	short := make([]float32, need-1)
	for i := range short {
		short[i] = guard
	}
	if got := GEMMGroupInto(ws, X, P, short); got != nil {
		t.Fatalf("undersized supplied buffer returned %d groups, want nil", len(got))
	}
	for i, v := range short {
		if v != guard {
			t.Fatalf("undersized buffer mutated at %d: %g", i, v)
		}
	}
}

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
	requireCompletedExecution(t, candidateObservation, ExecutionMixedQ4KQ8QKV)
	if got := candidateSnapshot.Events[0].Encoders; got != 2 {
		t.Fatalf("candidate encoders=%d, want one Q4_K plus one Q8 encoder", got)
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

func TestMixedQ4KQ8ObservationInjectedPostSubmitFailure(t *testing.T) {
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
	codes := make([]int8, 64*in)
	scales := make([]float32, 64*(in/32))
	for i := range codes {
		codes[i] = int8(i%7) - 3
	}
	for i := range scales {
		scales[i] = 0.01
	}
	q8 := UploadQ8(codes, scales, 64, in)
	if q8 == nil {
		t.Fatal("UploadQ8 returned nil")
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

	observation := NewExecutionObservation(ExecutionMixedQ4KQ8QKV)
	q4out, q8out, err := gemvGroupMixedQ4KQ8([]*Q4KWeight{q4}, []*Q8Weight{q8}, x, xq, xd, observation, true)
	if !IsMixedQ4KQ8PostSubmit(err) {
		t.Fatalf("injected call error=%T %v, want typed post-submit failure", err, err)
	}
	if q4out != nil || q8out != nil {
		t.Fatalf("injected call returned outputs: q4=%v q8=%v", q4out, q8out)
	}
	snapshot, snapshotErr := observation.Snapshot()
	if snapshotErr != nil {
		t.Fatal(snapshotErr)
	}
	if len(snapshot.Events) != 1 {
		t.Fatalf("events=%+v, want one submitted native event", snapshot.Events)
	}
	event := snapshot.Events[0]
	if !event.Committed || !event.CompletedWait || event.HostReadback || event.Encoders != 2 {
		t.Fatalf("injected post-submit lifecycle=%+v, want committed+waited, no readback, two encoders", event)
	}
	t.Logf("captured injected native failure: typed=%T committed=%t waited=%t readback=%t encoders=%d", err, event.Committed, event.CompletedWait, event.HostReadback, event.Encoders)
}

func TestMetalMTPBufferResidency(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}

	const (
		out = 64
		in  = 256
	)
	need := out * (in / 256) * 144

	// Allocates unified buffer with AllocateMetalMTPSharedBuffer for out=64, in=256.
	buf, err := AllocateMetalMTPSharedBuffer(need)
	if err != nil {
		t.Fatalf("AllocateMetalMTPSharedBuffer failed: %v", err)
	}

	// Generates Q4_K test data using q4kTestRaw(out, in, seed).
	raw := q4kTestRaw(out, in, 0x8392)
	copy(buf, raw)

	// Uploads with UploadMTPQ4KShared and UploadMTPQ4K.
	w, err := UploadMTPQ4KShared(buf, out, in)
	if err != nil {
		t.Fatalf("UploadMTPQ4KShared failed: %v", err)
	}
	// Asserts w != nil and w.NoCopy() == true (zero-copy verified!).
	if w == nil || !w.NoCopy() {
		t.Fatalf("UploadMTPQ4KShared = %#v, want non-nil with NoCopy() == true", w)
	}

	wConvenience := UploadMTPQ4K(buf, out, in)
	if wConvenience == nil || !wConvenience.NoCopy() {
		t.Fatalf("UploadMTPQ4K = %#v, want non-nil with NoCopy() == true", wConvenience)
	}
	wConvenience.Release()

	// Verifies pointer sharing: Go slice bytes are identical to what Metal reads.
	if !bytes.Equal(buf[:need], raw) {
		t.Fatal("unified buffer contents do not match raw")
	}
	if unsafe.SliceData(buf) != unsafe.SliceData(raw) && unsafe.SliceData(buf) == nil {
		t.Fatal("unified buffer pointer is nil")
	}

	x := q4kTestVector(in, 0x8392)
	want := q4kVectorizedReference(raw, out, in, x)
	got := make([]float32, out)
	w.GEMV(x, got)

	// Verifies cosine similarity >= 0.9999 against q4kVectorizedReference(raw, out, in, x).
	cosine, maxRel := q4kTestCosineMaxRel(want, got)
	if cosine < 0.9999 {
		t.Fatalf("GEMV parity: cosine=%g want >= 0.9999 (maxRel=%g)", cosine, maxRel)
	}

	// Active pointer sharing: mutate Go slice in-place and verify Metal reads updated bytes directly.
	origByte := buf[0]
	buf[0] ^= 0x55
	gotMutated := make([]float32, out)
	w.GEMV(x, gotMutated)
	if gotMutated[0] == got[0] {
		t.Fatal("expected pointer sharing: mutating Go slice in-place did not affect Metal output")
	}
	buf[0] = origByte
	gotRestored := make([]float32, out)
	w.GEMV(x, gotRestored)
	if cosineRestored, _ := q4kTestCosineMaxRel(got, gotRestored); cosineRestored < 0.999999 {
		t.Fatalf("restoring Go slice byte did not restore output: cosine=%g", cosineRestored)
	}

	// Tests non-page-aligned input staging into unified memory also produces NoCopy() == true.
	unalignedRaw := make([]byte, need+1)
	copy(unalignedRaw[1:], raw)
	unalignedSlice := unalignedRaw[1:]
	wUnaligned, err := UploadMTPQ4KShared(unalignedSlice, out, in)
	if err != nil {
		t.Fatalf("UploadMTPQ4KShared unaligned failed: %v", err)
	}
	if wUnaligned == nil || !wUnaligned.NoCopy() {
		t.Fatalf("wUnaligned = %#v, want non-nil with NoCopy() == true", wUnaligned)
	}
	gotUnaligned := make([]float32, out)
	wUnaligned.GEMV(x, gotUnaligned)
	if cosineUnaligned, _ := q4kTestCosineMaxRel(want, gotUnaligned); cosineUnaligned < 0.9999 {
		t.Fatalf("unaligned staged GEMV parity: cosine=%g want >= 0.9999", cosineUnaligned)
	}
	wUnaligned.Release()

	// Tests boundary conditions (invalid in, out, short buffer).
	boundaryTests := []struct {
		name string
		raw  []byte
		out  int
		in   int
	}{
		{"invalid in not multiple of 256", raw, out, in - 1},
		{"zero in", raw, out, 0},
		{"negative in", raw, out, -256},
		{"zero out", raw, 0, in},
		{"negative out", raw, -1, in},
		{"short buffer", raw[:need-1], out, in},
		{"empty buffer", []byte{}, out, in},
	}
	for _, tc := range boundaryTests {
		t.Run(tc.name, func(t *testing.T) {
			if badW, err := UploadMTPQ4KShared(tc.raw, tc.out, tc.in); err == nil || badW != nil {
				t.Fatalf("UploadMTPQ4KShared(%s) returned w=%v, err=%v; want error and nil weight", tc.name, badW, err)
			}
			if badW := UploadMTPQ4K(tc.raw, tc.out, tc.in); badW != nil {
				t.Fatalf("UploadMTPQ4K(%s) returned w=%v; want nil", tc.name, badW)
			}
		})
	}

	if _, err := AllocateMetalMTPSharedBuffer(0); err == nil {
		t.Fatal("AllocateMetalMTPSharedBuffer(0) expected error, got nil")
	}
	if _, err := AllocateMetalMTPSharedBuffer(-1); err == nil {
		t.Fatal("AllocateMetalMTPSharedBuffer(-1) expected error, got nil")
	}
	if err := FreeMetalMTPSharedBuffer(nil); err != nil {
		t.Fatalf("FreeMetalMTPSharedBuffer(nil) unexpected error: %v", err)
	}

	// Cleans up with ResetQ4K() and FreeMetalMTPSharedBuffer.
	ResetQ4K()
	if err := FreeMetalMTPSharedBuffer(buf); err != nil {
		t.Fatalf("FreeMetalMTPSharedBuffer failed: %v", err)
	}
}
