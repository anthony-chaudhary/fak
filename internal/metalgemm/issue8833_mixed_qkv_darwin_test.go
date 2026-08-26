//go:build darwin && arm64 && cgo

package metalgemm

import (
	"os"
	"syscall"
	"testing"
)

const (
	issue8833Hidden = 4096
	issue8833QOut   = 8192
	issue8833KOut   = 1024
	issue8833VOut   = 1024
)

type issue8833WeightID int

func (id issue8833WeightID) ID() int { return int(id) }

func issue8833Q8Weight(t *testing.T, out int, seed int) *Q8Weight {
	t.Helper()
	codes := make([]int8, out*issue8833Hidden)
	for i := range codes {
		codes[i] = int8((i+seed)%15) - 7
	}
	scales := make([]float32, out*(issue8833Hidden/32))
	for i := range scales {
		scales[i] = 0.005 + float32((i+seed)%7)*0.001
	}
	w := UploadQ8(codes, scales, out, issue8833Hidden)
	if w == nil {
		t.Fatalf("UploadQ8(%d,%d) returned nil", out, issue8833Hidden)
	}
	return w
}

func issue8833MappedQ4KWeight(t *testing.T) (*Q4KWeight, []byte) {
	t.Helper()
	const offset = 32
	raw := q4kTestRaw(issue8833VOut, issue8833Hidden, 0x9246)
	page := os.Getpagesize()
	spanBytes := offset + len(raw)
	spanBytes = ((spanBytes + page - 1) / page) * page
	span, err := syscall.Mmap(-1, 0, spanBytes, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	copy(span[offset:], raw)
	if err := syscall.Mprotect(span, syscall.PROT_READ); err != nil {
		_ = syscall.Munmap(span)
		t.Fatal(err)
	}
	w := UploadQ4KMappedSpan(span, offset, issue8833VOut, issue8833Hidden)
	if w == nil || !w.NoCopy() {
		_ = syscall.Munmap(span)
		t.Fatalf("mapped Q4_K upload = %#v, want no-copy weight", w)
	}
	return w, span
}

func issue8833Input(q, k, v MixedQKVWeight, observer ExecutionObserver) MixedQKVInput {
	xq := make([]int8, issue8833Hidden)
	xd := make([]float32, issue8833Hidden/32)
	xf := q4kTestVector(issue8833Hidden, 9246)
	for i := range xq {
		xq[i] = int8(i%13) - 6
	}
	for i := range xd {
		xd[i] = 0.0075 + float32(i%5)*0.001
	}
	return MixedQKVInput{Q: q, K: k, V: v, XQ: xq, XD: xd, XF: xf,
		Hidden: issue8833Hidden, Observer: observer}
}

func issue8833RequireLifecycle(t *testing.T, selector MixedQKVSelector, result MixedQKVResult, observed []ScopedExecutionEvent) {
	t.Helper()
	wantEvents := 1
	wantEncoders := []int{3}
	if selector == MixedQKVControl {
		wantEvents = 2
		wantEncoders = []int{2, 1}
	}
	if !result.Submitted || len(result.Observation.Events) != wantEvents || len(observed) != wantEvents {
		t.Fatalf("selector=%d submitted=%t events=%+v observed=%+v", selector, result.Submitted,
			result.Observation.Events, observed)
	}
	for i, event := range result.Observation.Events {
		if !event.Committed || !event.CompletedWait || event.Encoders != wantEncoders[i] {
			t.Fatalf("selector=%d event[%d]=%+v", selector, i, event)
		}
		wantReadback := i == wantEvents-1
		if event.HostReadback != wantReadback {
			t.Fatalf("selector=%d event[%d] readback=%t want %t", selector, i, event.HostReadback, wantReadback)
		}
		if observed[i].CallID != result.CallID || observed[i].Event != event {
			t.Fatalf("selector=%d observed[%d]=%+v result=%+v", selector, i, observed[i], event)
		}
	}
}

func issue8833RequireQ4KParity(t *testing.T, name string, wantV []float32, got MixedQKVResult) {
	t.Helper()
	cosine, maxRel := q4kTestCosineMaxRel(wantV, got.V)
	if cosine < 0.9999 || maxRel > 0.02 {
		t.Fatalf("%s mapped-span Q4_K parity cosine=%g maxRel=%g", name, cosine, maxRel)
	}
}

func TestIssue8833MixedQKVNativeBridge(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ8()

	q := issue8833Q8Weight(t, issue8833QOut, 1)
	k := issue8833Q8Weight(t, issue8833KOut, 2)
	v, span := issue8833MappedQ4KWeight(t)
	defer func() {
		ResetQ4K()
		if err := syscall.Munmap(span); err != nil {
			t.Errorf("unmap mapped Q4_K span: %v", err)
		}
	}()

	base := issue8833Input(q, k, v, nil)
	wantV := make([]float32, issue8833VOut)
	v.GEMV(base.XF, wantV)

	for _, selector := range []MixedQKVSelector{MixedQKVControl, MixedQKVCandidate} {
		var observed []ScopedExecutionEvent
		in := base
		in.Observer = ExecutionObserverFunc(func(event ScopedExecutionEvent) {
			observed = append(observed, event)
		})
		got, err := ExecuteMixedQKV(selector, in)
		if err != nil {
			t.Fatalf("selector=%d: %v", selector, err)
		}
		issue8833RequireLifecycle(t, selector, got, observed)
		issue8833RequireQ4KParity(t, issue8833SelectorName(selector), wantV, got)
	}

	injected := base
	injected.injectSetup = true
	got, err := ExecuteMixedQKV(MixedQKVCandidate, injected)
	if !IsMixedQKVDecline(err) || got.Submitted || len(got.Observation.Events) != 0 {
		t.Fatalf("setup decline: result=%+v err=%v", got, err)
	}

	// Keep a later table entry live so v's released slot is an interior tombstone. The bridge must
	// reject the stale native ID before creating a Q4_K encoder or submitting the caller's buffer.
	staleID := issue8833WeightID(v.ID())
	if sentinel := UploadQ4K(q4kTestRaw(1, 256, 0x9246), 1, 256); sentinel == nil {
		t.Fatal("sentinel Q4_K upload returned nil")
	}
	v.Release()
	var staleObserved []ScopedExecutionEvent
	stale := issue8833Input(q, k, staleID, ExecutionObserverFunc(func(event ScopedExecutionEvent) {
		staleObserved = append(staleObserved, event)
	}))
	got, err = ExecuteMixedQKV(MixedQKVCandidate, stale)
	if !IsMixedQKVDecline(err) || got.Submitted || len(got.Observation.Events) != 0 || len(staleObserved) != 0 {
		t.Fatalf("stale-handle decline: result=%+v observed=%+v err=%v", got, staleObserved, err)
	}
}

func issue8833SelectorName(selector MixedQKVSelector) string {
	if selector == MixedQKVControl {
		return "control"
	}
	return "candidate"
}
