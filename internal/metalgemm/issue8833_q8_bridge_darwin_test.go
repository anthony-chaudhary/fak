//go:build darwin && arm64 && cgo

package metalgemm

import (
	"slices"
	"syscall"
	"testing"
)

func issue8833RequireQ8Parity(t *testing.T, selector MixedQKVSelector, wantQ, wantK []float32, got MixedQKVResult) {
	t.Helper()
	if !slices.Equal(wantQ, got.Q) {
		t.Fatalf("selector=%d query output differs from independent Q8 GEMV", selector)
	}
	if !slices.Equal(wantK, got.K) {
		t.Fatalf("selector=%d key output differs from independent Q8 GEMV", selector)
	}
}

func TestIssue8833Q8CallerOwnedBridge(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ8()

	q := issue8833Q8Weight(t, issue8833QOut, 49)
	k := issue8833Q8Weight(t, issue8833KOut, 50)
	v, span := issue8833MappedQ4KWeight(t)
	defer func() {
		ResetQ4K()
		if err := syscall.Munmap(span); err != nil {
			t.Errorf("unmap mapped Q4_K span: %v", err)
		}
	}()

	base := issue8833Input(q, k, v, nil)
	wantQ := make([]float32, issue8833QOut)
	wantK := make([]float32, issue8833KOut)
	q.GEMV(base.XQ, base.XD, wantQ)
	k.GEMV(base.XQ, base.XD, wantK)

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
		// Control's first event has two Q8 encoders; candidate's sole event has both Q8 encoders
		// plus Q4_K. The owner's exact 2->1 event shape, final-only readback, and parity prove the
		// encode helpers appended work without taking commit, wait, or readback ownership.
		issue8833RequireLifecycle(t, selector, got, observed)
		issue8833RequireQ8Parity(t, selector, wantQ, wantK, got)
	}

	// K remains live after Q is released, leaving Q's native slot as an interior tombstone. The
	// caller-owned encoder must decline that stale ID without submitting or publishing lifecycle.
	staleQ := issue8833WeightID(q.ID())
	q.Release()
	var observed []ScopedExecutionEvent
	stale := issue8833Input(staleQ, k, v, ExecutionObserverFunc(func(event ScopedExecutionEvent) {
		observed = append(observed, event)
	}))
	got, err := ExecuteMixedQKV(MixedQKVCandidate, stale)
	if !IsMixedQKVDecline(err) || got.Submitted || len(got.Observation.Events) != 0 || len(observed) != 0 {
		t.Fatalf("stale Q8 handle: result=%+v observed=%+v err=%v", got, observed, err)
	}
}
