package model

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_ring_model_scope_test.go — a serve builds one session per request. With a session-scoped
// ring every request paid the checkpoint page-in again; the model-scoped ring keeps residency
// across requests, so the second request's identical expert is a hit with no checkpoint read and
// byte-identical output.

func modelScopeCheckOutput(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len=%d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: out[%d]=%v, want %v", label, i, got[i], want[i])
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestModelSharedExpertRingSecondRequestHits(t *testing.T) {
	t.Parallel()
	f := newCheckpointSessionFixture(t, true)
	sh, err := f.m.EnableSharedExpertRing(f.be)
	if err != nil {
		t.Fatal(err)
	}
	if again, err := f.m.EnableSharedExpertRing(f.be); err != nil || again != sh {
		t.Fatalf("second enable returned %p err=%v, want the same ring %p", again, err, sh)
	}
	want := f.reference(0, checkpointSessionInput())

	first := f.session(t)
	if first.sharedRing != sh || first.ExpertRingBytes != f.budget {
		t.Fatalf("session did not attach to the model ring: shared=%p budget=%d", first.sharedRing, first.ExpertRingBytes)
	}
	modelScopeCheckOutput(t, "first request", checkpointSessionForward(t, first, 0), want)
	firstStats := sh.Stats().Ring
	quantUploads := func() int { return f.be.uploads[compute.Q2_K] + f.be.uploads[compute.Q3_K] }
	readsAfterFirst, uploadsAfterFirst := f.reads(), quantUploads()
	first.Close()
	if st := sh.Stats(); st.Agents != 0 || st.Ring.ResidentBytes != firstStats.ResidentBytes {
		t.Fatalf("closing the request freed model residency: agents=%d resident=%d, want 0 and %d", st.Agents, st.Ring.ResidentBytes, firstStats.ResidentBytes)
	}

	second := f.session(t)
	modelScopeCheckOutput(t, "second request", checkpointSessionForward(t, second, 0), want)
	st := sh.Stats().Ring
	if hits := st.Hits - firstStats.Hits; hits <= 0 {
		t.Fatalf("second request hits=%d, want > 0", hits)
	}
	if st.PageIns != firstStats.PageIns || f.reads() != readsAfterFirst || quantUploads() != uploadsAfterFirst {
		t.Fatalf("second request paged in again: page_ins %d->%d reads %d->%d quant uploads %d->%d",
			firstStats.PageIns, st.PageIns, readsAfterFirst, f.reads(), uploadsAfterFirst, quantUploads())
	}
	rep := second.MoEResidency(MoEResidencyOptions{Tokens: 1})
	if rep.Shared == nil || !rep.Reconciliation.OK {
		t.Fatalf("model-ring report shared=%v reconciliation=%+v", rep.Shared != nil, rep.Reconciliation)
	}
	second.Close()
	if err := f.m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if f.m.SharedExpertRingFor(f.be) != nil {
		t.Fatal("weight close kept the model ring registered")
	}
	if err := sh.Attach(&Session{M: f.m, Backend: f.be}, "late"); !errors.Is(err, ErrSharedRingClosed) {
		t.Fatalf("attach after weight close err=%v, want ErrSharedRingClosed", err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestSessionScopedRingStillPagesInPerRequest(t *testing.T) {
	t.Parallel()
	f := newCheckpointSessionFixture(t, true)
	first := f.session(t)
	checkpointSessionForward(t, first, 0)
	reads := f.reads()
	first.Close()
	second := f.session(t)
	if second.sharedRing != nil {
		t.Fatal("a model without an enabled ring attached a session")
	}
	checkpointSessionForward(t, second, 0)
	if f.reads() != 2*reads || second.ExpertRing().Hits != 0 {
		t.Fatalf("control arm reads=%d hits=%d, want %d reads and 0 hits", f.reads(), second.ExpertRing().Hits, 2*reads)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestEnableSharedExpertRingRefusesUndeclaredBudget(t *testing.T) {
	t.Parallel()
	f := newCheckpointSessionFixture(t, false)
	if _, err := f.m.EnableSharedExpertRing(f.be); !errors.Is(err, ErrSharedRingUndeclared) {
		t.Fatalf("unbounded tier err=%v, want ErrSharedRingUndeclared", err)
	}
	if _, err := f.m.EnableSharedExpertRing(nil); !errors.Is(err, ErrSharedRingBackend) {
		t.Fatalf("nil backend err=%v, want ErrSharedRingBackend", err)
	}
}

// lockedExpertHALBackend serializes the recording backend so concurrent sessions measure the
// ring's discipline rather than the test double's counters.
type lockedExpertHALBackend struct {
	*expertHALRecordingBackend
	mu sync.Mutex
}

func (b *lockedExpertHALBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.expertHALRecordingBackend.Upload(t, as)
}

func (b *lockedExpertHALBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.expertHALRecordingBackend.MatMul(w, x)
}

func (b *lockedExpertHALBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.expertHALRecordingBackend.SwiGLU(g, u)
}

func (b *lockedExpertHALBackend) Read(t compute.Tensor) []float32 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]float32(nil), b.expertHALRecordingBackend.Read(t)...)
}

func (b *lockedExpertHALBackend) Free(t compute.Tensor) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.expertHALRecordingBackend.Free(t)
}

// fak-test:runtime fast est=2s lane=default
func TestModelSharedExpertRingConcurrentAttachDetach(t *testing.T) {
	t.Parallel()
	const agents, rounds = 4, 3
	// Each agent holds its expert's three projections pinned while it computes, so a shared
	// budget must exceed agents x one expert or a peer's staging finds every resident pinned.
	f := newCheckpointSessionFixture(t, false)
	if err := f.tier.SetDeviceRingBudget(agents * 256 * (84 + 84 + 110)); err != nil {
		t.Fatal(err)
	}
	be := &lockedExpertHALBackend{expertHALRecordingBackend: f.be}
	sh, err := f.m.EnableSharedExpertRing(be)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{f.reference(0, checkpointSessionInput()), f.reference(1, checkpointSessionInput())}
	var wg sync.WaitGroup
	errs := make(chan error, agents*rounds)
	for a := 0; a < agents; a++ {
		wg.Add(1)
		go func(a int) {
			defer wg.Done()
			for r := 0; r < rounds; r++ {
				s, err := f.m.NewBackendSessionChecked(be)
				if err != nil {
					errs <- err
					return
				}
				expert := (a + r) % 2
				names := checkpointSessionNames(expert)
				got, ok := s.expertSwiGLUHAL(names[0], names[1], names[2], checkpointSessionInput())
				_ = sh.Stats()
				s.Close()
				if !ok {
					errs <- errors.New("expert route declined")
					return
				}
				for i := range want[expert] {
					if got[i] != want[expert][i] {
						errs <- errors.New("concurrent output differs from the reference")
						return
					}
				}
			}
		}(a)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	st := sh.Stats()
	if st.Agents != 0 || st.PeakAgents < 1 || st.Ring.PeakBytes > st.Ring.BudgetBytes {
		t.Fatalf("after concurrent attach/detach: agents=%d peak_agents=%d peak=%d budget=%d", st.Agents, st.PeakAgents, st.Ring.PeakBytes, st.Ring.BudgetBytes)
	}
	if err := f.m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
}
