package model

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v4_live_expert_test.go — witness for perf(model) fak#13479: the DeepSeek V4 routed-expert
// residency must live for the (Model, Backend) pair, not for one Session. Before the fix,
// Session.ensureV4LiveExpert built a fresh v4ExpertRuntime per session and Session.Close freed
// its ring, so two sequential sessions over the same model paid the expert page-in twice and
// model teardown had no single owner to free exactly once.
//
// The test drives the real Session.ensureV4LiveExpert / Session.Close / Model.CloseWeights
// seams through a counting fake runtime, so it measures the lifetime contract itself rather
// than a simulation of it. [SW-VERIFIED]: counters and free counts, no hardware.

// countingV4Runtime is the counting fake backing the shared-lifetime witness. It records how
// many distinct runtime objects were constructed (one per page-in of the whole ring) and how
// many times a ring was freed, which is exactly the quantity the issue's acceptance gate names:
// one page-in across sequential sessions, one shared budget under concurrency, one final free.
type countingV4Runtime struct {
	mu       sync.Mutex
	built    int
	freed    int
	closed   int
	hid      int
	forwards int
}

func (c *countingV4Runtime) forward(layer, tokenID int, x, logits, correctionBias []float32) ([]float32, error) {
	c.mu.Lock()
	c.forwards++
	c.mu.Unlock()
	return make([]float32, c.hid), nil
}

func (c *countingV4Runtime) Close() error {
	c.mu.Lock()
	c.closed++
	c.mu.Unlock()
	return nil
}

func (c *countingV4Runtime) Stats() v4ExpertRuntimeStats { return v4ExpertRuntimeStats{} }

// v4LiveExpertTestFactory is the injection seam the witness installs: it replaces the concrete
// newV4ExpertRuntime construction with the counting fake while still flowing through the real
// lifetime code under test.
type v4LiveExpertTestFactory struct {
	mu      sync.Mutex
	rt      *countingV4Runtime
	builds  int
	frees   int
	failNth int
}

func (f *v4LiveExpertTestFactory) build(dir string, cfg Config, be compute.Backend) (v4LiveExpertRuntime, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.builds++
	if f.failNth > 0 && f.builds == f.failNth {
		return nil, errors.New("injected build failure")
	}
	f.rt = &countingV4Runtime{hid: cfg.HiddenSize}
	return f.rt, nil
}

func (f *v4LiveExpertTestFactory) counts() (builds, frees int, rt *countingV4Runtime) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.builds, f.frees, f.rt
}

// withV4LiveExpertFactory swaps the package construction seam for the duration of fn.
func withV4LiveExpertFactory(t *testing.T, fn func(*v4LiveExpertTestFactory)) {
	t.Helper()
	f := &v4LiveExpertTestFactory{}
	prev := v4LiveExpertRuntimeBuilder
	v4LiveExpertRuntimeBuilder = f.build
	t.Cleanup(func() { v4LiveExpertRuntimeBuilder = prev })
	fn(f)
}

// v4LifecycleFixture builds an admitted Flash-profile model with a fake source directory,
// mirroring the shared-expert fixture so no real weights are ever touched.
func v4LifecycleFixture(t *testing.T) (Config, *Model) {
	t.Helper()
	cfg, m := v4SharedExpertTestFixture(t, false)
	return cfg, m
}

func TestV4ExpertRuntimeSharedLifetime(t *testing.T) {
	cfg, m := v4LifecycleFixture(t)
	be := compute.Default()

	withV4LiveExpertFactory(t, func(f *v4LiveExpertTestFactory) {
		// --- Sequential sessions reuse one resident runtime: one build, not two. ---
		s1 := m.NewSession()
		s1.Backend = be
		rt1, err := s1.ensureV4LiveExpert()
		if err != nil {
			t.Fatalf("session 1 ensureV4LiveExpert: %v", err)
		}
		if got := rt1.Stats(); got.RingBudget < 0 {
			t.Fatalf("session 1 runtime has negative ring budget %d", got.RingBudget)
		}
		s1.Close()

		s2 := m.NewSession()
		s2.Backend = be
		rt2, err := s2.ensureV4LiveExpert()
		if err != nil {
			t.Fatalf("session 2 ensureV4LiveExpert: %v", err)
		}
		if rt2 != rt1 {
			t.Fatalf("second sequential session did not reuse the resident runtime: got a different object")
		}
		s2.Close()

		builds, _, rt := f.counts()
		if builds != 1 {
			t.Fatalf("sequential sessions built %d runtimes, want exactly 1 (no repeated page-in)", builds)
		}
		if rt == nil || rt.closed != 0 {
			t.Fatalf("closing a session freed the shared runtime: closed=%v", rt.closed)
		}

		// --- A third session after two closes still sees the same resident runtime. ---
		s3 := m.NewSession()
		s3.Backend = be
		rt3, err := s3.ensureV4LiveExpert()
		if err != nil {
			t.Fatalf("session 3 ensureV4LiveExpert: %v", err)
		}
		if rt3 != rt1 {
			t.Fatalf("resident runtime was not retained across three sequential sessions")
		}
		s3.Close()

		// --- Model teardown frees the shared ring exactly once. ---
		if err := m.CloseWeights(); err != nil {
			t.Fatalf("CloseWeights: %v", err)
		}
		builds, _, rt = f.counts()
		if rt.closed != 1 {
			t.Fatalf("model teardown freed the V4 ring %d time(s), want exactly 1", rt.closed)
		}
		if builds != 1 {
			t.Fatalf("model teardown rebuilt the runtime: builds=%d want 1", builds)
		}

		// After weight teardown, NewSession is refused by design (weights closing). The owner
		// is gone, so a late attach would have no runtime to reach.
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("NewSession after weight teardown did not refuse")
				}
			}()
			_ = m.NewSession()
		}()
	})

	_ = cfg
}

func TestV4ExpertRuntimeConcurrentSessionsShareOneRuntime(t *testing.T) {
	_, m := v4LifecycleFixture(t)
	be := compute.Default()

	withV4LiveExpertFactory(t, func(f *v4LiveExpertTestFactory) {
		const n = 8
		sessions := make([]*Session, n)
		for i := range sessions {
			sessions[i] = m.NewSession()
			sessions[i].Backend = be
		}
		var wg sync.WaitGroup
		runtimes := make([]v4LiveExpertRuntime, n)
		for i, s := range sessions {
			wg.Add(1)
			go func(i int, s *Session) {
				defer wg.Done()
				rt, err := s.ensureV4LiveExpert()
				if err != nil {
					t.Errorf("concurrent ensureV4LiveExpert[%d]: %v", i, err)
					return
				}
				runtimes[i] = rt
			}(i, s)
		}
		wg.Wait()

		builds, _, _ := f.counts()
		if builds != 1 {
			t.Fatalf("concurrent sessions built %d runtimes, want exactly 1 shared budget", builds)
		}
		for i := 1; i < n; i++ {
			if runtimes[i] != runtimes[0] {
				t.Fatalf("concurrent session %d got a duplicate runtime (budget multiplication)", i)
			}
		}

		// Closing a peer must not free bytes another live session is using.
		sessions[0].Close()
		if _, _, rt := f.counts(); rt.closed != 0 {
			t.Fatalf("closing one of several live sessions freed the shared ring")
		}
		for _, s := range sessions[1:] {
			s.Close()
		}
		if _, _, rt := f.counts(); rt.closed != 0 {
			t.Fatalf("the shared ring was freed before model teardown")
		}
		if err := m.CloseWeights(); err != nil {
			t.Fatalf("CloseWeights: %v", err)
		}
		if _, _, rt := f.counts(); rt.closed != 1 {
			t.Fatalf("model teardown freed the shared ring %d time(s), want exactly 1", rt.closed)
		}
	})
}

func TestV4ExpertRuntimeIdentityRefused(t *testing.T) {
	// A second Backend over the same model must not attach to the first backend's ring: a
	// device handle is only valid on the device that produced it.
	_, m := v4LifecycleFixture(t)
	withV4LiveExpertFactory(t, func(f *v4LiveExpertTestFactory) {
		beA := compute.Default()
		sA := m.NewSession()
		sA.Backend = beA
		rtA, err := sA.ensureV4LiveExpert()
		if err != nil {
			t.Fatalf("session A ensureV4LiveExpert: %v", err)
		}
		sA.Close()

		beB := &identityOtherBackend{Backend: compute.Default()}
		sB := m.NewSession()
		sB.Backend = beB
		rtB, err := sB.ensureV4LiveExpert()
		if err != nil {
			t.Fatalf("session B ensureV4LiveExpert: %v", err)
		}
		if rtB == rtA {
			t.Fatalf("a session on a different backend attached to the wrong backend's ring")
		}
		builds, _, _ := f.counts()
		if builds != 2 {
			t.Fatalf("distinct backends built %d runtime(s), want 2 (one owner per model/backend)", builds)
		}
		sB.Close()
		if err := m.CloseWeights(); err != nil {
			t.Fatalf("CloseWeights: %v", err)
		}
	})
}

// identityOtherBackend is a distinct identity wrapping the same default backend. It is only a
// distinct pointer, which is precisely the identity the owner registry keys on.
type identityOtherBackend struct{ compute.Backend }
