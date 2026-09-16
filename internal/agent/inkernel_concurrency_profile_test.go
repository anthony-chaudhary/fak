package agent

import (
	"context"
	"os"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInKernelConcurrencyProfileNamesDeviceMutex is the deterministic witness for
// issue #1589: a 4-way same-prefix fan-out driven through the resident device
// planner must produce a phase timeline that NAMES the serialization mechanism as
// the whole-forward device mutex (devMu), not a guessed one. It is GPU-free and
// replica-stable: four goroutines are released together by a start barrier, and
// the backend is the real cpu-ref so decode actually runs.
func TestInKernelConcurrencyProfileNamesDeviceMutex(t *testing.T) {
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend not registered")
	}
	be := &overlapBackend{Backend: base}

	m := model.NewSynthetic(tinyConcurrencyConfig())
	tok := loadProbeTok(t)

	p := NewInKernelPlanner(m, tok, "tiny-gpu", false, be, false)
	p.maxNew = 8
	p.EnableConcurrencyProfile()

	msgs := []Message{{Role: "user", Content: "hello there, decode a few tokens please"}}

	const N = 4
	var wg sync.WaitGroup
	start := make(chan struct{})
	errs := make([]error, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			_, err := p.Complete(context.Background(), msgs, nil)
			errs[i] = err
		}(i)
	}
	close(start)
	wg.Wait()

	for i := 0; i < N; i++ {
		if errs[i] != nil {
			t.Fatalf("Complete[%d] errored: %v", i, errs[i])
		}
	}

	prof := p.ConcurrencyProfile()
	if prof == nil {
		t.Fatal("ConcurrencyProfile() returned nil after EnableConcurrencyProfile")
	}
	if prof.Schema != ConcurrencyProfileSchema {
		t.Fatalf("schema = %q, want %q", prof.Schema, ConcurrencyProfileSchema)
	}
	if prof.Concurrency != N {
		t.Fatalf("recorded %d phases, want %d", prof.Concurrency, N)
	}
	if prof.Mechanism != MechanismDeviceMutex {
		t.Fatalf("mechanism = %q, want %q (witness: %s)", prof.Mechanism, MechanismDeviceMutex, prof.Witness)
	}
	if prof.MaxConcurrent != 1 {
		t.Fatalf("max_concurrent_forward = %d, want 1 (devMu serialized the whole forward)", prof.MaxConcurrent)
	}
	for _, ph := range prof.Phases {
		if !ph.Serialized {
			t.Fatalf("phase %d not marked serialized: the device mutex was not observed", ph.RequestID)
		}
		if ph.ForwardOutNS < ph.ForwardInNS {
			t.Fatalf("phase %d has forward_out_ns < forward_in_ns", ph.RequestID)
		}
	}
	// The device ops must still never overlap: naming the mechanism does not
	// relax the single-stream invariant this planner already enforces.
	if got := be.overlaps.Load(); got != 0 {
		t.Fatalf("device ops overlapped %d times with the profile enabled", got)
	}

	// Optionally mint the durable artifact the batched-path child (#1590)
	// consumes. Gated on an env var so the default test run writes nothing.
	if out := os.Getenv("FAK_CONCURRENCY_PROFILE_OUT"); out != "" {
		if err := prof.Save(out); err != nil {
			t.Fatalf("save profile: %v", err)
		}
		t.Logf("wrote concurrency profile to %s (mechanism=%s)", out, prof.Mechanism)
	}
}

// TestInKernelConcurrencyProfileNamesConcurrentUnfused is the negative control:
// when the profiler records a path that does NOT hold devMu (a coalescing/CPU
// planner), the timeline must name single-queue-no-batch rather than fabricate
// the device-mutex verdict. It drives the profiler directly so the naming rule
// is pinned independently of the device path.
func TestInKernelConcurrencyProfileNamesConcurrentUnfused(t *testing.T) {
	c := newConcurrencyProfiler()
	// Two requests admitted and held open simultaneously -> maxOpen == 2.
	p1 := c.admit()
	p2 := c.admit()
	c.forwardEnter(p1, false)
	if got := c.forwardEnter(p2, false); got != 2 {
		t.Fatalf("concurrent forward count = %d, want 2", got)
	}
	c.forwardExit(p1)
	c.forwardExit(p2)

	prof := c.build("cpu-ref")
	if prof.Mechanism != MechanismSingleQueueNoBatch {
		t.Fatalf("mechanism = %q, want %q", prof.Mechanism, MechanismSingleQueueNoBatch)
	}
	if prof.MaxConcurrent != 2 {
		t.Fatalf("max_concurrent_forward = %d, want 2", prof.MaxConcurrent)
	}
}

// TestInKernelConcurrencyProfileZeroValueInert pins the byte-for-byte no-op
// guarantee: a planner that never enables the profile records nothing and
// ConcurrencyProfile() returns nil, so no existing caller changes behavior.
func TestInKernelConcurrencyProfileZeroValueInert(t *testing.T) {
	var p *InKernelPlanner
	if prof := p.ConcurrencyProfile(); prof != nil {
		t.Fatalf("nil planner profile = %+v, want nil", prof)
	}
	c := &concurrencyProfiler{}
	if ph := c.admit(); ph != nil {
		t.Fatalf("disabled profiler admitted a phase: %+v", ph)
	}
	if prof := c.build("cpu-ref"); prof == nil || prof.Mechanism != MechanismUnknown {
		t.Fatalf("disabled profiler build = %+v, want unknown", prof)
	}
}
