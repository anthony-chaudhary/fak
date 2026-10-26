package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
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
// fak-test:runtime fast est=1s lane=default
func TestInKernelConcurrencyProfileNamesDeviceMutex(t *testing.T) {
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend not registered")
	}
	be := &overlapBackend{Backend: base}
	prof := runInKernelConcurrencyProfile(t, be)
	if got := be.overlaps.Load(); got != 0 {
		t.Fatalf("device ops overlapped %d times with the profile enabled", got)
	}
	assertInKernelDeviceMutexProfile(t, prof)
	assertInKernelConcurrencyProfileSaved(t, prof)
	if out := os.Getenv("FAK_CONCURRENCY_PROFILE_OUT"); out != "" {
		if err := prof.Save(out); err != nil {
			t.Fatalf("save profile: %v", err)
		}
		t.Logf("wrote concurrency profile to %s (backend=%s mechanism=%s)", out, prof.Backend, prof.Mechanism)
	}
}

func runInKernelConcurrencyProfile(t *testing.T, be compute.Backend) *ConcurrencyProfile {
	t.Helper()
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
	generated := make([]int, N)
	native := make([]*model.NativeInferenceReceipt, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			comp, err := p.Complete(context.Background(), msgs, nil, WithNativeInferenceReceipt(true))
			errs[i] = err
			if comp != nil {
				generated[i] = comp.Usage.CompletionTokens
				native[i] = comp.NativeInference
			}
		}(i)
	}
	close(start)
	wg.Wait()

	for i := 0; i < N; i++ {
		if errs[i] != nil {
			t.Fatalf("Complete[%d] errored: %v", i, errs[i])
		}
		if generated[i] == 0 {
			t.Fatalf("Complete[%d] generated no tokens", i)
		}
		if native[i] == nil || native[i].Backend != be.Name() || len(native[i].TokenIDs) == 0 {
			t.Fatalf("Complete[%d] lacks a native token receipt from %s", i, be.Name())
		}
		if be.Caps().DeviceMemory && native[i].FallbackActive {
			t.Fatalf("Complete[%d] used host fallback on device backend %s", i, be.Name())
		}
	}

	prof := p.ConcurrencyProfile()
	if prof == nil {
		t.Fatal("ConcurrencyProfile() returned nil after EnableConcurrencyProfile")
	}
	backend, path := p.ExecutionIdentity()
	if backend != be.Name() || path != "device/generic" {
		t.Fatalf("execution identity = %s/%s, want %s/device/generic", backend, path, be.Name())
	}
	prof.Backend, prof.BackendTier = backend, be.Tier()
	return prof
}

func assertInKernelDeviceMutexProfile(t *testing.T, prof *ConcurrencyProfile) {
	t.Helper()
	if prof.Schema != ConcurrencyProfileSchema {
		t.Fatalf("schema = %q, want %q", prof.Schema, ConcurrencyProfileSchema)
	}
	if prof.Concurrency != 4 {
		t.Fatalf("recorded %d phases, want 4", prof.Concurrency)
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
	if prof.ForwardPath != "device/generic" {
		t.Fatalf("forward_path = %q, want device/generic", prof.ForwardPath)
	}
}

func assertInKernelConcurrencyProfileSaved(t *testing.T, prof *ConcurrencyProfile) {
	t.Helper()
	out := filepath.Join(t.TempDir(), "profile.json")
	if err := prof.Save(out); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var saved ConcurrencyProfile
	if err := json.Unmarshal(data, &saved); err != nil {
		t.Fatal(err)
	}
	assertInKernelDeviceMutexProfile(t, &saved)
	if saved.Backend != prof.Backend || saved.BackendTier != prof.BackendTier || len(saved.Phases) != len(prof.Phases) {
		t.Fatalf("saved profile lost backend identity or phases: backend=%q tier=%q phases=%d", saved.Backend, saved.BackendTier, len(saved.Phases))
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
