package compute

import (
	"slices"
	"sync"
	"sync/atomic"
	"testing"
)

type deferredTestBackend struct {
	Backend
	name string
	tag  int
}

func (b deferredTestBackend) Name() string            { return b.name }
func (b deferredTestBackend) Class() CorrectnessClass { return Approx }

// isolateDeferredRegistry settles any real deferred backend, then gives the test a
// fresh join so it can reserve its own slots; the registry is restored afterwards.
// It resets deferredOnce without a lock, so tests using it must not run in parallel.
func isolateDeferredRegistry(t *testing.T) {
	t.Helper()
	joinDeferredBackends()
	registryMu.Lock()
	saved := slices.Clone(registry)
	deferredBackends = nil
	registryMu.Unlock()
	deferredOnce = sync.Once{}
	t.Cleanup(func() {
		joinDeferredBackends()
		registryMu.Lock()
		registry = saved
		deferredBackends = nil
		registryMu.Unlock()
	})
}

func TestDeferredBackendKeepsRegistrationOrder(t *testing.T) {
	isolateDeferredRegistry(t)
	before := Registered()
	deferredOnce = sync.Once{}
	var starts atomic.Int32
	registerDeferred(func() Backend {
		starts.Add(1)
		return deferredTestBackend{name: "deferred-test-dev"}
	})
	Register(deferredTestBackend{name: "deferred-test-after"})
	if n := starts.Load(); n != 0 {
		t.Fatalf("start ran %d times before any registry read", n)
	}
	got := Registered()
	want := append(slices.Clone(before), "deferred-test-dev", "deferred-test-after")
	if !slices.Equal(got, want) {
		t.Fatalf("Registered() = %v, want %v", got, want)
	}
	if b, ok := Lookup("deferred-test-dev"); !ok || b.Name() != "deferred-test-dev" {
		t.Fatalf("Lookup(deferred-test-dev) = %v, %v", b, ok)
	}
	if n := starts.Load(); n != 1 {
		t.Fatalf("start ran %d times, want 1", n)
	}
}

func TestDeferredBackendWithoutDeviceStaysUnregistered(t *testing.T) {
	isolateDeferredRegistry(t)
	before := Registered()
	deferredOnce = sync.Once{}
	registerDeferred(func() Backend { return nil })
	if _, ok := Lookup("deferred-test-dev"); ok {
		t.Fatal("a backend whose device never came up is registered")
	}
	if got := Registered(); !slices.Equal(got, before) {
		t.Fatalf("Registered() = %v, want %v", got, before)
	}
	if got, want := Pick("deferred-test-dev"), Default(); got != want {
		t.Fatalf("Pick fell back to %v, want Default %v", got, want)
	}
}

func TestDeferredBackendYieldsToLaterSameNameRegister(t *testing.T) {
	isolateDeferredRegistry(t)
	registerDeferred(func() Backend { return deferredTestBackend{name: "deferred-test-dev"} })
	double := deferredTestBackend{name: "deferred-test-dev", tag: 1}
	Register(double)
	b, ok := Lookup("deferred-test-dev")
	if !ok || b != Backend(double) {
		t.Fatalf("Lookup = %v, %v; want the later registered double", b, ok)
	}
	n := 0
	for _, name := range Registered() {
		if name == "deferred-test-dev" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("deferred-test-dev registered %d times", n)
	}
}

func TestStartBackendInitJoinsOnFirstRead(t *testing.T) {
	isolateDeferredRegistry(t)
	release := make(chan struct{})
	var starts atomic.Int32
	registerDeferred(func() Backend {
		starts.Add(1)
		<-release
		return deferredTestBackend{name: "deferred-test-dev"}
	})
	StartBackendInit()
	var wg sync.WaitGroup
	found := make([]bool, 8)
	for i := range found {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, found[i] = Lookup("deferred-test-dev")
		}()
	}
	close(release)
	wg.Wait()
	for i, ok := range found {
		if !ok {
			t.Fatalf("reader %d returned before bring-up finished", i)
		}
	}
	if n := starts.Load(); n != 1 {
		t.Fatalf("start ran %d times, want 1", n)
	}
}
