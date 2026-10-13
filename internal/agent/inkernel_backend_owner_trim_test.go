package agent

import (
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Keep the owner capability off this base type so the legacy case exercises
// absence, not an implemented provider that returns nil.
type inKernelTrimProbeBackend struct {
	compute.Backend
	calls []string
	keep  int
	hook  func(string)
}

func (b *inKernelTrimProbeBackend) note(site string) {
	b.calls = append(b.calls, site)
	if b.hook != nil {
		b.hook(site)
	}
}

func (b *inKernelTrimProbeBackend) Recycle() { b.note("recycle") }
func (b *inKernelTrimProbeBackend) Trim()    { b.note("trim") }
func (b *inKernelTrimProbeBackend) TrimLarge(keep int) {
	b.keep = keep
	b.note("trim-large")
}

type inKernelTrimOwnerBackend struct {
	*inKernelTrimProbeBackend
	owner   *sync.Mutex
	queries int
}

func (b *inKernelTrimOwnerBackend) ForwardExecutionOwner() *sync.Mutex {
	b.queries++
	return b.owner
}

// fak-test:runtime fast est=5ms lane=default
func TestInKernelBackendOwnerTrim(t *testing.T) {
	newProbe := func() *inKernelTrimProbeBackend {
		return &inKernelTrimProbeBackend{Backend: compute.Default(), keep: -1}
	}
	wantCalls := []string{"recycle", "trim", "trim-large"}
	assertCalls := func(t *testing.T, b *inKernelTrimProbeBackend, want []string) {
		t.Helper()
		if !reflect.DeepEqual(b.calls, want) {
			t.Fatalf("maintenance hooks = %v, want %v", b.calls, want)
		}
	}
	assertReleased := func(t *testing.T, owner *sync.Mutex) {
		t.Helper()
		if !owner.TryLock() {
			t.Fatal("maintenance retained the backend owner")
		}
		owner.Unlock()
	}

	t.Run("absent capability keeps legacy hooks", func(t *testing.T) {
		b := newProbe()
		if _, ok := any(b).(compute.ForwardExecutionOwnerBackend); ok {
			t.Fatal("legacy fixture unexpectedly implements the owner capability")
		}
		if !(&InKernelPlanner{backend: b}).trimBackendIdlePools() {
			t.Fatal("legacy maintenance was declined")
		}
		assertCalls(t, b, wantCalls)
		if b.keep != 0 {
			t.Fatalf("legacy TrimLarge argument = %d, want 0", b.keep)
		}
	})

	t.Run("implemented nil owner refuses without hooks", func(t *testing.T) {
		b := &inKernelTrimOwnerBackend{inKernelTrimProbeBackend: newProbe()}
		if (&InKernelPlanner{backend: b}).trimBackendIdlePools() {
			t.Fatal("nil-owner maintenance was admitted")
		}
		assertCalls(t, b.inKernelTrimProbeBackend, nil)
		if b.queries != 1 || b.keep != -1 {
			t.Fatalf("nil admission queried %d times or invoked TrimLarge(%d)", b.queries, b.keep)
		}
	})

	t.Run("busy shared domain refuses distinct clients", func(t *testing.T) {
		var owner sync.Mutex
		owner.Lock()
		defer owner.Unlock()
		for i := 0; i < 2; i++ {
			b := &inKernelTrimOwnerBackend{inKernelTrimProbeBackend: newProbe(), owner: &owner}
			if (&InKernelPlanner{backend: b}).trimBackendIdlePools() {
				t.Fatalf("busy maintenance client %d was admitted", i)
			}
			assertCalls(t, b.inKernelTrimProbeBackend, nil)
			if b.queries != 1 || b.keep != -1 {
				t.Fatalf("busy client queried %d times or invoked TrimLarge(%d)", b.queries, b.keep)
			}
		}
	})

	t.Run("one lease spans every hook", func(t *testing.T) {
		var owner sync.Mutex
		b := &inKernelTrimOwnerBackend{inKernelTrimProbeBackend: newProbe(), owner: &owner}
		b.hook = func(site string) {
			if owner.TryLock() {
				owner.Unlock()
				t.Fatalf("%s ran without owning its backend domain", site)
			}
		}
		if !(&InKernelPlanner{backend: b}).trimBackendIdlePools() {
			t.Fatal("idle owned maintenance was declined")
		}
		assertCalls(t, b.inKernelTrimProbeBackend, wantCalls)
		if b.queries != 1 || b.keep != 0 {
			t.Fatalf("owned cleanup queried %d times or invoked TrimLarge(%d)", b.queries, b.keep)
		}
		assertReleased(t, &owner)
	})

	t.Run("hook panic releases and preserves cause", func(t *testing.T) {
		var owner sync.Mutex
		b := &inKernelTrimOwnerBackend{inKernelTrimProbeBackend: newProbe(), owner: &owner}
		cause := &struct{ site string }{"trim"}
		b.hook = func(site string) {
			if site == "trim" {
				panic(cause)
			}
		}
		var got any
		func() {
			defer func() { got = recover() }()
			(&InKernelPlanner{backend: b}).trimBackendIdlePools()
		}()
		if got != cause {
			t.Fatalf("panic = %v, want original cause %v", got, cause)
		}
		assertCalls(t, b.inKernelTrimProbeBackend, wantCalls[:2])
		assertReleased(t, &owner)
	})
}
