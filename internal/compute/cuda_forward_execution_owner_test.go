//go:build cuda && cgo

package compute

import (
	"sync"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestCUDAForwardExecutionOwnerDomain(t *testing.T) {
	type embeddedCUDAOwner struct{ *cudaBackend }
	a, b := &cudaBackend{}, &cudaBackend{}
	var owner *sync.Mutex
	for _, handle := range []any{a, b, &embeddedCUDAOwner{a}} {
		// A test-only parent still compiles: absence is a behavioral assertion,
		// not a reference to a missing method or new mutex variable.
		provider, ok := handle.(ForwardExecutionOwnerBackend)
		if !ok {
			t.Fatalf("CUDA handle %T does not expose its recording owner", handle)
		}
		got := provider.ForwardExecutionOwner()
		if got == nil || got == &cudaMu {
			t.Fatal("CUDA owner is nil or aliases the per-operation mutex")
		}
		if owner == nil {
			owner = got
		}
		if got != owner || provider.ForwardExecutionOwner() != owner {
			t.Fatal("CUDA handles or wrappers do not share one stable owner")
		}
	}
	owner.Lock()
	locked := true
	defer func() {
		if locked {
			owner.Unlock()
		}
	}()
	if !cudaMu.TryLock() {
		t.Fatal("outer ownership prevents independent operation locking")
	}
	cudaMu.Unlock()
	other := any(b).(ForwardExecutionOwnerBackend).ForwardExecutionOwner()
	if other.TryLock() {
		other.Unlock()
		t.Fatal("another CUDA handle entered an owned recording domain")
	}
	owner.Unlock()
	locked = false
	if !other.TryLock() {
		t.Fatal("released CUDA owner is not reusable")
	}
	other.Unlock()
}
