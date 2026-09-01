package compute

import (
	"errors"
	"testing"
)

type countedRequestBackend struct {
	fakeDevice
	outstanding int
	retires     int
}

func (b *countedRequestBackend) allocateRequestResource() { b.outstanding++ }
func (b *countedRequestBackend) RetireRequestResources() {
	b.retires++
	b.outstanding = 0
}

func runCountedRequest(backend *countedRequestBackend, stage string) (err error) {
	request := BeginRequest(backend)
	defer request.Retire()
	if stage == "before-allocation" {
		return errors.New("interrupted before allocation")
	}
	backend.allocateRequestResource()
	switch stage {
	case "after-allocation", "during-execution", "before-delivery":
		return errors.New("interrupted request")
	case "panic":
		panic("interrupted request")
	case "complete":
		return nil
	default:
		return errors.New("unknown stage")
	}
}

func TestRequestLifetimeInterruptedCleanupInvariant(t *testing.T) {
	stages := []string{"before-allocation", "after-allocation", "during-execution", "before-delivery", "panic", "complete"}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			backend := &countedRequestBackend{}
			func() {
				defer func() { _ = recover() }()
				_ = runCountedRequest(backend, stage)
			}()
			if backend.outstanding != 0 {
				t.Fatalf("outstanding resources = %d, want baseline 0", backend.outstanding)
			}
			if backend.retires != 1 {
				t.Fatalf("retire calls = %d, want exactly 1", backend.retires)
			}
			if err := runCountedRequest(backend, "complete"); err != nil {
				t.Fatalf("clean follow-up request: %v", err)
			}
			if backend.outstanding != 0 || backend.retires != 2 {
				t.Fatalf("follow-up state = outstanding %d, retires %d; want 0, 2", backend.outstanding, backend.retires)
			}
		})
	}
}

func TestRequestLifetimeRetireIsIdempotent(t *testing.T) {
	backend := &countedRequestBackend{outstanding: 1}
	request := BeginRequest(backend)
	request.Retire()
	request.Retire()
	if backend.outstanding != 0 || backend.retires != 1 {
		t.Fatalf("state = outstanding %d, retires %d; want 0, 1", backend.outstanding, backend.retires)
	}
}
