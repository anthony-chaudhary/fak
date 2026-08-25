package metalgemm

import (
	"errors"
	"sync"
)

// ExecutionOperation identifies the native operation observed by one call.
type ExecutionOperation string

const (
	ExecutionQ4KGEMV      ExecutionOperation = "q4_k-gemv"
	ExecutionQ4KGEMVBatch ExecutionOperation = "q4_k-gemv-batch"
	ExecutionQ4KGEMVGroup ExecutionOperation = "q4_k-gemv-group"
	ExecutionQ4KGEMM      ExecutionOperation = "q4_k-gemm"
	ExecutionQ4KGEMMGroup ExecutionOperation = "q4_k-gemm-group"
	ExecutionQ8GEMV       ExecutionOperation = "q8-gemv"
	ExecutionQ8GEMVGroup  ExecutionOperation = "q8-gemv-group"
	ExecutionQ8GEMM       ExecutionOperation = "q8-gemm"
	ExecutionQ8GEMMGroup  ExecutionOperation = "q8-gemm-group"
)

// ExecutionEventsUnavailableError reports that this build cannot observe Metal execution.
type ExecutionEventsUnavailableError struct{}

func (ExecutionEventsUnavailableError) Error() string { return "Metal execution events unavailable" }

// IsExecutionEventsUnavailable reports whether err is the typed unsupported-build result.
func IsExecutionEventsUnavailable(err error) bool {
	var unavailable ExecutionEventsUnavailableError
	return errors.As(err, &unavailable)
}

// ExecutionEvent contains lifecycle facts witnessed inside one native Metal call. CommandBufferID
// is opaque and local to this observation; it never exposes a process-global native identity.
type ExecutionEvent struct {
	Operation       ExecutionOperation `json:"operation"`
	CommandBufferID uint64             `json:"command_buffer_id"`
	Committed       bool               `json:"committed"`
	CompletedWait   bool               `json:"completed_wait"`
	HostReadback    bool               `json:"host_readback"`
}

// ExecutionSnapshot is an immutable copy of one call's observations.
type ExecutionSnapshot struct {
	Events []ExecutionEvent `json:"events"`
}

// ExecutionObservation owns events for exactly one native call. It is safe to snapshot while
// unrelated calls execute; no package-global attribution state or whole-token lock is used.
type ExecutionObservation struct {
	operation ExecutionOperation
	available bool
	mu        sync.Mutex
	events    []ExecutionEvent
}

// NewExecutionObservation starts a call-local observation.
func NewExecutionObservation(operation ExecutionOperation) *ExecutionObservation {
	return newExecutionObservation(operation, executionEventsAvailable())
}

func newExecutionObservation(operation ExecutionOperation, available bool) *ExecutionObservation {
	return &ExecutionObservation{operation: operation, available: available}
}

func (o *ExecutionObservation) record(commandBuffer uintptr, committed, completedWait, hostReadback bool) {
	if o == nil || !o.available || commandBuffer == 0 {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.events = append(o.events, ExecutionEvent{
		Operation:       o.operation,
		CommandBufferID: uint64(len(o.events) + 1),
		Committed:       committed,
		CompletedWait:   completedWait,
		HostReadback:    hostReadback,
	})
}

// Snapshot returns only events owned by this observation.
func (o *ExecutionObservation) Snapshot() (ExecutionSnapshot, error) {
	if o == nil || !o.available {
		return ExecutionSnapshot{}, ExecutionEventsUnavailableError{}
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	return ExecutionSnapshot{Events: append([]ExecutionEvent(nil), o.events...)}, nil
}
