package metalgemm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync"
)

// ExecutionOperation identifies the native operation observed by one call.
type ExecutionOperation string

const (
	ExecutionQ4KGEMV                ExecutionOperation = "q4_k-gemv"
	ExecutionQ4KGEMVBatch           ExecutionOperation = "q4_k-gemv-batch"
	ExecutionQ4KGEMVGroup           ExecutionOperation = "q4_k-gemv-group"
	ExecutionQ4KGEMM                ExecutionOperation = "q4_k-gemm"
	ExecutionQ4KGEMMGroup           ExecutionOperation = "q4_k-gemm-group"
	ExecutionQ8GEMV                 ExecutionOperation = "q8-gemv"
	ExecutionQ8GEMVGroup            ExecutionOperation = "q8-gemv-group"
	ExecutionQ8GEMM                 ExecutionOperation = "q8-gemm"
	ExecutionQ8GEMMGroup            ExecutionOperation = "q8-gemm-group"
	ExecutionQ6KGEMV                ExecutionOperation = "q6_k-gemv"
	ExecutionQ6KGEMM                ExecutionOperation = "q6_k-gemm"
	ExecutionQ4KFusedMLP            ExecutionOperation = "q4_k-fused-mlp"
	ExecutionQ4KFusedMLPQ6Down      ExecutionOperation = "q4_k-fused-mlp-q6_k-down"
	ExecutionQ4KFusedMLPQ6DownBatch ExecutionOperation = "q4_k-fused-mlp-q6_k-down-batch"
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
	Operation        ExecutionOperation `json:"operation"`
	CommandBufferID  uint64             `json:"command_buffer_id"`
	Committed        bool               `json:"committed"`
	CompletedWait    bool               `json:"completed_wait"`
	HostReadback     bool               `json:"host_readback"`
	Encoders         int                `json:"encoders"`
	GPUMilliseconds  float64            `json:"gpu_milliseconds"`
	WaitMilliseconds float64            `json:"wait_milliseconds"`
	TimingAvailable  bool               `json:"timing_available"`
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

func (o *ExecutionObservation) record(commandBuffer uintptr, committed, completedWait, hostReadback bool, encoders int, gpuMS, waitMS float64, timingAvailable bool) {
	if o == nil || !o.available || commandBuffer == 0 {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.events = append(o.events, ExecutionEvent{
		Operation:        o.operation,
		CommandBufferID:  uint64(len(o.events) + 1),
		Committed:        committed,
		CompletedWait:    completedWait,
		HostReadback:     hostReadback,
		Encoders:         encoders,
		GPUMilliseconds:  gpuMS,
		WaitMilliseconds: waitMS,
		TimingAvailable:  timingAvailable,
	})
}

// ExecutionCounters is the lossless Metal subset required by the native-performance v1 profile.
type ExecutionCounters struct {
	CommandBuffers       int     `json:"command_buffers"`
	Encoders             int     `json:"encoders"`
	DispatchMilliseconds float64 `json:"dispatch_milliseconds"`
	WaitMilliseconds     float64 `json:"wait_milliseconds"`
}

// ExecutionReceipt preserves the raw ordered event stream behind an aggregate.
type ExecutionReceipt struct {
	Schema       string            `json:"schema"`
	Events       []ExecutionEvent  `json:"events"`
	EventsSHA256 string            `json:"events_sha256"`
	Counters     ExecutionCounters `json:"counters"`
}

// ExecutionCountersIncompleteError reports the first lifecycle fact that could not be captured.
type ExecutionCountersIncompleteError struct{ Detail string }

func (e ExecutionCountersIncompleteError) Error() string {
	return "Metal execution counters incomplete: " + e.Detail
}

// IsExecutionCountersIncomplete reports whether err is a typed incomplete-capture result.
func IsExecutionCountersIncomplete(err error) bool {
	var incomplete ExecutionCountersIncompleteError
	return errors.As(err, &incomplete)
}

// ExecutionSession owns every observation for one model session. It never aggregates process-wide.
type ExecutionSession struct {
	mu     sync.Mutex
	events []ExecutionEvent
	err    error
}

// NewExecutionSession creates an empty session-local aggregate.
func NewExecutionSession() *ExecutionSession { return &ExecutionSession{} }

// Record appends one native call, poisoning the aggregate when capture is unavailable or empty.
func (s *ExecutionSession) Record(snapshot ExecutionSnapshot, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return
	}
	if err != nil {
		s.err = err
		return
	}
	if len(snapshot.Events) == 0 {
		s.err = ExecutionCountersIncompleteError{Detail: "native call emitted no command-buffer event"}
		return
	}
	s.events = append(s.events, snapshot.Events...)
}

// Counters validates every captured lifecycle before returning the session aggregate.
func (s *ExecutionSession) Counters() (ExecutionCounters, error) {
	if s == nil {
		return ExecutionCounters{}, ExecutionCountersIncompleteError{Detail: "session capture is nil"}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return ExecutionCounters{}, s.err
	}
	return countersForEvents(s.events)
}

// Receipt returns the raw stream, its deterministic digest, and independently recomputed totals.
func (s *ExecutionSession) Receipt() (ExecutionReceipt, error) {
	if s == nil {
		return ExecutionReceipt{}, ExecutionCountersIncompleteError{Detail: "session capture is nil"}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return ExecutionReceipt{}, s.err
	}
	events := append([]ExecutionEvent(nil), s.events...)
	counters, err := countersForEvents(events)
	if err != nil {
		return ExecutionReceipt{}, err
	}
	raw, err := json.Marshal(events)
	if err != nil {
		return ExecutionReceipt{}, err
	}
	digest := sha256.Sum256(raw)
	return ExecutionReceipt{Schema: "fak-metal-execution-receipt/v1", Events: events, EventsSHA256: hex.EncodeToString(digest[:]), Counters: counters}, nil
}

// ValidateExecutionReceipt recomputes both digest and totals from the raw event stream.
func ValidateExecutionReceipt(receipt ExecutionReceipt) error {
	if receipt.Schema != "fak-metal-execution-receipt/v1" {
		return fmt.Errorf("unexpected execution receipt schema %q", receipt.Schema)
	}
	raw, err := json.Marshal(receipt.Events)
	if err != nil {
		return err
	}
	digest := sha256.Sum256(raw)
	if receipt.EventsSHA256 != hex.EncodeToString(digest[:]) {
		return fmt.Errorf("execution event digest mismatch")
	}
	counters, err := countersForEvents(receipt.Events)
	if err != nil {
		return err
	}
	if counters != receipt.Counters {
		return fmt.Errorf("execution counter aggregate mismatch")
	}
	return nil
}

func countersForEvents(events []ExecutionEvent) (ExecutionCounters, error) {
	if len(events) == 0 {
		return ExecutionCounters{}, ExecutionCountersIncompleteError{Detail: "session captured no command buffers"}
	}
	var out ExecutionCounters
	for i, event := range events {
		if !event.Committed || !event.CompletedWait || !event.HostReadback || event.Encoders <= 0 || !event.TimingAvailable || math.IsNaN(event.GPUMilliseconds) || math.IsInf(event.GPUMilliseconds, 0) || event.GPUMilliseconds <= 0 || math.IsNaN(event.WaitMilliseconds) || math.IsInf(event.WaitMilliseconds, 0) || event.WaitMilliseconds < 0 {
			return ExecutionCounters{}, ExecutionCountersIncompleteError{Detail: fmt.Sprintf("event %d (%s) lacks a complete committed/waited/readback/timed lifecycle", i+1, event.Operation)}
		}
		out.CommandBuffers++
		out.Encoders += event.Encoders
		out.DispatchMilliseconds += event.GPUMilliseconds
		out.WaitMilliseconds += event.WaitMilliseconds
	}
	return out, nil
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
