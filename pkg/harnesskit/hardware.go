package harnesskit

import (
	"context"
	"fmt"
	"slices"
	"time"
)

// Precision is a device data representation advertised by an adapter.
type Precision string

// Device describes one execution target without exposing implementation handles.
type Device struct {
	ID            string      `json:"id"`
	Architecture  string      `json:"architecture"`
	Precisions    []Precision `json:"precisions"`
	MemoryBytes   uint64      `json:"memory_bytes"`
	Concurrency   int         `json:"concurrency"`
	Deterministic bool        `json:"deterministic"`
}

// HardwareCapabilities are negotiated before an execution request is scheduled.
type HardwareCapabilities struct {
	Devices         []Device `json:"devices"`
	Kernels         []string `json:"kernels"`
	FallbackAdapter string   `json:"fallback_adapter,omitempty"`
}

// Buffer is adapter storage. Owner identifies who must call Release; adapters must
// reject a released buffer and Release must be safe to call more than once.
type Buffer interface {
	ID() string
	Owner() Ownership
	Bytes() uint64
	Release(context.Context) error
}

// BufferSpec requests explicitly owned storage.
type BufferSpec struct {
	Shape     []int
	Bytes     uint64
	Precision Precision
	Owner     Ownership
}

// ExecutionRequest is the portable kernel boundary. Buffer ownership does not
// transfer during Execute; outputs remain owned by their declared owner.
type ExecutionRequest struct {
	Kernel        string
	Precision     Precision
	Inputs        []Buffer
	Outputs       []Buffer
	Deterministic bool
}

// ExecutionTelemetry separates queueing and adapter overhead from device time so
// performance comparisons can report net-true cost against a tuned baseline.
type ExecutionTelemetry struct {
	Adapter         string
	Device          string
	FallbackAdapter string
	Queued          time.Duration
	AdapterOverhead time.Duration
	Execution       time.Duration
	PeakMemoryBytes uint64
}

// HardwareAdapter is the public discovery, negotiation, memory, and execution seam.
// Every blocking operation is cancellation-aware; Validate must not execute work.
type HardwareAdapter interface {
	Name() string
	Discover(context.Context) ([]Device, error)
	Capabilities(context.Context) (HardwareCapabilities, error)
	Allocate(context.Context, BufferSpec) (Buffer, error)
	Validate(context.Context, ExecutionRequest) error
	Execute(context.Context, ExecutionRequest) (ExecutionTelemetry, error)
}

// Scheduler controls admission and queueing independently of the hardware adapter.
type Scheduler interface {
	Schedule(context.Context, HardwareAdapter, ExecutionRequest) (ExecutionTelemetry, error)
}

// DirectScheduler is the reference scheduler. It guarantees unsupported work fails
// in Validate before Execute is entered.
type DirectScheduler struct{}

// Schedule validates then executes one request, preserving cancellation causes.
func (DirectScheduler) Schedule(ctx context.Context, adapter HardwareAdapter, req ExecutionRequest) (ExecutionTelemetry, error) {
	if err := context.Cause(ctx); err != nil {
		return ExecutionTelemetry{}, &Error{Code: CodeCanceled, Op: "hardware.schedule", Err: err}
	}
	if adapter == nil {
		return ExecutionTelemetry{}, &Error{Code: CodeInvalid, Op: "hardware.schedule", Err: fmt.Errorf("nil adapter")}
	}
	start := time.Now()
	if err := adapter.Validate(ctx, req); err != nil {
		return ExecutionTelemetry{}, err
	}
	validated := time.Now()
	telemetry, err := adapter.Execute(ctx, req)
	telemetry.AdapterOverhead += max(validated.Sub(start), time.Nanosecond)
	if err != nil {
		return telemetry, err
	}
	return telemetry, nil
}

// SupportsKernel reports whether caps advertise the exact kernel and precision.
func SupportsKernel(caps HardwareCapabilities, kernel string, precision Precision) bool {
	if !slices.Contains(caps.Kernels, kernel) {
		return false
	}
	for _, device := range caps.Devices {
		if slices.Contains(device.Precisions, precision) {
			return true
		}
	}
	return false
}

// HardwareContract is the machine-readable compatibility and promotion policy.
type HardwareContract struct {
	Cancellation           string `json:"cancellation"`
	BufferOwnership        string `json:"buffer_ownership"`
	Unsupported            string `json:"unsupported"`
	Performance            string `json:"performance"`
	PromotionEvidence      string `json:"promotion_evidence"`
	DemotionEvidence       string `json:"demotion_evidence"`
	InvalidatingAssumption string `json:"invalidating_assumption"`
}

// PublicHardwareContract returns the normative hardware adapter policy.
func PublicHardwareContract() HardwareContract {
	return HardwareContract{
		Cancellation:           "every blocking operation accepts context.Context and returns its cause",
		BufferOwnership:        "Buffer.Owner names who must call idempotent Release; Execute never transfers ownership",
		Unsupported:            "Scheduler calls side-effect-free Validate before Execute",
		Performance:            "claims compare tuned baselines and report adapter overhead separately from execution",
		PromotionEvidence:      "reference parity plus sanctioned-device correctness, cancellation, memory-pressure, fallback, and tuned net-true performance captures",
		DemotionEvidence:       "remove or gate an adapter when parity, cancellation, ownership, or net-true performance regresses",
		InvalidatingAssumption: "device discovery and capability negotiation remain truthful for the lifetime of a scheduled request",
	}
}
