package harnesskit

import (
	"context"
	"errors"
	"testing"
	"time"
)

type fixtureBuffer struct {
	id       string
	owner    Ownership
	data     []float32
	released bool
}

func (b *fixtureBuffer) ID() string                    { return b.id }
func (b *fixtureBuffer) Owner() Ownership              { return b.owner }
func (b *fixtureBuffer) Bytes() uint64                 { return uint64(len(b.data) * 4) }
func (b *fixtureBuffer) Release(context.Context) error { b.released = true; return nil }

type fixtureAdapter struct {
	name      string
	precision Precision
	executed  int
}

func (a *fixtureAdapter) Name() string { return a.name }
func (a *fixtureAdapter) Discover(context.Context) ([]Device, error) {
	return []Device{{ID: a.name + "-0", Architecture: a.name, Precisions: []Precision{a.precision}, MemoryBytes: 1 << 20, Concurrency: 1, Deterministic: true}}, nil
}
func (a *fixtureAdapter) Capabilities(ctx context.Context) (HardwareCapabilities, error) {
	devices, err := a.Discover(ctx)
	return HardwareCapabilities{Devices: devices, Kernels: []string{"add"}, FallbackAdapter: "reference"}, err
}
func (a *fixtureAdapter) Allocate(_ context.Context, spec BufferSpec) (Buffer, error) {
	return &fixtureBuffer{id: a.name, owner: spec.Owner, data: make([]float32, spec.Bytes/4)}, nil
}
func (a *fixtureAdapter) Validate(ctx context.Context, req ExecutionRequest) error {
	if err := context.Cause(ctx); err != nil {
		return &Error{Code: CodeCanceled, Op: "hardware.validate", Err: err}
	}
	caps, _ := a.Capabilities(ctx)
	if !SupportsKernel(caps, req.Kernel, req.Precision) {
		return &Error{Code: CodeUnsupported, Op: "hardware.validate", Err: errors.New("kernel or precision")}
	}
	return nil
}
func (a *fixtureAdapter) Execute(ctx context.Context, req ExecutionRequest) (ExecutionTelemetry, error) {
	a.executed++
	if err := context.Cause(ctx); err != nil {
		return ExecutionTelemetry{}, &Error{Code: CodeCanceled, Op: "hardware.execute", Err: err}
	}
	in := req.Inputs[0].(*fixtureBuffer)
	out := req.Outputs[0].(*fixtureBuffer)
	for i := range out.data {
		out.data[i] = in.data[i] + 1
	}
	return ExecutionTelemetry{Adapter: a.name, Device: a.name + "-0", Execution: time.Microsecond, PeakMemoryBytes: in.Bytes() + out.Bytes()}, nil
}

// RunHardwareCorrectnessFixture is shared by reference and accelerated adapter
// tests so both paths must satisfy the same observable contract.
func RunHardwareCorrectnessFixture(t *testing.T, adapter HardwareAdapter) []float32 {
	t.Helper()
	ctx := context.Background()
	input, err := adapter.Allocate(ctx, BufferSpec{Bytes: 8, Precision: "f32", Owner: OwnershipCaller})
	if err != nil {
		t.Fatal(err)
	}
	output, err := adapter.Allocate(ctx, BufferSpec{Bytes: 8, Precision: "f32", Owner: OwnershipCaller})
	if err != nil {
		t.Fatal(err)
	}
	in := input.(*fixtureBuffer)
	in.data = []float32{1, 2}
	req := ExecutionRequest{Kernel: "add", Precision: "f32", Inputs: []Buffer{input}, Outputs: []Buffer{output}, Deterministic: true}
	telemetry, err := (DirectScheduler{}).Schedule(ctx, adapter, req)
	if err != nil {
		t.Fatal(err)
	}
	if telemetry.Adapter != adapter.Name() || telemetry.AdapterOverhead <= 0 || telemetry.Execution <= 0 {
		t.Fatalf("incomplete telemetry: %#v", telemetry)
	}
	return output.(*fixtureBuffer).data
}

func TestHardwareReferenceAndAcceleratedPathsShareFixture(t *testing.T) {
	for _, name := range []string{"reference", "accelerated"} {
		t.Run(name, func(t *testing.T) {
			got := RunHardwareCorrectnessFixture(t, &fixtureAdapter{name: name, precision: "f32"})
			if got[0] != 2 || got[1] != 3 {
				t.Fatalf("got %v", got)
			}
		})
	}
}

func TestHardwareUnsupportedFailsBeforeExecution(t *testing.T) {
	adapter := &fixtureAdapter{name: "accelerated", precision: "f32"}
	_, err := (DirectScheduler{}).Schedule(context.Background(), adapter, ExecutionRequest{Kernel: "unknown", Precision: "f32"})
	var contractErr *Error
	if !errors.As(err, &contractErr) || contractErr.Code != CodeUnsupported {
		t.Fatalf("got %v", err)
	}
	if adapter.executed != 0 {
		t.Fatal("unsupported kernel reached Execute")
	}
}

func TestHardwareCancellationAndOwnershipAreExplicit(t *testing.T) {
	adapter := &fixtureAdapter{name: "accelerated", precision: "f32"}
	ctx, cancel := context.WithCancelCause(context.Background())
	cause := errors.New("operator canceled")
	cancel(cause)
	_, err := (DirectScheduler{}).Schedule(ctx, adapter, ExecutionRequest{Kernel: "add", Precision: "f32"})
	if !errors.Is(err, cause) || adapter.executed != 0 {
		t.Fatalf("cancellation: %v executed=%d", err, adapter.executed)
	}
	buffer, err := adapter.Allocate(context.Background(), BufferSpec{Bytes: 16, Precision: "f32", Owner: OwnershipCaller})
	if err != nil {
		t.Fatal(err)
	}
	if buffer.Owner() != OwnershipCaller {
		t.Fatalf("owner = %s", buffer.Owner())
	}
	if err := buffer.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := buffer.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestHardwareContractCarriesGenerationEvidence(t *testing.T) {
	contract := PublicHardwareContract()
	if contract.PromotionEvidence == "" || contract.DemotionEvidence == "" || contract.InvalidatingAssumption == "" || contract.Performance == "" {
		t.Fatalf("incomplete hardware contract: %#v", contract)
	}
}
