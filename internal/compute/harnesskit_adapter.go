package compute

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// HarnessAdapter exposes a compute Backend through the stable public harness-kit
// seam without making the internal Backend or Tensor contracts public.
type HarnessAdapter struct {
	backend Backend
	nextID  atomic.Uint64
}

// NewHarnessAdapter binds an already selected compute backend. Lookup("cuda") is the
// real non-default path in CUDA builds; unknown selectors must fail before this call.
func NewHarnessAdapter(backend Backend) (*HarnessAdapter, error) {
	if backend == nil {
		return nil, fmt.Errorf("compute harness adapter: nil backend")
	}
	return &HarnessAdapter{backend: backend}, nil
}

func (a *HarnessAdapter) Name() string { return a.backend.Name() }

func (a *HarnessAdapter) Discover(ctx context.Context) ([]harnesskit.Device, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, canceled("discover", err)
	}
	total, _, known := DeviceMemoryInfo(a.backend)
	var memory uint64
	if known && total > 0 {
		memory = uint64(total)
	}
	return []harnesskit.Device{{
		ID:            a.backend.Name() + "-0",
		Architecture:  a.backend.Tier(),
		Precisions:    []harnesskit.Precision{"f32"},
		MemoryBytes:   memory,
		Concurrency:   1,
		Deterministic: a.backend.Class() == Reference,
	}}, nil
}

func (a *HarnessAdapter) Capabilities(ctx context.Context) (harnesskit.HardwareCapabilities, error) {
	devices, err := a.Discover(ctx)
	if err != nil {
		return harnesskit.HardwareCapabilities{}, err
	}
	fallback := ""
	if a.backend.Class() != Reference {
		fallback = Default().Name()
	}
	return harnesskit.HardwareCapabilities{Devices: devices, Kernels: []string{"matmul"}, FallbackAdapter: fallback}, nil
}

func (a *HarnessAdapter) Allocate(ctx context.Context, spec harnesskit.BufferSpec) (harnesskit.Buffer, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, canceled("allocate", err)
	}
	if spec.Precision != "f32" || spec.Owner == "" || len(spec.Shape) == 0 {
		return nil, contractError(harnesskit.CodeInvalid, "allocate", "f32 precision, owner, and shape are required")
	}
	n := 1
	for _, dim := range spec.Shape {
		if dim <= 0 {
			return nil, contractError(harnesskit.CodeInvalid, "allocate", "shape dimensions must be positive")
		}
		n *= dim
	}
	bytes := uint64(n * 4)
	if spec.Bytes != 0 && spec.Bytes != bytes {
		return nil, contractError(harnesskit.CodeInvalid, "allocate", "bytes do not match shape")
	}
	_, free, known := DeviceMemoryInfo(a.backend)
	if known && free >= 0 && bytes > uint64(free) {
		return nil, contractError(harnesskit.CodeBackpressure, "allocate", "device memory pressure")
	}
	tensor := a.backend.Upload(Tensor{Dtype: F32, Layout: RowMajor, Shape: append([]int(nil), spec.Shape...), buf: &hostBuf{f32: make([]float32, n)}, be: Default()}, F32)
	return &harnessBuffer{id: fmt.Sprintf("%s-%d", a.Name(), a.nextID.Add(1)), owner: spec.Owner, bytes: bytes, adapter: a, tensor: tensor}, nil
}

func (a *HarnessAdapter) Validate(ctx context.Context, req harnesskit.ExecutionRequest) error {
	if err := context.Cause(ctx); err != nil {
		return canceled("validate", err)
	}
	caps, err := a.Capabilities(ctx)
	if err != nil {
		return err
	}
	if !harnesskit.SupportsKernel(caps, req.Kernel, req.Precision) {
		return contractError(harnesskit.CodeUnsupported, "validate", "kernel or precision")
	}
	if req.Deterministic && a.backend.Class() != Reference {
		return contractError(harnesskit.CodeUnsupported, "validate", "backend does not guarantee determinism")
	}
	if len(req.Inputs) != 2 || len(req.Outputs) != 1 {
		return contractError(harnesskit.CodeInvalid, "validate", "matmul requires weight, input, and one output")
	}
	for _, buffer := range append(append([]harnesskit.Buffer(nil), req.Inputs...), req.Outputs...) {
		b, ok := buffer.(*harnessBuffer)
		if !ok || b.adapter != a || b.isReleased() {
			return contractError(harnesskit.CodeInvalid, "validate", "buffer is released or belongs to another adapter")
		}
	}
	return nil
}

func (a *HarnessAdapter) Execute(ctx context.Context, req harnesskit.ExecutionRequest) (harnesskit.ExecutionTelemetry, error) {
	request := BeginRequest(a.backend)
	defer request.Retire()
	if err := a.Validate(ctx, req); err != nil {
		return harnesskit.ExecutionTelemetry{}, err
	}
	weight := req.Inputs[0].(*harnessBuffer)
	input := req.Inputs[1].(*harnessBuffer)
	output := req.Outputs[0].(*harnessBuffer)
	start := time.Now()
	result := a.backend.MatMul(weight.current(), input.current())
	a.backend.Read(result) // fence asynchronous backends so Execution is device-complete
	execution := max(time.Since(start), time.Nanosecond)
	if uint64(result.Numel()*4) != output.Bytes() {
		a.backend.Free(result)
		return harnesskit.ExecutionTelemetry{}, contractError(harnesskit.CodeInvalid, "execute", "output shape does not match result")
	}
	output.replace(result)
	if err := context.Cause(ctx); err != nil {
		return harnesskit.ExecutionTelemetry{}, canceled("execute", err)
	}
	return harnesskit.ExecutionTelemetry{Adapter: a.Name(), Device: a.Name() + "-0", Execution: execution, PeakMemoryBytes: weight.Bytes() + input.Bytes() + output.Bytes()}, nil
}

type harnessBuffer struct {
	mu       sync.Mutex
	id       string
	owner    harnesskit.Ownership
	bytes    uint64
	adapter  *HarnessAdapter
	tensor   Tensor
	released bool
}

func (b *harnessBuffer) ID() string                  { return b.id }
func (b *harnessBuffer) Owner() harnesskit.Ownership { return b.owner }
func (b *harnessBuffer) Bytes() uint64               { return b.bytes }
func (b *harnessBuffer) isReleased() bool            { b.mu.Lock(); defer b.mu.Unlock(); return b.released }
func (b *harnessBuffer) current() Tensor             { b.mu.Lock(); defer b.mu.Unlock(); return b.tensor }
func (b *harnessBuffer) replace(t Tensor) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.adapter.backend.Free(b.tensor)
	b.tensor = t
}
func (b *harnessBuffer) Release(ctx context.Context) error {
	if err := context.Cause(ctx); err != nil {
		return canceled("release", err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.released {
		b.adapter.backend.Free(b.tensor)
		b.released = true
	}
	return nil
}

func canceled(op string, err error) error {
	return &harnesskit.Error{Code: harnesskit.CodeCanceled, Op: "hardware." + op, Err: err}
}

func contractError(code harnesskit.Code, op, message string) error {
	return &harnesskit.Error{Code: code, Op: "hardware." + op, Err: fmt.Errorf("%s", message)}
}
