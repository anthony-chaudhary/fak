package compute

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestHarnessAdapterRunsComputeBackendThroughPublicScheduler(t *testing.T) {
	adapter, err := NewHarnessAdapter(Default())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	weight, err := adapter.Allocate(ctx, harnesskit.BufferSpec{Shape: []int{2, 2}, Bytes: 16, Precision: "f32", Owner: harnesskit.OwnershipCaller})
	if err != nil {
		t.Fatal(err)
	}
	input, err := adapter.Allocate(ctx, harnesskit.BufferSpec{Shape: []int{2}, Bytes: 8, Precision: "f32", Owner: harnesskit.OwnershipCaller})
	if err != nil {
		t.Fatal(err)
	}
	output, err := adapter.Allocate(ctx, harnesskit.BufferSpec{Shape: []int{2}, Bytes: 8, Precision: "f32", Owner: harnesskit.OwnershipCaller})
	if err != nil {
		t.Fatal(err)
	}
	w := weight.(*harnessBuffer)
	x := input.(*harnessBuffer)
	copy(w.tensor.buf.(HostBuffer).F32(), []float32{1, 2, 3, 4})
	copy(x.tensor.buf.(HostBuffer).F32(), []float32{5, 6})
	telemetry, err := (harnesskit.DirectScheduler{}).Schedule(ctx, adapter, harnesskit.ExecutionRequest{Kernel: "matmul", Precision: "f32", Inputs: []harnesskit.Buffer{weight, input}, Outputs: []harnesskit.Buffer{output}, Deterministic: true})
	if err != nil {
		t.Fatal(err)
	}
	got := adapter.backend.Read(output.(*harnessBuffer).current())
	if got[0] != 17 || got[1] != 39 {
		t.Fatalf("got %v", got)
	}
	if telemetry.Adapter != Default().Name() || telemetry.Execution <= 0 || telemetry.AdapterOverhead <= 0 {
		t.Fatalf("telemetry: %#v", telemetry)
	}
}

func TestHarnessAdapterRejectsUnsupportedBeforeBackendExecution(t *testing.T) {
	backend := &countingBackend{Backend: Default()}
	adapter, err := NewHarnessAdapter(backend)
	if err != nil {
		t.Fatal(err)
	}
	_, err = (harnesskit.DirectScheduler{}).Schedule(context.Background(), adapter, harnesskit.ExecutionRequest{Kernel: "attention", Precision: "f32"})
	var contractErr *harnesskit.Error
	if !errors.As(err, &contractErr) || contractErr.Code != harnesskit.CodeUnsupported {
		t.Fatalf("got %v", err)
	}
	if backend.matmulCalls != 0 {
		t.Fatal("unsupported kernel reached backend")
	}
}

type countingBackend struct {
	Backend
	matmulCalls int
}

func (b *countingBackend) MatMul(w, x Tensor) Tensor {
	b.matmulCalls++
	return b.Backend.MatMul(w, x)
}
