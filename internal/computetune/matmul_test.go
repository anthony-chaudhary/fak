package computetune

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestMatMulImplementationRunsComputeOperationAndDispatches(t *testing.T) {
	backend := compute.Pick("cpu-ref")
	inputs := func(Profile) (compute.Tensor, compute.Tensor, error) {
		weight := compute.NewF32(backend, []int{2, 3}, []float32{1, 0, 0, 0, 2, 0})
		input := compute.NewF32(backend, []int{3}, []float32{4, 5, 6})
		return weight, input, nil
	}
	fallback := MatMulImplementation{CandidateID: "cpu-ref", Backend: backend, Inputs: inputs}
	selected := MatMulImplementation{CandidateID: "cpu-selected", Backend: backend, Inputs: inputs}
	p := profile(1, 2, 3)
	manifest, err := NewManifest([]Entry{{Profile: p, CandidateID: selected.ID()}})
	if err != nil {
		t.Fatal(err)
	}

	got, id, err := DispatchMatMul(context.Background(), p, manifest, map[string]Candidate{selected.ID(): selected}, fallback)
	if err != nil {
		t.Fatal(err)
	}
	if id != selected.ID() || !reflect.DeepEqual(got, []float32{4, 10}) {
		t.Fatalf("real MatMul dispatch got=%v id=%q", got, id)
	}
}
