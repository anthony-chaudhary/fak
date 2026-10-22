package compute

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestQwen35IndependentDecodeBatchRejectsOversizeBeforeMutation(t *testing.T) {
	lanes := make([]Qwen35SequencePrefillRequest, 9)
	for i := range lanes {
		lanes[i] = independentDecodeLane()
	}
	req := Qwen35IndependentDecodeBatchRequest{Path: Qwen35IndependentDecodeBatchPath, Lanes: lanes}
	before := append([]Qwen35SequencePrefillRequest(nil), req.Lanes...)
	err := validateQwen35IndependentDecodeBatchContract(req)
	var batchErr *Qwen35IndependentDecodeBatchError
	if !errors.As(err, &batchErr) || batchErr.Stage != "request" || batchErr.Mutated {
		t.Fatalf("oversize refusal = %#v, want pre-mutation request error", err)
	}
	if !reflect.DeepEqual(req.Lanes, before) {
		t.Fatal("oversize preflight mutated a lane")
	}
}
