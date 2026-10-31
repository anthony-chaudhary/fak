package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// Missing Q8 support must refuse float projections after load-time conversion;
// a true host-only norm must not mask that device gap.
// fak-test:runtime fast est=1ms lane=default
func TestModelPreflightGGUFFloatProjectionMissingQ8Refuses(t *testing.T) {
	table := filterTableRows(compute.CUDADeviceOpTable(), func(op, format string) bool {
		return op != "matmul" || format != "q8_0"
	})
	tensors := []ggufload.TensorInfo{
		{Name: "blk.0.attn_q.weight", Dims: []uint64{32, 32}, Type: ggufload.TensorF32},
		{Name: "blk.0.attn_k.weight", Dims: []uint64{32, 32}, Type: ggufload.TensorF16},
		{Name: "blk.0.attn_v.weight", Dims: []uint64{32, 32}, Type: ggufload.TensorBF16},
		{Name: "output_norm.weight", Dims: []uint64{32}, Type: ggufload.TensorF32},
	}
	cfg := fakmodel.Config{ModelType: "llama", NumLayers: 1, HiddenSize: 32, NumHeads: 1}
	block, refs := buildDeviceReport(table, "sm_103", denseKQuantDecision{}.retained, tensors, nil, cfg)
	for _, tensor := range tensors[:3] {
		if got := block.Classes[tensor.Name]; got != deviceClassAbsent {
			t.Errorf("%s class = %q, want absent", tensor.Name, got)
		}
	}
	if got := block.Classes["output_norm.weight"]; got != deviceClassHost {
		t.Errorf("norm class = %q, want host", got)
	}
	if len(block.Gaps) != 1 || block.Gaps[0].Group != "cuda:matmul(q8_0)" || len(block.Gaps[0].Tensors) != 3 {
		t.Errorf("gaps = %+v, want one Q8 gap containing all three projections", block.Gaps)
	}
	if len(refs) != 1 || string(refs[0].Reason) != "UNSUPPORTED_DTYPE" || refs[0].Axis != "cuda:matmul(q8_0)" {
		t.Errorf("refusals = %+v, want one unsupported Q8 matmul refusal", refs)
	}
}
