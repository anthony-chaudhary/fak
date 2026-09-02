package compute

import (
	"errors"
	"reflect"
	"testing"
)

// The entry helper must validate geometry over the caller's tensor (the
// sequence path flattens the panel) while the returned operand list keeps the
// struct's own normalizedInput, so backend residency checks see the tensor the
// kernel launch will actually read.
func TestQwen35GDNEntryValidatesGeometryAndBuildsOperands(t *testing.T) {
	const hidden, nK, nV, kHd, vHd, kernel = 8, 1, 2, 2, 2, 1
	tensors := qwen35GDNGeometryTensors(hidden, nK, nV, kHd, vHd, kernel)
	in := qwen35GDNInputs{
		tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
		tensors[5], tensors[6], tensors[7], tensors[8], tensors[9],
		tensors[10], tensors[11],
	}

	panel := tensors[0]
	flattened := panel
	flattened.Shape = []int{hidden}

	gotHidden, gotKeyDim, gotValueDim, gotConvDim, operands, err := in.entry(flattened, nK, nV, kHd, vHd, kernel, 1e-5)
	if err != nil {
		t.Fatalf("entry refused valid geometry: %v", err)
	}
	wantHidden, wantKeyDim, wantValueDim, wantConvDim, directErr := validateQwen35GDNGeometry(
		flattened, tensors[1], tensors[2], tensors[3], tensors[4],
		tensors[5], tensors[6], tensors[7], tensors[8], tensors[9],
		tensors[10], tensors[11], nK, nV, kHd, vHd, kernel, 1e-5,
	)
	if directErr != nil {
		t.Fatalf("direct geometry validation refused the same tensors: %v", directErr)
	}
	if gotHidden != wantHidden || gotKeyDim != wantKeyDim || gotValueDim != wantValueDim || gotConvDim != wantConvDim {
		t.Fatalf("entry dims = (%d,%d,%d,%d), want (%d,%d,%d,%d)",
			gotHidden, gotKeyDim, gotValueDim, gotConvDim, wantHidden, wantKeyDim, wantValueDim, wantConvDim)
	}

	want := qwen35GDNOperands(
		panel, tensors[1], tensors[2], tensors[3], tensors[4],
		tensors[5], tensors[6], tensors[7], tensors[8], tensors[9],
		tensors[10], tensors[11],
	)
	if len(operands) != len(want) {
		t.Fatalf("entry built %d operands, want %d", len(operands), len(want))
	}
	for i := range want {
		if operands[i].name != want[i].name || !reflect.DeepEqual(operands[i].t, want[i].t) {
			t.Fatalf("operand %d = {%s %#v}, want {%s %#v}", i, operands[i].name, operands[i].t, want[i].name, want[i].t)
		}
	}
}

func TestQwen35GDNEntryPropagatesGeometryRefusal(t *testing.T) {
	const hidden, nK, nV, kHd, vHd, kernel = 8, 1, 2, 2, 2, 1
	tensors := qwen35GDNGeometryTensors(hidden, nK, nV, kHd, vHd, kernel)
	in := qwen35GDNInputs{
		tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
		tensors[5], tensors[6], tensors[7], tensors[8], tensors[9],
		tensors[10], tensors[11],
	}

	_, _, _, _, operands, err := in.entry(tensors[0], nK, nV, kHd, vHd, kernel, float32(0))
	if err == nil {
		t.Fatal("entry accepted a non-positive rms_norm_epsilon")
	}
	var geometry *Qwen35GDNGeometryError
	if !errors.As(err, &geometry) || geometry.Operand != "rms_norm_epsilon" {
		t.Fatalf("entry error = %T %v, want rms_norm_epsilon geometry refusal", err, err)
	}
	if operands != nil {
		t.Fatalf("entry returned operands alongside a refusal: %d", len(operands))
	}
}
