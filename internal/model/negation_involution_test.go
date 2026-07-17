package model

import (
	"encoding/json"
	"os"
	"testing"
)

type involutionABRow struct {
	Variant             string  `json:"variant"`
	InvolutionError     float64 `json:"involution_error"`
	DoubleNegationError float64 `json:"double_negation_error"`
	SyntheticAccuracy   float64 `json:"synthetic_accuracy"`
	Conclusion          string  `json:"conclusion"`
}

func TestInvolution(t *testing.T) {
	const tolerance = 1e-12
	for _, kind := range []NegationOperatorKind{NegationStrict, NegationPiRotation} {
		op, err := NewNegationOperator(kind, true)
		if err != nil {
			t.Fatal(err)
		}
		reg, err := op.InvolutionError()
		if err != nil {
			t.Fatal(err)
		}
		double, err := op.DoubleNegationError([]float64{.6, -.8})
		if err != nil {
			t.Fatal(err)
		}
		if reg > tolerance || double > tolerance {
			t.Fatalf("%s: ||N^2-I||=%g double=%g tolerance=%g", kind, reg, double, tolerance)
		}
		t.Logf("%s ||N^2-I||=%g ||N(N(x))-x||=%g tolerance=%g", kind, reg, double, tolerance)
	}
}
func TestInvolutionAB(t *testing.T) {
	data, err := os.ReadFile("testdata/negation_involution_ab.json")
	if err != nil {
		t.Fatal(err)
	}
	var want []involutionABRow
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	if len(want) != 3 {
		t.Fatalf("A/B rows=%d", len(want))
	}
	for _, row := range want {
		op, err := NewNegationOperator(NegationOperatorKind(row.Variant), true)
		if err != nil {
			t.Fatal(err)
		}
		reg, _ := op.InvolutionError()
		double, _ := op.DoubleNegationError([]float64{1, 0})
		if abs(reg-row.InvolutionError) > 1e-12 || abs(double-row.DoubleNegationError) > 1e-12 {
			t.Fatalf("%s witness drift: reg=%g double=%g want=%+v", row.Variant, reg, double, row)
		}
		once, _ := op.Apply([]float64{1, 0})
		twice, _ := op.Apply(once)
		acc := 0.0
		if once[0] < 0 {
			acc += .5
		}
		if abs(twice[0]-1) < 1e-12 && abs(twice[1]) < 1e-12 {
			acc += .5
		}
		if acc != row.SyntheticAccuracy {
			t.Fatalf("%s accuracy=%g want=%g", row.Variant, acc, row.SyntheticAccuracy)
		}
		t.Logf("variant=%s involution_error=%g double_negation_error=%g synthetic_accuracy=%.2f conclusion=%s", row.Variant, reg, double, acc, row.Conclusion)
	}
}
func TestInvolutionFlagOffIsIdentity(t *testing.T) {
	op, err := NewNegationOperator(NegationStrict, false)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := op.Apply([]float64{2, -3})
	if got[0] != 2 || got[1] != -3 {
		t.Fatalf("disabled operator=%v", got)
	}
}
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
