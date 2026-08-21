package agenticbench

import (
	"math"
	"strings"
	"testing"
)

func TestEvaluatorReceiptDistinguishesValidZeroFromFailure(t *testing.T) {
	zero := 0.0
	if err := (EvaluatorReceipt{Status: EvaluatorSuccess, Score: &zero, MinScore: 0, MaxScore: 1, DurationMS: 12}).Validate(); err != nil {
		t.Fatalf("valid zero refused: %v", err)
	}
	one := 1.0
	cases := []struct {
		name    string
		receipt EvaluatorReceipt
		reason  string
	}{
		{"failure with score", EvaluatorReceipt{Status: EvaluatorFailure, Score: &zero, ErrorRef: "log:1"}, "cannot carry score"},
		{"timeout without evidence", EvaluatorReceipt{Status: EvaluatorTimeout}, "requires error evidence"},
		{"success missing score", EvaluatorReceipt{Status: EvaluatorSuccess, MinScore: 0, MaxScore: 1}, "requires score"},
		{"nan", EvaluatorReceipt{Status: EvaluatorSuccess, Score: floatPtr(math.NaN()), MinScore: 0, MaxScore: 1}, "finite"},
		{"infinite", EvaluatorReceipt{Status: EvaluatorSuccess, Score: floatPtr(math.Inf(1)), MinScore: 0, MaxScore: 1}, "finite"},
		{"out of range", EvaluatorReceipt{Status: EvaluatorSuccess, Score: &one, MinScore: 0, MaxScore: .5}, "outside declared domain"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.receipt.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.reason) {
				t.Fatalf("err=%v, want %q", err, tc.reason)
			}
		})
	}
}
func floatPtr(v float64) *float64 { return &v }
