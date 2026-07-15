package metrics

import "testing"

func TestPromptNegationTaxDerivedFromClassifier(t *testing.T) {
	low := PromptNegationTax("List apple and cherry.")
	high := PromptNegationTax("Do not include banana. Never mention banana.")
	if low != 0 || high <= low {
		t.Fatalf("low=%d high=%d", low, high)
	}
	if high != MeasureNegationTax("Do not include banana. Never mention banana.").Total {
		t.Fatalf("per-prompt tax diverged from MeasureNegationTax")
	}
}
