package amdgpu

import (
	"slices"
	"testing"
)

// TestValidationSubkernelSelectionDefaultEarnsCredit binds the default
// --subkernels selection to the CreditEligible contract: a fully passing run of
// exactly the default selection must earn physical credit, while "all" stays
// the broad list and an explicit list passes through unchanged.
func TestValidationSubkernelSelectionDefaultEarnsCredit(t *testing.T) {
	receiptFor := func(t *testing.T, names []string) *StrixValidationReceipt {
		t.Helper()
		receipt := NewStrixValidationReceipt(testStrixTarget(), "HEAD", testTip, "fak validate --strix")
		for _, name := range names {
			event := NewCosineMaxAbsParityEvent("fak-native/vulkan", 1, true, 0.999995, 0.999900, 0.0012, 0.01)
			if name == "argmax" {
				event = NewExactArgmaxParityEvent("fak-native/vulkan", 1, true, true)
			}
			receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
				Name: name, Status: "PASS", DurationUS: 1, Iterations: 1,
				ParityEvents: []StrixParityEvent{event},
			})
		}
		sealReceipt(t, receipt)
		if err := receipt.Validate(); err != nil {
			t.Fatalf("test receipt must remain generally valid: %v", err)
		}
		return receipt
	}

	def := ValidationSubkernelSelection("")
	if len(def) != 1 || def[0] != "argmax" {
		t.Fatalf("default selection = %q, want [argmax]", def)
	}
	if r := receiptFor(t, def); !r.CreditEligible() || r.SelectedSubkernels != 1 {
		t.Fatalf("default selection receipt: credit=%v selected=%d, want credit with 1 selected", r.CreditEligible(), r.SelectedSubkernels)
	}

	all := ValidationSubkernelSelection("all")
	if !slices.Equal(all, DefaultCreditableSubkernelSelectors) || len(all) < 2 {
		t.Fatalf("\"all\" selection = %q, want the broad DefaultCreditableSubkernelSelectors", all)
	}

	if got := ValidationSubkernelSelection("argmax,rope"); !slices.Equal(got, []string{"argmax", "rope"}) {
		t.Fatalf("explicit selection = %q, want [argmax rope]", got)
	}

	def[0] = "mutated"
	if CreditSubkernelSelectors[0] != "argmax" {
		t.Fatal("ValidationSubkernelSelection must return a copy of CreditSubkernelSelectors")
	}
}
