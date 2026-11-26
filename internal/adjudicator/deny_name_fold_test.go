package adjudicator

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=100ms lane=default
func TestIsNameDeniedFoldsRegisteredCase(t *testing.T) {
	a := New(Policy{Deny: map[string]abi.ReasonCode{"Bash": abi.ReasonPolicyBlock}})
	for _, name := range []string{"Bash", "bash", "BASH"} {
		if !a.IsNameDenied(name) {
			t.Errorf("IsNameDenied(%q) = false, want true for Deny on \"Bash\"", name)
		}
	}
	if a.IsNameDenied("cat") {
		t.Errorf("IsNameDenied(\"cat\") = true, want false")
	}
}
