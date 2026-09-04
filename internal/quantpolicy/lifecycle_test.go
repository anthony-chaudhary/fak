package quantpolicy

import (
	"encoding/json"
	"os"
	"testing"
)

// Invariant: Quantization policy evaluations must strictly enforce format and precision bounds.
// Guard: Evaluate refuses requests outside declared min/max precision boundaries.

func TestQuantPolicyLifecycle(t *testing.T) {
	t.Parallel()

	policyRaw, err := os.ReadFile("testdata/policy.json")
	if err != nil {
		t.Fatalf("failed reading policy: %v", err)
	}
	requestRaw, err := os.ReadFile("testdata/allow.json")
	if err != nil {
		t.Fatalf("failed reading allow request: %v", err)
	}

	var policy Policy
	if err := json.Unmarshal(policyRaw, &policy); err != nil {
		t.Fatalf("failed unmarshaling policy: %v", err)
	}
	var request Request
	if err := json.Unmarshal(requestRaw, &request); err != nil {
		t.Fatalf("failed unmarshaling request: %v", err)
	}

	result := Evaluate(policy, request)
	if result.Outcome != OutcomeAllow {
		t.Fatalf("expected OutcomeAllow, got %s: %s", result.Outcome, result.Reason)
	}
}
