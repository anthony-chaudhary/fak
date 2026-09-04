package quantpolicy

import (
	"encoding/json"
	"os"
	"testing"
)

func TestQuantPolicySanity(t *testing.T) {
	policyRaw, err := os.ReadFile("testdata/policy.json")
	if err != nil {
		t.Fatal(err)
	}
	allowRaw, err := os.ReadFile("testdata/allow.json")
	if err != nil {
		t.Fatal(err)
	}
	var policy Policy
	if err := json.Unmarshal(policyRaw, &policy); err != nil {
		t.Fatal(err)
	}
	var request Request
	if err := json.Unmarshal(allowRaw, &request); err != nil {
		t.Fatal(err)
	}
	res := Evaluate(policy, request)
	if res.Outcome != OutcomeAllow {
		t.Fatalf("expected allow, got %v", res.Outcome)
	}
}

func BenchmarkQuantPolicy(b *testing.B) {
	policyRaw, err := os.ReadFile("testdata/policy.json")
	if err != nil {
		b.Fatal(err)
	}
	allowRaw, err := os.ReadFile("testdata/allow.json")
	if err != nil {
		b.Fatal(err)
	}
	var policy Policy
	if err := json.Unmarshal(policyRaw, &policy); err != nil {
		b.Fatal(err)
	}
	var request Request
	if err := json.Unmarshal(allowRaw, &request); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Evaluate(policy, request)
		if res.Outcome != OutcomeAllow {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}
