package policy

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestApplySuppressionsExactScopeAndStale(t *testing.T) {
	got := ApplySuppressions([]string{"secret", "destructive"},
		[]Finding{{Rule: "secret", Scope: "src/a.go"}, {Rule: "secret", Scope: "src/b.go"}, {Rule: "destructive", Scope: "cmd/x"}},
		[]Suppression{{Rule: "secret", Scope: "src/a.go", Reason: "test fixture"}, {Rule: "secret", Scope: "missing.go", Reason: "migration complete"}})
	if len(got.Remaining) != 2 || got.Remaining[0].Scope != "src/b.go" {
		t.Fatalf("exact scope not preserved: %+v", got.Remaining)
	}
	if !got.Suppressions[0].Used || got.Suppressions[0].Status != "used" {
		t.Fatalf("used record: %+v", got.Suppressions[0])
	}
	if got.Suppressions[1].Used || got.Suppressions[1].Status != "unused" {
		t.Fatalf("stale record: %+v", got.Suppressions[1])
	}
	if text := got.Text(); !strings.Contains(text, "USED rule=secret") || !strings.Contains(text, "UNUSED rule=secret") {
		t.Fatalf("human output hid records: %q", text)
	}
}

func TestApplySuppressionsRefusesInvalidWithoutSuppressing(t *testing.T) {
	finding := Finding{Rule: "secret", Scope: "src/a.go"}
	cases := []struct {
		name string
		s    Suppression
		want SuppressionRefusal
	}{
		{"unknown rule", Suppression{Rule: "secrett", Scope: "src/a.go", Reason: "typo"}, SuppressionUnknownRule},
		{"missing reason", Suppression{Rule: "secret", Scope: "src/a.go"}, SuppressionMissingReason},
		{"missing scope", Suppression{Rule: "secret", Reason: "broad"}, SuppressionMalformed},
		{"missing rule", Suppression{Scope: "src/a.go", Reason: "broad"}, SuppressionMalformed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ApplySuppressions([]string{"secret"}, []Finding{finding}, []Suppression{tc.s})
			if len(got.Remaining) != 1 || got.Remaining[0] != finding {
				t.Fatalf("invalid suppression hid finding: %+v", got)
			}
			if got.Suppressions[0].Status != "refused" || got.Suppressions[0].Refusal != tc.want {
				t.Fatalf("record=%+v want=%s", got.Suppressions[0], tc.want)
			}
		})
	}
}

func TestSuppressionReceiptFixtures(t *testing.T) {
	for _, name := range []string{"suppression_known_positive.json", "suppression_adversarial.json"} {
		b, err := os.ReadFile("testdata/" + name)
		if err != nil {
			t.Fatal(err)
		}
		var receipt SuppressionResult
		if err := json.Unmarshal(b, &receipt); err != nil {
			t.Fatal(err)
		}
		if receipt.Schema != "fak-policy-suppressions/1" || len(receipt.Suppressions) == 0 {
			t.Fatalf("invalid receipt %s: %+v", name, receipt)
		}
		if name == "suppression_known_positive.json" && (!receipt.Suppressions[0].Used || len(receipt.Remaining) != 0) {
			t.Fatalf("positive cannot succeed: %+v", receipt)
		}
		if name == "suppression_adversarial.json" && (receipt.Suppressions[0].Refusal != SuppressionUnknownRule || len(receipt.Remaining) != 1) {
			t.Fatalf("adversarial passed vacuously: %+v", receipt)
		}
	}
}

func TestApplySuppressionsMultipleRulesPublicJSON(t *testing.T) {
	got := ApplySuppressions([]string{"a", "b"}, []Finding{{"a", "x"}, {"b", "y"}}, []Suppression{{"a", "x", "one"}, {"b", "y", "two"}})
	b, err := got.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var shape map[string]any
	if err := json.Unmarshal(b, &shape); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"schema", "findings", "remaining", "suppressions"} {
		if _, ok := shape[key]; !ok {
			t.Fatalf("missing public field %q: %s", key, b)
		}
	}
	if len(got.Remaining) != 0 {
		t.Fatalf("multiple rules not honored: %+v", got)
	}
}
