package issueorchestrator

import (
	"encoding/json"
	"testing"
)

// TestDecodeIssues_GitHubBodyWithEmptyLabels pins issue #1123: a real GitHub
// issue row carrying a routing contract in "body" must be decoded through the
// IssueDraft path even when "labels" is an empty array. The internal []Issue
// transport has no body field, so the prior first-match decode silently
// discarded key/lane/paths and produced a zero-step, unenforced candidate.
func TestDecodeIssues_GitHubBodyWithEmptyLabels(t *testing.T) {
	body := "## Observed failure\n\nRouter admits roster-bound requests late.\n\n" +
		"```routing\n" +
		"lane: platform/gateway\n" +
		"paths: platform/gateway/admission.go, platform/gateway/admission_test.go\n" +
		"expected_steps: 3\n" +
		"priority: P1\n" +
		"```\n\n" +
		"## Acceptance Criteria\n- [ ] 1. Admit roster-bound requests\n"

	row := map[string]any{
		"number": 12886,
		"state":  "OPEN",
		"title":  "feat(gateway): admit roster-bound chat requests before EP fanout",
		"url":    "https://github.com/anthony-chaudhary/fak/issues/12886",
		"labels": []any{},
		"body":   body,
	}
	encoded, err := json.Marshal([]any{row})
	if err != nil {
		t.Fatal(err)
	}

	issues, err := DecodeIssues(encoded)
	if err != nil {
		t.Fatalf("DecodeIssues: %v", err)
	}
	if len(issues) != 1 {
		t.Fatalf("want 1 issue, got %d", len(issues))
	}
	got := issues[0]
	if got.Number != 12886 {
		t.Errorf("number = %d, want 12886", got.Number)
	}
	if got.Key == "" {
		t.Errorf("key is empty; body contract was not parsed")
	}
	if got.Lane != "platform/gateway" {
		t.Errorf("lane = %q, want platform/gateway", got.Lane)
	}
	if len(got.Paths) != 2 {
		t.Errorf("paths = %v, want two exact paths", got.Paths)
	}
	if got.ExpectedSteps != 3 {
		t.Errorf("expected_steps = %d, want 3", got.ExpectedSteps)
	}
	if got.Dispatchability == "" {
		t.Errorf("dispatchability is empty; contract not enforced")
	}
}

// TestDecodeIssues_DirectTransportUnchanged keeps a literal internal []Issue
// payload (no body field) decoding through the direct transport, so the
// body-aware routing does not regress that schema.
func TestDecodeIssues_DirectTransportUnchanged(t *testing.T) {
	row := []byte(`[
  {
    "number": 42,
    "key": "issue-42",
    "title": "direct transport",
    "lane": "engine",
    "paths": ["internal/engine/batcher.go"],
    "expected_steps": 2,
    "labels": ["alpha", "beta"]
  }
]`)

	issues, err := DecodeIssues(row)
	if err != nil {
		t.Fatalf("DecodeIssues: %v", err)
	}
	if len(issues) != 1 {
		t.Fatalf("want 1 issue, got %d", len(issues))
	}
	if issues[0].Lane != "engine" || issues[0].ExpectedSteps != 2 {
		t.Fatalf("direct transport changed: %+v", issues[0])
	}
	if len(issues[0].Labels) != 2 {
		t.Fatalf("labels = %v, want 2", issues[0].Labels)
	}
}
