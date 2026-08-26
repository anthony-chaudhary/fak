package trajectory

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestApplyQwenToolErrorAttribution(t *testing.T) {
	events := []QwenToolErrorEvent{
		{Content: "permission denied", Index: 2, Tokens: 30},
		{Content: "permission denied", Index: 4, Tokens: 20},
		{Content: "deadline exceeded", Index: 6, Tokens: 10},
	}
	attrs := []qwenToolErrorAttribution{
		{failureKey: "edit\x00same", mutationTarget: "a.go"},
		{failureKey: "edit\x00same", mutationTarget: "a.go"},
		{failureKey: "read\x00other"},
	}
	applyQwenToolErrorAttribution(events, attrs, map[string]int{"a.go": 3})
	got := rankQwenToolErrorFamilies(events)
	if len(got) != 2 {
		t.Fatalf("families = %#v, want 2", got)
	}
	if got[0].Family != "permission" || got[0].Count != 2 || got[0].Tokens != 50 || got[0].RepeatedFailures != 1 || got[0].MutationChurn != 2 {
		t.Fatalf("permission family = %#v", got[0])
	}
	if got[1].Family != "timeout" || got[1].RepeatedFailures != 0 || got[1].MutationChurn != 0 {
		t.Fatalf("timeout family = %#v", got[1])
	}
}

func TestApplyQwenToolErrorAttributionRequiresJoinedEvidence(t *testing.T) {
	events := []QwenToolErrorEvent{{Content: "unknown failure", Index: 1}}
	applyQwenToolErrorAttribution(events, nil, map[string]int{"a.go": 9})
	got := rankQwenToolErrorFamilies(events)
	if len(got) != 1 || got[0].Family != "unknown" || got[0].MutationChurn != 0 || got[0].RepeatedFailures != 0 {
		t.Fatalf("families = %#v", got)
	}
}

func TestAuditAttributesRepeatedFailuresAndMutationChurnToFamilies(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "qwen.jsonl")
	rows := []string{
		`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"c1","name":"edit_file","input":{"path":"a.go"}}]}}`,
		`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"c1","is_error":true,"content":"permission denied"}]}}`,
		`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"c2","name":"edit_file","input":{"path":"a.go"}}]}}`,
		`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"c2","is_error":true,"content":"permission denied"}]}}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(rows, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceClaude, Root: dir}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolErrorFamilies) != 1 {
		t.Fatalf("families = %#v", result.ToolErrorFamilies)
	}
	got := result.ToolErrorFamilies[0]
	if got.Family != "permission" || got.Count != 2 || got.RepeatedFailures != 1 || got.MutationChurn != 1 {
		t.Fatalf("family = %#v", got)
	}
	if result.Summary.RepeatedFailures != got.RepeatedFailures || result.Summary.MutationChurn != got.MutationChurn {
		t.Fatalf("summary repeats/churn = %d/%d, family = %#v", result.Summary.RepeatedFailures, result.Summary.MutationChurn, got)
	}
}
