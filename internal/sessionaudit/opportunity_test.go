package sessionaudit

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func opportunityRecord(t *testing.T, blocks []map[string]any) string {
	t.Helper()
	content, err := json.Marshal(blocks)
	if err != nil {
		t.Fatal(err)
	}
	record, err := json.Marshal(map[string]any{"message": map[string]any{"content": json.RawMessage(content)}})
	if err != nil {
		t.Fatal(err)
	}
	return string(record) + "\n"
}

func TestOpportunityByToolAttributesResultsAndRanksStably(t *testing.T) {
	var transcript strings.Builder
	transcript.WriteString(opportunityRecord(t, []map[string]any{
		{"type": "tool_use", "id": "go-1", "name": "Bash", "input": map[string]any{"command": "GOTOOLCHAIN=auto go test ./..."}},
		{"type": "tool_use", "id": "tf-1", "name": "Bash", "input": map[string]any{"command": "terraform plan"}},
		{"type": "tool_use", "id": "read-1", "name": "Read", "input": map[string]any{"file_path": "README.md"}},
	}))
	goBody := strings.Repeat("=== RUN   TestRoutine\n", 12) + "ok  example/pkg\n"
	tfBody := strings.Repeat("Refreshing state...\n", 5) + "Plan: 1 to add.\n"
	transcript.WriteString(opportunityRecord(t, []map[string]any{
		{"type": "tool_result", "tool_use_id": "tf-1", "content": tfBody},
		{"type": "tool_result", "tool_use_id": "go-1", "content": []map[string]any{{"type": "text", "text": goBody}}},
		{"type": "tool_result", "tool_use_id": "read-1", "content": "same\nsame\n"},
		{"type": "tool_result", "tool_use_id": "missing", "content": strings.Repeat("unpaired\n", 100)},
	}))

	got := OpportunityByTool([]byte(transcript.String()))
	if len(got) != 3 {
		t.Fatalf("rows = %#v", got)
	}
	if got[0].Tool != "go" || got[0].Calls != 1 || got[0].RawBytes != int64(len(goBody)) {
		t.Fatalf("first row = %#v", got[0])
	}
	if got[0].EstCompressible != int64(11*len("=== RUN   TestRoutine\n")) {
		t.Fatalf("go estimate = %d", got[0].EstCompressible)
	}
	if got[1].Tool != "terraform" || got[2].Tool != "read" {
		t.Fatalf("ordering = %#v", got)
	}
}

func TestOpportunityByToolAggregatesCallsAndBreaksTiesByName(t *testing.T) {
	var transcript strings.Builder
	for i, command := range []string{"beta run", "alpha run", "alpha again"} {
		id := fmt.Sprintf("id-%d", i)
		transcript.WriteString(opportunityRecord(t, []map[string]any{{"type": "tool_use", "id": id, "name": "shell", "input": map[string]any{"command": command}}}))
		transcript.WriteString(opportunityRecord(t, []map[string]any{{"type": "tool_result", "tool_use_id": id, "content": "repeat\nrepeat\n"}}))
	}
	got := OpportunityByTool([]byte(transcript.String()))
	if len(got) != 2 || got[0].Tool != "alpha" || got[0].Calls != 2 || got[1].Tool != "beta" {
		t.Fatalf("rows = %#v", got)
	}
}

func TestOpportunityByToolIsPureAndMalformedInputIsIgnored(t *testing.T) {
	input := []byte("not json\n" + opportunityRecord(t, []map[string]any{{"type": "tool_result", "tool_use_id": "absent", "content": "secret"}}))
	before := append([]byte(nil), input...)
	first := OpportunityByTool(input)
	second := OpportunityByTool(input)
	if len(first) != 0 || len(second) != 0 || string(input) != string(before) {
		t.Fatalf("first=%#v second=%#v input mutated=%v", first, second, string(input) != string(before))
	}
}

func TestLeadingCommandToken(t *testing.T) {
	for input, want := range map[string]string{
		"GOTOOLCHAIN=auto go test ./...": "go",
		"terraform plan":                 "terraform",
		"pnpm.exe test":                  "pnpm",
		"":                               "",
	} {
		if got := leadingCommandToken(input); got != want {
			t.Errorf("leadingCommandToken(%q) = %q, want %q", input, got, want)
		}
	}
}
