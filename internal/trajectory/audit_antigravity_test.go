package trajectory

import (
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=50ms
func TestAuditAntigravityFixture(t *testing.T) {
	root := filepath.Join("testdata", "audit", "antigravity", "brain")
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceAntigravity, Root: root, RootLabel: "antigravity/brain"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 0 {
		t.Fatalf("fixture must parse without refusals: %+v", result.Refusals)
	}
	denominator := result.Denominators[0]
	if denominator.Source != AuditSourceAntigravity || !denominator.RootPresent || denominator.FilesDiscovered != 1 || denominator.RefusedRecords != 0 {
		t.Fatalf("denominator = %+v", denominator)
	}
	if denominator.UsageRecordsExact != 4 || denominator.UsageRecordsSeen != 4 {
		t.Fatalf("usage records exact/seen = %d/%d, want 4/4 (three planner turns + one checkpoint)", denominator.UsageRecordsExact, denominator.UsageRecordsSeen)
	}
	row := auditSession(t, result, AuditSourceAntigravity)
	if row.TranscriptID != "11111111-2222-3333-4444-555555555555" {
		t.Fatalf("transcript id = %q", row.TranscriptID)
	}
	wantTokens := AuditTokens{InputTokens: 1700, OutputTokens: 300, CacheReadTokens: 5900}
	if row.Tokens != wantTokens {
		t.Fatalf("tokens = %+v, want %+v", row.Tokens, wantTokens)
	}
	if row.ToolCalls != 5 {
		t.Fatalf("tool calls = %d, want 5", row.ToolCalls)
	}
	if row.ToolErrors != 2 {
		t.Fatalf("tool errors = %d, want 2 (two go test failures)", row.ToolErrors)
	}
	if row.ToolErrorsBenignNegative != 1 {
		t.Fatalf("benign negatives = %d, want 1 (rg no-match exit 1)", row.ToolErrorsBenignNegative)
	}
	if row.RepeatedFailures != 1 {
		t.Fatalf("repeated failures = %d, want 1", row.RepeatedFailures)
	}
	if len(row.CompactionEvents) != 1 || row.CompactionEvents[0].Origin != "antigravity_checkpoint" {
		t.Fatalf("compaction events = %+v", row.CompactionEvents)
	}
	if len(row.Models) != 1 || row.Models[0] != "gemini-4-neon-medium" {
		t.Fatalf("models = %v", row.Models)
	}
	if result.ConclusionStatus.BreakingSchemaDrift != 0 {
		t.Fatalf("breaking schema drift = %d, want 0 (%+v)", result.ConclusionStatus.BreakingSchemaDrift, result.SchemaDrift)
	}

	results := map[string]*AuditToolResultRow{}
	for i := range row.ToolResults {
		results[row.ToolResults[i].Name] = &row.ToolResults[i]
	}
	rc, ok := results["run_command"]
	if !ok || rc.Truncated != 2 || rc.ExitNonzero != 3 {
		t.Fatalf("run_command tool results = %+v", rc)
	}

	// UserContains must match only USER_EXPLICIT/USER_INPUT, not SYSTEM_MESSAGE.
	matched, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceAntigravity, Root: root, RootLabel: "antigravity/brain"}}, UserContains: "flaky test in cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if len(matched.Transcripts) != 1 {
		t.Fatalf("user prompt match = %d, want 1", len(matched.Transcripts))
	}
	systemOnly, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceAntigravity, Root: root, RootLabel: "antigravity/brain"}}, UserContains: "qwen3.8"})
	if err != nil {
		t.Fatal(err)
	}
	if len(systemOnly.Transcripts) != 0 {
		t.Fatalf("system message must not match UserContains, got %d", len(systemOnly.Transcripts))
	}
}

// fak-test:runtime fast est=50ms
func TestAuditAntigravityRefusesMalformedUsage(t *testing.T) {
	dir := t.TempDir()
	logDir := filepath.Join(dir, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", ".system_generated", "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatal(err)
	}
	bad := `{"step_index":0,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DONE","input_tokens":-5,"output_tokens":10,"cache_read_tokens":0}` + "\n"
	if err := os.WriteFile(filepath.Join(logDir, "transcript.jsonl"), []byte(bad), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceAntigravity, Root: dir, RootLabel: "antigravity/brain"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 1 || result.Refusals[0].Code != "antigravity_usage_input_invalid" {
		t.Fatalf("refusals = %+v, want antigravity_usage_input_invalid", result.Refusals)
	}
}
