package trajectory

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
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

// fak-test:runtime fast est=50ms
func TestAuditAntigravityOverheadAndChainedBenignNegative(t *testing.T) {
	dir := t.TempDir()
	logDir := filepath.Join(dir, "bbbbbbbb-cccc-dddd-eeee-ffffffffffff", ".system_generated", "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatal(err)
	}
	lines := strings.Join([]string{
		`{"step_index":0,"source":"SYSTEM","type":"ERROR_MESSAGE","status":"DONE","error":"API error (attempt 1): HTTP 429","created_at":"2026-11-27T06:43:42Z"}`,
		`{"step_index":1,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DONE","model":"gemini-4-neon-medium","input_tokens":500,"output_tokens":50,"cache_read_tokens":1000,"tool_calls":[{"name":"run_command","args":{"CommandLine":"cd c:\\work\\fak; rg -n MissingSymbol internal/"}},{"name":"run_command","args":{"CommandLine":"go test ./internal/trajectory"}},{"name":"manage_task","args":{"Action":"status","TaskId":"t1"}},{"name":"read_file","args":{"Path":"C:\\Users\\USER\\.gemini\\antigravity\\brain\\id\\.system_generated\\steps\\9\\output.txt"}}]}`,
		`{"step_index":2,"source":"MODEL","type":"GENERIC","status":"DONE","content":"Command exited with code 1."}`,
		`{"step_index":3,"source":"MODEL","type":"GENERIC","status":"DONE","content":"fak-run: STALE fak-sync: slot abc is 2 commit(s) behind HEAD def\nTool is running as a background task with task id: t1"}`,
		`{"step_index":4,"source":"MODEL","type":"GENERIC","status":"DONE","content":"Task status: running"}`,
		`{"step_index":5,"source":"MODEL","type":"GENERIC","status":"DONE","content":"file contents"}`,
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(logDir, "transcript.jsonl"), []byte(lines), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceAntigravity, Root: dir, RootLabel: "antigravity/brain"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 0 {
		t.Fatalf("unexpected refusals: %+v", result.Refusals)
	}
	for _, d := range result.SchemaDrift {
		if d.Source == AuditSourceAntigravity && d.EventType == "ERROR_MESSAGE" {
			t.Fatalf("unexpected ERROR_MESSAGE schema drift: %+v", d)
		}
	}
	row := auditSession(t, result, AuditSourceAntigravity)
	if row.ToolErrors != 0 || row.ToolErrorsBenignNegative != 1 {
		t.Fatalf("errors=%d benign=%d, want 0/1 (chained rg exit 1 must be benign)", row.ToolErrors, row.ToolErrorsBenignNegative)
	}
	if row.AntigravityBackgroundedCommands != 1 || row.AntigravityManageTaskStatusCalls != 1 || row.AntigravityStaleBannerHits != 1 || row.AntigravitySpillOutputReads != 1 {
		t.Fatalf("antigravity overhead = bg:%d mt:%d stale:%d spill:%d, want 1/1/1/1",
			row.AntigravityBackgroundedCommands, row.AntigravityManageTaskStatusCalls, row.AntigravityStaleBannerHits, row.AntigravitySpillOutputReads)
	}
	if result.Summary.UnknownExemplars.Retained != 0 {
		t.Fatalf("ERROR_MESSAGE must not produce unknown exemplars, got %+v", result.Summary.UnknownExemplars)
	}
	var md bytes.Buffer
	if err := WriteAuditMarkdown(&md, result); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(md.String(), "Antigravity overhead: backgrounded commands: 1; manage_task(status) polls: 1; fak-run STALE banners: 1; output.txt spill reads: 1.") {
		t.Fatalf("markdown missing Antigravity overhead line:\n%s", md.String())
	}
}
