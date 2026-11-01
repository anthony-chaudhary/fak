package trajectory

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms
func TestAuditPiFixture(t *testing.T) {
	root := filepath.Join("testdata", "audit", "pi", "sessions")
	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourcePi, Root: root, RootLabel: "pi/sessions"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Refusals) != 0 {
		t.Fatalf("fixture must parse without refusals: %+v", result.Refusals)
	}
	denominator := result.Denominators[0]
	if denominator.Source != AuditSourcePi || !denominator.RootPresent || denominator.FilesDiscovered != 1 || denominator.RefusedRecords != 0 {
		t.Fatalf("denominator = %+v", denominator)
	}
	if denominator.UsageRecordsExact != 8 || denominator.UsageRecordsSeen != 8 {
		t.Fatalf("usage records exact/seen = %d/%d, want 8/8 (seven assistant turns + one compaction)", denominator.UsageRecordsExact, denominator.UsageRecordsSeen)
	}
	row := auditSession(t, result, AuditSourcePi)
	if row.TranscriptID != "pi-fixture-0001" {
		t.Fatalf("transcript id = %q", row.TranscriptID)
	}
	wantTokens := AuditTokens{InputTokens: 1840, OutputTokens: 480, CacheReadTokens: 3140, CacheCreateTokens: 15}
	if row.Tokens != wantTokens {
		t.Fatalf("tokens = %+v, want %+v", row.Tokens, wantTokens)
	}
	if row.ToolCalls != 5 {
		t.Fatalf("tool calls = %d, want 5", row.ToolCalls)
	}
	if row.ToolErrors != 1 {
		t.Fatalf("tool errors = %d, want 1 (the go test failure)", row.ToolErrors)
	}
	if row.ToolErrorsBenignNegative != 2 {
		t.Fatalf("benign negatives = %d, want 2 (rg no-match exit 1 + git lock notice)", row.ToolErrorsBenignNegative)
	}
	if len(row.CompactionEvents) != 1 || row.CompactionEvents[0].Unit != "pi_compaction_record_utf8_bytes" || row.CompactionEvents[0].Origin != "pi_compaction_record" {
		t.Fatalf("compaction events = %+v", row.CompactionEvents)
	}
	if row.RepeatedFailures != 0 {
		t.Fatalf("repeated failures = %d, want 0 (every failing call is unique)", row.RepeatedFailures)
	}
	if len(row.Models) != 1 || row.Models[0] != "claude-opus-4-8" {
		t.Fatalf("models = %v", row.Models)
	}
	if result.ConclusionStatus.SchemaDriftCount != 0 || result.ConclusionStatus.BreakingSchemaDrift != 0 {
		t.Fatalf("schema drift status = %+v, want zero drift against the embedded baseline", result.ConclusionStatus)
	}
	assertPiDistribution(t, row)
}

func assertPiDistribution(t *testing.T, row AuditTranscriptRow) {
	t.Helper()
	category := map[string]int64{}
	for _, d := range row.Distribution {
		category[d.Name] += d.Bytes
	}
	for _, want := range []string{"user_message", "assistant_message", "reasoning", "tool_call", "tool_result"} {
		if category[want] == 0 {
			t.Fatalf("distribution missing visible category %q: %+v", want, category)
		}
	}
	storage := map[string]*AuditStorageRow{}
	for i := range row.StorageDistribution {
		storage[row.StorageDistribution[i].Subtype] = &row.StorageDistribution[i]
	}
	synthetic, ok := storage["message/user_synthetic"]
	if !ok || synthetic.Records != 1 {
		t.Fatalf("synthetic user line must be storage-only with one record: %+v", storage["message/user_synthetic"])
	}
	if system, ok := storage["message/system"]; !ok || system.Records != 1 {
		t.Fatalf("system preamble must be storage-only: %+v", storage["message/system"])
	}
	for _, subtype := range []string{"session", "model_change", "thinking_level_change", "compaction", "custom/fak-router-autocontinue"} {
		if row_, ok := storage[subtype]; !ok || row_.Records != 1 {
			t.Fatalf("envelope %q must be stored exactly once: %+v", subtype, storage[subtype])
		}
	}
	results := map[string]*AuditToolResultRow{}
	for i := range row.ToolResults {
		results[row.ToolResults[i].Name] = &row.ToolResults[i]
	}
	read, ok := results["read"]
	if !ok || read.Success != 1 {
		t.Fatalf("read result = %+v", read)
	}
	bash, ok := results["bash"]
	// All three bash probes report wire errors (isError); the parse plane
	// splits them into one real failure plus two benign negatives.
	if !ok || bash.Results != 3 || bash.Success != 0 || bash.Errors != 3 {
		t.Fatalf("bash results = %+v, want 3 results / 0 success / 3 wire errors", bash)
	}
	write, ok := results["write"]
	if !ok || write.Success != 1 {
		t.Fatalf("write result = %+v", write)
	}
}

func TestAuditPiUserContainsFiltersSynthetic(t *testing.T) {
	root := filepath.Join("testdata", "audit", "pi", "sessions")
	matched, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourcePi, Root: root, RootLabel: "pi/sessions"}}, UserContains: "flaky test in cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if len(matched.Transcripts) != 1 {
		t.Fatalf("operator prompt selection = %d transcripts, want 1", len(matched.Transcripts))
	}
	synthetic, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourcePi, Root: root, RootLabel: "pi/sessions"}}, UserContains: "fak-router:auto-continue"})
	if err != nil {
		t.Fatal(err)
	}
	if len(synthetic.Transcripts) != 0 {
		t.Fatalf("synthetic harness injection selected %d transcripts, want 0", len(synthetic.Transcripts))
	}
}

// fak-test:runtime fast est=50ms
func TestAuditPiParseTable(t *testing.T) {
	session := `{"type":"session","id":"pi-table","cwd":"/w","timestamp":"2026-11-01T00:00:00Z","version":3}`
	assistant := func(id string, usage string, content string) string {
		return `{"type":"message","id":"` + id + `","parentId":"pi-table","timestamp":"2026-11-01T00:00:01Z","message":{"role":"assistant","model":"m","provider":"p","stopReason":"end_turn","timestamp":1,"content":` + content + `,` + usage + `}}`
	}
	usage := `"usage":{"input":1,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":2,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}`
	for _, tc := range []struct {
		name         string
		lines        []string
		wantRefusal  string
		wantTeleport bool
		wantBenign   int
		wantErrors   int
		wantRepeated int
	}{
		{
			name:        "malformed_line_refuses",
			lines:       []string{session, `{not json`},
			wantRefusal: "malformed_json",
		},
		{
			name:         "unknown_record_type_breaks_schema",
			lines:        []string{session, `{"type":"teleport","id":"t1","timestamp":"2026-11-01T00:00:02Z"}`},
			wantTeleport: true,
		},
		{
			name:        "unsupported_session_version_refuses",
			lines:       []string{`{"type":"session","id":"pi-v99","cwd":"/w","timestamp":"2026-11-01T00:00:00Z","version":99}`},
			wantRefusal: "pi_session_version_unsupported",
		},
		{
			name: "duplicate_usage_mismatch_refuses",
			lines: []string{
				session,
				assistant("a1", usage, `[{"type":"text","text":"one"}]`),
				assistant("a1", `"usage":{"input":9,"output":9,"cacheRead":0,"cacheWrite":0,"totalTokens":18,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}`, `[{"type":"text","text":"two"}]`),
			},
			wantRefusal: "pi_duplicate_usage_mismatch",
		},
		{
			name:       "search_no_match_is_benign_negative",
			lines:      []string{session, assistant("a1", usage, `[{"type":"toolCall","id":"c1","name":"bash","arguments":{"command":"rg -n nope nowhere"}}]`), `{"type":"message","id":"r1","parentId":"a1","timestamp":"2026-11-01T00:00:02Z","message":{"role":"toolResult","toolCallId":"c1","toolName":"bash","isError":true,"timestamp":2,"content":[{"type":"text","text":"(no output)\n\nCommand exited with code 1"}]}}`},
			wantBenign: 1,
		},
		{
			name:       "real_failure_stays_error",
			lines:      []string{session, assistant("a1", usage, `[{"type":"toolCall","id":"c1","name":"bash","arguments":{"command":"go test ./..."}}]`), `{"type":"message","id":"r1","parentId":"a1","timestamp":"2026-11-01T00:00:02Z","message":{"role":"toolResult","toolCallId":"c1","toolName":"bash","isError":true,"timestamp":2,"content":[{"type":"text","text":"--- FAIL: TestX (0.1s)\nFAIL\n\nCommand exited with code 1"}]}}`},
			wantErrors: 1,
		},
		{
			name: "timeout_warning_is_benign_negative",
			lines: []string{
				session,
				assistant("a1", usage, `[{"type":"toolCall","id":"c1","name":"bash","arguments":{"command":"sleep 999"}}]`),
				`{"type":"message","id":"r1","parentId":"a1","timestamp":"2026-11-01T00:00:02Z","message":{"role":"toolResult","toolCallId":"c1","toolName":"bash","isError":true,"timestamp":2,"content":[{"type":"text","text":"Command timed out after 30 seconds"}]}}`,
			},
			wantBenign: 1,
		},
		{
			name: "git_lock_notice_is_benign_negative",
			lines: []string{
				session,
				assistant("a1", usage, `[{"type":"toolCall","id":"c1","name":"bash","arguments":{"command":"git commit -m x"}}]`),
				`{"type":"message","id":"r1","parentId":"a1","timestamp":"2026-11-01T00:00:02Z","message":{"role":"toolResult","toolCallId":"c1","toolName":"bash","isError":true,"timestamp":2,"content":[{"type":"text","text":"fatal: Unable to create '.git/index.lock': File exists.\nAnother git process seems to be running in this repository.\n\nCommand exited with code 1"}]}}`,
			},
			wantBenign: 1,
		},
		{
			name: "non_search_exit_one_with_output_stays_error",
			lines: []string{
				session,
				assistant("a1", usage, `[{"type":"toolCall","id":"c1","name":"bash","arguments":{"command":"make build"}}]`),
				`{"type":"message","id":"r1","parentId":"a1","timestamp":"2026-11-01T00:00:02Z","message":{"role":"toolResult","toolCallId":"c1","toolName":"bash","isError":true,"timestamp":2,"content":[{"type":"text","text":"# github.com/x/y\ny.go:1:2: undefined: z\n\nCommand exited with code 2"}]}}`,
			},
			wantErrors: 1,
		},
		{
			name: "compaction_resets_repeat_family",
			lines: []string{
				session,
				assistant("a1", usage, `[{"type":"toolCall","id":"c1","name":"bash","arguments":{"command":"false"}}]`),
				`{"type":"message","id":"r1","parentId":"a1","timestamp":"2026-11-01T00:00:02Z","message":{"role":"toolResult","toolCallId":"c1","toolName":"bash","isError":true,"timestamp":2,"content":[{"type":"text","text":"boom\n\nCommand exited with code 1"}]}}`,
				assistant("a2", usage, `[{"type":"toolCall","id":"c2","name":"bash","arguments":{"command":"false"}}]`),
				`{"type":"message","id":"r2","parentId":"a2","timestamp":"2026-11-01T00:00:03Z","message":{"role":"toolResult","toolCallId":"c2","toolName":"bash","isError":true,"timestamp":3,"content":[{"type":"text","text":"boom\n\nCommand exited with code 1"}]}}`,
				`{"type":"compaction","id":"c1x","parentId":"r2","firstKeptEntryId":"a2","summary":"compacted","tokensBefore":100,"timestamp":"2026-11-01T00:00:04Z"}`,
				assistant("a3", usage, `[{"type":"toolCall","id":"c3","name":"bash","arguments":{"command":"false"}}]`),
				`{"type":"message","id":"r3","parentId":"a3","timestamp":"2026-11-01T00:00:05Z","message":{"role":"toolResult","toolCallId":"c3","toolName":"bash","isError":true,"timestamp":5,"content":[{"type":"text","text":"boom\n\nCommand exited with code 1"}]}}`,
			},
			// Two identical failures before the boundary settle into the
			// accumulated total; the post-boundary failure starts a fresh
			// segment and cannot repeat it. All three are real errors: the
			// body is not empty and the command is not a search probe.
			wantErrors:   3,
			wantRepeated: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "table.jsonl")
			var builder strings.Builder
			for _, line := range tc.lines {
				builder.WriteString(line)
				builder.WriteByte('\n')
			}
			if err := os.WriteFile(path, []byte(builder.String()), 0o600); err != nil {
				t.Fatal(err)
			}
			result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourcePi, Root: dir, RootLabel: "pi/table"}}})
			if err != nil {
				t.Fatal(err)
			}
			if tc.wantRefusal != "" {
				if len(result.Refusals) == 0 || result.Refusals[0].Code != tc.wantRefusal {
					t.Fatalf("refusals = %+v, want code %q", result.Refusals, tc.wantRefusal)
				}
				return
			}
			if len(result.Refusals) != 0 {
				t.Fatalf("unexpected refusals: %+v", result.Refusals)
			}
			if tc.wantTeleport {
				var found bool
				for _, row := range result.SchemaDrift {
					if row.Source == AuditSourcePi && row.EventType == "teleport" && row.Change == "new_event" && row.Compatibility == "breaking" {
						found = true
						if row.ParserSurface != "internal/trajectory/audit_parse.go:parsePiAuditRecord" || row.FixtureSurface != "internal/trajectory/testdata/audit/pi/sessions" {
							t.Fatalf("drift ownership = %q / %q", row.ParserSurface, row.FixtureSurface)
						}
					}
				}
				if !found {
					t.Fatalf("missing breaking new_event drift for teleport: %+v", result.SchemaDrift)
				}
				return
			}
			row := auditSession(t, result, AuditSourcePi)
			if row.ToolErrorsBenignNegative != tc.wantBenign {
				t.Fatalf("benign negatives = %d, want %d", row.ToolErrorsBenignNegative, tc.wantBenign)
			}
			if row.ToolErrors != tc.wantErrors {
				t.Fatalf("tool errors = %d, want %d", row.ToolErrors, tc.wantErrors)
			}
			if row.RepeatedFailures != tc.wantRepeated {
				t.Fatalf("repeated failures = %d, want %d", row.RepeatedFailures, tc.wantRepeated)
			}
		})
	}
}

func TestAuditPiBenignNegativeDirect(t *testing.T) {
	call := func(name, args string) auditToolCall { return auditToolCall{name: name, args: args} }
	for _, tc := range []struct {
		name       string
		call       auditToolCall
		content    string
		wantBenign bool
	}{
		{name: "rg_direct_tool_no_match", call: call("rg", `null`), content: "(no output)\n\nCommand exited with code 1", wantBenign: true},
		{name: "grep_direct_tool_no_match", call: call("grep", `null`), content: "\nCommand exited with code 1", wantBenign: true},
		{name: "bash_rg_pipe_no_match", call: call("bash", `{"command":"rg -n flaky | wc -l"}`), content: "(no output)\n\nCommand exited with code 1", wantBenign: true},
		{name: "bash_grep_no_match", call: call("bash", `{"command":"grep -rn needle ."}`), content: "(no output)\n\nCommand exited with code 1", wantBenign: true},
		{name: "bash_rmdir_exit_one_empty_body_not_probe", call: call("bash", `{"command":"rmdir empty-dir"}`), content: "(no output)\n\nCommand exited with code 1", wantBenign: false},
		{name: "exit_one_with_output_never_benign", call: call("rg", `null`), content: "error: path 'x' does not exist\n\nCommand exited with code 1", wantBenign: false},
		{name: "zero_exit_never_benign", call: call("rg", `null`), content: "(no output)", wantBenign: false},
		{name: "timeout_warning", call: call("bash", `{"command":"sleep 999"}`), content: "Command timed out after 30 seconds", wantBenign: true},
		{name: "git_lock_on_git_command", call: call("bash", `{"command":"git commit -m x"}`), content: "fatal: Unable to create '.git/index.lock': File exists.\nAnother git process seems to be running.", wantBenign: true},
		{name: "lock_text_without_git_command_stays_error", call: call("bash", `{"command":"python build.py"}`), content: "fatal: Unable to create '.git/index.lock': File exists.\nAnother git process seems to be running.", wantBenign: false},
		{name: "empty_content", call: call("bash", `{"command":"false"}`), content: "", wantBenign: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := auditPiBenignNegative(tc.call, tc.content); got != tc.wantBenign {
				t.Fatalf("auditPiBenignNegative(%q, %q) = %t, want %t", tc.call.name, tc.content, got, tc.wantBenign)
			}
		})
	}
}
