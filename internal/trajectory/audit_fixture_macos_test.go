package trajectory

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=20ms
func TestAuditIssue9828MacOSPytestClassification(t *testing.T) {
	t.Parallel()
	const filename = "00000000-aaaa-bbbb-cccc-000000000000.jsonl"
	const windowsProject = "C--Users-U-AppData-Local-Temp-pytest-of-U-pytest-1-test_main_json_runs_and_emits_0-ws"
	const windowsCWD = `C:\Users\U\AppData\Local\Temp\pytest-of-U\pytest-1\test_main_json_runs_and_emits_0\ws`
	// Relocate the pinned hermetic fixture, not a captured user's transcript.
	const macCWD = "/private/var/folders/zz/fixture_hash/T/pytest-of-U/pytest-1/test_main_json_runs_and_emits_0/ws"
	data, err := os.ReadFile(filepath.Join("testdata", "a", "i8508", "c", "p", windowsProject, filename))
	if err != nil {
		t.Fatal(err)
	}
	windowsJSON, err := json.Marshal(windowsCWD)
	if err != nil || !bytes.Contains(data, windowsJSON) {
		t.Fatalf("pinned fixture cwd changed: %v", err)
	}
	baseline := runPinnedAudit(t, nil)
	for _, tc := range []struct {
		name       string
		cwd        string
		old, new   string
		excluded   int
		exact      int
		refused    int
		transcript int
	}{
		{name: "hermetic_macos", cwd: macCWD, excluded: 1},
		{name: "exact_usage_is_counted", cwd: macCWD, old: `"usage": {}`, new: `"usage": {"input_tokens": 7, "output_tokens": 2}`, exact: 4, transcript: 1},
		{name: "live_identity_still_refuses", cwd: macCWD, old: `"type": "assistant"`, new: `"type": "assistant", "sessionId": "live-session"`, refused: 4, transcript: 1},
		{name: "other_pytest_test_still_refuses", cwd: strings.ReplaceAll(macCWD, "test_main_json_runs_and_emits_0", "test_unrelated0"), refused: 4, transcript: 1},
		{name: "other_temporary_directory_still_refuses", cwd: strings.ReplaceAll(macCWD, "pytest-of-U", "scratch-U"), refused: 4, transcript: 1},
		{name: "mismatched_cwd_still_refuses", cwd: macCWD, old: `"cwd": "/private/var/folders/zz/fixture_hash/T/pytest-of-U/pytest-1/test_main_json_runs_and_emits_0/ws"`, new: `"cwd": "/different/workspace"`, refused: 4, transcript: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cwdJSON, err := json.Marshal(tc.cwd)
			if err != nil {
				t.Fatal(err)
			}
			variant := bytes.Replace(data, windowsJSON, cwdJSON, 1)
			if tc.old != "" {
				if !bytes.Contains(variant, []byte(tc.old)) {
					t.Fatalf("fixture mutation target %q missing", tc.old)
				}
				variant = bytes.ReplaceAll(variant, []byte(tc.old), []byte(tc.new))
			}
			root := t.TempDir()
			path := filepath.Join(root, strings.ReplaceAll(tc.cwd, "/", "-"), filename)
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, variant, 0o644); err != nil {
				t.Fatal(err)
			}
			result, err := RunAudit(AuditOptions{Sources: []AuditSource{
				{Name: AuditSourceClaude, Root: filepath.Join("testdata", "audit", "claude", "projects"), RootLabel: "claude/projects"},
				{Name: AuditSourceCodex, Root: filepath.Join("testdata", "audit", "codex", "sessions"), RootLabel: "codex/sessions"},
				{Name: AuditSourceClaude, Root: root, RootLabel: "claude/projects/issue-9828"},
			}})
			if err != nil {
				t.Fatal(err)
			}
			wantTokens := baseline.Summary.Tokens
			wantTokens.InputTokens += int64(tc.exact * 7)
			wantTokens.OutputTokens += int64(tc.exact * 2)
			if result.Summary.Tokens != wantTokens || result.Summary.Transcripts != baseline.Summary.Transcripts+tc.transcript ||
				result.Summary.UsageRecordsExact != baseline.Summary.UsageRecordsExact+tc.exact ||
				result.Summary.FixtureFilesExcluded != tc.excluded || result.Summary.RefusedRecords != tc.refused || len(result.Refusals) != tc.refused {
				t.Fatalf("summary = %+v, refusals = %+v; want baseline plus excluded=%d exact=%d refused=%d transcripts=%d tokens=%+v", result.Summary, result.Refusals, tc.excluded, tc.exact, tc.refused, tc.transcript, wantTokens)
			}
			for _, refusal := range result.Refusals {
				if refusal.Code != "claude_input_tokens" {
					t.Fatalf("refusal = %+v, want fail-closed missing usage", refusal)
				}
			}
			for _, denominator := range result.Denominators {
				if denominator.Root == "claude/projects/issue-9828" && (denominator.FilesDiscovered != 1 || denominator.FilesScanned != 1-tc.excluded || denominator.FixtureFilesExcluded != tc.excluded) {
					t.Fatalf("fixture denominator = %+v", denominator)
				}
			}
			if tc.refused > 0 && result.ConclusionStatus.BroadEfficiencySupported {
				t.Fatal("unsupported live usage must block broad efficiency conclusions")
			}
		})
	}
}
