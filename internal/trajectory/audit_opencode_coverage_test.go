package trajectory

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=20ms
func TestAuditOpenCodeUnsupported(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name    string
		present bool
		payload bool
	}{
		{name: "missing"},
		{name: "present_empty", present: true},
		{name: "present_corpus", present: true, payload: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			root := t.TempDir()
			opencodeRoot := filepath.Join(root, "opencode")
			if tc.present {
				if err := os.Mkdir(opencodeRoot, 0o700); err != nil {
					t.Fatal(err)
				}
			}
			if tc.payload {
				if err := os.WriteFile(filepath.Join(opencodeRoot, "private.jsonl"), []byte("UNREAD_OPENCODE_CONTENT"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			claudeRoot := filepath.Join(root, "claude")
			codexRoot := filepath.Join(root, "codex")
			for _, path := range []string{claudeRoot, codexRoot} {
				if err := os.Mkdir(path, 0o700); err != nil {
					t.Fatal(err)
				}
			}
			claude := `{"type":"assistant","sessionId":"claude-owned","message":{"id":"owned-message","model":"claude-fixture","usage":{"input_tokens":10,"output_tokens":2,"cache_read_input_tokens":0,"cache_creation_input_tokens":0},"content":[]}}` + "\n"
			if err := os.WriteFile(filepath.Join(claudeRoot, "owned.jsonl"), []byte(claude), 0o600); err != nil {
				t.Fatal(err)
			}
			result, err := RunAudit(AuditOptions{Sources: []AuditSource{
				{Name: AuditSourceOpencode, Root: opencodeRoot, RootLabel: "opencode/storage"},
				{Name: AuditSourceClaude, Root: claudeRoot, RootLabel: "claude/projects"},
				{Name: AuditSourceCodex, Root: codexRoot, RootLabel: "codex/sessions"},
			}})
			if err != nil {
				t.Fatalf("coverage refusal must preserve the report: %v", err)
			}
			wantRefusals := 0
			if tc.present {
				wantRefusals = 1
			}
			if len(result.Refusals) != wantRefusals || result.Summary.RefusedRecords != wantRefusals || result.ConclusionStatus.RefusalCount != wantRefusals {
				t.Fatalf("refusal counts = rows:%d summary:%d conclusion:%d, want %d", len(result.Refusals), result.Summary.RefusedRecords, result.ConclusionStatus.RefusalCount, wantRefusals)
			}
			if result.ConclusionStatus.BroadEfficiencySupported == tc.present {
				t.Fatalf("broad conclusions supported = %v, root present = %v", result.ConclusionStatus.BroadEfficiencySupported, tc.present)
			}
			if len(result.Denominators) != 3 || result.Summary.Transcripts != 1 || result.Summary.Tokens != (AuditTokens{InputTokens: 10, OutputTokens: 2}) {
				t.Fatalf("supported-source evidence lost: %+v", result.Summary)
			}
			for _, d := range result.Denominators {
				if d.Source == AuditSourceOpencode {
					if d.RootPresent != tc.present || d.RefusedRecords != wantRefusals || d.FilesDiscovered != 0 || d.FilesScanned != 0 || d.Records != 0 || !strings.Contains(d.TokenSemantics, "unsupported") {
						t.Fatalf("unsupported coverage denominator = %+v", d)
					}
				}
				if d.Source == AuditSourceCodex && (!d.RootPresent || d.RefusedRecords != 0 || d.FilesScanned != 0) {
					t.Fatalf("empty supported source confused with unsupported coverage: %+v", d)
				}
			}
			if tc.present {
				refusal := result.Refusals[0]
				if refusal.Schema != AuditSchema || refusal.Kind != "refusal" || refusal.Source != AuditSourceOpencode || refusal.Code != "unsupported_coverage" || refusal.SourcePath != "." || refusal.Line != 0 {
					t.Fatalf("typed source-level refusal = %+v", refusal)
				}
			}
			var artifact bytes.Buffer
			if err := WriteAuditJSONL(&artifact, result); err != nil {
				t.Fatal(err)
			}
			if err := WriteAuditMarkdown(&artifact, result); err != nil {
				t.Fatal(err)
			}
			for _, private := range []string{root, "UNREAD_OPENCODE_CONTENT", "private.jsonl"} {
				if strings.Contains(artifact.String(), private) {
					t.Fatalf("audit exported private source data %q", private)
				}
			}
		})
	}
}
