package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=20ms
func TestRunTrajectoryAuditOpenCodeUnsupported(t *testing.T) {
	for _, tc := range []struct {
		name     string
		present  bool
		explicit bool
	}{
		{name: "explicit_present", present: true, explicit: true},
		{name: "discovered_present", present: true},
		{name: "explicit_missing", explicit: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			opencodeRoot := filepath.Join(root, "opencode")
			t.Setenv("HOME", root)
			t.Setenv("USERPROFILE", root)
			t.Setenv("CLAUDE_CONFIG_DIR", filepath.Join(root, "claude"))
			t.Setenv("CODEX_HOME", filepath.Join(root, "codex"))
			t.Setenv("OPENCODE_HOME", opencodeRoot)
			t.Setenv("XDG_DATA_HOME", filepath.Join(root, "data"))
			if tc.present {
				if err := os.Mkdir(opencodeRoot, 0o700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(opencodeRoot, "opencode.db"), []byte("UNREAD_OPENCODE_CONTENT"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			jsonlPath := filepath.Join(root, "report.jsonl")
			markdownPath := filepath.Join(root, "report.md")
			args := []string{"audit", "--since", "0", "--jsonl", jsonlPath, "--md", markdownPath}
			if tc.explicit {
				args = append(args, "--opencode-root", opencodeRoot)
			}
			var stdout, stderr bytes.Buffer
			rc := runTrajectory(&stdout, &stderr, args)
			wantRC := 0
			if tc.present {
				wantRC = 1
			}
			if rc != wantRC {
				t.Fatalf("exit=%d, want %d; stderr=%s", rc, wantRC, stderr.String())
			}
			jsonl, err := os.ReadFile(jsonlPath)
			if err != nil {
				t.Fatalf("refusal must preserve JSONL: %v", err)
			}
			markdown, err := os.ReadFile(markdownPath)
			if err != nil {
				t.Fatalf("refusal must preserve Markdown: %v", err)
			}
			if tc.present {
				if !bytes.Contains(jsonl, []byte(`"code":"unsupported_coverage"`)) || !bytes.Contains(jsonl, []byte(`"refused_records":1`)) || !strings.Contains(stderr.String(), "TRAJECTORY_SCHEMA_REFUSED opencode .:0 unsupported_coverage") || !bytes.Contains(markdown, []byte("Broad efficiency conclusions: **blocked** (refusals: 1).")) {
					t.Fatalf("missing incomplete coverage evidence: stderr=%s\nJSONL=%s\nMarkdown=%s", stderr.String(), jsonl, markdown)
				}
			} else if bytes.Contains(jsonl, []byte(`"kind":"refusal"`)) || !bytes.Contains(markdown, []byte("Broad efficiency conclusions: **supported** (refusals: 0).")) {
				t.Fatalf("absent optional root fabricated a refusal: %s\n%s", jsonl, markdown)
			}
			for _, artifact := range []string{string(jsonl), string(markdown), stderr.String()} {
				if strings.Contains(artifact, root) || strings.Contains(artifact, "UNREAD_OPENCODE_CONTENT") {
					t.Fatal("audit exported private source data")
				}
			}
		})
	}
}
