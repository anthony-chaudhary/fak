package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionmine"
)

func TestRunSessionMineSpine(t *testing.T) {
	root := t.TempDir()
	codex := filepath.Join(root, "codex")
	claude := filepath.Join(root, "claude")
	if err := os.MkdirAll(codex, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(claude, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(codex, "a.jsonl"), []byte("{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"shell_command\",\"arguments\":\"{\\\"command\\\":\\\"git status --short\\\"}\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"function_call\",\"name\":\"view_image\"}}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(claude, "b.jsonl"), []byte("{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"name\":\"Bash\",\"input\":{\"command\":\"git status --short\"}},{\"type\":\"tool_use\",\"name\":\"view_image\"}]}}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runSessionMine(&out, &errOut, []string{"--codex-root", codex, "--claude-root", claude, "--days", "0"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var report sessionmine.Report
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != sessionmine.Schema || len(report.Candidates) != 1 {
		t.Fatalf("report=%+v", report)
	}
	if got := report.Candidates[0].Trajectory; len(got) != 2 || got[0] != "git_status" || got[1] != "view_image" {
		t.Fatalf("trajectory=%v", got)
	}
}
