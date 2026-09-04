package framevisibility

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestFoldMeasuresMasterVisibilityAndRelevance(t *testing.T) {
	root := t.TempDir()
	project := "C--work-fak"
	session := "session-1"
	dir := filepath.Join(root, ".claude-test", "projects", project, session)
	if err := os.MkdirAll(filepath.Join(dir, "subagents"), 0o755); err != nil {
		t.Fatal(err)
	}
	master := filepath.Join(filepath.Dir(dir), session+".jsonl")
	masterRows := []byte(
		`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Task","input":{"prompt":"delegate"}}]}}` + "\n" +
			`{"type":"assistant","message":{"content":[{"type":"tool_result","tool_use_id":"spawn","content":"agent done"}]}}` + "\n")
	if err := os.WriteFile(master, masterRows, 0o600); err != nil {
		t.Fatal(err)
	}
	subRows := []byte(
		`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","input":{"file_path":"x"}}]}}` + "\n" +
			`{"type":"assistant","message":{"content":[{"type":"tool_result","tool_use_id":"read","content":"hidden"}]}}` + "\n" +
			`{"type":"assistant","message":{"content":[{"type":"text","text":"agent done"}]}}` + "\n")
	if err := os.WriteFile(filepath.Join(dir, "subagents", "agent-a.jsonl"), subRows, 0o600); err != nil {
		t.Fatal(err)
	}

	totals, rows, err := Fold(root, project)
	if err != nil {
		t.Fatal(err)
	}
	if totals.Sessions != 1 || totals.SubFiles != 1 || totals.MasterSpawns != 1 {
		t.Fatalf("totals = %+v", totals)
	}
	if totals.SubEvents != 3 || totals.SubVisible != 1 {
		t.Fatalf("sub visibility = %d/%d, want 1/3", totals.SubVisible, totals.SubEvents)
	}
	if totals.MasterEvents != 1 || totals.MasterRelevant != 1 {
		t.Fatalf("master relevance = %d/%d, want 1/1", totals.MasterRelevant, totals.MasterEvents)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
}

func TestRun(t *testing.T) {
	root := t.TempDir()
	project := "C--work-fak"
	session := "session-1"
	dir := filepath.Join(root, ".claude-test", "projects", project, session)
	if err := os.MkdirAll(filepath.Join(dir, "subagents"), 0o755); err != nil {
		t.Fatal(err)
	}
	master := filepath.Join(filepath.Dir(dir), session+".jsonl")
	if err := os.WriteFile(master, []byte(`{"type":"assistant","message":{"content":[{"type":"text","text":"done"}]}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	outFile := filepath.Join(root, "out.json")

	var stdout, stderr bytes.Buffer
	code := Run(&stdout, &stderr, []string{"-homes", root, "-project", project, "-out", outFile})
	if code != 0 {
		t.Fatalf("Run returned code %d, stderr: %s", code, stderr.String())
	}
	if _, err := os.Stat(outFile); err != nil {
		t.Fatalf("expected output file %s: %v", outFile, err)
	}

	code = Run(&stdout, &stderr, []string{"-unrecognized-flag"})
	if code != 2 {
		t.Fatalf("Run with invalid flags returned %d, want 2", code)
	}

	code = Run(&stdout, &stderr, []string{"-homes", filepath.Join(root, "nonexistent")})
	if code != 1 {
		t.Fatalf("Run with nonexistent homes returned %d, want 1", code)
	}
}
