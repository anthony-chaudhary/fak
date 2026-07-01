package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestToolCoverageAuditJSONReportsLoadBearingDebt(t *testing.T) {
	root := t.TempDir()
	writeToolCoverageFixture(t, root, "tools/foo.py", "x=1\n")
	writeToolCoverageFixture(t, root, "tools/foo_test.py", "x=1\n")
	writeToolCoverageFixture(t, root, "tools/bar.py", "x=1\n")
	writeToolCoverageFixture(t, root, ".claude/skills/audit/SKILL.md", "calls tools/foo.py and tools/bar.py\n")

	var stdout, stderr bytes.Buffer
	code := runToolCoverageAudit(&stdout, &stderr, []string{
		"--workspace", root,
		"--min-coverage", "90",
		"--json",
	})
	if code != 1 {
		t.Fatalf("exit code = %d, stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var payload struct {
		Schema              string   `json:"schema"`
		OK                  bool     `json:"ok"`
		Verdict             string   `json:"verdict"`
		LoadBearingUntested []string `json:"load_bearing_untested"`
		Debt                int      `json:"debt"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v\n%s", err, stdout.String())
	}
	if payload.Schema != "fleet-tool-coverage-audit/1" || payload.OK || payload.Verdict != "BELOW_FLOOR" {
		t.Fatalf("payload header = %+v", payload)
	}
	if payload.Debt != 1 || len(payload.LoadBearingUntested) != 1 || payload.LoadBearingUntested[0] != "bar" {
		t.Fatalf("payload debt = %+v", payload)
	}
}

func writeToolCoverageFixture(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
