package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCentralityAuditFixtureCLI(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	data := `[{"number":1,"title":"kernel","body":"Centrality: Core\nP1: advanced - measured context\nP2: advanced - measured net value\nP3: preserved - unchanged adaptation\nP4: advanced - live operation"},{"number":2,"title":"legacy","body":"old"}]`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	err := runCentralityAudit([]string{"--input", path}, &out, &stderr, func() time.Time { return time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC) })
	if err != nil {
		t.Fatalf("run: %v; stderr=%s", err, stderr.String())
	}
	if got := out.String(); !strings.Contains(got, "CENTRALITY COVERAGE 50.0% (1/2)") || !strings.Contains(got, "Complete P1-P4 frame 1/2") {
		t.Fatalf("output:\n%s", got)
	}
}

func TestCentralityAuditJSONSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(path, []byte(`[]`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := runCentralityAudit([]string{"--input", path, "--json"}, &out, &bytes.Buffer{}, time.Now); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); !strings.Contains(got, `"schema": "fak-issue-centrality-audit/1"`) || !strings.Contains(got, `"errors": []`) {
		t.Fatalf("json:\n%s", got)
	}
}
