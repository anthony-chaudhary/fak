package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStudyMonitorReadsRegistry(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "repos.json")
	data := `{"schema":"fak-monitored-repositories/1","updated_at":"2026-08-14","methodology":"ranked","repositories":[{"repository":"owner/repo","url":"https://github.com/owner/repo","status":"candidate","priority":1,"why":"fresh seam","last_checked":"2026-08-01","checked_revision":"abcdef1234567890","stars_at_check":42,"last_push_at_check":"2026-08-13T00:00:00Z"}]}`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runStudyMonitor(&stdout, &stderr, []string{"--registry", path, "--as-of", "2026-08-14", "--due-days", "7"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if got := stdout.String(); !strings.Contains(got, "owner/repo status=candidate checked=2026-08-01 age_days=13 due=true") {
		t.Fatalf("unexpected output:\n%s", got)
	}
}

func TestStudyMonitorRejectsMalformedRegistry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "repos.json")
	if err := os.WriteFile(path, []byte(`{"schema":"wrong"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runStudyMonitor(&stdout, &stderr, []string{"--registry", path}); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "schema must be") {
		t.Fatalf("unexpected error: %s", stderr.String())
	}
}
