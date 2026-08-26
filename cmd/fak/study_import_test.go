package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func TestStudyImportCommandDryRun(t *testing.T) {
	repo := t.TempDir()
	path := filepath.Join(repo, "docs", "research", "study.md")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("# Study\nsource: https://example.test/repo\nsource-revision: abc\nobserved-at: 2026-08-20\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"init", "-q"}, {"add", "--", "."}} {
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	var stdout, stderr bytes.Buffer
	if code := runStudyMonitor(&stdout, &stderr, []string{"import", "--repo", repo, "--dry-run"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var ledger studymonitor.ImportLedger
	if err := json.Unmarshal(stdout.Bytes(), &ledger); err != nil {
		t.Fatal(err)
	}
	if ledger.Attempted != 1 || ledger.Candidate != 1 {
		t.Fatalf("unexpected ledger: %+v", ledger)
	}
}

func TestStudyImportCommandRequiresStoreForLiveImport(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runStudyImport(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}
