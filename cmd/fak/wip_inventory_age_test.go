package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestWIPInventoryAgeGateRefusesStaleUntrackedSource(t *testing.T) {
	repo := initWipAdmitRepo(t)
	path := filepath.Join(repo, "stale.go")
	if err := os.WriteFile(path, []byte("package stale\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runWIPInventory([]string{"--root", repo, "--max-untracked-age", "1h"}, &out, &errOut)
	if code != 3 {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	for _, want := range []string{"STALE_UNTRACKED_SOURCE", "stale.go", "fak wip autocheckpoint", "fak worktree worker prepare"} {
		if !strings.Contains(errOut.String(), want) {
			t.Fatalf("missing %q: %s", want, errOut.String())
		}
	}
}

func TestWIPInventoryAgeGateAllowsFreshAndIgnoredFiles(t *testing.T) {
	repo := initWipAdmitRepo(t)
	if err := os.WriteFile(filepath.Join(repo, ".gitignore"), []byte("ignored.tmp\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	ignored := filepath.Join(repo, "ignored.tmp")
	if err := os.WriteFile(ignored, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-24 * time.Hour)
	_ = os.Chtimes(ignored, old, old)
	fresh := filepath.Join(repo, "fresh.go")
	if err := os.WriteFile(fresh, []byte("package fresh\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runWIPInventory([]string{"--root", repo, "--max-untracked-age", "1h"}, &out, &errOut); code != 0 {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	if strings.Contains(out.String(), "ignored.tmp") {
		t.Fatalf("ignored file leaked into source population: %s", out.String())
	}
}

func TestWIPInventoryAgeGateRejectsUnprotectedPathInMixedPopulation(t *testing.T) {
	repo := initWipAdmitRepo(t)
	protected := filepath.Join(repo, "protected.go")
	if err := os.WriteFile(protected, []byte("package protected\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runWIPInventoryGit(t, repo, "add", "protected.go")
	runWIPInventoryGit(t, repo, "commit", "-m", "checkpoint source")
	checkpoint := strings.TrimSpace(runWIPInventoryGit(t, repo, "rev-parse", "HEAD"))
	runWIPInventoryGit(t, repo, "update-ref", "refs/fak/wip/test", checkpoint)
	runWIPInventoryGit(t, repo, "reset", "--hard", "HEAD^")
	if err := os.WriteFile(protected, []byte("package protected\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	unprotected := filepath.Join(repo, "unprotected.go")
	if err := os.WriteFile(unprotected, []byte("package unprotected\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(protected, old.Add(-time.Hour), old.Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(unprotected, old, old); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	code := runWIPInventory([]string{"--root", repo, "--max-untracked-age", "1h", "--json"}, &out, &errOut)
	if code != 3 {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), `"protection": "mixed"`) || !strings.Contains(out.String(), `"oldest_unprotected_path": "unprotected.go"`) {
		t.Fatalf("mixed protection provenance missing: %s", out.String())
	}
	if !strings.Contains(errOut.String(), "unprotected.go") {
		t.Fatalf("gate did not identify stale unprotected path: %s", errOut.String())
	}
}

func runWIPInventoryGit(t *testing.T, repo string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
	}
	return string(out)
}
