package main

import (
	"bytes"
	"os"
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
