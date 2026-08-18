package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestWIPAdmitDefaultBudgetAutomaticallyHoldsStaleUntrackedSource(t *testing.T) {
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
	code := runWipAdmit(&out, &errOut, []string{"-C", repo, "--session", "age-gate"})
	if code != wipAdmitHoldExit {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	for _, want := range []string{"STALE_UNTRACKED_SOURCE", "stale.go", "fak wip autocheckpoint"} {
		if !strings.Contains(errOut.String(), want) {
			t.Fatalf("missing %q: %s", want, errOut.String())
		}
	}
}
