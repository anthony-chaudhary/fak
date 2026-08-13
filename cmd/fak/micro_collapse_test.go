package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunMicroCollapseMeasuresSavedParentContext(t *testing.T) {
	r, err := runMicroCollapse(3, 2048)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "PASS" || r.Allowed != 3 || r.Denied != 0 || r.Errored != 0 || r.JournalRows != 3 {
		t.Fatalf("receipt=%+v", r)
	}
	if r.IntermediateTokens <= r.FoldedTokens || r.SavedTokens != r.IntermediateTokens-r.FoldedTokens {
		t.Fatalf("context accounting=%+v", r)
	}
}

func TestMicroCollapseJSONIsCapturedReceipt(t *testing.T) {
	r, err := runMicroCollapse(2, 1024)
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(r); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"schema":"fak-micro-collapse/1"`, `"verdict":"PASS"`, `"saved_tokens":`} {
		if !strings.Contains(b.String(), want) {
			t.Fatalf("receipt missing %s: %s", want, b.String())
		}
	}
}

func TestRunRepoPulseCollapsesRealGitReads(t *testing.T) {
	dir := t.TempDir()
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	run("init", "-q")
	run("config", "user.email", "pulse@example.invalid")
	run("config", "user.name", "Pulse Test")
	if err := os.WriteFile(filepath.Join(dir, "tracked.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	run("add", "tracked.txt")
	run("commit", "-q", "-m", "base")
	if err := os.WriteFile(filepath.Join(dir, "tracked.txt"), []byte("base\nchanged\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	r, err := runRepoPulse(dir)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "PASS" || r.Calls != 3 || r.Allowed != 3 || r.JournalRows != 3 {
		t.Fatalf("receipt=%+v", r)
	}
	if r.InlineTokens <= r.FoldedTokens || r.SavedTokens != r.InlineTokens-r.FoldedTokens || r.ToolTurnsSkipped != 2 {
		t.Fatalf("accounting=%+v", r)
	}
	for _, want := range []string{"status:", "head:", "diff:", "tracked.txt"} {
		if !strings.Contains(r.Collapsed, want) {
			t.Fatalf("collapsed missing %q: %s", want, r.Collapsed)
		}
	}
}
