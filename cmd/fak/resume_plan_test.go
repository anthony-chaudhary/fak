package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestResumePlanCommitPushForcedTimeout is the #2086 end-to-end fixture: a real
// commit+push sequence whose PUSH half was interrupted (the local commit landed,
// the remote never received it) must report which half applied plus a safe
// resume command — the structured answer that replaces a bare exit-143.
//
// The kill is reproduced by its observable post-kill tree state (commit present
// locally, push withheld), which is byte-identical to a push SIGTERM'd on
// timeout — the reader inspects the tree, not the dead process.
func TestResumePlanCommitPushForcedTimeout(t *testing.T) {
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	work := filepath.Join(root, "work")

	runGitFixture(t, root, "init", "--bare", "-b", "main", remote)
	runGitFixture(t, root, "init", "-b", "main", work)
	runGitFixture(t, work, "config", "user.email", "fixture@example.com")
	runGitFixture(t, work, "config", "user.name", "fixture")
	runGitFixture(t, work, "config", "commit.gpgsign", "false")
	runGitFixture(t, work, "remote", "add", "origin", remote)

	// Baseline commit + push establishes the upstream tracking ref.
	writeAndCommit(t, work, "a.txt", "one\n", "chore: seed")
	runGitFixture(t, work, "push", "-u", "origin", "main")

	// The op's commit half lands; its push half is the interrupted one.
	writeAndCommit(t, work, "b.txt", "two\n", "feat: second change")

	steps, err := commitPushSteps(work)
	if err != nil {
		t.Fatalf("commitPushSteps: %v", err)
	}
	if len(steps) != 2 || !steps[0].Applied || steps[1].Applied {
		t.Fatalf("steps = %+v, want commit applied + push pending", steps)
	}

	var out, errb bytes.Buffer
	if code := runToolResumePlan(&out, &errb, "commit-push", work, false); code != 0 {
		t.Fatalf("runResumePlan exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"applied so far: commit",
		"pending: push",
		"commit created locally",
		"push NOT sent",
		"safe resume: git push",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("resume report missing %q:\n%s", want, got)
		}
	}

	// JSON surface carries the same partial-state contract the agent branches on.
	var jout, jerr bytes.Buffer
	if code := runToolResumePlan(&jout, &jerr, "commit-push", work, true); code != 0 {
		t.Fatalf("runResumePlan --json exit=%d stderr=%s", code, jerr.String())
	}
	for _, want := range []string{
		`"applied_so_far"`, `"commit"`,
		`"safe_resume_cmd": "git push"`,
		`"complete": false`,
		`"retryable": true`,
	} {
		if !strings.Contains(jout.String(), want) {
			t.Fatalf("resume JSON missing %q:\n%s", want, jout.String())
		}
	}

	// After the push lands, the same op reads Complete with nothing to resume.
	runGitFixture(t, work, "push", "origin", "main")
	steps2, err := commitPushSteps(work)
	if err != nil {
		t.Fatalf("commitPushSteps after push: %v", err)
	}
	if !steps2[0].Applied || !steps2[1].Applied {
		t.Fatalf("after push both halves must read applied: %+v", steps2)
	}
}

// TestResumePlanCommitPushDirtyTree pins the earliest-kill case: the commit half
// never completed (changes still uncommitted), so the resume starts at commit.
func TestResumePlanCommitPushDirtyTree(t *testing.T) {
	root := t.TempDir()
	work := filepath.Join(root, "work")
	runGitFixture(t, root, "init", "-b", "main", work)
	runGitFixture(t, work, "config", "user.email", "fixture@example.com")
	runGitFixture(t, work, "config", "user.name", "fixture")
	runGitFixture(t, work, "config", "commit.gpgsign", "false")
	writeAndCommit(t, work, "a.txt", "one\n", "chore: seed")

	// Uncommitted change: the commit half is pending, not applied.
	if err := os.WriteFile(filepath.Join(work, "a.txt"), []byte("one\ntwo\n"), 0o644); err != nil {
		t.Fatalf("dirty write: %v", err)
	}

	steps, err := commitPushSteps(work)
	if err != nil {
		t.Fatalf("commitPushSteps: %v", err)
	}
	if steps[0].Applied {
		t.Fatalf("dirty tree must read commit as pending: %+v", steps[0])
	}

	var out, errb bytes.Buffer
	if code := runToolResumePlan(&out, &errb, "commit-push", work, false); code != 0 {
		t.Fatalf("runResumePlan exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "safe resume: git commit") {
		t.Fatalf("dirty-tree resume must start at commit:\n%s", out.String())
	}
}
