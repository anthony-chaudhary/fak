package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// hooks_claim_reclass_test.go — end-to-end exit-contract tests for `fak hooks claim-reclass`
// against a REAL temp git repo (#5434).
//
// These drive the push-seam rung the shell hook calls, not a stand-in: the commit is real, its
// subject and diff are read back out of git, the ledger is a real file on disk, and the review
// text arrives on stdin exactly as the hook pipes it. Exit contract: 0 = every residual cured,
// 1 = something uncured (the standing refusal stands), 2 = could-not-run.

// claimReclassRepo builds a repo whose HEAD is the wedge shape: a code-effect subject over a
// prose-only diff. It returns the repo path and the landed commit's full sha.
func claimReclassRepo(t *testing.T, subject string, files map[string]string) (string, string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	gitHook(t, repo, "init", "-q", "-b", "main")
	gitHook(t, repo, "config", "user.email", "t@t")
	gitHook(t, repo, "config", "user.name", "t")
	writeRepoFile(t, repo, "README.md", "base\n")
	gitHook(t, repo, "add", "--", "README.md")
	gitHook(t, repo, "commit", "-q", "-m", "chore(alpha): seed the tree")
	for p, content := range files {
		writeRepoFile(t, repo, p, content)
		gitHook(t, repo, "add", "--", p)
	}
	gitHook(t, repo, "commit", "-q", "-m", subject)
	c := exec.Command("git", "-C", repo, "rev-parse", "HEAD")
	out, err := c.Output()
	if err != nil {
		t.Skipf("git rev-parse: %v", err)
	}
	return repo, strings.TrimSpace(string(out))
}

func writeRepoFile(t *testing.T, repo, p, content string) {
	t.Helper()
	full := filepath.Join(repo, filepath.FromSlash(p))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// claimReview renders the claim-honesty review output the push gate would emit for one residual.
func claimReview(sha, subject string) string {
	return "CLAIM REVIEW  origin/main..HEAD  (2 commits)\n" +
		"RESIDUAL — your 100% (1)  [a CLAIM the kernel could not witness]\n" +
		"  " + sha[:9] + "  subject-only   " + subject + "\n" +
		"             |- code-effect claim but the diff touches no SOURCE file\n" +
		"CLEARED (1)\n" +
		"  0000000  diff-witnessed  chore(alpha): seed the tree\n"
}

// TestHooksClaimReclass_wedgedRangeClearedByForwardOnlyRecord is the end-to-end wedge witness: the
// range is refused with no ledger (today's dead end, whose only exit is FLEET_ALLOW_RESIDUAL=1),
// the refusal hands back the exact record that cures it, and appending that record clears the SAME
// rung without touching the landed commit.
func TestHooksClaimReclass_wedgedRangeClearedByForwardOnlyRecord(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	subject := "fix(alpha): correct the offline-check wording (fak alpha)"
	repo, sha := claimReclassRepo(t, subject, map[string]string{"docs/alpha-notes.md": "the offline check runs once per turn.\n"})
	review := claimReview(sha, subject)

	var out, errb bytes.Buffer
	code := runHooksClaimReclass(&out, &errb, strings.NewReader(review), []string{"--root", repo})
	if code != 1 {
		t.Fatalf("no ledger: exit = %d, want 1 (the standing refusal stands)\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "type: docs") || !strings.Contains(out.String(), "witness: docs/alpha-notes.md") {
		t.Fatalf("the refusal does not hand back a reachable cure:\n%s", out.String())
	}

	// The forward-only cure: a new file in the working tree, no history rewritten.
	writeRepoFile(t, repo, "docs/claim-reclass.txt",
		"# forward-only claim reclassification (#5434)\n"+
			"commit: "+sha[:9]+"\n"+
			"type: docs\n"+
			"witness: docs/alpha-notes.md\n"+
			"reason: the diff edits only a note; the landed subject typed a prose edit as a code effect\n")

	out.Reset()
	errb.Reset()
	code = runHooksClaimReclass(&out, &errb, strings.NewReader(review), []string{"--root", repo})
	if code != 0 {
		t.Fatalf("verified record: exit = %d, want 0 (range cleared)\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "cleared") {
		t.Fatalf("clear verdict not reported:\n%s", out.String())
	}

	// The landed commit is untouched — the cure is forward-only, never a rewrite.
	c := exec.Command("git", "-C", repo, "log", "-1", "--format=%H%n%s")
	got, err := c.Output()
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.ReplaceAll(string(got), "\r\n", "\n"), "\n")
	if strings.TrimSpace(lines[0]) != sha || strings.TrimSpace(lines[1]) != subject {
		t.Fatalf("history moved under the cure: %q / %q", lines[0], lines[1])
	}
}

// TestHooksClaimReclass_launderingRefusedThroughTheCLI is the end-to-end anti-laundering witness:
// the same rung, the same real commit, ledgers that try to wave a genuinely mis-described commit
// through. Every one must leave the range refused (exit 1).
func TestHooksClaimReclass_launderingRefusedThroughTheCLI(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	subject := "feat(alpha): add the retry loop (fak alpha)"
	repo, sha := claimReclassRepo(t, subject, map[string]string{"docs/alpha-notes.md": "the retry loop is planned.\n"})
	review := claimReview(sha, subject)

	cases := []struct {
		name   string
		ledger string
		want   string
	}{
		{"restate the code-effect claim", "commit: " + sha[:9] + "\ntype: feat\nwitness: docs/alpha-notes.md\nreason: it really is a feature\n", "itself a code-effect claim"},
		{"sidestep into perf", "commit: " + sha[:9] + "\ntype: perf\nwitness: docs/alpha-notes.md\nreason: it is faster now\n", "itself a code-effect claim"},
		{"demote to an unwitnessed type", "commit: " + sha[:9] + "\ntype: test\nwitness: docs/alpha-notes.md\nreason: really a test drop\n", "no test or CI witness file"},
		{"cite a file the commit never touched", "commit: " + sha[:9] + "\ntype: docs\nwitness: internal/alpha/loop.go\nreason: the note explains the loop\n", "is not in commit"},
		{"no rationale", "commit: " + sha[:9] + "\ntype: docs\nwitness: docs/alpha-notes.md\n", "no `reason:`"},
		{"a record for someone else's commit", "commit: 0123456789\ntype: docs\nwitness: docs/alpha-notes.md\nreason: a note elsewhere\n", "no reclassification record names this commit"},
	}
	for _, tc := range cases {
		writeRepoFile(t, repo, "docs/claim-reclass.txt", tc.ledger)
		var out, errb bytes.Buffer
		code := runHooksClaimReclass(&out, &errb, strings.NewReader(review), []string{"--root", repo})
		if code != 1 {
			t.Errorf("%s: exit = %d, want 1 (the cure must not launder)\nstdout:\n%s", tc.name, code, out.String())
			continue
		}
		if !strings.Contains(out.String(), tc.want) {
			t.Errorf("%s: refusal does not explain itself (want %q):\n%s", tc.name, tc.want, out.String())
		}
	}
}

// TestHooksClaimReclass_unreadableReviewIsCouldNotRun pins the fail-closed edge at the CLI: review
// text the rung cannot read yields exit 2, and the shell hook keeps its standing refusal.
func TestHooksClaimReclass_unreadableReviewIsCouldNotRun(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo, _ := claimReclassRepo(t, "docs(alpha): note the offline check (fak alpha)", map[string]string{"docs/alpha-notes.md": "x\n"})
	var out, errb bytes.Buffer
	code := runHooksClaimReclass(&out, &errb, strings.NewReader("nothing the parser recognizes\n"), []string{"--root", repo})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (could-not-run)\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
	}
}
