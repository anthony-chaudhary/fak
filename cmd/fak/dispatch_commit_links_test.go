package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseCommitLinkLog(t *testing.T) {
	raw := "aaa111" + commitLinkFieldSep + "feat(audit): add rollup #1612 (fak audit)\n" + commitLinkRecordSep +
		"bbb222" + commitLinkFieldSep + "feat(audit): add rollup\n\nFixes #1612\n\n(fak audit)\n" + commitLinkRecordSep

	commits := parseCommitLinkLog(raw)
	if len(commits) != 2 {
		t.Fatalf("want 2 commits, got %d: %+v", len(commits), commits)
	}
	if commits[0].SHA != "aaa111" || commits[0].Subject != "feat(audit): add rollup #1612 (fak audit)" || commits[0].Body != "" {
		t.Errorf("commit 0 mismatch: %+v", commits[0])
	}
	if commits[1].SHA != "bbb222" || commits[1].Subject != "feat(audit): add rollup" {
		t.Errorf("commit 1 mismatch: %+v", commits[1])
	}
	if !strings.Contains(commits[1].Body, "Fixes #1612") {
		t.Errorf("commit 1 body should retain the Fixes trailer, got %q", commits[1].Body)
	}
}

func TestParseCommitLinkLog_Empty(t *testing.T) {
	if commits := parseCommitLinkLog(""); len(commits) != 0 {
		t.Fatalf("want no commits for empty input, got %+v", commits)
	}
}

// TestRunDispatchCommitLinks_RealGitRepo builds a tiny real git repo (not a
// mock) with one good-subject commit and one missing-link commit, and
// asserts the CLI shell's rendered output flags exactly the latter -- the
// same "real fixtures over mocks" bar cmd/fak/audit_usage_test.go uses.
func TestRunDispatchCommitLinks_RealGitRepo(t *testing.T) {
	root := t.TempDir()
	runGitFixture(t, root, "init", "-q", ".")
	runGitFixture(t, root, "config", "user.email", "fixture@example.com")
	runGitFixture(t, root, "config", "user.name", "Fixture")

	writeAndCommit(t, root, "base.txt", "base", "chore: base commit")
	writeAndCommit(t, root, "a.txt", "a", "feat(audit): add rollup #1612 (fak audit)")
	writeAndCommit(t, root, "b.txt", "b", "feat(audit): add another rollup\n\nFixes #1612\n\n(fak audit)")
	writeAndCommit(t, root, "c.txt", "c", "typo: fix a comment")

	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	defer os.Chdir(oldwd)

	var stdout, stderr bytes.Buffer
	code := runDispatchCommitLinks(&stdout, &stderr, []string{"--range", "HEAD~3..HEAD", "--json"})
	if code != 0 {
		t.Fatalf("runDispatchCommitLinks exit=%d, stderr=%s", code, stderr.String())
	}

	out := stdout.String()
	if strings.Contains(out, "\"subject\": \"feat(audit): add rollup #1612 (fak audit)\"") {
		t.Errorf("good subject (has #1612 already) must not be a finding: %s", out)
	}
	if !strings.Contains(out, "\"guessed_issue\": \"1612\"") && !strings.Contains(out, "\"GuessedIssue\": \"1612\"") {
		t.Errorf("want a finding guessing #1612 from the body trailer, got: %s", out)
	}
	if strings.Contains(out, "typo: fix a comment") {
		t.Errorf("commit with no ship-stamp trailer must not be flagged: %s", out)
	}
}

func TestRunDispatchCommitLinksWitnessJSONBucketsUnresolved(t *testing.T) {
	path := filepath.Join(t.TempDir(), "witness.json")
	fixture := `{"issues":[
		{"number":10,"sha":"aaa111","subject":"fix(dispatch): close work (fak cmd)","audit_verdict":"OK","audit_witness":"diff-witnessed","reachable":true},
		{"number":11,"sha":"bbb222","subject":"fix(dispatch): close #11 (fak cmd)","audit_verdict":"FAIL","audit_witness":"diff-witnessed","reachable":true},
		{"number":12,"sha":"ccc333","subject":"fix(dispatch): close #12 (fak cmd)","audit_verdict":"OK","audit_witness":"diff-witnessed","reachable":false},
		{"number":13,"sha":"ddd444","subject":"fix(dispatch): close #13 (fak cmd)","audit_verdict":"OK","audit_witness":"subject-only","reachable":true}
	]}`
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runDispatchCommitLinks(&stdout, &stderr, []string{"--witness-json", path})
	if code != 0 {
		t.Fatalf("runDispatchCommitLinks exit=%d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"witness-failures: scanned 4 issue(s)",
		"#10  aaa111  missing_issue_link",
		"#11  bbb222  failed_audit",
		"#12  ccc333  stale_sha",
		"#13  ddd444  insufficient_diff_evidence",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("witness report missing %q:\n%s", want, out)
		}
	}
}

func runGitFixture(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out.String())
	}
}

func writeAndCommit(t *testing.T, root, name, content, message string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	runGitFixture(t, root, "add", name)
	runGitFixture(t, root, "commit", "-q", "-m", message)
}
