package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Fixture SHAs: 40 lowercase hex chars, distinct per commit.
const (
	staleFixedSHAOld = "ffeeddccbbaa99887766554433221100ffeeddcc"
	staleFixedSHAMid = "00112233445566778899aabbccddeeff00112233"
	staleFixedSHANew = "aa11bb22cc33dd44ee55ff66aa77bb88cc99dd00"
	staleFixedTrunk  = "111122223333444455556666777788889999aaaa"
)

// staleFixedFixtureCommits is the injected history, newest first: the newest
// commit carries no keyword, the middle one Fixes #8252, the oldest one claims
// both #8252 and #9513 (with an owner/repo qualifier on the second).
func staleFixedFixtureCommits() []staleFixedCommit {
	return []staleFixedCommit{
		{SHA: staleFixedSHANew, Message: "chore(deps): bump runtime deps\n\nNo closing keywords in this body.\n"},
		{SHA: staleFixedSHAMid, Message: "fix(gateway): drop stale sessions Fixes #8252\n\nMore body text follows.\n"},
		{SHA: staleFixedSHAOld, Message: "Resolves: #8252 and Closes o/r#9513\n"},
	}
}

func staleFixedFixtureLog() string {
	var b strings.Builder
	for _, c := range staleFixedFixtureCommits() {
		b.WriteString("\n")
		b.WriteString(c.SHA)
		b.WriteRune('\x00')
		b.WriteString(c.Message)
		b.WriteRune('\x1e')
	}
	return b.String()
}

func writeStaleFixedIssues(t *testing.T) string {
	t.Helper()
	const issues = `[
	  {"number": 8252, "title": "stale session leak", "url": "https://github.com/o/r/issues/8252"},
	  {"number": 9513, "title": "router stall", "url": "https://github.com/o/r/issues/9513"},
	  {"number": 7777, "title": "unrelated open issue"}
	]`
	path := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(path, []byte(issues), 0o644); err != nil {
		t.Fatalf("write issues fixture: %v", err)
	}
	return path
}

// staleFixedGHRecorder captures every gh argv the apply path builds and lets a
// test refuse specific calls.
type staleFixedGHRecorder struct {
	calls       [][]string
	failCloses  bool
	failComment bool
}

func (r *staleFixedGHRecorder) run(args []string) (string, string, bool) {
	r.calls = append(r.calls, append([]string(nil), args...))
	verb := args[0] + " " + args[1]
	if r.failComment && verb == "issue comment" {
		return "", "comment boom", false
	}
	if r.failCloses && verb == "issue close" {
		return "", "close boom", false
	}
	return "", "", true
}

func staleFixedScannerFixture() staleFixedScanner {
	return func(root, ref string, depth int) (string, []staleFixedCommit, error) {
		return staleFixedTrunk, staleFixedFixtureCommits(), nil
	}
}

func TestIssueStaleFixedDryRunJSON(t *testing.T) {
	issues := writeStaleFixedIssues(t)
	var stdout, stderr bytes.Buffer
	rec := &staleFixedGHRecorder{}
	code := runIssueStaleFixedWith(&stdout, &stderr, []string{"--from-issues", issues, "--json"}, rec.run, staleFixedScannerFixture())
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	if len(rec.calls) != 0 {
		t.Fatalf("dry-run must not call gh, got %d calls: %v", len(rec.calls), rec.calls)
	}
	var res staleFixedResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("decode result: %v\n%s", err, stdout.String())
	}
	if res.Schema != staleFixedSchema || res.TrunkSHA != staleFixedTrunk || res.ScannedCommits != 3 || res.OpenIssues != 3 {
		t.Fatalf("header fields wrong: %+v", res)
	}
	if len(res.Rows) != 2 {
		t.Fatalf("want 2 candidate rows (7777 has no closing commit), got %d: %+v", len(res.Rows), res.Rows)
	}
	if res.Rows[0].Number != 8252 || res.Rows[0].Title != "stale session leak" {
		t.Fatalf("row 0 wrong: %+v", res.Rows[0])
	}
	want8252 := []string{staleFixedSHAMid, staleFixedSHAOld} // newest-first
	if strings.Join(res.Rows[0].Shas, ",") != strings.Join(want8252, ",") {
		t.Fatalf("#8252 shas = %v, want %v", res.Rows[0].Shas, want8252)
	}
	if res.Rows[1].Number != 9513 || len(res.Rows[1].Shas) != 1 || res.Rows[1].Shas[0] != staleFixedSHAOld {
		t.Fatalf("row 1 wrong: %+v", res.Rows[1])
	}
	if res.AppliedCount != 0 || res.FailedCount != 0 {
		t.Fatalf("dry-run counts wrong: applied=%d failed=%d", res.AppliedCount, res.FailedCount)
	}
}

func TestIssueStaleFixedApply(t *testing.T) {
	issues := writeStaleFixedIssues(t)
	var stdout, stderr bytes.Buffer
	rec := &staleFixedGHRecorder{}
	code := runIssueStaleFixedWith(&stdout, &stderr, []string{"--from-issues", issues, "--apply", "--repo", "o/r", "--json"}, rec.run, staleFixedScannerFixture())
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	if len(rec.calls) != 4 {
		t.Fatalf("want 4 gh calls (comment+close per row), got %d: %v", len(rec.calls), rec.calls)
	}
	wantOrder := [][]string{
		{"issue", "comment", "8252"},
		{"issue", "close", "8252"},
		{"issue", "comment", "9513"},
		{"issue", "close", "9513"},
	}
	for i, want := range wantOrder {
		got := rec.calls[i]
		if len(got) < 3 || !strings.EqualFold(got[0]+" "+got[1]+" "+got[2], want[0]+" "+want[1]+" "+want[2]) {
			t.Fatalf("call %d = %v, want head %v", i, got, want)
		}
	}
	comment8252 := rec.calls[0]
	bodyIdx := -1
	for i, a := range comment8252 {
		if a == "--body" {
			bodyIdx = i + 1
		}
	}
	if bodyIdx <= 0 {
		t.Fatalf("comment argv lacks --body: %v", comment8252)
	}
	body := comment8252[bodyIdx]
	for _, want := range []string{staleFixedSHAMid, staleFixedSHAOld, "fak-private#3392"} {
		if !strings.Contains(body, want) {
			t.Fatalf("evidence comment missing %q:\n%s", want, body)
		}
	}
	repoIdx := -1
	for i, a := range rec.calls[1] {
		if a == "--repo" {
			repoIdx = i + 1
		}
	}
	if repoIdx <= 0 || rec.calls[1][repoIdx] != "o/r" {
		t.Fatalf("close argv lost --repo o/r: %v", rec.calls[1])
	}
	var res staleFixedResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if res.AppliedCount != 2 || res.FailedCount != 0 {
		t.Fatalf("applied=%d failed=%d, want 2/0", res.AppliedCount, res.FailedCount)
	}
	for _, row := range res.Rows {
		if !row.Applied || !row.CommentOK || !row.CloseOK {
			t.Fatalf("row %+v not fully applied", row)
		}
	}
}

func TestIssueStaleFixedApplyCloseFailure(t *testing.T) {
	issues := writeStaleFixedIssues(t)
	var stdout, stderr bytes.Buffer
	rec := &staleFixedGHRecorder{failCloses: true}
	code := runIssueStaleFixedWith(&stdout, &stderr, []string{"--from-issues", issues, "--apply", "--json"}, rec.run, staleFixedScannerFixture())
	if code != 1 {
		t.Fatalf("exit=%d, want 1 on gh close failure (stderr=%s)", code, stderr.String())
	}
	var res staleFixedResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if res.AppliedCount != 0 || res.FailedCount != 2 {
		t.Fatalf("applied=%d failed=%d, want 0/2", res.AppliedCount, res.FailedCount)
	}
	for _, row := range res.Rows {
		if !row.CommentOK || row.CloseOK || !strings.Contains(row.Error, "gh issue close failed") {
			t.Fatalf("row %+v: comment should stick, close should fail with reason", row)
		}
	}
}

func TestIssueStaleFixedFlagErrors(t *testing.T) {
	issues := writeStaleFixedIssues(t)
	var stdout, stderr bytes.Buffer
	if code := runIssueStaleFixedWith(&stdout, &stderr, []string{"--from-issues", issues, "stray"}, nil, staleFixedScannerFixture()); code != 2 {
		t.Fatalf("positional argument: exit=%d, want 2", code)
	}
	stderr.Reset()
	if code := runIssueStaleFixedWith(&stdout, &stderr, []string{"--bogus-flag"}, nil, staleFixedScannerFixture()); code != 2 {
		t.Fatalf("unknown flag: exit=%d, want 2", code)
	}
	stderr.Reset()
	if code := runIssueStaleFixedWith(&stdout, &stderr, []string{"--from-issues", filepath.Join(t.TempDir(), "missing.json")}, nil, staleFixedScannerFixture()); code != 2 {
		t.Fatalf("missing --from-issues file: exit=%d, want 2", code)
	}
}

func TestIssueStaleFixedDispatchRouting(t *testing.T) {
	var stdout, stderr bytes.Buffer
	// An unknown top-level subcommand reports "unknown subcommand"; reaching
	// our FlagSet instead proves the switch routes stale-fixed correctly.
	code := RunIssue(&stdout, &stderr, []string{"stale-fixed", "--bogus-flag"})
	if code != 2 {
		t.Fatalf("exit=%d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "Usage of issue stale-fixed") {
		t.Fatalf("stderr did not come from the stale-fixed FlagSet:\n%s", stderr.String())
	}
}

func TestStaleFixedCloseRegex(t *testing.T) {
	pos := []struct {
		msg  string
		want string
	}{
		{"Fixes #123", "123"},
		{"fixes: #123", "123"},
		{"CLOSES o/r#456", "456"},
		{"fixed #789\n", "789"},
		{"Resolved:\t#5", "5"},
		{"body line\nCloses #42\nnext", "42"},
	}
	for _, tc := range pos {
		m := staleFixedCloseRE.FindStringSubmatch(tc.msg)
		if m == nil {
			t.Errorf("positive case %q: no match", tc.msg)
			continue
		}
		if m[2] != tc.want {
			t.Errorf("positive case %q: captured %q, want %q", tc.msg, m[2], tc.want)
		}
	}
	neg := []string{
		"prefixes #123",
		"closeout #123",
		"resolution #123",
		"see #123",
		"Fixes 123",
		"fixes #",
		"",
	}
	for _, msg := range neg {
		if m := staleFixedCloseRE.FindStringSubmatch(msg); m != nil {
			t.Errorf("negative case %q: unexpected match %q", msg, m[2])
		}
	}
}

func TestParseStaleFixedLog(t *testing.T) {
	raw := "\n" + staleFixedSHAOld + "\x00subject A\n\nbody A\n\x1e" +
		"\n" + staleFixedSHAMid + "\x00subject B\x1e" +
		"garbage chunk without null\x1e" +
		"\nshortsha\x00not forty hex digits\x1e"
	commits := parseStaleFixedLog(raw)
	if len(commits) != 2 {
		t.Fatalf("want 2 well-formed records, got %d: %+v", len(commits), commits)
	}
	if commits[0].SHA != staleFixedSHAOld || commits[0].Message != "subject A\n\nbody A\n" {
		t.Fatalf("record 0 wrong: %+v", commits[0])
	}
	if commits[1].SHA != staleFixedSHAMid || commits[1].Message != "subject B" {
		t.Fatalf("record 1 wrong: %+v", commits[1])
	}
}

// fak-test:runtime medium est=10s lane=default
func TestIssueStaleFixedRealGitRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := t.TempDir()
	git := func(args ...string) string {
		out, err := exec.Command("git", append([]string{"-C", root}, args...)...).CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return string(out)
	}
	initOut, err := exec.Command("git", append([]string{"-C", root}, "init", "-q", "-b", "main")...).CombinedOutput()
	if err != nil {
		git("init", "-q") // older git without -b
		git("symbolic-ref", "HEAD", "refs/heads/main")
		_ = initOut
	}
	git("config", "user.email", "stale-fixed@test.invalid")
	git("config", "user.name", "stale fixed test")
	if err := os.WriteFile(filepath.Join(root, "probe.txt"), []byte("probe\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "probe.txt")
	git("commit", "-q", "-m", "feat(probe): add probe\n\nFixes #42\n")
	sha := strings.TrimSpace(git("rev-parse", "HEAD"))

	issues := filepath.Join(root, "issues.json")
	if err := os.WriteFile(issues, []byte(`[{"number": 42, "title": "probe gap"}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	rec := &staleFixedGHRecorder{}
	code := runIssueStaleFixedWith(&stdout, &stderr, []string{"--from-issues", issues, "--root", root, "--json"}, rec.run, nil)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	if len(rec.calls) != 0 {
		t.Fatalf("dry-run called gh: %v", rec.calls)
	}
	var res staleFixedResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("decode result: %v\n%s", err, stdout.String())
	}
	if res.TrunkSHA != sha {
		t.Fatalf("trunk_sha=%s, want %s", res.TrunkSHA, sha)
	}
	if len(res.Rows) != 1 || res.Rows[0].Number != 42 || len(res.Rows[0].Shas) != 1 || res.Rows[0].Shas[0] != sha {
		t.Fatalf("rows wrong: %+v (want #42 closed-by %s)", res.Rows, sha)
	}
}
