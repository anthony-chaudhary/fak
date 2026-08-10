package flowmetrics

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// tempRepo builds a throwaway git repo so the gather path is exercised against
// real git output rather than a hand-written fixture of what git is believed to
// print. Format drift in `git log --pretty` is exactly the class of bug a
// string fixture cannot catch.
func tempRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	ctx := context.Background()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "test@example.invalid"},
		{"config", "user.name", "flowmetrics test"},
		{"config", "commit.gpgsign", "false"},
	} {
		if out, err := runIn(ctx, root, "git", args...); err != nil {
			t.Skipf("git %v unavailable in this environment: %v: %s", args, err, out)
		}
	}
	return root
}

func writeFile(t *testing.T, root, rel, body string) {
	t.Helper()
	full := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

func gitCommit(t *testing.T, root, message string, paths ...string) {
	t.Helper()
	ctx := context.Background()
	if out, err := runIn(ctx, root, "git", append([]string{"add", "--"}, paths...)...); err != nil {
		t.Fatalf("git add: %v: %s", err, out)
	}
	args := append([]string{"commit", "-q", "-m", message, "--"}, paths...)
	if out, err := runIn(ctx, root, "git", args...); err != nil {
		t.Fatalf("git commit: %v: %s", err, out)
	}
}

func TestGatherCommitsParsesRealGitOutput(t *testing.T) {
	root := tempRepo(t)
	ctx := context.Background()

	writeFile(t, root, "a.go", "package a\n")
	gitCommit(t, root, "feat(a): add a (#41) (fak a)", "a.go")

	writeFile(t, root, "b.go", "package b\n")
	// A multi-line body with a real close trailer plus prose that must NOT
	// attribute, and a blank line that would break a newline-delimited parse.
	gitCommit(t, root, "fix(b): repair b (fak b)\n\nThe #999 shape is prose.\n\nFixes #42\n", "b.go")

	writeFile(t, root, "c.go", "package c\n")
	gitCommit(t, root, "chore: no leaf and no refs", "c.go")

	commits, err := GatherCommits(ctx, root, 0)
	if err != nil {
		t.Fatalf("GatherCommits: %v", err)
	}
	if len(commits) != 3 {
		t.Fatalf("commits = %d, want 3: %+v", len(commits), commits)
	}
	byLeaf := map[string]Commit{}
	for _, c := range commits {
		byLeaf[c.Leaf] = c
		if c.SHA == "" || c.When.IsZero() {
			t.Fatalf("commit missing sha or time: %+v", c)
		}
	}
	if got := byLeaf["a"].Issues; len(got) != 1 || got[0] != 41 {
		t.Fatalf("subject refs = %v, want [41]", got)
	}
	b := byLeaf["b"]
	if len(b.Issues) != 1 || b.Issues[0] != 42 {
		t.Fatalf("body refs = %v, want [42] — the trailer counts, the prose does not", b.Issues)
	}
	if !strings.HasPrefix(b.Subject, "fix(b):") {
		t.Fatalf("subject was contaminated by the body: %q", b.Subject)
	}
	// A commit with no `(fak <leaf>)` trailer must parse with an empty leaf,
	// not fail: its absence is a hygiene fact this package reports on.
	if c, ok := byLeaf[""]; !ok || len(c.Issues) != 0 {
		t.Fatalf("leafless commit mishandled: %+v (present=%v)", c, ok)
	}

	rev, err := HeadRev(ctx, root)
	if err != nil || len(rev) < 7 {
		t.Fatalf("HeadRev = %q, %v", rev, err)
	}
	// The limit must actually bound the read, or a windowed report silently
	// folds the whole history.
	if got, err := GatherCommits(ctx, root, 1); err != nil || len(got) != 1 {
		t.Fatalf("limited gather = %d commits, %v; want 1", len(got), err)
	}
}

func TestGatherTreeCensusesUnlandedWork(t *testing.T) {
	root := tempRepo(t)
	ctx := context.Background()

	writeFile(t, root, "tracked.go", "package t\n\nvar A = 1\n")
	writeFile(t, root, "notes.txt", "not source\n")
	gitCommit(t, root, "chore: seed (fak x)", "tracked.go", "notes.txt")

	// Modify the tracked file, and leave three untracked sources: one normal,
	// one scratch probe, one dot-prefixed (invisible to the go tool).
	writeFile(t, root, "tracked.go", "package t\n\nvar A = 1\nvar B = 2\nvar C = 3\n")
	writeFile(t, root, "newwork.go", "package t\n")
	writeFile(t, root, "zz_probe.go", "package t\n")
	writeFile(t, root, ".hidden.go", "package t\n")
	// A non-source untracked file must not be censused at all.
	writeFile(t, root, "scratch.txt", "ignore me\n")

	tree, err := GatherTree(ctx, root, time.Now())
	if err != nil {
		t.Fatalf("GatherTree: %v", err)
	}
	if !tree.Measured {
		t.Fatalf("tree must report Measured after a successful gather")
	}
	if tree.Rev == "" {
		t.Fatalf("tree census must pin the rev it was taken at")
	}
	if tree.UntrackedGo != 3 {
		t.Fatalf("untracked_go = %d, want 3 (newwork, zz_probe, .hidden — not scratch.txt)", tree.UntrackedGo)
	}
	if tree.ModifiedGo != 1 {
		t.Fatalf("modified_go = %d, want 1", tree.ModifiedGo)
	}
	// zz_probe.go and .hidden.go are both litter; .hidden.go is also hidden.
	if tree.ScratchLitter != 2 {
		t.Fatalf("scratch_litter = %d, want 2", tree.ScratchLitter)
	}
	if tree.HiddenGo != 1 {
		t.Fatalf("hidden_go = %d, want 1", tree.HiddenGo)
	}
	if tree.AddedLines != 2 {
		t.Fatalf("added_lines = %d, want 2 (tracked diff only; untracked files are not in git diff)", tree.AddedLines)
	}
	if tree.RecentWriters < 1 {
		t.Fatalf("recent_writers = %d, want at least 1 for files just written", tree.RecentWriters)
	}
	// Buildability was never probed, so it must not claim a verdict.
	if tree.BuildProbed || tree.Buildable {
		t.Fatalf("unprobed tree must not report a build verdict: %+v", tree)
	}
	if tree.StatFailures != 0 {
		t.Fatalf("stat_failures = %d, want 0 — every dirty path must resolve", tree.StatFailures)
	}
	if tree.RecentWriters != 4 {
		t.Fatalf("recent_writers = %d, want 4 (all dirty sources were just written)", tree.RecentWriters)
	}
}

func TestGatherTreeResolvesPathsFromASubdirectory(t *testing.T) {
	// git status prints paths relative to the repo TOPLEVEL, not to the
	// directory git ran in. Gathering from a subdirectory used to fail every
	// os.Stat, and because the mtime fields default to 0 the result was
	// indistinguishable from a pristine tree — a silent false clean.
	root := tempRepo(t)
	ctx := context.Background()

	writeFile(t, root, "sub/pkg/tracked.go", "package pkg\n")
	gitCommit(t, root, "chore: seed (fak x)", "sub/pkg/tracked.go")
	writeFile(t, root, "sub/pkg/newwork.go", "package pkg\n")

	sub := filepath.Join(root, "sub", "pkg")
	tree, err := GatherTree(ctx, sub, time.Now())
	if err != nil {
		t.Fatalf("GatherTree from subdir: %v", err)
	}
	if tree.StatFailures != 0 {
		t.Fatalf("stat_failures = %d from a subdirectory, want 0", tree.StatFailures)
	}
	if tree.UntrackedGo != 1 {
		t.Fatalf("untracked_go = %d, want 1", tree.UntrackedGo)
	}
	if tree.RecentWriters != 1 {
		t.Fatalf("recent_writers = %d, want 1 — an unresolvable path reads as a quiet tree", tree.RecentWriters)
	}
}

func TestIsScratchNameRequiresASeparatorAfterZZ(t *testing.T) {
	cases := map[string]bool{
		"zz_probe.go": true,
		"zz-probe.go": true,
		"zz1.go":      true,
		".hidden.go":  true,
		"zzip.go":     false, // a legitimate name, not litter
		"puzzle.go":   false,
		"gateway.go":  false,
		"zz.go":       false, // no separator or digit after the prefix
	}
	for name, want := range cases {
		if got := isScratchName(name); got != want {
			t.Fatalf("isScratchName(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestDecodeIssuesMapsTheGhWireShape(t *testing.T) {
	raw := []byte(`[
	  {"number":1,"title":"open one","createdAt":"2026-08-01T00:00:00Z","closedAt":null,
	   "labels":[{"name":"epic"},{"name":""}],"body":"- [ ] #2"},
	  {"number":2,"title":"closed one","createdAt":"2026-08-01T00:00:00Z",
	   "closedAt":"2026-08-02T00:00:00Z","labels":[],"body":""},
	  {"number":3,"title":"zero closedAt","createdAt":"2026-08-01T00:00:00Z",
	   "closedAt":"0001-01-01T00:00:00Z","labels":null,"body":""},
	  {"number":0,"title":"bogus","createdAt":"2026-08-01T00:00:00Z"}
	]`)
	issues, err := DecodeIssues(raw)
	if err != nil {
		t.Fatalf("DecodeIssues: %v", err)
	}
	if len(issues) != 3 {
		t.Fatalf("issues = %d, want 3 (number 0 dropped)", len(issues))
	}
	if issues[0].Closed() {
		t.Fatalf("null closedAt must read as open")
	}
	if len(issues[0].Labels) != 1 || issues[0].Labels[0] != "epic" {
		t.Fatalf("labels = %v, want [epic] with the empty name dropped", issues[0].Labels)
	}
	if !issues[1].Closed() {
		t.Fatalf("issue 2 must read as closed")
	}
	// The zero-time close is the trap: read literally it makes the issue look
	// closed two millennia before it opened.
	if issues[2].Closed() {
		t.Fatalf("a zero closedAt must read as OPEN, not as a year-1 close")
	}

	if _, err := DecodeIssues([]byte(`{"not":"an array"}`)); err == nil {
		t.Fatalf("a non-array payload must be an error, not an empty backlog")
	}
}

func TestLoadIssuesFileRoundTrips(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "issues.json")
	writeFile(t, dir, "issues.json",
		`[{"number":7,"title":"t","createdAt":"2026-08-01T00:00:00Z","closedAt":null,"body":""}]`)
	issues, err := LoadIssuesFile(path)
	if err != nil {
		t.Fatalf("LoadIssuesFile: %v", err)
	}
	if len(issues) != 1 || issues[0].Number != 7 {
		t.Fatalf("issues = %+v, want one issue #7", issues)
	}
	if _, err := LoadIssuesFile(filepath.Join(dir, "absent.json")); err == nil {
		t.Fatalf("a missing dump must error rather than yield an empty backlog")
	}
}

// TestLiveGitGatherIsWellFormed runs the real git path against this checkout and
// pins the SHAPE of what comes back. It deliberately asserts no thresholds: the
// numbers move with every commit, and a shape test that also gated tree state
// would fail for reasons unrelated to this package.
//
// The issue side is not exercised — it needs `gh` and the network — so this
// covers git and the filesystem only.
func TestLiveGitGatherIsWellFormed(t *testing.T) {
	if testing.Short() {
		t.Skip("shells out to git; run without -short")
	}
	ctx := context.Background()
	root := ".."
	commits, err := GatherCommits(ctx, root, 200)
	if err != nil {
		t.Skipf("live git gather unavailable: %v", err)
	}
	if len(commits) == 0 {
		t.Fatalf("live gather returned no commits")
	}
	withLeaf, withRefs := 0, 0
	for _, c := range commits {
		if c.SHA == "" || c.When.IsZero() || c.Subject == "" {
			t.Fatalf("live commit missing a required field: %+v", c)
		}
		if c.Leaf != "" {
			withLeaf++
		}
		for _, n := range c.Issues {
			if n <= 0 || n >= ForeignRefFloor {
				t.Fatalf("commit %s yielded an out-of-range ref %d", c.SHA, n)
			}
			withRefs++
		}
	}
	// The repo mandates the `(fak <leaf>)` trailer, so a run that parsed zero
	// leaves out of 200 real commits means the trailer regex has drifted.
	if withLeaf == 0 {
		t.Fatalf("parsed 0 leaf trailers from %d live commits — leafRE has drifted", len(commits))
	}
	if withRefs == 0 {
		t.Fatalf("parsed 0 issue refs from %d live commits — issueRefRE has drifted", len(commits))
	}

	tree, err := GatherTree(ctx, root, time.Now())
	if err != nil {
		t.Skipf("live tree census unavailable: %v", err)
	}
	if !tree.Measured || tree.Rev == "" {
		t.Fatalf("live census must be Measured and rev-pinned: %+v", tree)
	}

	// The whole fold must run on live git facts without panicking, and must
	// emit a well-formed envelope even with no issue records at all.
	rep := Build(Input{Commits: commits, Tree: tree, Now: time.Now(), WindowDays: 30})
	if rep.Schema != Schema || rep.Verdict == "" || rep.Reason == "" || rep.NextAction == "" {
		t.Fatalf("live report missing a top-level field: %+v", rep)
	}
	if len(rep.KPIs) != 8 {
		t.Fatalf("live report has %d KPIs, want 8", len(rep.KPIs))
	}
	for _, k := range rep.KPIs {
		if k.KPI == "" || k.Group == "" {
			t.Fatalf("live KPI missing kpi/group: %+v", k)
		}
		if k.Defects == nil || k.Soft == nil {
			t.Fatalf("KPI %q must carry non-nil defects/soft (JSON [] not null)", k.KPI)
		}
	}
}
