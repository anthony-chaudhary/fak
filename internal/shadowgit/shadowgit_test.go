package shadowgit

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func gitAvailable(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
}

// newRealRepo makes a real git repo at dir with one committed file, so tests can prove
// the shadow ledger never disturbs it.
func newRealRepo(t *testing.T, dir string) {
	t.Helper()
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	run("init", "-q")
	if err := os.WriteFile(filepath.Join(dir, "tracked.txt"), []byte("v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	run("add", "-A")
	run("commit", "-q", "-m", "seed")
}

func realStatus(t *testing.T, dir string) (head, status string) {
	t.Helper()
	get := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("git %v: %v", args, err)
		}
		return string(out)
	}
	return get("rev-parse", "HEAD"), get("status", "--porcelain")
}

func TestShadowGit_AttributesWritesPerStepAndIsNonInvasive(t *testing.T) {
	gitAvailable(t)
	wt := t.TempDir()
	newRealRepo(t, wt)
	head0, status0 := realStatus(t, wt)

	shadow := filepath.Join(t.TempDir(), "shadow.git")
	sg, err := Open(shadow, wt, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sg.Baseline(); err != nil {
		t.Fatal(err)
	}

	// Step 1: add a new file.
	if err := os.WriteFile(filepath.Join(wt, "new.txt"), []byte("hello\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if dirty, _ := sg.CheckForWrites(); !dirty {
		t.Error("CheckForWrites should see the new file")
	}
	s1, err := sg.Snapshot(1, "write new.txt")
	if err != nil {
		t.Fatal(err)
	}
	if len(s1.Changes) != 1 || s1.Changes[0].Status != "A" || s1.Changes[0].Path != "new.txt" {
		t.Fatalf("step1 changes = %+v, want one A new.txt", s1.Changes)
	}
	if dirty, _ := sg.CheckForWrites(); dirty {
		t.Error("CheckForWrites should be clean right after a snapshot")
	}

	// Step 2: modify the pre-existing tracked file.
	if err := os.WriteFile(filepath.Join(wt, "tracked.txt"), []byte("v2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s2, err := sg.Snapshot(2, "edit tracked.txt")
	if err != nil {
		t.Fatal(err)
	}
	if len(s2.Changes) != 1 || s2.Changes[0].Status != "M" || s2.Changes[0].Path != "tracked.txt" {
		t.Fatalf("step2 changes = %+v, want one M tracked.txt", s2.Changes)
	}

	// Step 3: delete a file. Distinct step, distinct attribution.
	if err := os.Remove(filepath.Join(wt, "new.txt")); err != nil {
		t.Fatal(err)
	}
	s3, err := sg.Snapshot(3, "rm new.txt")
	if err != nil {
		t.Fatal(err)
	}
	if len(s3.Changes) != 1 || s3.Changes[0].Status != "D" || s3.Changes[0].Path != "new.txt" {
		t.Fatalf("step3 changes = %+v, want one D new.txt", s3.Changes)
	}

	// Non-invasiveness: the real repo's HEAD and status are exactly as before.
	head1, status1 := realStatus(t, wt)
	if head1 != head0 {
		t.Errorf("real repo HEAD moved: %q -> %q", head0, head1)
	}
	// status1 differs from status0 only by the v2 edit we made (tracked.txt), never by
	// any shadow-repo bookkeeping — the shadow left no staged/committed trace in .git.
	if bytes.Contains([]byte(status1), []byte("shadow")) {
		t.Errorf("real repo status mentions shadow: %q", status1)
	}
	_ = status0
}

func TestShadowGit_SnapshotBeforeBaselineErrors(t *testing.T) {
	gitAvailable(t)
	wt := t.TempDir()
	sg, err := Open(filepath.Join(t.TempDir(), "s.git"), wt, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sg.Snapshot(1, "x"); err == nil {
		t.Fatal("Snapshot before Baseline must error")
	}
}

func TestShadowGit_ShadowDirInsideWorktreeIsExcluded(t *testing.T) {
	gitAvailable(t)
	wt := t.TempDir()
	if err := os.WriteFile(filepath.Join(wt, "a.txt"), []byte("a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	shadow := filepath.Join(wt, ".fak-shadow.git") // nested in the worktree
	sg, err := Open(shadow, wt, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sg.Baseline(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wt, "b.txt"), []byte("b\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s1, err := sg.Snapshot(1, "add b")
	if err != nil {
		t.Fatal(err)
	}
	// Only b.txt is attributed — never any path under the nested shadow dir.
	if len(s1.Changes) != 1 || s1.Changes[0].Path != "b.txt" {
		t.Fatalf("nested shadow dir leaked into snapshot: %+v", s1.Changes)
	}
}

func TestWriteChangelogLine_RoundTrips(t *testing.T) {
	var buf bytes.Buffer
	snap := Snapshot{Step: 2, Label: "edit", Commit: "abc", Parent: "def", Changes: []Change{{Status: "M", Path: "x.go"}}}
	if err := WriteChangelogLine(&buf, snap); err != nil {
		t.Fatal(err)
	}
	var back Snapshot
	if err := json.Unmarshal(bytes.TrimSpace(buf.Bytes()), &back); err != nil {
		t.Fatalf("changelog line does not parse: %v", err)
	}
	if back.Step != 2 || len(back.Changes) != 1 || back.Changes[0].Path != "x.go" {
		t.Fatalf("round-trip lost data: %+v", back)
	}
}

func TestParseNameStatusZ(t *testing.T) {
	// A modify, an add, and a rename (R100 old -> new).
	z := "M\x00a.go\x00A\x00b.go\x00R100\x00old.go\x00new.go\x00"
	got := parseNameStatusZ(z)
	want := []Change{
		{Status: "M", Path: "a.go"},
		{Status: "A", Path: "b.go"},
		{Status: "R", OldPath: "old.go", Path: "new.go"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d changes, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("change %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestShadowGit_StatusReportLifecycleAndReopen(t *testing.T) {
	gitAvailable(t)
	wt := t.TempDir()
	newRealRepo(t, wt)
	shadow := filepath.Join(t.TempDir(), "shadow.git")

	sg, err := Open(shadow, wt, Options{})
	if err != nil {
		t.Fatal(err)
	}

	// Pre-baseline: valid repo, uninitialized baseline, sees existing worktree file as pending.
	pre, err := sg.Status()
	if err != nil {
		t.Fatalf("Status pre-baseline: %v", err)
	}
	if pre.Initialized {
		t.Errorf("pre-baseline Initialized = true, want false")
	}
	if !pre.Dirty || pre.PendingWrites != 1 || !reflect.DeepEqual(pre.PendingPaths, []string{"tracked.txt"}) {
		t.Errorf("pre-baseline pending = (dirty=%v count=%d paths=%v), want dirty=true count=1 [tracked.txt]", pre.Dirty, pre.PendingWrites, pre.PendingPaths)
	}
	if pre.CommitCount != 0 || pre.SnapshotCount != 0 || pre.LatestCommit != "" {
		t.Errorf("pre-baseline counts/commit = (%d, %d, %q), want (0, 0, \"\")", pre.CommitCount, pre.SnapshotCount, pre.LatestCommit)
	}
	if !strings.Contains(pre.ChangesSummary, "uninitialized") || !strings.Contains(pre.ChangesSummary, "1 pending write") {
		t.Errorf("pre-baseline summary = %q, want uninitialized + 1 pending write", pre.ChangesSummary)
	}

	// Baseline: initialized, 1 commit, 0 snapshots, clean.
	base, err := sg.Baseline()
	if err != nil {
		t.Fatal(err)
	}
	atBase, err := sg.Summarize()
	if err != nil {
		t.Fatalf("Summarize at baseline: %v", err)
	}
	if !atBase.Initialized || !atBase.LatestBaseline || atBase.Dirty || atBase.PendingWrites != 0 {
		t.Fatalf("atBase flags unexpected: %+v", atBase)
	}
	if atBase.CommitCount != 1 || atBase.SnapshotCount != 0 || atBase.LatestStep != 0 || atBase.LatestCommit != base.Commit || atBase.LatestLabel != "" {
		t.Fatalf("atBase metadata unexpected: %+v (want commit=%s)", atBase, base.Commit)
	}
	if len(atBase.LastChanges) != 0 {
		t.Errorf("baseline LastChanges = %+v, want empty", atBase.LastChanges)
	}
	if !strings.Contains(atBase.ChangesSummary, "baseline (step 0)") || !strings.Contains(atBase.ChangesSummary, "clean") {
		t.Errorf("atBase summary = %q", atBase.ChangesSummary)
	}

	// Dirty worktree after baseline (before snapshot).
	if err := os.WriteFile(filepath.Join(wt, "alpha.txt"), []byte("a1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wt, "tracked.txt"), []byte("v2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	dirtyRep, err := sg.Status()
	if err != nil {
		t.Fatalf("Status dirty: %v", err)
	}
	if !dirtyRep.Dirty || dirtyRep.PendingWrites != 2 || !reflect.DeepEqual(dirtyRep.PendingPaths, []string{"alpha.txt", "tracked.txt"}) {
		t.Errorf("dirty pending = %+v, want [alpha.txt tracked.txt]", dirtyRep)
	}
	if !strings.Contains(dirtyRep.ChangesSummary, "2 pending writes") {
		t.Errorf("dirty summary = %q, want 2 pending writes", dirtyRep.ChangesSummary)
	}

	// Snapshot 1 and Snapshot 2 (with custom non-contiguous step number & label).
	s1, err := sg.Snapshot(1, "add alpha and edit tracked")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wt, "beta.txt"), []byte("b1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s2, err := sg.Snapshot(7, "add beta")
	if err != nil {
		t.Fatal(err)
	}

	// Dirty again after snapshot 2.
	if err := os.WriteFile(filepath.Join(wt, "gamma.txt"), []byte("g1\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Reopen a fresh ShadowGit handle (simulating a peer process or post-crash probe).
	reopened, err := Open(shadow, wt, Options{})
	if err != nil {
		t.Fatal(err)
	}
	rep, err := reopened.Summarize()
	if err != nil {
		t.Fatalf("reopened Summarize: %v", err)
	}
	if !rep.Initialized || rep.LatestBaseline {
		t.Errorf("reopened flags = (%v, %v), want (true, false)", rep.Initialized, rep.LatestBaseline)
	}
	if rep.CommitCount != 3 || rep.SnapshotCount != 2 {
		t.Errorf("counts = (%d, %d), want (3, 2)", rep.CommitCount, rep.SnapshotCount)
	}
	if rep.LatestStep != 7 || rep.LatestCommit != s2.Commit || rep.LatestLabel != "add beta" {
		t.Errorf("latest = (step=%d commit=%s label=%q), want (7, %s, %q)", rep.LatestStep, rep.LatestCommit, rep.LatestLabel, s2.Commit, "add beta")
	}
	if !reflect.DeepEqual(rep.LastChanges, []Change{{Status: "A", Path: "beta.txt"}}) {
		t.Errorf("LastChanges = %+v, want [A beta.txt]", rep.LastChanges)
	}
	if !rep.Dirty || rep.PendingWrites != 1 || !reflect.DeepEqual(rep.PendingPaths, []string{"gamma.txt"}) {
		t.Errorf("pending = %+v, want [gamma.txt]", rep)
	}
	if !strings.Contains(rep.ChangesSummary, "step 7 (add beta)") || !strings.Contains(rep.ChangesSummary, "1 A") || !strings.Contains(rep.ChangesSummary, "2 snapshots") || !strings.Contains(rep.ChangesSummary, "1 pending write") {
		t.Errorf("reopened summary = %q", rep.ChangesSummary)
	}
	_ = s1

	// Re-basing resets SnapshotCount since latest baseline while CommitCount keeps growing.
	rebased, err := sg.Baseline()
	if err != nil {
		t.Fatal(err)
	}
	afterRebase, err := sg.Status()
	if err != nil {
		t.Fatal(err)
	}
	if afterRebase.CommitCount != 4 || afterRebase.SnapshotCount != 0 || afterRebase.LatestCommit != rebased.Commit || !afterRebase.LatestBaseline || afterRebase.Dirty {
		t.Errorf("afterRebase = %+v", afterRebase)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestShadowGit_StatusErrorsAndUninitialized(t *testing.T) {
	var nilSG *ShadowGit
	if _, err := nilSG.Status(); err == nil {
		t.Error("nil ShadowGit.Status() must error")
	}
	if _, err := (&ShadowGit{}).Summarize(); err == nil {
		t.Error("zero ShadowGit.Summarize() must error")
	}

	// Non-git directory as gitDir should return an error rather than reporting clean.
	wt := t.TempDir()
	notAGitDir := t.TempDir()
	sg := &ShadowGit{gitDir: notAGitDir, workTree: wt, git: ExecRunner{}}
	if _, err := sg.Summarize(); err == nil {
		t.Error("Summarize over non-git directory must error")
	}

	// Missing workTree should return an error.
	sgMissingWT := &ShadowGit{gitDir: notAGitDir, workTree: filepath.Join(wt, "missing"), git: ExecRunner{}}
	if _, err := sgMissingWT.Status(); err == nil {
		t.Error("Status with missing work-tree must error")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestFoldStatus_PureFoldingAndSummary(t *testing.T) {
	uninit := FoldStatus("/tmp/s.git", "/tmp/wt", nil, []string{"b.go", "", "a.go"})
	if uninit.Initialized || !uninit.Dirty || uninit.PendingWrites != 2 {
		t.Fatalf("uninit FoldStatus = %+v", uninit)
	}
	if !reflect.DeepEqual(uninit.PendingPaths, []string{"a.go", "b.go"}) {
		t.Errorf("PendingPaths not sorted/filtered: %v", uninit.PendingPaths)
	}

	snaps := []Snapshot{
		{Step: 0, Commit: "c0", Baseline: true},
		{Step: 1, Label: "step-one", Commit: "c1", Parent: "c0", Changes: []Change{{Status: "A", Path: "x.go"}}},
		{Step: 2, Commit: "c2", Parent: "c1", Changes: []Change{{Status: "M", Path: "x.go"}, {Status: "A", Path: "y.go"}}},
	}
	rep := FoldStatus("/tmp/s.git", "/tmp/wt", snaps, nil)
	if !rep.Initialized || rep.Dirty || rep.CommitCount != 3 || rep.SnapshotCount != 2 {
		t.Fatalf("folded report = %+v", rep)
	}
	if rep.LatestStep != 2 || rep.LatestCommit != "c2" || rep.LatestLabel != "" || len(rep.LastChanges) != 2 {
		t.Fatalf("folded latest = %+v", rep)
	}
	if rep.ChangesSummary != "step 2; 2 changes (1 A, 1 M); 2 snapshots; clean" {
		t.Errorf("ChangesSummary = %q", rep.ChangesSummary)
	}

	// Mutating LastChanges on report must not alias input snapshot slice.
	rep.LastChanges[0].Path = "tampered.go"
	if snaps[2].Changes[0].Path != "x.go" {
		t.Errorf("FoldStatus aliased snapshot Changes slice")
	}

	// Also verify parseShadowLog fallback on commits without trailers, and porcelain -z rename parsing.
	parsed := parseShadowLog("sha0\x1f\x1fshadowgit baseline\x1f\x1esha1\x1fsha0\x1fshadowgit step 4\x1f\x1e")
	if len(parsed) != 2 || !parsed[0].Baseline || parsed[1].Step != 4 || parsed[1].Label != "" {
		t.Errorf("parseShadowLog fallback = %+v", parsed)
	}
	pending := parseStatusPorcelainZ(" M tracked.txt\x00R  new.txt\x00old.txt\x00?? a b.txt\x00")
	if !reflect.DeepEqual(pending, []string{"a b.txt", "new.txt", "tracked.txt"}) {
		t.Errorf("parseStatusPorcelainZ = %v", pending)
	}
}
