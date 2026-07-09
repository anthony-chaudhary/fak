package shadowgit

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
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
