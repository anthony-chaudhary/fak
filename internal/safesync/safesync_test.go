package safesync

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseNameStatusZ(t *testing.T) {
	got := parseNameStatusZ([]byte("M\x00a.txt\x00A\x00new.txt\x00R100\x00old.txt\x00renamed.txt\x00"))
	want := []Entry{
		{Status: "M", Path: "a.txt"},
		{Status: "A", Path: "new.txt"},
		{Status: "R100", Path: "old.txt"},
		{Status: "R100", Path: "renamed.txt"},
	}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%+v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("entry %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestBehindDirtyIdenticalIsSafeAndApplies(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, "a.txt"), "v2\n")     // M, identical to target
	writeFile(t, filepath.Join(clone, "new.txt"), "n1\n")   // A, identical to target
	writeFile(t, filepath.Join(clone, "mine.txt"), "local") // unrelated dirty work

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateBehind || !info.OK || len(info.Divergent) != 0 {
		t.Fatalf("assessment = %+v, want safe behind", info)
	}
	if info.WriteCount != 2 {
		t.Fatalf("write count = %d, want 2", info.WriteCount)
	}

	applied, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if !applied.Applied {
		t.Fatalf("apply did not apply: %+v", applied)
	}
	if got, want := revString(t, clone, "HEAD"), revString(t, clone, "origin/work"); got != want {
		t.Fatalf("HEAD = %s, want origin/work %s", got, want)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "v2\n" {
		t.Fatalf("a.txt = %q", got)
	}
	if got := readFile(t, filepath.Join(clone, "new.txt")); got != "n1\n" {
		t.Fatalf("new.txt = %q", got)
	}
	if got := readFile(t, filepath.Join(clone, "mine.txt")); got != "local" {
		t.Fatalf("unrelated work was not preserved: %q", got)
	}
}

func TestBehindCleanTrackedUpdateIsSafeAndApplies(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, "mine.txt"), "local") // unrelated dirty work

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateBehind || !info.OK {
		t.Fatalf("assessment = %+v, want safe behind with clean tracked update", info)
	}
	if len(info.Identical) != 2 {
		t.Fatalf("identical/safe entries = %+v, want a.txt M and new.txt A", info.Identical)
	}

	applied, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if !applied.Applied {
		t.Fatalf("apply did not apply: %+v", applied)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "v2\n" {
		t.Fatalf("a.txt = %q", got)
	}
	if got := readFile(t, filepath.Join(clone, "new.txt")); got != "n1\n" {
		t.Fatalf("new.txt = %q", got)
	}
	if got := readFile(t, filepath.Join(clone, "mine.txt")); got != "local" {
		t.Fatalf("unrelated work was not preserved: %q", got)
	}
}

func TestBehindDivergentTrackedRefusesAndDoesNotMoveHead(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, "a.txt"), "LOCAL EDIT\n")
	headBefore := revString(t, clone, "HEAD")

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.OK || len(info.Divergent) != 1 || info.Divergent[0].Path != "a.txt" {
		t.Fatalf("assessment = %+v, want a.txt divergent", info)
	}

	applied, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if applied.Applied {
		t.Fatalf("apply should have refused: %+v", applied)
	}
	if got := revString(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("HEAD moved on refusal: got %s want %s", got, headBefore)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "LOCAL EDIT\n" {
		t.Fatalf("worktree changed on refusal: %q", got)
	}
}

func TestInSyncIsOKNoop(t *testing.T) {
	clone := behindClone(t)
	git(t, clone, "merge", "--ff-only", "origin/work")

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateInSync || !info.OK || info.Applied {
		t.Fatalf("apply in sync = %+v, want ok noop", info)
	}
}

func TestDivergedRefuses(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, "local.txt"), "local\n")
	git(t, clone, "add", "local.txt")
	git(t, clone, "commit", "-m", "local")

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateDiverged || info.OK {
		t.Fatalf("assessment = %+v, want diverged refusal", info)
	}
}

func TestRenameWriteSetRefuses(t *testing.T) {
	clone := behindClone(t)
	head := revString(t, clone, "HEAD")
	target := revString(t, clone, "origin/work")
	entries := []Entry{{Status: "R100", Path: "old.txt"}, {Status: "R100", Path: "new.txt"}}
	_, divergent := classify(clone, RealRunner, context.Background(), head, target, entries)
	if len(divergent) != 2 {
		t.Fatalf("rename entries should be divergent/refused, got %+v", divergent)
	}
}

func behindClone(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	origin := filepath.Join(tmp, "origin")
	mkdir(t, origin)
	git(t, origin, "init", "-b", "work")
	git(t, origin, "config", "core.autocrlf", "false")
	git(t, origin, "config", "user.name", "test")
	git(t, origin, "config", "user.email", "test@example.com")
	writeFile(t, filepath.Join(origin, "a.txt"), "v1\n")
	writeFile(t, filepath.Join(origin, "keep.txt"), "keep\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "c1")

	clone := filepath.Join(tmp, "clone")
	git(t, tmp, "clone", origin, clone)
	git(t, clone, "config", "core.autocrlf", "false")
	git(t, clone, "config", "user.name", "test")
	git(t, clone, "config", "user.email", "test@example.com")

	writeFile(t, filepath.Join(origin, "a.txt"), "v2\n")
	writeFile(t, filepath.Join(origin, "new.txt"), "n1\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "c2")
	git(t, clone, "fetch", "origin")
	return clone
}

func git(t *testing.T, cwd string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, cwd, err, out)
	}
}

func revString(t *testing.T, cwd, ref string) string {
	t.Helper()
	cmd := exec.Command("git", "rev-parse", "--verify", ref)
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("rev-parse %s: %v", ref, err)
	}
	return strings.TrimSpace(string(out))
}

func mkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
}

func writeFile(t *testing.T, path, text string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
