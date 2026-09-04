package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestLiveEndToEnd drives Prepare -> edit-in-worktree -> Land -> Reap against a
// REAL throwaway git repo (no fake): proves the detached worktree is created, a
// worker's in-worktree commit lands onto the trunk keeping its own subject, and
// the worktree is reaped. The #3168 done-condition, witnessed.
func TestLiveEndToEnd(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	run := func(args ...string) string {
		c := exec.Command("git", args...)
		c.Dir = repo
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	run("init", "-q", "-b", "main")
	run("config", "user.email", "e2e@test")
	run("config", "user.name", "e2e")
	run("config", "commit.gpgsign", "false")
	os.WriteFile(filepath.Join(repo, "app.go"), []byte("package app\n\nfunc A() int { return 1 }\n"), 0o644)
	run("add", "app.go")
	run("commit", "-q", "-m", "base")

	base := TrunkHeadSHA(repo, nil)
	if base == "" {
		t.Fatal("no trunk head")
	}

	wtRoot := t.TempDir()
	res := Prepare(repo, "app", "3168", base, wtRoot, nil)
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	// The worktree exists and is enumerated from git.
	n, paths := Count(repo, nil)
	if n != 1 {
		t.Fatalf("count = %d (%v), want 1 live worker worktree", n, paths)
	}

	// Worker edits AND commits INSIDE its detached worktree (the ship-when-green
	// path), with its own stamped subject citing the issue.
	os.WriteFile(filepath.Join(res.Path, "app.go"), []byte("package app\n\nfunc A() int { return 42 }\n"), 0o644)
	wc := func(args ...string) {
		c := exec.Command("git", args...)
		c.Dir = res.Path
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("worktree git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
	}
	wc("config", "user.email", "worker@test")
	wc("config", "user.name", "worker")
	wc("config", "commit.gpgsign", "false")
	wc("add", "app.go")
	wc("commit", "-q", "-m", "fix(app): return 42 (#3168) (fak app)")

	// git diff HEAD in the worktree is EMPTY (worker committed) — the base-diff is
	// what captures the change. Land onto the trunk scoped to app.go.
	land := Land(repo, res.Path, base, "", []string{"app.go"}, nil, nil)
	if !land.OK || !land.Committed {
		t.Fatalf("land: %+v", land)
	}

	// The trunk now carries the worker's OWN subject as a real commit.
	subj := strings.TrimSpace(run("log", "-1", "--format=%s"))
	if subj != "fix(app): return 42 (#3168) (fak app)" {
		t.Fatalf("landed subject = %q, want the worker's own stamped subject", subj)
	}
	// And it is signed off (DCO) on main.
	body := run("log", "-1", "--format=%B")
	if !strings.Contains(body, "Signed-off-by:") {
		t.Fatalf("landed commit not signed off:\n%s", body)
	}
	// The trunk file actually changed.
	got, _ := os.ReadFile(filepath.Join(repo, "app.go"))
	if !strings.Contains(string(got), "return 42") {
		t.Fatalf("trunk app.go did not get the worker edit:\n%s", got)
	}

	// Reap removes the worktree.
	reap := Reap(repo, res.Path, nil)
	if !reap.OK || !reap.Removed {
		t.Fatalf("reap: %+v", reap)
	}
	if n, _ := Count(repo, nil); n != 0 {
		t.Fatalf("worktree still present after reap: %d", n)
	}
}

func TestLiveLandNewDirectoryContentsAtomically(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	t.Setenv(IsolatedLandEnv, "1")
	repo := t.TempDir()
	run := func(dir string, args ...string) string {
		c := exec.Command("git", args...)
		c.Dir = dir
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	run(repo, "init", "-q", "-b", "main")
	run(repo, "config", "user.email", "e2e@test")
	run(repo, "config", "user.name", "e2e")
	run(repo, "config", "commit.gpgsign", "false")
	run(repo, "config", "core.autocrlf", "false")
	os.WriteFile(filepath.Join(repo, "base.txt"), []byte("base\n"), 0o644)
	run(repo, "add", "base.txt")
	run(repo, "commit", "-q", "-m", "base")
	base := TrunkHeadSHA(repo, nil)
	res := Prepare(repo, "workerworktree", "9129", base, t.TempDir(), nil)
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	dir := filepath.Join(res.Path, "proof", "nested")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "witness.txt"), []byte("atomic\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	msg := filepath.Join(res.Path, "message.txt")
	os.WriteFile(msg, []byte("fix(workerworktree): atomic directory (#9129) (fak workerworktree)\n"), 0o644)
	land := Land(repo, res.Path, base, msg, []string{"proof"}, nil, nil)
	if !land.OK || !land.Committed {
		t.Fatalf("land: %+v", land)
	}
	for _, check := range [][]string{{"show", "HEAD:proof/nested/witness.txt"}, {"show", ":proof/nested/witness.txt"}} {
		if got := run(repo, check...); got != "atomic\n" {
			t.Fatalf("git %v = %q", check, got)
		}
	}
	got, err := os.ReadFile(filepath.Join(repo, "proof", "nested", "witness.txt"))
	if err != nil || string(got) != "atomic\n" {
		t.Fatalf("worktree witness = %q, %v", got, err)
	}
}

// TestLiveLandCommittedBinaryAddition preserves the binary patch payload and
// full object IDs needed to apply a file that exists only in the worker commit.
func TestLiveLandCommittedBinaryAddition(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	run := func(dir string, args ...string) string {
		c := exec.Command("git", args...)
		c.Dir = dir
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	run(repo, "init", "-q", "-b", "main")
	run(repo, "config", "user.email", "e2e@test")
	run(repo, "config", "user.name", "e2e")
	run(repo, "config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(repo, "base.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	run(repo, "add", "base.txt")
	run(repo, "commit", "-q", "-m", "base")
	base := TrunkHeadSHA(repo, nil)

	res := Prepare(repo, "workerworktree", "9821", base, t.TempDir(), nil)
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	binaryPath := filepath.Join(res.Path, "proof.png")
	wantBytes := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0x00, 0xff, 0x10, 0x80}
	if err := os.WriteFile(binaryPath, wantBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	run(res.Path, "config", "user.email", "worker@test")
	run(res.Path, "config", "user.name", "worker")
	run(res.Path, "config", "commit.gpgsign", "false")
	run(res.Path, "add", "proof.png")
	run(res.Path, "commit", "-q", "-m", "fix(workerland): add binary proof (#9821) (fak workerworktree)")
	workerBlob := strings.TrimSpace(run(res.Path, "rev-parse", "HEAD:proof.png"))

	land := Land(repo, res.Path, base, "", []string{"proof.png"}, nil, nil)
	if !land.OK || !land.Committed {
		t.Fatalf("land committed binary addition: %+v", land)
	}
	landedBlob := strings.TrimSpace(run(repo, "rev-parse", "HEAD:proof.png"))
	if landedBlob != workerBlob {
		t.Fatalf("landed blob = %s, worker blob = %s", landedBlob, workerBlob)
	}
	gotBytes, err := os.ReadFile(filepath.Join(repo, "proof.png"))
	if err != nil {
		t.Fatal(err)
	}
	if string(gotBytes) != string(wantBytes) {
		t.Fatalf("landed bytes = %x, want %x", gotBytes, wantBytes)
	}
}
