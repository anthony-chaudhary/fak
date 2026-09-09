package workerworktree

import (
	"fmt"
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

// TestLiveLandWithDirtyWorkerEditsVerified proves #11978:
// managed worktree land with compilation verification completes successfully when
// the worker has uncommitted dirty edits.
// Post-merge validation runs against an isolated candidate checkout without
// attempting to checkout over the worker's dirty files, preserving the worker's
// uncommitted WIP while ensuring failed validation still refuses CAS.
func TestLiveLandWithDirtyWorkerEditsVerified(t *testing.T) {
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
	if err := os.WriteFile(filepath.Join(repo, "app.go"), []byte("package app\n\nfunc Version() int { return 1 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	run(repo, "add", "app.go")
	run(repo, "commit", "-q", "-m", "base")
	base := TrunkHeadSHA(repo, nil)

	res := Prepare(repo, "app", "11978", base, t.TempDir(), nil)
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}

	// Worker makes uncommitted edits in its worktree (dirty worker edits, NOT committed)
	dirtyContent := "package app\n\nfunc Version() int { return 2 }\n"
	if err := os.WriteFile(filepath.Join(res.Path, "app.go"), []byte(dirtyContent), 0o644); err != nil {
		t.Fatal(err)
	}

	// Part 1: Validation failure refuses CAS and preserves dirty edits in worker worktree
	failingVerify := func(candDir string) (bool, string) {
		if candDir == res.Path {
			// Prospective validation on the worker worktree passes
			return true, ""
		}
		return false, "simulated compile error in candidate"
	}
	landFail := Land(repo, res.Path, base, "", []string{"app.go"}, failingVerify, nil)
	if landFail.OK || landFail.Committed {
		t.Fatalf("expected land to fail on verify error, got %+v", landFail)
	}
	if !strings.Contains(landFail.Reason, "simulated compile error in candidate") {
		t.Fatalf("expected compilation error reason, got %q", landFail.Reason)
	}
	// Verify dirty edits in worker worktree were preserved
	workerGot, err := os.ReadFile(filepath.Join(res.Path, "app.go"))
	if err != nil || string(workerGot) != dirtyContent {
		t.Fatalf("worker dirty edits lost on failed verify: %v, content=%q", err, string(workerGot))
	}
	// Trunk HEAD must still be base
	if TrunkHeadSHA(repo, nil) != base {
		t.Fatalf("trunk HEAD moved despite failed verify")
	}

	// Part 2: Successful validation passes and preserves dirty edits in worker worktree
	validatorObservedCandidate := false
	passingVerify := func(candDir string) (bool, string) {
		if candDir == res.Path {
			// Prospective validation on the worker worktree passes
			return true, ""
		}
		candBytes, err := os.ReadFile(filepath.Join(candDir, "app.go"))
		if err != nil {
			return false, "read candidate app.go: " + err.Error()
		}
		if string(candBytes) != dirtyContent {
			return false, fmt.Sprintf("candidate app.go mismatch: got %q, want %q", string(candBytes), dirtyContent)
		}
		validatorObservedCandidate = true
		return true, ""
	}

	landPass := Land(repo, res.Path, base, "", []string{"app.go"}, passingVerify, nil)
	if !landPass.OK || !landPass.Committed {
		t.Fatalf("land failed with dirty worker edits: %+v", landPass)
	}
	if !validatorObservedCandidate {
		t.Fatalf("validator did not observe isolated candidate checkout")
	}

	// Verify worker worktree still has its uncommitted edits intact
	workerGotAfter, err := os.ReadFile(filepath.Join(res.Path, "app.go"))
	if err != nil || string(workerGotAfter) != dirtyContent {
		t.Fatalf("worker dirty edits overwritten during successful land: %v, content=%q", err, string(workerGotAfter))
	}

	// Verify trunk now has the landed change
	trunkGot, err := os.ReadFile(filepath.Join(repo, "app.go"))
	if err != nil || string(trunkGot) != dirtyContent {
		t.Fatalf("trunk did not receive candidate edits: %v, content=%q", err, string(trunkGot))
	}
}

// TestLiveLandWithSiblingWorkspaceVerified proves #12447: post-merge
// verification keeps the selected repository's parent-directory topology, so
// relative sibling modules named by a checked-in go.work remain resolvable.
func TestLiveLandWithSiblingWorkspaceVerified(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}

	parent := t.TempDir()
	dependency := filepath.Join(parent, "support")
	repo := filepath.Join(parent, "selected")
	for _, dir := range []string{dependency, repo} {
		if err := os.Mkdir(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dependency, "go.mod"), []byte("module example.test/support\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dependency, "support.go"), []byte("package support\n\nconst Value = 40\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	runGit := func(dir string, args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	runGit(repo, "init", "-q", "-b", "main")
	runGit(repo, "config", "user.email", "e2e@test")
	runGit(repo, "config", "user.name", "e2e")
	runGit(repo, "config", "commit.gpgsign", "false")
	files := map[string]string{
		"go.mod":  "module example.test/selected\n\ngo 1.26\n\nrequire example.test/support v0.0.0\n",
		"go.work": "go 1.26\n\nuse (\n\t.\n\t../support\n)\n",
		"app.go":  "package selected\n\nimport \"example.test/support\"\n\nfunc Value() int { return support.Value + 1 }\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(repo, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGit(repo, "add", "app.go", "go.mod", "go.work")
	runGit(repo, "commit", "-q", "-m", "base")
	base := TrunkHeadSHA(repo, nil)

	prepared := Prepare(repo, "workerworktree", "12447", base, t.TempDir(), nil)
	if !prepared.OK {
		t.Fatalf("prepare: %+v", prepared)
	}
	workerBody := "package selected\n\nimport \"example.test/support\"\n\nfunc Value() int { return support.Value + 2 }\n"
	if err := os.WriteFile(filepath.Join(prepared.Path, "app.go"), []byte(workerBody), 0o644); err != nil {
		t.Fatal(err)
	}

	var validatedCandidates []string
	verify := func(dir string) (bool, string) {
		validatedCandidates = append(validatedCandidates, dir)
		cmd := exec.Command("go", "build", "./...")
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=auto")
		out, err := cmd.CombinedOutput()
		if err != nil {
			return false, strings.TrimSpace(string(out))
		}
		return true, ""
	}

	landed := Land(repo, prepared.Path, base, "", []string{"app.go"}, verify, nil)
	if !landed.OK || !landed.Committed {
		t.Fatalf("sibling-workspace land failed: %+v", landed)
	}
	if len(validatedCandidates) != 2 {
		t.Fatalf("verified candidates = %v, want prospective and post-merge candidates", validatedCandidates)
	}
	for _, candidate := range validatedCandidates {
		if filepath.Clean(filepath.Dir(candidate)) != filepath.Clean(filepath.Dir(repo)) {
			t.Fatalf("candidate parent = %q, want selected-root parent %q", filepath.Dir(candidate), filepath.Dir(repo))
		}
	}
	got, err := os.ReadFile(filepath.Join(repo, "app.go"))
	if err != nil || string(got) != workerBody {
		t.Fatalf("trunk app.go = %q, %v; want landed worker bytes", got, err)
	}
}

// TestLiveLandWithoutDisambiguationContract proves #12457: a repository that
// has never carried fak's concept analyzer can still use the same root-portable
// managed lander for an ordinary Go command change.
func TestLiveLandWithoutDisambiguationContract(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}

	repo := t.TempDir()
	runGit := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	runGit("init", "-q", "-b", "main")
	runGit("config", "user.email", "e2e@test")
	runGit("config", "user.name", "e2e")
	runGit("config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example.test/portable-land\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	commandDir := filepath.Join(repo, "cmd", "demo")
	if err := os.MkdirAll(commandDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(commandDir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit("add", "go.mod", "cmd/demo/main.go")
	runGit("commit", "-q", "-m", "base")
	base := TrunkHeadSHA(repo, nil)

	prepared := Prepare(repo, "workerworktree", "12457", base, t.TempDir(), nil)
	if !prepared.OK {
		t.Fatalf("prepare: %+v", prepared)
	}
	t.Cleanup(func() { _ = Reap(repo, prepared.Path, nil) })
	workerBody := "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"portable\") }\n"
	if err := os.WriteFile(filepath.Join(prepared.Path, "cmd", "demo", "main.go"), []byte(workerBody), 0o644); err != nil {
		t.Fatal(err)
	}

	verify := func(dir string) (bool, string) {
		cmd := exec.Command("go", "build", "./...")
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off")
		out, err := cmd.CombinedOutput()
		if err != nil {
			return false, strings.TrimSpace(string(out))
		}
		return true, ""
	}
	landed := Land(repo, prepared.Path, base, "", []string{"cmd/demo/main.go"}, verify, nil)
	if !landed.OK || !landed.Committed {
		t.Fatalf("portable land without analyzer contract failed: %+v", landed)
	}
	got, err := os.ReadFile(filepath.Join(repo, "cmd", "demo", "main.go"))
	if err != nil || string(got) != workerBody {
		t.Fatalf("trunk command = %q, %v; want landed worker bytes", got, err)
	}
}
