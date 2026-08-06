package gitbroker

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// runnerTestRepo builds a hermetic repository with one object of every type the
// pool can be asked about: a blob (deliberately WITHOUT a trailing newline, so a
// payload off by one byte cannot pass), a tree, a commit, and an annotated tag.
func runnerTestRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	none := filepath.Join(dir, "no-such-gitconfig")
	git := func(args ...string) {
		t.Helper()
		c := exec.Command("git", append([]string{"-C", dir}, args...)...)
		c.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=fak test", "GIT_AUTHOR_EMAIL=t@fak.invalid",
			"GIT_COMMITTER_NAME=fak test", "GIT_COMMITTER_EMAIL=t@fak.invalid",
			"GIT_AUTHOR_DATE=2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE=2026-01-01T00:00:00Z",
			// Neutralize the host's git config: an inherited hooksPath or
			// signing key would make this fixture depend on the machine.
			"GIT_CONFIG_GLOBAL="+none, "GIT_CONFIG_SYSTEM="+none,
		)
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
	}
	git("init", "-q", "-b", "main")
	git("config", "user.name", "fak test")
	git("config", "user.email", "t@fak.invalid")
	git("config", "commit.gpgsign", "false")
	git("config", "tag.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(dir, "hello.txt"), []byte("hello, batch"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "hello.txt")
	git("commit", "-q", "-m", "seed")
	git("tag", "-a", "v1", "-m", "annotated")
	return dir
}

// runnerGit runs git the plain way — this is the oracle every equivalence test
// in this file compares against.
func runnerGit(t *testing.T, repo string, args ...string) []byte {
	t.Helper()
	out, code := runnerGitCode(t, repo, args...)
	if code != 0 {
		t.Fatalf("git %s: exit %d", strings.Join(args, " "), code)
	}
	return out
}

func runnerGitCode(t *testing.T, repo string, args ...string) ([]byte, int) {
	t.Helper()
	c := exec.Command("git", append([]string{"-C", repo}, args...)...)
	out, err := c.Output()
	if err == nil {
		return out, 0
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return out, ee.ExitCode()
	}
	t.Fatalf("git %s: %v", strings.Join(args, " "), err)
	return nil, -1
}

func runnerRev(t *testing.T, repo, rev string) string {
	t.Helper()
	return strings.TrimSpace(string(runnerGit(t, repo, "rev-parse", rev)))
}

func runnerCatFile(repo string, args ...string) Invocation {
	return Invocation{Dir: repo, DirAsFlag: true, Args: args, DiscardStderr: true}
}

// TestRunObjectReadsMatchGitExactly is the no-behaviour-change gate for the
// fast path: for every argument shape the pool is allowed to serve, and every
// object type, the bytes Run returns must be the bytes git prints. A refactor
// that is 78x faster and one byte different is a regression, not a win.
func TestRunObjectReadsMatchGitExactly(t *testing.T) {
	repo := runnerTestRepo(t)
	blobOID := runnerRev(t, repo, "HEAD:hello.txt")
	treeOID := runnerRev(t, repo, "HEAD^{tree}")
	commitOID := runnerRev(t, repo, "HEAD")
	tagOID := runnerRev(t, repo, "v1")

	e := New()
	t.Cleanup(func() { _ = e.Close() })

	shapes := [][]string{}
	for _, oid := range []string{blobOID, treeOID, commitOID, tagOID} {
		shapes = append(shapes,
			[]string{"cat-file", "-t", oid},
			[]string{"cat-file", "-s", oid},
			[]string{"cat-file", "-p", oid},
			[]string{"rev-parse", "--verify", oid},
			[]string{"rev-parse", "--verify", "--quiet", oid},
		)
	}
	shapes = append(shapes,
		[]string{"rev-parse", "--verify", "HEAD^{commit}"},
		[]string{"rev-parse", "--verify", "v1^{commit}"},
		[]string{"rev-parse", "--verify", "--quiet", commitOID + "^{tree}"},
		[]string{"cat-file", "-t", "HEAD"},
		[]string{"cat-file", "-p", "HEAD:hello.txt"},
	)

	for _, args := range shapes {
		want := runnerGit(t, repo, args...)
		got, err := e.Run(context.Background(), runnerCatFile(repo, args...))
		if err != nil {
			t.Fatalf("Run(%v): %v", args, err)
		}
		if string(got.Stdout) != string(want) {
			t.Fatalf("Run(%v) = %q, git says %q", args, got.Stdout, want)
		}
	}
}

// TestWarmPoolCollapsesObjectReadsToOneProcess is the measurement #5621 exists
// for, as a test rather than a claim: the SAME reads that cost one git process
// each before this package cost exactly one process in total behind the pool.
func TestWarmPoolCollapsesObjectReadsToOneProcess(t *testing.T) {
	repo := runnerTestRepo(t)
	oid := runnerRev(t, repo, "HEAD:hello.txt")
	const reads = 200

	warm := New()
	t.Cleanup(func() { _ = warm.Close() })
	before := Spawns()
	for i := 0; i < reads; i++ {
		out, err := warm.Run(context.Background(), runnerCatFile(repo, "cat-file", "-t", oid))
		if err != nil {
			t.Fatalf("warm read %d: %v", i, err)
		}
		if string(out.Stdout) != "blob\n" {
			t.Fatalf("warm read %d = %q", i, out.Stdout)
		}
	}
	warmProcs := Spawns() - before
	if warmProcs != 1 {
		t.Fatalf("%d reads behind the warm pool created %d git processes, want exactly 1", reads, warmProcs)
	}

	// The same reads with no pool available: this is the pre-#5621 cost, and
	// it is measured here rather than quoted so the ratio cannot go stale.
	cold := New()
	if err := cold.Close(); err != nil {
		t.Fatal(err)
	}
	before = Spawns()
	for i := 0; i < reads; i++ {
		out, err := cold.Run(context.Background(), runnerCatFile(repo, "cat-file", "-t", oid))
		if err != nil {
			t.Fatalf("cold read %d: %v", i, err)
		}
		if string(out.Stdout) != "blob\n" {
			t.Fatalf("cold read %d = %q", i, out.Stdout)
		}
	}
	coldProcs := Spawns() - before
	if coldProcs != reads {
		t.Fatalf("%d reads with no pool created %d git processes, want %d", reads, coldProcs, reads)
	}
	t.Logf("git processes for %d object reads: %d warm vs %d spawning (%dx fewer)", reads, warmProcs, coldProcs, coldProcs/warmProcs)
}

// TestKilledPoolFallsBackTransparently is the fail-open gate: kill the warm
// process out from under a live Exec and the caller must still get the same
// bytes. A dead pool may cost a spawn; it may never cost an answer.
func TestKilledPoolFallsBackTransparently(t *testing.T) {
	repo := runnerTestRepo(t)
	oid := runnerRev(t, repo, "HEAD:hello.txt")

	e := New()
	t.Cleanup(func() { _ = e.Close() })

	want, err := e.Run(context.Background(), runnerCatFile(repo, "cat-file", "-p", oid))
	if err != nil {
		t.Fatalf("warm read: %v", err)
	}

	p := e.pool(repo, true)
	p.mu.Lock()
	proc := p.cmd.Process
	p.mu.Unlock()
	if proc == nil {
		t.Fatal("no warm process to kill")
	}
	// The whole TREE, not just the pid this package holds. On Windows the `git`
	// on PATH is usually a launcher that runs the real git as a child, and a
	// single kill leaves that child answering our pipe — so a plain
	// proc.Kill() would leave the pool WORKING and this test would be asserting
	// against a pool that was never killed. See killGitProcessTree.
	if err := killGitProcessTree(proc.Pid); err != nil {
		t.Fatalf("kill the pool: %v", err)
	}
	for i := 0; i < 200 && !gitProcessGone(proc.Pid); i++ {
		time.Sleep(25 * time.Millisecond)
	}

	before := Spawns()
	got, err := e.Run(context.Background(), runnerCatFile(repo, "cat-file", "-p", oid))
	if err != nil {
		t.Fatalf("read after the pool was killed: %v", err)
	}
	if string(got.Stdout) != string(want.Stdout) {
		t.Fatalf("after the kill Run = %q, before it %q", got.Stdout, want.Stdout)
	}
	if string(got.Stdout) != "hello, batch" {
		t.Fatalf("payload %q is not the blob's exact bytes", got.Stdout)
	}
	if n := Spawns() - before; n == 0 {
		t.Fatal("a killed pool must fall back by SPAWNING; no git process was created")
	}
	p.mu.Lock()
	dead := p.dead
	p.mu.Unlock()
	if !dead {
		t.Fatal("a pool that failed mid-stream must stay dead, not be reused at an unknown offset")
	}
}

// TestHungPoolFallsBackWithinTheDeadline covers the other half of fail-open: a
// pool that never answers is indistinguishable, to the caller, from one that
// was never there. The wedge is a read that no writer will ever satisfy, which
// is exactly what a hung git looks like from this side of the pipe.
func TestHungPoolFallsBackWithinTheDeadline(t *testing.T) {
	repo := runnerTestRepo(t)
	oid := runnerRev(t, repo, "HEAD:hello.txt")

	restore := batchDeadline
	batchDeadline = 200 * time.Millisecond
	t.Cleanup(func() { batchDeadline = restore })

	e := New()
	t.Cleanup(func() { _ = e.Close() })
	if _, err := e.ObjectInfo(context.Background(), repo, oid); err != nil {
		t.Fatalf("warm the pool: %v", err)
	}

	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pw.Close(); _ = pr.Close() })
	p := e.pool(repo, false)
	p.mu.Lock()
	p.out = bufio.NewReader(pr)
	p.mu.Unlock()

	start := time.Now()
	got, err := e.Run(context.Background(), runnerCatFile(repo, "cat-file", "-t", oid))
	if err != nil {
		t.Fatalf("read through a wedged pool: %v", err)
	}
	if string(got.Stdout) != "blob\n" {
		t.Fatalf("wedged-pool read = %q, want %q", got.Stdout, "blob\n")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("a wedged pool held the caller for %s; the deadline must bound it", elapsed)
	}
	p.mu.Lock()
	dead := p.dead
	p.mu.Unlock()
	if !dead {
		t.Fatal("a wedged pool must not be retried")
	}
}

// TestCloseReapsTheWarmProcess pins the deliberate teardown: Close closes the
// child's stdin, which is what `cat-file --batch` exits on.
func TestCloseReapsTheWarmProcess(t *testing.T) {
	repo := runnerTestRepo(t)
	e := New()
	if _, err := e.ObjectInfo(context.Background(), repo, "HEAD"); err != nil {
		t.Fatalf("warm the pool: %v", err)
	}
	p := e.pool(repo, false)
	p.mu.Lock()
	pid := p.cmd.Process.Pid
	p.mu.Unlock()

	if err := e.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	for i := 0; i < 200 && !gitProcessGone(pid); i++ {
		time.Sleep(25 * time.Millisecond)
	}
	if !gitProcessGone(pid) {
		t.Fatalf("git cat-file pid %d survived Close", pid)
	}
	// It must have left on stdin EOF, not on the kill backstop: EOF is the
	// mechanism that also reaps a pool whose parent exited without calling
	// anything, so a child that stopped honouring it would mean orphans.
	p.mu.Lock()
	killed := p.killedOnClose
	p.mu.Unlock()
	if killed {
		t.Fatal("the pool had to be KILLED; `cat-file --batch` must exit on stdin EOF")
	}
	// A closed Exec must not quietly start a new pool behind the caller's back.
	before := Spawns()
	if _, err := e.ObjectInfo(context.Background(), repo, "HEAD"); err != nil {
		t.Fatalf("read after Close: %v", err)
	}
	if Spawns()-before != 1 {
		t.Fatal("a closed Exec must serve object reads by spawning, one process per read")
	}
}

const (
	reapHelperEnv = "FAK_GITBROKER_REAP_HELPER"
	reapRepoEnv   = "FAK_GITBROKER_REAP_REPO"
	reapPIDPrefix = "fak-gitbroker-pool-pid: "
)

// TestNoCatFileSurvivesProcessExit is the orphan gate, and it is deliberately a
// REAL subprocess: an orphaned `git cat-file` is worse than the churn this
// package removes, so the claim is proven against the operating system rather
// than against a mock. The helper starts a pool and exits WITHOUT calling
// Close — the OS closing the write end of the child's stdin must be enough.
func TestNoCatFileSurvivesProcessExit(t *testing.T) {
	if os.Getenv(reapHelperEnv) != "" {
		t.Skip("running as the helper process")
	}
	repo := runnerTestRepo(t)

	cmd := exec.Command(os.Args[0], "-test.run=^TestPoolReapHelperProcess$", "-test.v")
	cmd.Env = append(os.Environ(), reapHelperEnv+"=1", reapRepoEnv+"="+repo)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("helper process: %v\n%s", err, out)
	}
	var pid int
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if rest, ok := strings.CutPrefix(line, reapPIDPrefix); ok {
			if pid, err = strconv.Atoi(strings.TrimSpace(rest)); err != nil {
				t.Fatalf("unreadable pid line %q: %v", line, err)
			}
		}
	}
	if pid == 0 {
		t.Fatalf("helper never reported a pool pid:\n%s", out)
	}

	// Generous: the child exits on EOF, which it observes on its own schedule.
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if gitProcessGone(pid) {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("git cat-file pid %d outlived the process that started it", pid)
}

// TestPoolReapHelperProcess is the body of the subprocess above, not a test in
// its own right; it skips unless it is running as the helper.
func TestPoolReapHelperProcess(t *testing.T) {
	if os.Getenv(reapHelperEnv) == "" {
		t.Skip("helper for TestNoCatFileSurvivesProcessExit")
	}
	repo := os.Getenv(reapRepoEnv)
	e := New()
	if _, err := e.ObjectInfo(context.Background(), repo, "HEAD"); err != nil {
		fmt.Printf("helper: warm the pool: %v\n", err)
		os.Exit(3)
	}
	p := e.pool(repo, false)
	p.mu.Lock()
	pid := p.cmd.Process.Pid
	p.mu.Unlock()
	fmt.Printf("%s%d\n", reapPIDPrefix, pid)
	_ = os.Stdout.Sync()
	// Exit without reaping anything on purpose. If the pool needs a polite
	// caller to avoid leaking a process, it is not safe to ship.
	os.Exit(0)
}

// TestOnlyProvableShapesReachThePool pins the fast path's closed set. Every
// false here is a spawn, i.e. the exact pre-refactor behaviour, so this test is
// really about what the pool must NOT try to answer.
func TestOnlyProvableShapesReachThePool(t *testing.T) {
	const repo = "/repo"
	served := []struct {
		name string
		inv  Invocation
		rev  string
		kind int
	}{
		{"type", Invocation{Dir: repo, Args: []string{"cat-file", "-t", "deadbeef"}}, "deadbeef", readType},
		{"size", Invocation{Dir: repo, Args: []string{"cat-file", "-s", "deadbeef"}}, "deadbeef", readSize},
		{"print", Invocation{Dir: repo, Args: []string{"cat-file", "-p", "deadbeef"}}, "deadbeef", readPrint},
		{"verify", Invocation{Dir: repo, Args: []string{"rev-parse", "--verify", "x^{commit}"}}, "x^{commit}", readVerify},
		{"verify quiet", Invocation{Dir: repo, Args: []string{"rev-parse", "--verify", "--quiet", "x"}}, "x", readVerify},
	}
	for _, tc := range served {
		rev, kind, ok := objectReadShape(tc.inv)
		if !ok || rev != tc.rev || kind != tc.kind {
			t.Fatalf("%s: got (%q,%d,%v), want (%q,%d,true)", tc.name, rev, kind, ok, tc.rev, tc.kind)
		}
	}

	refused := []struct {
		why string
		inv Invocation
	}{
		{"no repo named: the pool runs -C dir and cannot guess which repo",
			Invocation{Args: []string{"cat-file", "-t", "deadbeef"}}},
		{"tuned environment: git would see a different environment than the pool has",
			Invocation{Dir: repo, Env: []string{"GIT_OPTIONAL_LOCKS=0"}, Args: []string{"cat-file", "-t", "x"}}},
		{"combined stderr: the caller asked for a stream the pool does not produce",
			Invocation{Dir: repo, Combined: true, Args: []string{"cat-file", "-t", "x"}}},
		{"stdin: git is being fed, not asked",
			Invocation{Dir: repo, Stdin: "x", Args: []string{"cat-file", "-t", "x"}}},
		{"cat-file -e is an existence probe, answered by the exit code, not stdout",
			Invocation{Dir: repo, Args: []string{"cat-file", "-e", "x"}}},
		{"rev-parse without --verify prints a different shape",
			Invocation{Dir: repo, Args: []string{"rev-parse", "HEAD"}}},
		{"rev-parse --short changes the output",
			Invocation{Dir: repo, Args: []string{"rev-parse", "--verify", "--short", "x"}}},
		{"a newline in the key would inject a second batch record",
			Invocation{Dir: repo, Args: []string{"cat-file", "-t", "a\nb"}}},
		{"a cwd-relative key resolves against a directory the pool does not share",
			Invocation{Dir: repo, Args: []string{"cat-file", "-p", "HEAD:./x"}}},
		{"a leading dash is a flag to the spawn fallback",
			Invocation{Dir: repo, Args: []string{"cat-file", "-t", "--x"}}},
		{"not an object read at all",
			Invocation{Dir: repo, Args: []string{"status", "--porcelain"}}},
	}
	for _, tc := range refused {
		if _, _, ok := objectReadShape(tc.inv); ok {
			t.Fatalf("the pool must refuse this and spawn instead (%s): %v", tc.why, tc.inv.Args)
		}
	}
}

// TestTreePrettyPrintNeverComesFromThePool guards the one object type whose
// `cat-file -p` output is NOT its payload: a tree pretty-prints as a listing.
// Serving it from --batch would hand the caller raw binary instead.
func TestTreePrettyPrintNeverComesFromThePool(t *testing.T) {
	repo := runnerTestRepo(t)
	treeOID := runnerRev(t, repo, "HEAD^{tree}")

	e := New()
	t.Cleanup(func() { _ = e.Close() })

	want := runnerGit(t, repo, "cat-file", "-p", treeOID)
	got, err := e.Run(context.Background(), runnerCatFile(repo, "cat-file", "-p", treeOID))
	if err != nil {
		t.Fatal(err)
	}
	if string(got.Stdout) != string(want) {
		t.Fatalf("tree -p = %q, git says %q", got.Stdout, want)
	}
	if !strings.Contains(string(got.Stdout), "hello.txt") {
		t.Fatalf("tree -p should be the pretty listing, got %q", got.Stdout)
	}
	if poolServiceableRead("tree") {
		t.Fatal("a tree must never be served from the payload pool")
	}
}

// TestMissingObjectBehavesExactlyLikeGit: when the pool cannot vouch for an
// answer the caller must see git's own, whatever it is.
//
// The `rev-parse --verify` cases are why this test compares against git instead
// of asserting a failure. `--verify` checks that a NAME is well formed, not
// that the object exists, so a well-formed-but-absent OID exits 0 and is echoed
// back — while `--batch-check` calls that same key missing. The two genuinely
// disagree, and the only reason that disagreement is invisible to callers is
// that a pool answer of "missing" is never trusted as a rev-parse result: it
// drops to the spawn, which is git's own opinion by construction.
func TestMissingObjectBehavesExactlyLikeGit(t *testing.T) {
	repo := runnerTestRepo(t)
	const ghost = "0123456789abcdef0123456789abcdef01234567"

	e := New()
	t.Cleanup(func() { _ = e.Close() })

	for _, args := range [][]string{
		{"cat-file", "-t", ghost},
		{"cat-file", "-s", ghost},
		{"cat-file", "-p", ghost},
		{"cat-file", "-t", "no-such-ref"},
		{"rev-parse", "--verify", ghost},
		{"rev-parse", "--verify", "--quiet", ghost},
		{"rev-parse", "--verify", "--quiet", "no-such-ref"},
		{"rev-parse", "--verify", "--quiet", ghost + "^{commit}"},
	} {
		wantOut, wantCode := runnerGitCode(t, repo, args...)
		got, err := e.Run(context.Background(), runnerCatFile(repo, args...))
		if (err != nil) != (wantCode != 0) {
			t.Fatalf("Run(%v) err=%v, git exit %d: error and exit status disagree", args, err, wantCode)
		}
		if got.ExitCode != wantCode {
			t.Fatalf("Run(%v) exit %d, git exit %d", args, got.ExitCode, wantCode)
		}
		if string(got.Stdout) != string(wantOut) {
			t.Fatalf("Run(%v) stdout %q, git %q", args, got.Stdout, wantOut)
		}
	}

	// ObjectInfo, the explicit surface, reports it as a real answer instead.
	if _, err := e.ObjectInfo(context.Background(), repo, ghost); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ObjectInfo on a missing object: %v, want ErrNotFound", err)
	}
}

// TestSpawnPreservesHelperSemantics pins the four axes the consolidated helpers
// disagreed on. Each assertion here is a behaviour some existing call site
// depends on, which is why Invocation carries the axis instead of picking one.
func TestSpawnPreservesHelperSemantics(t *testing.T) {
	repo := runnerTestRepo(t)
	e := New()
	t.Cleanup(func() { _ = e.Close() })
	ctx := context.Background()

	// -C dir (gitOut) and cmd.Dir (gitOutput, gitRunner) both aim git at the
	// repo, and both must work from a cwd that is not it.
	for _, asFlag := range []bool{true, false} {
		out, err := e.Run(ctx, Invocation{Dir: repo, DirAsFlag: asFlag, Args: []string{"rev-parse", "--is-inside-work-tree"}})
		if err != nil {
			t.Fatalf("DirAsFlag=%v: %v", asFlag, err)
		}
		if strings.TrimSpace(string(out.Stdout)) != "true" {
			t.Fatalf("DirAsFlag=%v did not aim git at the repo: %q", asFlag, out.Stdout)
		}
	}

	// A non-zero exit is data, not a transport failure: gitRunner returns the
	// code with a nil error, so ExitCode must be the real one.
	bad := []string{"cat-file", "-t", "definitely-not-a-rev"}
	out, err := e.Run(ctx, Invocation{Dir: repo, DirAsFlag: true, Args: bad})
	if err == nil {
		t.Fatal("a failing git must return an error")
	}
	if out.ExitCode <= 0 {
		t.Fatalf("ExitCode %d, want git's real non-zero status", out.ExitCode)
	}
	if len(out.Stderr) == 0 {
		t.Fatal("stderr must be captured by default")
	}
	var ee *exec.ExitError
	if !errors.As(err, &ee) || len(ee.Stderr) == 0 {
		t.Fatal("ExitError.Stderr must be populated, as exec.Cmd.Output used to do for runGitOutput")
	}

	// DiscardStderr (gitOut) must keep stderr invisible.
	out, _ = e.Run(ctx, Invocation{Dir: repo, DirAsFlag: true, Args: bad, DiscardStderr: true})
	if len(out.Stderr) != 0 {
		t.Fatalf("DiscardStderr still captured %q", out.Stderr)
	}

	// Combined (gitRunner) must fold stderr into stdout.
	out, _ = e.Run(ctx, Invocation{Dir: repo, Args: bad, Combined: true})
	if len(out.Stdout) == 0 {
		t.Fatal("Combined must fold stderr into Stdout")
	}
	if len(out.Stderr) != 0 {
		t.Fatalf("Combined must not also fill Stderr, got %q", out.Stderr)
	}

	// Env reaches git.
	out, err = e.Run(ctx, Invocation{Dir: repo, Args: []string{"config", "--get", "user.name"},
		Env: []string{"GIT_CONFIG_COUNT=1", "GIT_CONFIG_KEY_0=user.name", "GIT_CONFIG_VALUE_0=from-env"}})
	if err != nil {
		t.Fatalf("Env invocation: %v", err)
	}
	if strings.TrimSpace(string(out.Stdout)) != "from-env" {
		t.Fatalf("Env did not reach git: %q", out.Stdout)
	}

	// Stdin reaches git, and a batch read spawned this way parses.
	oid := runnerRev(t, repo, "HEAD:hello.txt")
	out, err = e.Run(ctx, Invocation{Dir: repo, DirAsFlag: true, Args: []string{"cat-file", "--batch-check"}, Stdin: oid + "\n"})
	if err != nil {
		t.Fatalf("Stdin invocation: %v", err)
	}
	if !strings.HasPrefix(string(out.Stdout), oid+" blob ") {
		t.Fatalf("Stdin did not reach git: %q", out.Stdout)
	}

	// git that cannot be executed at all is ExitCode -1, which is how gitRunner
	// tells "git failed" from "git said no".
	if out, err := e.Run(ctx, Invocation{Dir: filepath.Join(repo, "no-such-dir"), Args: []string{"status"}}); err == nil || out.ExitCode != -1 {
		t.Fatalf("unstartable git: exit %d err %v, want -1 and an error", out.ExitCode, err)
	}
}

// TestParseBatchReplyMatchesTheWarmParser: the fallback and the pool must never
// disagree about what a record means, only about how it was obtained.
func TestParseBatchReplyMatchesTheWarmParser(t *testing.T) {
	info, data, err := parseBatchReply([]byte("abc123 blob 5\nhello\n"), true)
	if err != nil {
		t.Fatal(err)
	}
	if info.OID != "abc123" || info.Type != "blob" || info.Size != 5 || string(data) != "hello" {
		t.Fatalf("got %+v %q", info, data)
	}
	if _, _, err := parseBatchReply([]byte("abc123 missing\n"), false); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing record: %v, want ErrNotFound", err)
	}
	if _, _, err := parseBatchReply([]byte("abc123 blob 99\nshort\n"), true); err == nil {
		t.Fatal("a truncated payload must be an error, not a short object")
	}
	if _, _, err := parseBatchReply([]byte("nonsense\n"), false); err == nil {
		t.Fatal("an unparseable header must be an error")
	}
}
