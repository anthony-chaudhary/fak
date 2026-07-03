package toolprocgate

import (
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// Env guards for the re-exec helpers. The test binary re-invokes itself with one
// of these set to become the child or the grandchild instead of running the
// suite. FAK_TPG_GRANDCHILD_FILE names the file the child writes the real
// grandchild pid into so the parent test can learn the pid it must witness dead.
const (
	tpgChildEnv          = "FAK_TPG_TREEKILL_CHILD"
	tpgGrandchildEnv     = "FAK_TPG_TREEKILL_GRANDCHILD"
	tpgGrandchildFileEnv = "FAK_TPG_GRANDCHILD_FILE"
)

// TestHelperGrandchildBlocks is the leaf of the spawned tree: a real OS process
// that blocks long enough to be observed alive and then witnessed dead. It is
// NOT a test — the env guard turns the re-exec'd binary into the grandchild.
func TestHelperGrandchildBlocks(t *testing.T) {
	if os.Getenv(tpgGrandchildEnv) != "1" {
		return
	}
	// Block well past the test's bounded poll. If the tree-kill fails to reach
	// this descendant, the parent test's liveness poll will still find it alive
	// and fail — which is exactly the security regression this witnesses.
	time.Sleep(5 * time.Minute)
	os.Exit(0)
}

// TestHelperChildSpawnsGrandchild is the middle of the tree: the bound pid. It
// spawns the real grandchild, records the grandchild pid where the parent test
// can read it, and then blocks so the child stays alive as the tree root the
// supervisor binds. Not a test — the env guard re-purposes the re-exec'd binary.
func TestHelperChildSpawnsGrandchild(t *testing.T) {
	if os.Getenv(tpgChildEnv) != "1" {
		return
	}
	gcFile := os.Getenv(tpgGrandchildFileEnv)
	if gcFile == "" {
		os.Exit(3)
	}
	gc := exec.Command(os.Args[0], "-test.run=^TestHelperGrandchildBlocks$", "-test.v")
	gc.Env = append(os.Environ(), tpgGrandchildEnv+"=1")
	gc.Env = envWithout(gc.Env, tpgChildEnv)
	if err := gc.Start(); err != nil {
		os.Exit(4)
	}
	// Publish the real grandchild pid atomically (temp + rename) so the parent
	// never reads a half-written pid.
	tmp := gcFile + ".tmp"
	if err := os.WriteFile(tmp, []byte(strconv.Itoa(gc.Process.Pid)), 0o644); err != nil {
		os.Exit(5)
	}
	if err := os.Rename(tmp, gcFile); err != nil {
		os.Exit(6)
	}
	// Reap our own zombie if the grandchild is killed out from under us, but do
	// not race the block: the child must stay alive as the bound tree root.
	go func() { _, _ = gc.Process.Wait() }()
	time.Sleep(5 * time.Minute)
	os.Exit(0)
}

// TestBindPIDDeadlineTreeKillReapsRealGrandchild is the descendant-containment
// security witness (issue #2357). Prior seam-6 tests inject a recordingReaper
// that only RECORDS which pid it was asked to reap — they never prove a real
// grandchild dies. This test launches a REAL child that spawns a REAL grandchild,
// binds the child pid to a Supervisor wired with the production procguard OS
// reaper, fires the deadline kill, and asserts via the OS process table that the
// GRANDCHILD is dead within a bounded timeout — proving no orphan survives the
// run boundary. procguard.KillPID terminates the whole tree (taskkill /T /F on
// Windows, process-group SIGKILL on POSIX).
func TestBindPIDDeadlineTreeKillReapsRealGrandchild(t *testing.T) {
	// procguard.KillPID is a real OS lever on both Windows (taskkill /T /F) and
	// POSIX (SIGKILL). It is exercised for real on this box; no OS is skipped.
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("no real tree-kill oracle for GOOS=%s; procguard.KillPID unproven here", runtime.GOOS)
	}

	Reset()
	t.Cleanup(Reset)

	gcFile := t.TempDir() + "/grandchild.pid"

	// Launch the real child, which spawns the real grandchild.
	child := exec.Command(os.Args[0], "-test.run=^TestHelperChildSpawnsGrandchild$", "-test.v")
	child.Env = append(os.Environ(), tpgChildEnv+"=1", tpgGrandchildFileEnv+"="+gcFile)
	child.Env = envWithout(child.Env, tpgGrandchildEnv)
	if err := child.Start(); err != nil {
		t.Fatalf("start child: %v", err)
	}
	childPID := child.Process.Pid

	// Belt-and-suspenders: whatever the assertions do, no test process survives.
	t.Cleanup(func() {
		_, _ = procguard.KillPID(childPID)
		if gc := readPIDFile(gcFile); gc > 0 {
			_, _ = procguard.KillPID(gc)
		}
		_, _ = child.Process.Wait()
	})

	// Learn the real grandchild pid the child published.
	grandchildPID := waitForPIDFile(t, gcFile, 30*time.Second)
	if grandchildPID <= 0 {
		t.Fatalf("child never published a grandchild pid to %s", gcFile)
	}
	if grandchildPID == childPID || grandchildPID == os.Getpid() {
		t.Fatalf("grandchild pid %d collides with child/self", grandchildPID)
	}

	// Precondition: the grandchild is really alive before we kill the tree, or
	// the "dead after kill" assertion below proves nothing.
	if !waitUntil(func() bool { return pidAlive(grandchildPID) }, 15*time.Second) {
		t.Fatalf("grandchild pid %d never came up alive; cannot witness containment", grandchildPID)
	}

	// Bind the CHILD (tree root) to a Supervisor wired with the PRODUCTION OS
	// reaper, then blow its deadline so the same Tick that cancels also fires the
	// real tree-kill against the bound pid.
	sup := NewOSSupervisor(toolproc.Config{})
	if err := sup.Spawn("tree", "bg_build", "s1", 10_000, 0, 1_000, nil); err != nil {
		t.Fatalf("spawn: %v", err)
	}
	if err := sup.BindPID("tree", childPID); err != nil {
		t.Fatalf("bind child pid %d: %v", childPID, err)
	}

	rep, err := sup.Tick(12_000)
	if err != nil {
		t.Fatalf("tick: %v", err)
	}
	if len(rep.Actions) != 1 {
		t.Fatalf("want exactly one kill action, got %+v", rep.Actions)
	}
	act := rep.Actions[0]
	if act.Advice != toolproc.AdviceKill {
		t.Fatalf("want a kill advice, got %+v", act)
	}
	if !act.Reaped {
		t.Fatalf("production reaper must report the bound tree terminated, got %+v", act)
	}

	// THE WITNESS: the grandchild — a descendant the supervisor never bound
	// directly — must be dead within a bounded timeout, proving /T reached the
	// whole tree and no orphan survives the run boundary.
	if !waitUntil(func() bool { return !pidAlive(grandchildPID) }, 30*time.Second) {
		t.Fatalf("SECURITY: grandchild pid %d still alive after tree-kill of bound child %d — an orphan survived the run boundary", grandchildPID, childPID)
	}

	// The bound child (the tree root) is dead too.
	if !waitUntil(func() bool { return !pidAlive(childPID) }, 30*time.Second) {
		t.Fatalf("bound child pid %d still alive after tree-kill", childPID)
	}
}

// envWithout returns env with any assignment of key removed, so a re-exec'd
// child never inherits the parent role's guard var.
func envWithout(env []string, key string) []string {
	out := env[:0:0]
	prefix := key + "="
	for _, kv := range env {
		if strings.HasPrefix(kv, prefix) {
			continue
		}
		out = append(out, kv)
	}
	return out
}

// waitForPIDFile polls until the pid file holds a positive int or the timeout
// elapses, returning 0 on timeout.
func waitForPIDFile(t *testing.T, path string, timeout time.Duration) int {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if pid := readPIDFile(path); pid > 0 {
			return pid
		}
		time.Sleep(50 * time.Millisecond)
	}
	return 0
}

// readPIDFile returns the pid written to path, or 0 if absent/unparsable.
func readPIDFile(path string) int {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil {
		return 0
	}
	return pid
}

// waitUntil polls cond every 50ms until it is true or the timeout elapses.
func waitUntil(cond func() bool, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(50 * time.Millisecond)
	}
	return cond()
}
