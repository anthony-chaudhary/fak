package fleetpane

import (
	"context"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// Env guards for the OSRunner descendant-containment witness (issue #3108).
// OSRunner.Run executes a caller-supplied argv; the test re-execs THIS test binary
// as that argv so the "command" is a real child that forks a real grandchild. The
// guard vars select the child/grandchild roles when the re-exec'd binary runs, while
// the top-level `go test` invocation (neither var set) skips both helpers. The
// grandchild-file var names the file the child writes the real grandchild pid into so
// the parent test learns the pid it must witness reaped. Same oracle as
// internal/toolprocgate/procbind_realtree_test.go and internal/swebench.
const (
	fpChildEnv       = "FAK_FP_TREEKILL_CHILD"
	fpGrandchildEnv  = "FAK_FP_TREEKILL_GRANDCHILD"
	fpGrandchildFile = "FAK_FP_TREEKILL_GC_FILE"
)

// TestHelperFleetpaneGrandchildBlocks is the leaf of the spawned tree: a real OS
// process that blocks long enough to be observed alive and then witnessed dead. It is
// NOT a test — the env guard turns the re-exec'd binary into the grandchild.
func TestHelperFleetpaneGrandchildBlocks(t *testing.T) {
	if os.Getenv(fpGrandchildEnv) != "1" {
		return
	}
	// Block well past the test's bounded poll. If the tree-kill fails to reach this
	// descendant, the parent test's liveness poll will still find it alive and fail —
	// exactly the security regression this witnesses.
	time.Sleep(5 * time.Minute)
	os.Exit(0)
}

// TestHelperFleetpaneChildSpawnsGrandchild is the middle of the tree — the direct
// child OSRunner.Run spawns. It forks the real grandchild, records the grandchild pid
// where the parent test can read it, then blocks so the child stays alive as the tree
// root the tree-cancel binds. Not a test — the env guard re-purposes the re-exec'd
// binary.
func TestHelperFleetpaneChildSpawnsGrandchild(t *testing.T) {
	if os.Getenv(fpChildEnv) != "1" {
		return
	}
	gcFile := os.Getenv(fpGrandchildFile)
	if gcFile == "" {
		os.Exit(3)
	}
	gc := exec.Command(os.Args[0], "-test.run=^TestHelperFleetpaneGrandchildBlocks$", "-test.v")
	gc.Env = append(os.Environ(), fpGrandchildEnv+"=1")
	// Strip the child guard so the grandchild's own run of the child helper is a no-op
	// and only the grandchild helper activates.
	gc.Env = envWithout(gc.Env, fpChildEnv)
	if err := gc.Start(); err != nil {
		os.Exit(4)
	}
	// Publish the real grandchild pid atomically (temp + rename) so the parent never
	// reads a half-written pid.
	tmp := gcFile + ".tmp"
	if err := os.WriteFile(tmp, []byte(strconv.Itoa(gc.Process.Pid)), 0o644); err != nil {
		os.Exit(5)
	}
	if err := os.Rename(tmp, gcFile); err != nil {
		os.Exit(6)
	}
	// Reap our own zombie if the grandchild is killed out from under us, but do not
	// race the block: the child must stay alive as the bound tree root.
	go func() { _, _ = gc.Process.Wait() }()
	time.Sleep(5 * time.Minute)
	os.Exit(0)
}

// TestOSRunnerReapsTreeOnCancel is the descendant-containment witness for issue
// #3108: OSRunner.Run executes a caller-supplied argv under a Timeout, and on
// cancel/timeout it must reap the whole subtree — not just the direct child — via
// procguard.ConfigureProcessTreeCancel. The pre-fix path leaned on bare
// exec.CommandContext ctx-cancel, which kills only the root pid and orphans the
// descendant tree; this test runs a real child->grandchild tree through OSRunner,
// cancels the ctx, and asserts via the OS process table that the grandchild is dead.
func TestOSRunnerReapsTreeOnCancel(t *testing.T) {
	// procguard tree-kill is a real OS lever on Windows (taskkill /T-class walk) and
	// POSIX (process-group SIGKILL). Exercised for real here; no OS is skipped.
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("no real tree-kill oracle for GOOS=%s; procguard tree-kill unproven here", runtime.GOOS)
	}

	self, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	gcFile := t.TempDir() + "/grandchild.pid"

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Run the caller-supplied argv (this very binary, re-exec'd as the child role) on a
	// generous Timeout so the ONLY teardown that fires within the witness window is our
	// explicit ctx cancel below — not the deadline.
	done := make(chan struct{})
	go func() {
		OSRunner{}.Run(ctx, RunRequest{
			Args:    []string{self, "-test.run=^TestHelperFleetpaneChildSpawnsGrandchild$", "-test.v"},
			Env:     map[string]string{fpChildEnv: "1", fpGrandchildFile: gcFile},
			Timeout: 2 * time.Minute,
		})
		close(done)
	}()

	// Learn the grandchild pid the child published, then witness it alive BEFORE the
	// cancel so the "dead after" assertion proves containment, not a process that never
	// started.
	grandchildPID := waitForPIDFile(t, gcFile, 30*time.Second)
	if grandchildPID <= 0 {
		t.Fatalf("child never published a grandchild pid to %s", gcFile)
	}
	if grandchildPID == os.Getpid() {
		t.Fatalf("grandchild pid %d collides with self", grandchildPID)
	}
	if !waitUntil(func() bool { return pidAlive(grandchildPID) }, 15*time.Second) {
		t.Fatalf("grandchild pid %d never came up alive; cannot witness containment", grandchildPID)
	}

	// Belt-and-suspenders: whatever the assertions do, no test process survives.
	t.Cleanup(func() {
		if pidAlive(grandchildPID) {
			_, _ = procguard.KillPID(grandchildPID)
		}
	})

	// Cancel the runner's context — the ONLY teardown path. Pre-fix this is single-PID
	// and orphans the grandchild; post-fix procguard tree-kills the whole subtree.
	cancel()

	// THE WITNESS: the grandchild — a descendant OSRunner never handed to any killer
	// directly — must be dead now that the ctx was cancelled, proving the whole subtree
	// was reaped and nothing orphaned.
	if !waitUntil(func() bool { return !pidAlive(grandchildPID) }, 30*time.Second) {
		t.Fatalf("SECURITY: grandchild pid %d still alive after OSRunner ctx cancel — a descendant was orphaned", grandchildPID)
	}

	// OSRunner.Run returns once its process is reaped and the WaitDelay-bounded output
	// copy drains.
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatalf("OSRunner.Run did not return after ctx cancel — the tree-kill reap did not bound the wait")
	}
}

// envWithout returns env with any assignment of key removed, so a re-exec'd child
// never inherits the parent role's guard var.
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

// waitForPIDFile polls until the pid file holds a positive int or the timeout elapses,
// returning 0 on timeout.
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
