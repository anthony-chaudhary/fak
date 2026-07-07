package accounts

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Env guards for the re-exec stand-ins. DefaultRefreshSpawn execs ClaudeExe()
// with claude's own flags (`-p ok --model ...`), so the stand-in cannot be the
// test binary in normal test mode (the unknown flags would fail the flag parse).
// Instead TestMain intercepts BEFORE flag parsing: with crChildEnv set the
// binary becomes the `claude` stand-in (spawns a child and blocks — the node/
// tool subtree of the real CLI); with crGrandchildEnv set it becomes that
// child. Same descendant-containment oracle as
// internal/toolprocgate/procbind_realtree_test.go (issue #2357), copied
// per-package because the helpers are test-scoped.
const (
	crChildEnv          = "FAK_CR_TREEKILL_CHILD"
	crGrandchildEnv     = "FAK_CR_TREEKILL_GRANDCHILD"
	crGrandchildFileEnv = "FAK_CR_GRANDCHILD_FILE"
)

func TestMain(m *testing.M) {
	switch {
	case os.Getenv(crChildEnv) == "1":
		runClaudeStandIn()
	case os.Getenv(crGrandchildEnv) == "1":
		// The leaf of the spawned tree: block long enough to be observed alive
		// and then witnessed dead. If the deadline kill fails to reach this
		// descendant, the parent test's liveness poll finds it alive and fails —
		// exactly the orphaning regression this witnesses (#3105).
		time.Sleep(5 * time.Minute)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// runClaudeStandIn plays the `claude` CLI: it spawns a child process (standing
// in for claude's node runtime / tool children), publishes both pids atomically
// where the driving test can read them, and blocks well past the test's bounded
// poll — a hung refresh probe, the exact case whose deadline teardown must reap
// the whole tree. Never returns.
func runClaudeStandIn() {
	gcFile := os.Getenv(crGrandchildFileEnv)
	if gcFile == "" {
		os.Exit(3)
	}
	gc := exec.Command(os.Args[0])
	gc.Env = append(crEnvWithout(os.Environ(), crChildEnv), crGrandchildEnv+"=1")
	if err := gc.Start(); err != nil {
		os.Exit(4)
	}
	// Publish both real pids atomically (temp + rename) so the parent never
	// reads a half-written file: line 1 = this stand-in, line 2 = its child.
	tmp := gcFile + ".tmp"
	payload := strconv.Itoa(os.Getpid()) + "\n" + strconv.Itoa(gc.Process.Pid) + "\n"
	if err := os.WriteFile(tmp, []byte(payload), 0o644); err != nil {
		os.Exit(5)
	}
	if err := os.Rename(tmp, gcFile); err != nil {
		os.Exit(6)
	}
	// Reap our own zombie if the child is killed out from under us, but do not
	// race the block: the stand-in must stay alive as the tree root the spawn
	// path tears down.
	go func() { _, _ = gc.Process.Wait() }()
	time.Sleep(5 * time.Minute)
	os.Exit(0)
}

// TestDefaultRefreshSpawnDeadlineReapsRealSubtree is the descendant-containment
// security witness for issue #3105: a timed-out credential-refresh probe must
// reap the whole `claude` subtree, not just the top pid. It injects a claude
// stand-in (via FLEET_CLAUDE_EXE) that spawns a child and does not exit, drives
// DefaultRefreshSpawn for real, fires the ctx deadline once the subtree is
// observed alive, and asserts via the OS process table that the DESCENDANT dies
// too — the pid ctx-cancel's default single-PID kill would orphan.
func TestDefaultRefreshSpawnDeadlineReapsRealSubtree(t *testing.T) {
	// refreshKillTree is a real OS lever on both Windows (taskkill /T /F) and
	// POSIX (process-group SIGKILL). It is exercised for real on this box.
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("no real tree-kill oracle for GOOS=%s; refresh teardown unproven here", runtime.GOOS)
	}

	cfgDir := t.TempDir()
	pidFile := filepath.Join(t.TempDir(), "standin.pids")
	t.Setenv("FLEET_CLAUDE_EXE", os.Args[0])
	t.Setenv(crChildEnv, "1")
	t.Setenv(crGrandchildFileEnv, pidFile)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	spawnErr := make(chan error, 1)
	go func() { spawnErr <- DefaultRefreshSpawn(ctx, cfgDir) }()

	// Learn the real pids the stand-in published.
	childPID, grandchildPID := waitForStandInPIDs(t, pidFile, 30*time.Second)
	if childPID <= 0 || grandchildPID <= 0 {
		t.Fatalf("stand-in never published its pids to %s", pidFile)
	}
	if grandchildPID == childPID || grandchildPID == os.Getpid() || childPID == os.Getpid() {
		t.Fatalf("published pids (child %d, grandchild %d) collide with self", childPID, grandchildPID)
	}

	// Belt-and-suspenders: whatever the assertions do, no test process survives.
	t.Cleanup(func() {
		_ = refreshKillTree(childPID)
		_ = refreshKillTree(grandchildPID)
	})

	// Precondition: the grandchild is really alive before the deadline fires, or
	// the "dead after kill" assertion below proves nothing.
	if !waitUntil(func() bool { return pidAlive(grandchildPID) }, 15*time.Second) {
		t.Fatalf("grandchild pid %d never came up alive; cannot witness containment", grandchildPID)
	}

	// THE DEADLINE: expire the ctx exactly as a slow/hung refresh probe would.
	cancel()

	// The spawn must come back promptly — the post-kill wait may not wedge the
	// periodic refresh loop even if the kill misbehaves — and must not report a
	// clean run after a deadline teardown.
	select {
	case err := <-spawnErr:
		if err == nil {
			t.Fatalf("DefaultRefreshSpawn returned nil after a deadline teardown")
		}
	case <-time.After(60 * time.Second):
		t.Fatalf("DefaultRefreshSpawn did not return within 60s of ctx expiry — the teardown wait wedged")
	}

	// THE WITNESS: the grandchild — a descendant the spawn never saw directly —
	// must be dead within a bounded timeout, proving the deadline teardown
	// reached the whole tree and no orphan survives the probe (#3105).
	if !waitUntil(func() bool { return !pidAlive(grandchildPID) }, 30*time.Second) {
		t.Fatalf("SECURITY: grandchild pid %d still alive after the refresh deadline killed child %d — an orphan survived the probe boundary", grandchildPID, childPID)
	}

	// The stand-in (the tree root) is dead too.
	if !waitUntil(func() bool { return !pidAlive(childPID) }, 30*time.Second) {
		t.Fatalf("claude stand-in pid %d still alive after the refresh deadline", childPID)
	}
}

// crEnvWithout returns env with any assignment of key removed, so a re-exec'd
// child never inherits the parent role's guard var.
func crEnvWithout(env []string, key string) []string {
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

// waitForStandInPIDs polls until the pid file holds two positive ints (stand-in
// pid, then its child's pid) or the timeout elapses, returning zeros on timeout.
func waitForStandInPIDs(t *testing.T, path string, timeout time.Duration) (int, int) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(path)
		if err == nil {
			lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
			if len(lines) == 2 {
				child, err1 := strconv.Atoi(strings.TrimSpace(lines[0]))
				gc, err2 := strconv.Atoi(strings.TrimSpace(lines[1]))
				if err1 == nil && err2 == nil && child > 0 && gc > 0 {
					return child, gc
				}
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	return 0, 0
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
