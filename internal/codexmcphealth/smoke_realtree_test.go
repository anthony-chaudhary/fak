package codexmcphealth

import (
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Env guards for the smoke-launch teardown witness (issue #3107). The test binary
// re-execs itself as the `fak serve --stdio` smoke SERVER: RunServerSmoke launches
// FakBinary(root), which falls back to os.Executable() when root holds no fak
// binary, so the server it spawns is this very test binary. cmhSmokeServerEnv turns
// that re-exec into a helper that spawns a real grandchild and then hangs past the
// smoke deadline; cmhSmokeGCFileEnv names the file it publishes the grandchild pid
// into so the parent test learns the descendant it must witness reaped.
const (
	cmhSmokeServerEnv = "FAK_CMH_SMOKE_SERVER"
	cmhSmokeGCFileEnv = "FAK_CMH_SMOKE_GC_FILE"
)

// TestMain intercepts the smoke-server re-exec BEFORE the testing flag parser runs,
// so RunServerSmoke's fixed `serve --stdio --policy` argv never has to parse as test
// flags. Without the env guard it is a plain passthrough to m.Run(), leaving every
// other test (including the reap_realtree re-exec helpers) untouched.
func TestMain(m *testing.M) {
	if os.Getenv(cmhSmokeServerEnv) == "1" {
		runSmokeServerHelper()
		return
	}
	os.Exit(m.Run())
}

// runSmokeServerHelper is the fake `fak serve --stdio` the smoke probe launches: it
// spawns a real grandchild (the descendant a live MCP server would fork while
// executing a tool call), publishes the grandchild pid atomically, then blocks well
// past the smoke deadline so RunServerSmoke MUST tree-kill it to tear the tree down.
func runSmokeServerHelper() {
	gcFile := os.Getenv(cmhSmokeGCFileEnv)
	if gcFile == "" {
		os.Exit(3)
	}
	gc := exec.Command(os.Args[0], "-test.run=^TestHelperGrandchildBlocks$", "-test.v")
	gc.Env = append(os.Environ(), cmhGrandchildEnv+"=1")
	// Strip the server/child guards so the grandchild does not recurse into another
	// server or the reap helper's child role.
	gc.Env = envWithout(gc.Env, cmhSmokeServerEnv)
	gc.Env = envWithout(gc.Env, cmhChildEnv)
	if err := gc.Start(); err != nil {
		os.Exit(4)
	}
	tmp := gcFile + ".tmp"
	if err := os.WriteFile(tmp, []byte(strconv.Itoa(gc.Process.Pid)), 0o644); err != nil {
		os.Exit(5)
	}
	if err := os.Rename(tmp, gcFile); err != nil {
		os.Exit(6)
	}
	// Reap our own zombie if the grandchild dies first, but do not race the block:
	// this server must stay alive as the tree root the smoke launch hands the killer.
	go func() { _, _ = gc.Process.Wait() }()
	time.Sleep(5 * time.Minute)
	os.Exit(0)
}

// TestRunServerSmokeReapsTreeOnTimeout is the descendant-containment witness for
// issue #3107: when the smoke deadline expires, RunServerSmoke must reap the whole
// `fak serve --stdio` subtree — not just the direct server pid — before it returns
// the timeout SmokeResult. The pre-fix path leaned on exec.CommandContext's
// ctx-cancel, which kills only the root pid and orphans the descendant tree; this
// test spawns a real server->grandchild tree, forces a timeout, and asserts via the
// OS process table that the grandchild is dead, reusing the child->grandchild oracle
// from reap_realtree_test.go / internal/toolprocgate/procbind_realtree_test.go.
func TestRunServerSmokeReapsTreeOnTimeout(t *testing.T) {
	// procguard.KillPID is a real OS lever on both Windows (taskkill /T /F) and
	// POSIX (process-group / descendant-walk SIGKILL). Exercised for real here.
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("no real tree-kill oracle for GOOS=%s; smoke teardown unproven here", runtime.GOOS)
	}

	// root holds no fak binary, so FakBinary falls back to os.Executable() — this
	// test binary, which the env guard turns into the smoke-server helper.
	root := t.TempDir()
	gcFile := t.TempDir() + "/grandchild.pid"
	t.Setenv(cmhSmokeServerEnv, "1")
	t.Setenv(cmhSmokeGCFileEnv, gcFile)

	// A deadline long enough to spawn + witness the tree alive, short enough to keep
	// the test quick. The helper server hangs for 5 minutes, so the ONLY way the
	// grandchild dies is the timeout tree-kill under test.
	const timeout = 8 * time.Second
	resCh := make(chan SmokeResult, 1)
	start := time.Now()
	go func() { resCh <- RunServerSmoke(root, DefaultPolicy, timeout) }()

	// Learn the grandchild pid the smoke server published, then witness it alive
	// BEFORE the reap so the "dead after" assertion proves containment, not just a
	// process that never started.
	grandchildPID := waitForPIDFile(t, gcFile, timeout)
	if grandchildPID <= 0 {
		t.Fatalf("smoke server never published a grandchild pid to %s", gcFile)
	}
	if grandchildPID == os.Getpid() {
		t.Fatalf("grandchild pid %d collides with self", grandchildPID)
	}
	if !waitUntil(func() bool { return pidAlive(grandchildPID) }, timeout) {
		t.Fatalf("grandchild pid %d never came up alive; cannot witness containment", grandchildPID)
	}

	// Belt-and-suspenders: no test process survives whatever the assertions do.
	t.Cleanup(func() {
		if pidAlive(grandchildPID) {
			_ = ReapChildren([]int{grandchildPID})
		}
	})

	// The smoke must run its full deadline and return a timeout result.
	res := <-resCh
	if res.OK {
		t.Fatalf("smoke should have timed out, got OK result: %+v", res)
	}
	if !strings.Contains(res.Reason, "timed out") {
		t.Fatalf("smoke result reason = %q, want a timeout reason", res.Reason)
	}
	if elapsed := time.Since(start); elapsed < timeout {
		t.Fatalf("smoke returned after %s, before its %s deadline — it did not actually wait", elapsed, timeout)
	}

	// THE WITNESS: the grandchild — a descendant the smoke launch never handed to
	// any killer directly — must be dead now that RunServerSmoke returned the
	// timeout result, proving the whole subtree was reaped and nothing orphaned.
	if !waitUntil(func() bool { return !pidAlive(grandchildPID) }, 30*time.Second) {
		t.Fatalf("SECURITY: grandchild pid %d still alive after smoke timeout — the smoke launch orphaned a descendant", grandchildPID)
	}
}
