package swebench

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

// Env guards for the DeepSWE-adapter teardown witness (issue #3106). RunInstance
// launches the adapter via exec.CommandContext + procguard.ConfigureProcessTreeCancel,
// forwarding the Instance fields to the child as FAK_SWEBENCH_* across the
// secretload.SandboxEnv (deny-by-default) boundary. Those explicit keys are the ONLY
// env that crosses, so the witness smuggles its signals through them: InstanceID
// carries the sentinel that selects the helper, and BaseCommit carries the file the
// helper publishes the real grandchild pid into so the parent test can witness it
// reaped. Same descendant-containment oracle as internal/codexmcphealth/#3107.
const (
	treekillSentinelID    = "__fak_swebench_treekill_helper__"
	swebenchGrandchildEnv = "FAK_SWEBENCH_TREEKILL_GRANDCHILD"
)

// TestMain intercepts the adapter re-exec BEFORE the testing flag parser runs, so
// RunInstance's fixed adapter argv never has to parse as test flags. The sentinel
// arrives as FAK_SWEBENCH_INSTANCE_ID (RunInstance forwards Instance.InstanceID as an
// explicit SandboxEnv key). Without the guard it is a plain passthrough to m.Run(),
// leaving every other swebench test untouched.
func TestMain(m *testing.M) {
	if os.Getenv("FAK_SWEBENCH_INSTANCE_ID") == treekillSentinelID {
		runAdapterTreeHelper()
		return
	}
	os.Exit(m.Run())
}

// runAdapterTreeHelper is the fake DeepSWE adapter RunInstance execs: it spawns a real
// grandchild (the descendant a live agent adapter forks while driving its multi-step
// loop), publishes the grandchild pid atomically, then blocks well past the test's
// bounded poll so the ONLY way the grandchild dies is the ctx-cancel tree-kill under
// test. It always terminates via os.Exit, so it never returns into the test suite.
func runAdapterTreeHelper() {
	gcFile := os.Getenv("FAK_SWEBENCH_BASE_COMMIT") // smuggled grandchild-pid file path
	if gcFile == "" {
		os.Exit(3)
	}
	gc := exec.Command(os.Args[0], "-test.run=^TestHelperSwebenchGrandchildBlocks$", "-test.v")
	gc.Env = append(os.Environ(), swebenchGrandchildEnv+"=1")
	// Strip the adapter sentinel so the grandchild's TestMain falls through to the
	// suite (filtered to the grandchild helper) instead of recursing into the adapter.
	gc.Env = envWithout(gc.Env, "FAK_SWEBENCH_INSTANCE_ID")
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
	// race the block: this adapter must stay alive as the tree root handed to the killer.
	go func() { _, _ = gc.Process.Wait() }()
	time.Sleep(5 * time.Minute)
	os.Exit(0)
}

// TestHelperSwebenchGrandchildBlocks is the leaf of the spawned tree: a real OS process
// that blocks long enough to be observed alive and then witnessed dead. It is NOT a
// test — the env guard turns the re-exec'd binary into the grandchild.
func TestHelperSwebenchGrandchildBlocks(t *testing.T) {
	if os.Getenv(swebenchGrandchildEnv) != "1" {
		return
	}
	time.Sleep(5 * time.Minute)
	os.Exit(0)
}

// TestRunInstanceReapsTreeOnCancel is the descendant-containment witness for issue
// #3106: when the adapter's context is cancelled (a bounded eval's normal outcome),
// RunInstance must reap the whole adapter subtree — not just the direct adapter pid —
// via procguard.ConfigureProcessTreeCancel. The pre-fix path leaned on bare
// exec.CommandContext ctx-cancel, which kills only the root pid and orphans the
// descendant tree; this test spawns a real adapter->grandchild tree, cancels the ctx,
// and asserts via the OS process table that the grandchild is dead.
func TestRunInstanceReapsTreeOnCancel(t *testing.T) {
	// procguard tree-kill is a real OS lever on Windows (taskkill /T-class walk) and
	// POSIX (process-group / descendant-walk SIGKILL). Exercised for real here.
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("no real tree-kill oracle for GOOS=%s; adapter teardown unproven here", runtime.GOOS)
	}

	self, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	gcFile := t.TempDir() + "/grandchild.pid"

	// adapterCommand() reads FAK_DEEPSWE_RUNNER from the parent env to pick the exe —
	// here, this very test binary, which the sentinel InstanceID turns into the helper.
	t.Setenv("FAK_DEEPSWE_RUNNER", self)
	t.Setenv("FAK_DEEPSWE_RUNNER_ARGS", "")

	d := &deepSWERunner{cfg: RunConfig{Model: "treekill-witness", MaxSteps: 1}}
	in := Instance{
		InstanceID: treekillSentinelID, // crosses as FAK_SWEBENCH_INSTANCE_ID -> TestMain selects the helper
		Repo:       "fak/treekill",
		BaseCommit: gcFile, // crosses as FAK_SWEBENCH_BASE_COMMIT -> helper publishes the grandchild pid here
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { _, _ = d.RunInstance(ctx, in); close(done) }()

	// Learn the grandchild pid the adapter published, then witness it alive BEFORE the
	// cancel so the "dead after" assertion proves containment, not a process that never
	// started.
	grandchildPID := waitForPIDFile(t, gcFile, 30*time.Second)
	if grandchildPID <= 0 {
		t.Fatalf("adapter never published a grandchild pid to %s", gcFile)
	}
	if grandchildPID == os.Getpid() {
		t.Fatalf("grandchild pid %d collides with self", grandchildPID)
	}
	if !waitUntil(func() bool { return pidAlive(grandchildPID) }, 15*time.Second) {
		t.Fatalf("grandchild pid %d never came up alive; cannot witness containment", grandchildPID)
	}

	// Belt-and-suspenders: no test process survives whatever the assertions do.
	t.Cleanup(func() {
		if pidAlive(grandchildPID) {
			_, _ = procguard.KillPID(grandchildPID)
		}
	})

	// Cancel the adapter's context — the ONLY teardown path. Pre-fix this is single-PID
	// and orphans the grandchild; post-fix procguard tree-kills the whole subtree.
	cancel()

	// THE WITNESS: the grandchild — a descendant RunInstance never handed to any killer
	// directly — must be dead now that the ctx was cancelled, proving the whole subtree
	// was reaped and nothing orphaned.
	if !waitUntil(func() bool { return !pidAlive(grandchildPID) }, 30*time.Second) {
		t.Fatalf("SECURITY: grandchild pid %d still alive after adapter ctx cancel — RunInstance orphaned a descendant", grandchildPID)
	}

	// The adapter run returns once its process is reaped.
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatalf("RunInstance did not return after ctx cancel — the tree-kill reap did not bound the wait")
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
