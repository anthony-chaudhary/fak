package webbench

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// Reap witness for #3481: runBoundedHarness's expiry kill must route through
// procguard.KillPID so the whole harness subtree is reaped, not just the direct
// PID. The harness (`python -m browser_use.eval`) fans out into browser
// subprocesses (Playwright, Chromium); the default CommandContext single-PID
// kill would return a timeout verdict while orphaning real browsers still
// driving the network. This test re-execs THIS test binary as the launched
// harness (child) and as a grandchild the harness spawns. The grandchild
// heartbeats a file; after the timeout reap it must STOP — a single-PID kill
// would orphan it and it would keep beating. Heartbeat-file liveness is fully
// portable (no pid syscalls), matching gardenbundle's #3103 reap witness.
const (
	reapHarnessEnv    = "FAK_WEBBENCHREAP_HARNESS"
	reapGrandchildEnv = "FAK_WEBBENCHREAP_GRANDCHILD"
	reapHeartbeatEnv  = "FAK_WEBBENCHREAP_HEARTBEAT"
)

// TestMain turns the test binary into the harness / grandchild stand-ins when
// the guard envs are set; otherwise it runs the package's tests normally (there
// is no other TestMain in this package). The grandchild branch is checked first
// so a process carrying both inherited guards resolves to the grandchild.
func TestMain(m *testing.M) {
	if os.Getenv(reapGrandchildEnv) != "" {
		beatUntilKilled(os.Getenv(reapHeartbeatEnv))
		os.Exit(0)
	}
	if os.Getenv(reapHarnessEnv) != "" {
		// The "harness": spawn a heartbeating grandchild, then hang past any deadline.
		gc := exec.Command(os.Args[0])
		gc.Env = append(os.Environ(), reapGrandchildEnv+"=1")
		_ = gc.Start()
		time.Sleep(2 * time.Minute)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// beatUntilKilled rewrites path every ~80ms until it is killed (or a safety
// ceiling elapses), so a live grandchild keeps the file's mtime advancing.
func beatUntilKilled(path string) {
	if path == "" {
		time.Sleep(2 * time.Minute)
		return
	}
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		_ = os.WriteFile(path, []byte(time.Now().Format(time.RFC3339Nano)), 0o644)
		time.Sleep(80 * time.Millisecond)
	}
}

func TestRunBoundedHarnessTimeoutReapsGrandchild3481(t *testing.T) {
	hb := filepath.Join(t.TempDir(), "grandchild.heartbeat")
	// runBoundedHarness inherits the test process env, so the re-exec'd binary
	// resolves to the hanging harness stand-in.
	t.Setenv(reapHarnessEnv, "1")
	t.Setenv(reapHeartbeatEnv, hb)

	start := time.Now()
	_, err := runBoundedHarness(400*time.Millisecond, os.Args[0])
	if err == nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected a timeout error wrapping context.DeadlineExceeded from the wedged harness, got %v", err)
	}
	if d := time.Since(start); d > 30*time.Second {
		t.Fatalf("runBoundedHarness hung %s past the 400ms deadline — the timeout kill did not return", d)
	}

	// The grandchild must have been reaped along with the harness: its heartbeat
	// must go stale. Under a single-PID kill it is orphaned and keeps beating,
	// which fails here.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if fi, err := os.Stat(hb); err == nil && time.Since(fi.ModTime()) > 1*time.Second {
			return // heartbeat went stale -> grandchild dead -> harness subtree reaped
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("grandchild heartbeat %s kept advancing after the timeout — the harness subtree was orphaned (the #3481 tree-kill gap)", hb)
}

// TestEvalConfigHarnessTimeout covers the per-benchmark deadline resolution:
// an explicit cfg.Timeout wins, the zero value falls back to the package
// default rather than running unbounded.
func TestEvalConfigHarnessTimeout(t *testing.T) {
	if got := (EvalConfig{}).harnessTimeout(); got != evalHarnessTimeout {
		t.Fatalf("zero cfg.Timeout should resolve to the package default %s, got %s", evalHarnessTimeout, got)
	}
	if got := (EvalConfig{Timeout: 3 * time.Second}).harnessTimeout(); got != 3*time.Second {
		t.Fatalf("explicit cfg.Timeout should win, got %s", got)
	}
}
