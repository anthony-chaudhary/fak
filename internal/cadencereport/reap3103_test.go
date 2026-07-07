package cadencereport

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Reap witness for #3103 (sibling of #2989): RunPyEnvelope's timeout branch must
// route through procguard.KillPID so the whole member subtree is reaped, not just
// the direct PID. The folded control-pane scripts spawn `git` grandchildren; the
// pre-#3103 `cmd.Process.Kill()` reaped only the immediate process and orphaned
// them. This test re-execs THIS test binary as the launched member (child) and as
// a grandchild the member spawns. The grandchild heartbeats a file; after the
// timeout reap it must STOP — a single-PID kill would orphan it and it would keep
// beating. Heartbeat-file liveness is fully portable (no pid syscalls), matching
// the other #2989 sibling reap witnesses.
const (
	reapChildEnv      = "FAK_CADENCEREAP_CHILD"
	reapGrandchildEnv = "FAK_CADENCEREAP_GRANDCHILD"
	reapHeartbeatEnv  = "FAK_CADENCEREAP_HEARTBEAT"
)

// TestMain turns the test binary into the member / grandchild stand-ins when the
// guard envs are set; otherwise it runs the package's tests normally (there is no
// other TestMain in this package). The grandchild branch is checked first so a
// process carrying both inherited guards resolves to the grandchild.
func TestMain(m *testing.M) {
	if os.Getenv(reapGrandchildEnv) != "" {
		beatUntilKilled(os.Getenv(reapHeartbeatEnv))
		os.Exit(0)
	}
	if os.Getenv(reapChildEnv) != "" {
		// The "member": spawn a heartbeating grandchild, then hang past any deadline.
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

func TestRunPyEnvelopeTimeoutReapsGrandchild3103(t *testing.T) {
	root := t.TempDir()
	// RunPyEnvelope stats root/argv[0] before launching; a stub member file must exist.
	if err := os.WriteFile(filepath.Join(root, "member_stub"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	hb := filepath.Join(t.TempDir(), "grandchild.heartbeat")
	t.Setenv(reapChildEnv, "1")
	t.Setenv(reapHeartbeatEnv, hb)

	// `python` = this test binary in child mode: it spawns the heartbeating
	// grandchild and hangs, so the only way RunPyEnvelope returns is the timeout kill.
	start := time.Now()
	_, errStr := RunPyEnvelope(root, []string{"member_stub"}, os.Args[0], 400*time.Millisecond)
	if !strings.Contains(errStr, "timed out") {
		t.Fatalf("expected a timeout result, got %q", errStr)
	}
	if d := time.Since(start); d > 30*time.Second {
		t.Fatalf("RunPyEnvelope hung %s past the 400ms deadline — the timeout kill did not return", d)
	}

	// The grandchild must have been reaped along with the member: its heartbeat
	// must go stale. Under the pre-#3103 single-PID kill it is orphaned and keeps
	// beating, which fails here.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if fi, err := os.Stat(hb); err == nil && time.Since(fi.ModTime()) > 1*time.Second {
			return // heartbeat went stale -> grandchild dead -> descendant subtree reaped
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("grandchild heartbeat %s kept advancing after the timeout — the descendant subtree was orphaned (the #3103 bug)", hb)
}
