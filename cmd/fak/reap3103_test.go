package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// reap3103_test.go — the Site-1 (keystone) reap witness for #3103 (sibling of #2989):
// runReleaseShipCommand's timeout branch must route through procguard.KillPID so the
// whole subprocess subtree is reaped, not just the direct PID. release_ship wraps
// git/gh OR the Python release-cut/tag/publish orchestrators, which spawn `git`/`npm`
// grandchildren; the pre-#3103 `cmd.Process.Kill()` reaped only the immediate process
// and orphaned them (release_ship.go:1470-1473). This test re-execs THIS test binary
// as the launched command (child) and as a grandchild the child spawns. The grandchild
// heartbeats a file; after the timeout reap it must STOP — a single-PID kill would
// orphan it and it would keep beating. Heartbeat-file liveness is fully portable (no
// pid syscalls), matching internal/cadencereport (RunPyEnvelope) and
// internal/gardenbundle (RunMember), the two sibling #3103 reap witnesses.
//
// Unlike those two internal packages, cmd/fak already declares a TestMain
// (guard_login_e2e_test.go); Go allows only one per package, so the child/grandchild
// env-dispatch is hooked into that existing TestMain via dispatchReleaseShipReap3103
// (called at the top of TestMain) rather than declaring a second TestMain here.
const (
	releaseShipReapChildEnv      = "FAK_RELEASESHIPREAP_CHILD"
	releaseShipReapGrandchildEnv = "FAK_RELEASESHIPREAP_GRANDCHILD"
	releaseShipReapHeartbeatEnv  = "FAK_RELEASESHIPREAP_HEARTBEAT"
)

// dispatchReleaseShipReap3103 turns the test binary into the child / grandchild
// stand-ins when the guard envs are set, then exits; in a normal test run neither env
// is set and it returns immediately (a no-op). TestMain (guard_login_e2e_test.go) calls
// it first. The grandchild branch is checked before the child branch so a process
// carrying BOTH inherited guards (the grandchild inherits the child's env) resolves to
// the grandchild.
func dispatchReleaseShipReap3103() {
	if os.Getenv(releaseShipReapGrandchildEnv) != "" {
		beatReleaseShipUntilKilled(os.Getenv(releaseShipReapHeartbeatEnv))
		os.Exit(0)
	}
	if os.Getenv(releaseShipReapChildEnv) != "" {
		// The wrapped command: spawn a heartbeating grandchild, then hang past any
		// deadline so the only way runReleaseShipCommand returns is the timeout kill.
		gc := exec.Command(os.Args[0])
		gc.Env = append(os.Environ(), releaseShipReapGrandchildEnv+"=1")
		_ = gc.Start()
		time.Sleep(2 * time.Minute)
		os.Exit(0)
	}
}

// beatReleaseShipUntilKilled rewrites path every ~80ms until it is killed (or a safety
// ceiling elapses), so a live grandchild keeps the file's mtime advancing.
func beatReleaseShipUntilKilled(path string) {
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

func TestRunReleaseShipCommandTimeoutReapsGrandchild3103(t *testing.T) {
	root := t.TempDir()
	hb := filepath.Join(t.TempDir(), "grandchild.heartbeat")
	t.Setenv(releaseShipReapChildEnv, "1")
	t.Setenv(releaseShipReapHeartbeatEnv, hb)

	// os.Args[0] (this test binary) in child mode: it spawns the heartbeating
	// grandchild and hangs. env=nil => runReleaseShipCommand leaves cmd.Env unset, so
	// the child inherits our process env (the reap guards above) and dispatches to
	// child mode. The 400ms timeout is the only path back out.
	start := time.Now()
	code, out := runReleaseShipCommand(root, os.Args[0], nil, nil, 400*time.Millisecond)
	if code != 124 || !strings.Contains(out, "timed out") {
		t.Fatalf("expected a timeout result (code 124, output containing %q), got code=%d out=%q", "timed out", code, out)
	}
	if d := time.Since(start); d > 30*time.Second {
		t.Fatalf("runReleaseShipCommand hung %s past the 400ms deadline — the timeout kill did not return", d)
	}

	// The grandchild must have been reaped along with the child: its heartbeat must go
	// stale. Under the pre-#3103 single-PID kill it is orphaned and keeps beating,
	// which fails here. (Negative control: reverting release_ship.go:1473 to a bare
	// cmd.Process.Kill() makes this loop time out with the advancing-heartbeat failure.)
	deadline := time.Now().Add(10 * time.Second)
	var observation releaseShipHeartbeatObservation
	for time.Now().Before(deadline) {
		fi, err := os.Stat(hb)
		var modified time.Time
		if err == nil {
			modified = fi.ModTime()
		}
		observation.observe(modified, err)
		if observation.stale(time.Now()) {
			return // heartbeat went stale -> grandchild dead -> descendant subtree reaped
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatal(observation.failure(hb))
}

// A missing heartbeat does not establish that the descendant survived cleanup:
// the timeout may have fired before the fixture could create it. Keep that failure
// distinct from a heartbeat observed advancing after the timeout.
type releaseShipHeartbeatObservation struct {
	seen     bool
	advanced bool
	modified time.Time
	err      error
}

func (o *releaseShipHeartbeatObservation) observe(modified time.Time, err error) {
	o.err = err
	if err != nil {
		return
	}
	if o.seen && modified.After(o.modified) {
		o.advanced = true
	}
	o.seen = true
	o.modified = modified
}

func (o releaseShipHeartbeatObservation) stale(now time.Time) bool {
	return o.seen && o.err == nil && now.Sub(o.modified) > time.Second
}

func (o releaseShipHeartbeatObservation) failure(path string) string {
	switch {
	case !o.seen:
		return fmt.Sprintf("grandchild heartbeat %s was never observed; fixture startup/write readiness is unproven (last stat error: %v)", path, o.err)
	case o.err != nil:
		return fmt.Sprintf("grandchild heartbeat %s was observed but is now unreadable; descendant cleanup is unproven (last stat error: %v)", path, o.err)
	case o.advanced:
		return fmt.Sprintf("grandchild heartbeat %s advanced after the timeout and did not go stale; descendant cleanup failed (the #3103 bug)", path)
	default:
		return fmt.Sprintf("grandchild heartbeat %s was observed but did not go stale; no advancing heartbeat was witnessed and descendant cleanup is unproven", path)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestReleaseShipHeartbeatObservation(t *testing.T) {
	now := time.Unix(100, 0)
	t.Run("never_observed", func(t *testing.T) {
		var o releaseShipHeartbeatObservation
		o.observe(time.Time{}, os.ErrNotExist)
		if o.stale(now) || !strings.Contains(o.failure("heartbeat"), "never observed") || !strings.Contains(o.failure("heartbeat"), os.ErrNotExist.Error()) {
			t.Fatalf("missing heartbeat misclassified: %+v: %s", o, o.failure("heartbeat"))
		}
	})
	t.Run("observed_then_unreadable", func(t *testing.T) {
		var o releaseShipHeartbeatObservation
		o.observe(now.Add(-2*time.Second), nil)
		o.observe(time.Time{}, os.ErrPermission)
		if o.stale(now) || !strings.Contains(o.failure("heartbeat"), "now unreadable") || !strings.Contains(o.failure("heartbeat"), os.ErrPermission.Error()) {
			t.Fatalf("unreadable heartbeat misclassified: %+v: %s", o, o.failure("heartbeat"))
		}
	})
	t.Run("recent_without_advance", func(t *testing.T) {
		var o releaseShipHeartbeatObservation
		o.observe(now, nil)
		o.observe(now, nil)
		if o.stale(now) || o.advanced || !strings.Contains(o.failure("heartbeat"), "no advancing heartbeat was witnessed") {
			t.Fatalf("recent heartbeat misclassified: %+v: %s", o, o.failure("heartbeat"))
		}
	})
	t.Run("advancing", func(t *testing.T) {
		var o releaseShipHeartbeatObservation
		o.observe(now.Add(-time.Millisecond), nil)
		o.observe(now, nil)
		if o.stale(now) || !strings.Contains(o.failure("heartbeat"), "advanced after the timeout") || !strings.Contains(o.failure("heartbeat"), "cleanup failed") {
			t.Fatalf("advancing heartbeat misclassified: %+v: %s", o, o.failure("heartbeat"))
		}
	})
	t.Run("strict_stale_boundary", func(t *testing.T) {
		var o releaseShipHeartbeatObservation
		o.observe(now.Add(-time.Second), nil)
		if o.stale(now) {
			t.Fatal("exactly one second must not establish a stale heartbeat")
		}
		if !o.stale(now.Add(time.Nanosecond)) {
			t.Fatal("a readable heartbeat older than one second must retain the original success condition")
		}
	})
}
