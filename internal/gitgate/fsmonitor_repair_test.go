package gitgate

import (
	"context"
	"strings"
	"testing"
)

// driftedFake returns a fakeMaint modeling the #5068 acceptance-gate clone: safe
// posture everywhere EXCEPT core.fsmonitor=true with a dead builtin daemon.
func driftedFake() *fakeMaint {
	posture := safePosture()
	posture["core.fsmonitor"] = "true"
	return &fakeMaint{
		posture: posture,
		fsmon:   "fsmonitor-daemon is not watching",
	}
}

// TestRepairFsmonitorOffClearsTrueButDead is the #5068 acceptance gate in test form:
// on a `true`-but-dead clone the default (off) repair unsets core.fsmonitor, the
// drift reads CLEARED, and a subsequent readPosture reads SAFE.
func TestRepairFsmonitorOffClearsTrueButDead(t *testing.T) {
	f := driftedFake()

	// Before: the detector flags the drift.
	if p := readPosture(context.Background(), f.run, "root"); p.Safe {
		t.Fatalf("precondition: posture should read DRIFT on a true-but-dead clone, got SAFE (%+v)", p)
	}

	res := RepairFsmonitor(context.Background(), f.run, FsmonitorRepairOptions{RepoRoot: "root", Apply: true})
	if res.Action != FsmonitorActionUnsetKey {
		t.Fatalf("action = %q, want %q", res.Action, FsmonitorActionUnsetKey)
	}
	if !res.Ran || res.Code != 0 || res.Err != "" {
		t.Fatalf("repair did not run cleanly: %+v", res)
	}
	if res.BeforeValue != "true" || res.BeforeDaemon != fsmonitorNotWatching {
		t.Fatalf("before witness = (%q, %q), want (true, not-watching)", res.BeforeValue, res.BeforeDaemon)
	}
	if res.AfterValue != "" || !res.Cleared {
		t.Fatalf("after witness = (value=%q, cleared=%v), want unset + cleared", res.AfterValue, res.Cleared)
	}

	// After: readPosture reads SAFE — the acceptance gate's witnessed posture line.
	if p := readPosture(context.Background(), f.run, "root"); !p.Safe {
		t.Fatalf("posture after repair should read SAFE, got drift: %s", p.Drift)
	}
}

// TestRepairFsmonitorStartClearsByStartingDaemon: the alternative mode starts the
// builtin daemon and the drift clears with core.fsmonitor kept enabled.
func TestRepairFsmonitorStartClearsByStartingDaemon(t *testing.T) {
	f := driftedFake()
	res := RepairFsmonitor(context.Background(), f.run, FsmonitorRepairOptions{
		RepoRoot: "root", Mode: FsmonitorRepairStart, Apply: true,
	})
	if res.Action != FsmonitorActionStartDaemon {
		t.Fatalf("action = %q, want %q", res.Action, FsmonitorActionStartDaemon)
	}
	if res.AfterValue != "true" || res.AfterDaemon != fsmonitorWatching || !res.Cleared {
		t.Fatalf("after witness = (%q, %q, cleared=%v), want (true, watching, true)", res.AfterValue, res.AfterDaemon, res.Cleared)
	}
	if p := readPosture(context.Background(), f.run, "root"); !p.Safe {
		t.Fatalf("posture after daemon start should read SAFE, got drift: %s", p.Drift)
	}
}

// TestRepairFsmonitorStartFailureNotCleared: the verdict comes from the RE-PROBED
// state, never the start command's outcome — a start that leaves no watching daemon
// reads NOT cleared (fail-closed), covering the daemon-cannot-stay-up clone shape.
func TestRepairFsmonitorStartFailureNotCleared(t *testing.T) {
	f := driftedFake()
	f.fsmonStartFails = true
	res := RepairFsmonitor(context.Background(), f.run, FsmonitorRepairOptions{
		RepoRoot: "root", Mode: FsmonitorRepairStart, Apply: true,
	})
	if res.Cleared {
		t.Fatalf("a failed daemon start must not read cleared: %+v", res)
	}
	if res.AfterDaemon != fsmonitorNotWatching {
		t.Fatalf("after daemon = %q, want %q", res.AfterDaemon, fsmonitorNotWatching)
	}
}

// TestRepairFsmonitorUnknownDaemonRepairs: the third daemon health class — an
// unprobeable daemon (unknown) — is drift too, and the default repair clears it.
func TestRepairFsmonitorUnknownDaemonRepairs(t *testing.T) {
	f := driftedFake()
	f.fsmonErr = true // status probe cannot run at all → unknown
	res := RepairFsmonitor(context.Background(), f.run, FsmonitorRepairOptions{RepoRoot: "root", Apply: true})
	if res.BeforeDaemon != fsmonitorUnknown {
		t.Fatalf("before daemon = %q, want %q", res.BeforeDaemon, fsmonitorUnknown)
	}
	if res.Action != FsmonitorActionUnsetKey || !res.Cleared {
		t.Fatalf("unknown-daemon drift should repair via unset-key and clear, got %+v", res)
	}
}

// TestRepairFsmonitorNoopWhenAlreadySafe: all three already-safe shapes — key
// unset, hook-program path, and a watching builtin daemon — repair nothing and
// touch no config.
func TestRepairFsmonitorNoopWhenAlreadySafe(t *testing.T) {
	cases := []struct {
		name   string
		value  string // "" = unset
		fsmon  string
		daemon string
	}{
		{name: "unset", value: ""},
		{name: "hook-program-path", value: "C:/hooks/fsmonitor.exe"},
		{name: "watching", value: "true", fsmon: "fsmonitor-daemon is watching 'C:/work/clone'", daemon: fsmonitorWatching},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			posture := safePosture()
			if tc.value != "" {
				posture["core.fsmonitor"] = tc.value
			}
			f := &fakeMaint{posture: posture, fsmon: tc.fsmon}
			res := RepairFsmonitor(context.Background(), f.run, FsmonitorRepairOptions{RepoRoot: "root", Apply: true})
			if res.Action != FsmonitorActionNone || !res.Cleared || res.Ran {
				t.Fatalf("already-safe shape %q should be a cleared no-op, got %+v", tc.name, res)
			}
			if res.BeforeDaemon != tc.daemon {
				t.Fatalf("before daemon = %q, want %q", res.BeforeDaemon, tc.daemon)
			}
			for _, c := range f.calls {
				if len(c) >= 2 && (c[1] == "config" && len(c) >= 3 && c[2] == "--unset" || c[1] == "fsmonitor--daemon" && len(c) >= 3 && c[2] == "start") {
					t.Fatalf("no-op repair issued a mutating call: %v", c)
				}
			}
		})
	}
}

// TestRepairFsmonitorDryRunPlansOnly: without Apply the repair reports the planned
// action but mutates nothing — the key survives and the drift stays uncleared.
func TestRepairFsmonitorDryRunPlansOnly(t *testing.T) {
	f := driftedFake()
	res := RepairFsmonitor(context.Background(), f.run, FsmonitorRepairOptions{RepoRoot: "root", Apply: false})
	if res.Action != FsmonitorActionUnsetKey || res.Ran {
		t.Fatalf("dry-run should plan unset-key without running, got %+v", res)
	}
	if strings.Join(res.Args, " ") != "config --unset core.fsmonitor" {
		t.Fatalf("planned args = %v", res.Args)
	}
	if res.Cleared {
		t.Fatalf("dry-run on a drifted clone must not read cleared")
	}
	if v, ok := f.posture["core.fsmonitor"]; !ok || v != "true" {
		t.Fatalf("dry-run mutated the config: core.fsmonitor=%q ok=%v", v, ok)
	}
}

// TestRepairFsmonitorUnknownModeRefuses: an unrecognized mode repairs nothing and
// fails closed (not cleared), so a typo cannot unset a key or start a daemon.
func TestRepairFsmonitorUnknownModeRefuses(t *testing.T) {
	f := driftedFake()
	res := RepairFsmonitor(context.Background(), f.run, FsmonitorRepairOptions{
		RepoRoot: "root", Mode: FsmonitorRepairMode("nuke"), Apply: true,
	})
	if res.Action != FsmonitorActionNone || res.Cleared || res.Err == "" {
		t.Fatalf("unknown mode should refuse with an error, got %+v", res)
	}
	if len(f.calls) != 0 {
		t.Fatalf("unknown mode issued git calls: %v", f.calls)
	}
}
