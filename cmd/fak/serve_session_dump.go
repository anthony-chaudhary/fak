package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/snapshot"
)

// restoreServeSessions re-attaches the persisted DRIVE state of every session (the COLD
// resume of #629) from a fleet-snapshot file a prior `fak serve` wrote on shutdown. It is
// the load-time inverse of dumpServeSessions: each session re-attaches at the budget /
// priority / run-state / pace it held — a STOPPED session reloads STOPPED with its reason
// (session.Table.Restore is the one write that re-establishes a terminal record), never
// silently resurrected as RUNNING. An empty path is off (no-op). A missing file is a clean
// first boot (not an error). A PRESENT-but-corrupt file fails loud — a tampered/truncated
// drive record is worse than none, the same fail-closed posture the policy/route loaders
// take, and the snapshot envelope's own sha256 body digest is what catches the tamper. This
// is the process-restart half the design note SESSION-CONTROL-STATE-AS-FIRST-CLASS §5
// named; it is DISTINCT from the live Paused→Running resume the control verbs already do.
func restoreServeSessions(tbl *session.Table, path string) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	snap, err := snapshot.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // first boot — nothing persisted yet
		}
		return fmt.Errorf("--session-state %s: %w", path, err)
	}
	n, err := snap.RestoreFleet(tbl)
	if err != nil {
		return fmt.Errorf("--session-state %s: %w", path, err)
	}
	if n > 0 {
		fmt.Fprintf(os.Stderr, "fak: cold resume (#629) — re-attached %d session(s) drive state from %s\n", n, path)
	}
	return nil
}

// dumpServeSessions writes the live DRIVE table to path as an integrity-checked fleet
// snapshot so the NEXT `fak serve` cold-resumes it (#629). An empty path is off (no-op).
// Best-effort on a clean shutdown: a write failure is logged, never fatal — a failed dump
// must not turn a graceful stop into a crash (worst case the next boot starts at defaults,
// exactly today's behavior). A hard kill skips the dump; the last clean shutdown's file
// stands. An empty table writes an empty (still valid) snapshot.
func dumpServeSessions(tbl *session.Table, path string) {
	path = strings.TrimSpace(path)
	if path == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "fak: create session-state parent for %s failed: %v\n", path, err)
		return
	}
	snap, err := snapshot.DumpFleet("serve", tbl, 0)
	if err == nil {
		var b []byte
		if b, err = snap.Encode(); err == nil {
			err = os.WriteFile(path, b, 0o644)
		}
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak: persist session state to %s failed: %v\n", path, err)
		return
	}
	fmt.Fprintf(os.Stderr, "fak: persisted live session drive state → %s (#629)\n", path)
}
