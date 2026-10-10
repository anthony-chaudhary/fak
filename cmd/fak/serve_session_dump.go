package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/snapshot"
)

// sessionStateCorruptCode is the typed warning a quarantined --session-state file emits.
const sessionStateCorruptCode = "SESSION_STATE_CORRUPT"

// restoreServeSessions re-attaches the persisted DRIVE state of every session (the COLD
// resume of #629) from a fleet-snapshot file a prior `fak serve` wrote on shutdown. It is
// the load-time inverse of dumpServeSessions: each session re-attaches at the budget /
// priority / run-state / pace it held — a STOPPED session reloads STOPPED with its reason
// (session.Table.Restore is the one write that re-establishes a terminal record), never
// silently resurrected as RUNNING. An empty path is off (no-op). A missing file is a clean
// first boot (not an error). A PRESENT-but-corrupt file (snapshot.ErrCorrupt: torn,
// concatenated, tampered, wrong kind) is never resumed as if whole: it is renamed to
// <path>.corrupt-<unix> with a SESSION_STATE_CORRUPT warning and serve starts with empty
// drive state, so one bad file cannot turn every restart into an exit. An I/O error
// reading the file still fails loud. This is the process-restart half the design note
// SESSION-CONTROL-STATE-AS-FIRST-CLASS §5 named; it is DISTINCT from the live
// Paused→Running resume the control verbs already do.
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
		if errors.Is(err, snapshot.ErrCorrupt) {
			return quarantineServeSessionState(path, err)
		}
		return fmt.Errorf("--session-state %s: %w", path, err)
	}
	n, err := snap.RestoreFleet(tbl)
	if err != nil {
		if errors.Is(err, snapshot.ErrCorrupt) {
			return quarantineServeSessionState(path, err)
		}
		return fmt.Errorf("--session-state %s: %w", path, err)
	}
	if n > 0 {
		fmt.Fprintf(os.Stderr, "fak: cold resume (#629) — re-attached %d session(s) drive state from %s\n", n, path)
	}
	return nil
}

func quarantineServeSessionState(path string, cause error) error {
	dst := fmt.Sprintf("%s.corrupt-%d", path, time.Now().Unix())
	if err := os.Rename(path, dst); err != nil {
		return fmt.Errorf("--session-state %s: %w (quarantine to %s failed: %v)", path, cause, dst, err)
	}
	fmt.Fprintf(os.Stderr, "fak serve: warning %s: --session-state %s: %v; moved to %s, starting with empty session state\n",
		sessionStateCorruptCode, path, cause, dst)
	return nil
}

// dumpServeSessions writes the live DRIVE table to path as an integrity-checked fleet
// snapshot so the NEXT `fak serve` cold-resumes it (#629). An empty path is off (no-op).
// Best-effort on a clean shutdown: a write failure is logged, never fatal — a failed dump
// must not turn a graceful stop into a crash (worst case the next boot starts at defaults,
// exactly today's behavior). The write is atomic (snapshot.WriteFile), so a hard kill
// mid-dump leaves the last clean shutdown's file. An empty table writes an empty (still
// valid) snapshot.
func dumpServeSessions(tbl *session.Table, path string) {
	path = strings.TrimSpace(path)
	if path == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "fak: create session-state parent for %s failed: %v\n", path, err)
		return
	}
	body := snapshot.FleetBody{Sessions: tbl.Snapshot()}
	if err := snapshot.WriteFile(path, snapshot.KindFleet, "serve", body, nil, 0); err != nil {
		fmt.Fprintf(os.Stderr, "fak: persist session state to %s failed: %v\n", path, err)
		return
	}
	fmt.Fprintf(os.Stderr, "fak: persisted live session drive state → %s (#629)\n", path)
}
