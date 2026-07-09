package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
)

// plantResolveSession writes a resolve-<issue>-<stamp> worker log + its .pid/.backend
// sidecars for a live pid, mirroring what spawnDispatchIssueWorker leaves behind, and
// stamps the log mtime so the age fold is deterministic.
func plantResolveSession(t *testing.T, runsDir, worker, header, body string, pid int, mtime time.Time) {
	t.Helper()
	log := filepath.Join(runsDir, worker+".log")
	content := header + "\n" + body
	if err := os.WriteFile(log, []byte(content), 0o644); err != nil {
		t.Fatalf("write log: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, worker+".pid"), []byte(strconv.Itoa(pid)), 0o644); err != nil {
		t.Fatalf("write pid: %v", err)
	}
	if err := os.Chtimes(log, mtime, mtime); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
}

// TestDispatchSessionsJoinsGuardAndOutcome is the spine witness: a live worker (alive
// pid, real log bytes) folds into a session row carrying its audit outcome, a nonzero
// age from the log mtime, and — the cross-plane join no other command does — the guard
// session (handle/trace/audit) resolved from guard_sessions.jsonl by pid.
func TestDispatchSessionsJoinsGuardAndOutcome(t *testing.T) {
	runsDir := t.TempDir()
	regDir := t.TempDir()

	pid := os.Getpid() // a genuinely-alive pid so liveResolutionScopes counts it live
	worker := "resolve-4242-20260708-101500"
	header := "# fak-spawn 20260708-101500 issue=4242 lane=docs backend=claude argv0=fak"
	body := "fak guard starting\nworking on the docs lane\nsome real streamed output here\n"
	logMtime := time.Date(2026, 7, 8, 10, 20, 0, 0, time.UTC)
	plantResolveSession(t, runsDir, worker, header, body, pid, logMtime)

	// A guard session recorded for the SAME pid — the join key.
	if err := guardsessions.Record(regDir, guardsessions.Row{
		Schema:    guardsessions.Schema,
		Handle:    "gdeadbeef",
		TraceID:   "trace-xyz",
		Agent:     "claude",
		PID:       pid,
		AuditPath: "/tmp/guard-audit.json",
		StartedAt: "2026-07-08T10:15:00Z",
	}); err != nil {
		t.Fatalf("record guard session: %v", err)
	}

	now := time.Date(2026, 7, 8, 10, 25, 0, 0, time.UTC) // 5 min after the log mtime
	snap := dispatchSessionsScan(runsDir, regDir, now)

	if snap.Schema != dispatchSessionsSchema {
		t.Fatalf("schema = %q, want %q", snap.Schema, dispatchSessionsSchema)
	}
	if snap.SessionCount != 1 || len(snap.Sessions) != 1 {
		t.Fatalf("session count = %d (%d rows), want 1", snap.SessionCount, len(snap.Sessions))
	}
	if snap.LiveCount != 1 {
		t.Fatalf("live count = %d, want 1", snap.LiveCount)
	}
	s := snap.Sessions[0]
	if !s.Live {
		t.Errorf("session not marked live")
	}
	if s.Issue != "4242" {
		t.Errorf("issue = %q, want 4242", s.Issue)
	}
	if s.Lane != "docs" {
		t.Errorf("lane = %q, want docs", s.Lane)
	}
	if s.Backend != "claude" {
		t.Errorf("backend = %q, want claude", s.Backend)
	}
	if s.Outcome == "" {
		t.Errorf("outcome empty; want a classified outcome")
	}
	if s.AgeSeconds != 300 {
		t.Errorf("age_seconds = %d, want 300 (5 min)", s.AgeSeconds)
	}
	if s.Started != "2026-07-08T10:15:00Z" {
		t.Errorf("started = %q, want 2026-07-08T10:15:00Z", s.Started)
	}
	if s.Guard == nil {
		t.Fatalf("guard session not joined (nil); the cross-plane join failed")
	}
	if s.Guard.Handle != "gdeadbeef" || s.Guard.TraceID != "trace-xyz" || s.Guard.AuditPath != "/tmp/guard-audit.json" {
		t.Errorf("guard join = %+v, want handle=gdeadbeef trace=trace-xyz audit=/tmp/guard-audit.json", s.Guard)
	}
}

// TestDispatchSessionsEmptyRunsDir fails soft on a missing runs dir: no sessions, no
// panic, a well-formed empty snapshot (matching `dispatch status`).
func TestDispatchSessionsEmptyRunsDir(t *testing.T) {
	snap := dispatchSessionsScan(filepath.Join(t.TempDir(), "nope"), t.TempDir(), time.Now().UTC())
	if snap.SessionCount != 0 || len(snap.Sessions) != 0 {
		t.Fatalf("want empty snapshot, got %d sessions", snap.SessionCount)
	}
	if snap.Schema != dispatchSessionsSchema {
		t.Fatalf("schema = %q, want %q", snap.Schema, dispatchSessionsSchema)
	}
}

// TestDispatchSessionsCLIJSON drives the command end to end and asserts it emits the
// fleet-dispatch-sessions/1 payload.
func TestDispatchSessionsCLIJSON(t *testing.T) {
	runsDir := t.TempDir()
	worker := "resolve-77-20260708-090000"
	header := "# fak-spawn 20260708-090000 issue=77 lane=cmd backend=opencode argv0=fak"
	plantResolveSession(t, runsDir, worker, header, "streamed work\n", os.Getpid(), time.Now().UTC())

	var stdout, stderr strings.Builder
	code := runDispatchSessions(&stdout, &stderr, []string{"--runs-dir", runsDir, "--reg-dir", t.TempDir(), "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	var snap dispatchSessionsSnapshot
	if err := json.Unmarshal([]byte(stdout.String()), &snap); err != nil {
		t.Fatalf("decode json: %v\n%s", err, stdout.String())
	}
	if snap.Schema != dispatchSessionsSchema || snap.SessionCount != 1 {
		t.Fatalf("snapshot = %+v", snap)
	}
	if snap.Sessions[0].Issue != "77" {
		t.Errorf("issue = %q, want 77", snap.Sessions[0].Issue)
	}
}
