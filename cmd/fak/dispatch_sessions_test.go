package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
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
	oldStart := dispatchProcessStart
	dispatchProcessStart = func(int) (time.Time, bool) { return time.Date(2026, 7, 8, 9, 30, 30, 0, time.UTC), true }
	t.Cleanup(func() { dispatchProcessStart = oldStart })
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
	snap := dispatchSessionsScan(runsDir, regDir, "", now)

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
	snap := dispatchSessionsScan(filepath.Join(t.TempDir(), "nope"), t.TempDir(), "", time.Now().UTC())
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

// TestDispatchSessionsFoldsTokenAccounting (#3329): a session whose guard trace-id
// matches a gateway-usage row folds its OWN token volume, input-token-equivalent
// cost, and cache-read share onto the row — and a scan with NO usage ledger leaves
// all three at zero (byte-identical to the pre-accounting shape).
func TestDispatchSessionsFoldsTokenAccounting(t *testing.T) {
	oldStart := dispatchProcessStart
	dispatchProcessStart = func(int) (time.Time, bool) { return time.Date(2026, 7, 8, 9, 30, 30, 0, time.UTC), true }
	t.Cleanup(func() { dispatchProcessStart = oldStart })
	runsDir := t.TempDir()
	regDir := t.TempDir()

	pid := os.Getpid()
	worker := "resolve-4242-20260708-101500"
	header := "# fak-spawn 20260708-101500 issue=4242 lane=docs backend=claude argv0=fak"
	logMtime := time.Date(2026, 7, 8, 10, 20, 0, 0, time.UTC)
	plantResolveSession(t, runsDir, worker, header, "streamed work\n", pid, logMtime)

	const trace = "trace-acct"
	if err := guardsessions.Record(regDir, guardsessions.Row{
		Schema: guardsessions.Schema, Handle: "gacct0001", TraceID: trace,
		Agent: "claude", PID: pid, StartedAt: "2026-07-08T10:15:00Z",
	}); err != nil {
		t.Fatalf("record guard session: %v", err)
	}

	// A gateway-usage ledger row keyed by the served trace-id.
	usagePath := filepath.Join(t.TempDir(), "gateway-usage.jsonl")
	row := gatewayusageledger.NewRow("exit", "guard", "claude", trace, 0, nil, gatewayusageledger.Counters{
		InputTokens: 1000, OutputTokens: 200, CachedPromptTokens: 4000, CacheCreationTokens: 500,
	}, time.Now())
	if err := gatewayusageledger.Append(usagePath, row); err != nil {
		t.Fatalf("append usage row: %v", err)
	}

	now := time.Date(2026, 7, 8, 10, 25, 0, 0, time.UTC)
	snap := dispatchSessionsScan(runsDir, regDir, usagePath, now)
	if len(snap.Sessions) != 1 {
		t.Fatalf("want 1 session, got %d", len(snap.Sessions))
	}
	s := snap.Sessions[0]
	if s.Tokens != 5700 {
		t.Errorf("tokens = %d, want 5700 (1000+200+4000+500)", s.Tokens)
	}
	// cost = 1000 + 200 + 4000*0.1 + 500*1.25 = 2225
	if math.Abs(s.Cost-2225) > 1e-6 {
		t.Errorf("cost = %v, want 2225", s.Cost)
	}
	// cache-read share = 4000 / (1000+4000+500) = 0.727272...
	if math.Abs(s.CacheReadShare-(4000.0/5500.0)) > 1e-9 {
		t.Errorf("cache_read_share = %v, want ~0.7273", s.CacheReadShare)
	}

	// No usage ledger → no fold; the fields stay zero (omitted in JSON).
	bare := dispatchSessionsScan(runsDir, regDir, "", now)
	if b := bare.Sessions[0]; b.Tokens != 0 || b.Cost != 0 || b.CacheReadShare != 0 {
		t.Errorf("no-ledger scan folded accounting: tokens=%d cost=%v share=%v", b.Tokens, b.Cost, b.CacheReadShare)
	}
}

// plantLogOnly writes just a worker .log (no .pid/.backend) — the "missing sidecars"
// fixture for the determinism sweep.
func plantLogOnly(t *testing.T, runsDir, worker, content string, mtime time.Time) {
	t.Helper()
	log := filepath.Join(runsDir, worker+".log")
	if err := os.WriteFile(log, []byte(content), 0o644); err != nil {
		t.Fatalf("write log: %v", err)
	}
	if err := os.Chtimes(log, mtime, mtime); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
}

// TestDispatchSessionsDeterministicEdgeSweep (#3332) plants a spread of ugly fixtures —
// a live worker, a finished worker on a dead pid, a log with NO sidecars, a log with no
// spawn header, and a guard row whose pid matches no worker — then asserts the scan is
// deterministic (same inputs → identical snapshot), stably ordered (live first, then by
// worker name), and joins/skips each fixture correctly.
func TestDispatchSessionsDeterministicEdgeSweep(t *testing.T) {
	oldStart := dispatchProcessStart
	dispatchProcessStart = func(int) (time.Time, bool) { return time.Date(2026, 7, 8, 10, 15, 30, 0, time.UTC), true }
	t.Cleanup(func() { dispatchProcessStart = oldStart })
	runsDir := t.TempDir()
	regDir := t.TempDir()
	mtime := time.Date(2026, 7, 8, 10, 20, 0, 0, time.UTC)
	now := time.Date(2026, 7, 8, 10, 25, 0, 0, time.UTC)

	deadPID := 2147483000 // implausibly high → not alive → finished

	// 1) live worker on a genuinely-alive pid.
	plantResolveSession(t, runsDir, "resolve-11-20260708-101500",
		"# fak-spawn 20260708-101500 issue=11 lane=docs backend=claude argv0=fak",
		"live streamed output\n", os.Getpid(), mtime)
	// 2) finished worker (dead pid).
	plantResolveSession(t, runsDir, "resolve-22-20260708-101500",
		"# fak-spawn 20260708-101500 issue=22 lane=cmd backend=opencode argv0=fak",
		"done streamed output\n", deadPID, mtime)
	// 3) log with NO sidecars (missing .pid) → pid 0, not live.
	plantLogOnly(t, runsDir, "resolve-33-20260708-101500",
		"# fak-spawn 20260708-101500 issue=33 lane=core backend=claude argv0=fak\nno pid sidecar\n", mtime)
	// 4) log with no spawn header at all.
	plantLogOnly(t, runsDir, "resolve-44-20260708-101500",
		"just some banner text with no header line\nand more\n", mtime)

	// 5) a guard row whose pid matches NO worker — must not crash the join or attach.
	if err := guardsessions.Record(regDir, guardsessions.Row{
		Schema: guardsessions.Schema, Handle: "gorphan1", TraceID: "trace-orphan",
		Agent: "claude", PID: deadPID + 12345, StartedAt: "2026-07-08T10:15:00Z",
	}); err != nil {
		t.Fatalf("record orphan guard: %v", err)
	}

	a := dispatchSessionsScan(runsDir, regDir, "", now)
	b := dispatchSessionsScan(runsDir, regDir, "", now)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("scan not deterministic:\n a=%+v\n b=%+v", a, b)
	}

	if a.SessionCount != 4 {
		t.Fatalf("session count = %d, want 4", a.SessionCount)
	}
	if a.LiveCount != 1 {
		t.Errorf("live count = %d, want 1", a.LiveCount)
	}
	// Stable order: the single live worker sorts first.
	if !a.Sessions[0].Live {
		t.Errorf("first session should be the live one, got %+v", a.Sessions[0])
	}
	for _, s := range a.Sessions[1:] {
		if s.Live {
			t.Errorf("non-first session unexpectedly live: %+v", s)
		}
	}
	// No orphan guard leaked onto any row.
	for _, s := range a.Sessions {
		if s.Guard != nil && s.Guard.Handle == "gorphan1" {
			t.Errorf("orphan guard attached to %s", s.Worker)
		}
	}
}

// TestDispatchSessionsTailScrubs (#3334): --tail resolves a worker by a guard
// handle prefix and prints its transcript tail with every raw secret masked through
// the shared canon matcher — the tail is never emitted with a live credential.
func TestDispatchSessionsTailScrubs(t *testing.T) {
	runsDir := t.TempDir()
	regDir := t.TempDir()
	pid := os.Getpid()
	// Build the secret by concatenation so no live-looking literal sits contiguous.
	secret := "ghp_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
	worker := "resolve-55-20260708-101500"
	header := "# fak-spawn 20260708-101500 issue=55 lane=docs backend=claude argv0=fak"
	body := "streamed line one\ntoken=" + secret + "\nstreamed line three\n"
	plantResolveSession(t, runsDir, worker, header, body, pid, time.Now())

	if err := guardsessions.Record(regDir, guardsessions.Row{
		Schema: guardsessions.Schema, Handle: "gcafef00d", TraceID: "trace-tail",
		Agent: "claude", PID: pid, StartedAt: "2026-07-08T10:15:00Z",
	}); err != nil {
		t.Fatalf("record guard session: %v", err)
	}

	var stdout, stderr strings.Builder
	code := runDispatchSessions(&stdout, &stderr, []string{"--runs-dir", runsDir, "--reg-dir", regDir, "--tail", "gcafe"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if strings.Contains(out, secret) {
		t.Fatalf("raw secret leaked into tail output:\n%s", out)
	}
	if !strings.Contains(out, "dispatch session tail") {
		t.Errorf("missing tail header:\n%s", out)
	}
	if !strings.Contains(out, "masked") {
		t.Errorf("expected a masked-secret notice:\n%s", out)
	}
	if !strings.Contains(out, "streamed line three") {
		t.Errorf("expected non-secret transcript lines to survive:\n%s", out)
	}
}

// TestDispatchSessionsTailAmbiguousAndMiss exercises the two soft-fail arms: an
// unknown prefix and (implicitly) a resolved session with no worker log both exit 1
// without printing a tail.
func TestDispatchSessionsTailMiss(t *testing.T) {
	var stdout, stderr strings.Builder
	code := runDispatchSessions(&stdout, &stderr, []string{"--runs-dir", t.TempDir(), "--reg-dir", t.TempDir(), "--tail", "nope"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1 for an unknown prefix; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "no guard session matches") {
		t.Errorf("stderr = %q, want a no-match message", stderr.String())
	}
}

// TestDispatchSessionsWatchBounded (#3334): --watch with a bounded iteration count
// renders exactly that many times and returns — the loop is never infinite in a test.
func TestDispatchSessionsWatchBounded(t *testing.T) {
	runsDir := t.TempDir()
	plantResolveSession(t, runsDir, "resolve-66-20260708-101500",
		"# fak-spawn 20260708-101500 issue=66 lane=docs backend=claude argv0=fak",
		"work\n", os.Getpid(), time.Now())

	var stdout, stderr strings.Builder
	code := runDispatchSessions(&stdout, &stderr, []string{
		"--runs-dir", runsDir, "--reg-dir", t.TempDir(),
		"--watch", "--watch-iterations", "2", "--watch-interval", "1ms",
	})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "watch 1") || !strings.Contains(out, "watch 2") {
		t.Fatalf("want two bounded renders (watch 1, watch 2):\n%s", out)
	}
	if strings.Contains(out, "watch 3") {
		t.Fatalf("watch exceeded its iteration bound:\n%s", out)
	}
}

func TestDispatchSessionsRejectsReusedPIDIdentity(t *testing.T) {
	runs := t.TempDir()
	reg := t.TempDir()
	pid := os.Getpid()
	logName := "resolve-8022-20260818-120000-codex.log"
	if err := os.WriteFile(filepath.Join(runs, logName), []byte{}, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runs, strings.TrimSuffix(logName, ".log")+".pid"), []byte(strconv.Itoa(pid)), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runs, strings.TrimSuffix(logName, ".log")+".backend"), []byte("codex\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runs, strings.TrimSuffix(logName, ".log")+".lease-scope.json"), []byte(`{"worker":"resolve-8022-20260818-120000-codex","lane":"cmd","lease_id":"lease-x","tree":"cmd/fak/**"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := guardsessions.Record(reg, guardsessions.Row{Handle: "wrong-guard", TraceID: "wrong-trace", PID: pid}); err != nil {
		t.Fatal(err)
	}
	old := dispatchProcessStart
	dispatchProcessStart = func(got int) (time.Time, bool) { return time.Date(2026, 8, 18, 13, 0, 0, 0, time.UTC), got == pid }
	t.Cleanup(func() { dispatchProcessStart = old })

	snap := dispatchSessionsScan(runs, reg, "", time.Date(2026, 8, 18, 13, 1, 0, 0, time.UTC))
	if snap.LiveCount != 0 || len(snap.Sessions) != 1 {
		t.Fatalf("unexpected snapshot: %+v", snap)
	}
	row := snap.Sessions[0]
	if row.Live || !row.PIDAlive || row.PIDIdentity != "stale" || row.Guard != nil {
		t.Fatalf("reused PID fabricated live/guard state: %+v", row)
	}
}

func TestDispatchSessionsKeepsLaunchCompatiblePIDLive(t *testing.T) {
	runs := t.TempDir()
	reg := t.TempDir()
	pid := os.Getpid()
	logName := "resolve-8022-20260818-120000-codex.log"
	base := strings.TrimSuffix(logName, ".log")
	for name, data := range map[string]string{logName: "", base + ".pid": strconv.Itoa(pid), base + ".backend": "codex\n", base + ".lease-scope.json": `{"worker":"` + base + `","lane":"cmd","lease_id":"lease-x","tree":"cmd/fak/**"}`} {
		if err := os.WriteFile(filepath.Join(runs, name), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := guardsessions.Record(reg, guardsessions.Row{Handle: "guard-ok", TraceID: "trace-ok", PID: pid}); err != nil {
		t.Fatal(err)
	}
	old := dispatchProcessStart
	dispatchProcessStart = func(got int) (time.Time, bool) { return time.Date(2026, 8, 18, 12, 0, 30, 0, time.UTC), got == pid }
	t.Cleanup(func() { dispatchProcessStart = old })

	snap := dispatchSessionsScan(runs, reg, "", time.Date(2026, 8, 18, 12, 1, 0, 0, time.UTC))
	row := snap.Sessions[0]
	if !row.Live || row.PIDIdentity != "launch-confirmed" || row.Guard == nil || row.Guard.Handle != "guard-ok" {
		t.Fatalf("matching PID identity lost live state: %+v", row)
	}
}

func TestMatchDispatchWorkerWorktreeProjectsEvidenceWithoutPaths(t *testing.T) {
	row := dispatchSessionRow{Issue: "10551", Lane: "workerworktree", LeaseID: "lease-10551"}
	got := matchDispatchWorkerWorktree(row, []workerworktree.StatusEvidence{{IssueNumber: 10551, Lane: "workerworktree", Session: "lease-10551", AssociationKnown: true, CleanupReady: true}})
	if got == nil || got.State != workerworktree.DisplayCleanupReady || got.Complete {
		t.Fatalf("projection = %#v", got)
	}
	rendered := renderDispatchSessions(dispatchSessionsSnapshot{SessionCount: 1, Sessions: []dispatchSessionRow{{Issue: "10551", Lane: "workerworktree", Backend: "codex", Worker: "resolve-10551", Outcome: "done", WorkerWorktree: got}}})
	markdown := renderDispatchSessionsMarkdown(dispatchSessionsSnapshot{SessionCount: 1, Sessions: []dispatchSessionRow{{Issue: "10551", Lane: "workerworktree", Backend: "codex", Worker: "resolve-10551", Outcome: "done", WorkerWorktree: got}}})
	if !strings.Contains(rendered, "worker-worktree=cleanup_ready  complete=false") || !strings.Contains(markdown, "cleanup_ready") {
		t.Fatalf("missing projection:\n%s\n%s", rendered, markdown)
	}
}

func TestMatchDispatchWorkerWorktreeRequiresSafeIdentity(t *testing.T) {
	got := matchDispatchWorkerWorktree(dispatchSessionRow{Issue: "99", Lane: "docs"}, []workerworktree.StatusEvidence{{IssueNumber: 10551, Lane: "workerworktree", AssociationKnown: true, Dirty: true}})
	if got != nil {
		t.Fatalf("unrelated projection = %#v", got)
	}
}

func TestMatchDispatchWorkerWorktreeFailsClosedOnAmbiguousAssociation(t *testing.T) {
	row := dispatchSessionRow{Issue: "10551", Lane: "workerworktree"}
	got := matchDispatchWorkerWorktree(row, []workerworktree.StatusEvidence{
		{IssueNumber: 10551, Lane: "workerworktree", AssociationKnown: true, Dirty: true},
		{IssueNumber: 10551, Lane: "workerworktree", AssociationKnown: true, CleanupReady: true},
	})
	if got != nil {
		t.Fatalf("ambiguous projection = %#v, want nil", got)
	}
}
