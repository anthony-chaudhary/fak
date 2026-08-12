package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume/stopped"
)

// dedupArgvSIDs are the three transcript stems the argv case stages: the LIVE owner of a
// /loop lane, its crashed twin on the SAME lane (the only row that may ever be tombstoned),
// and a crashed session doing genuinely distinct work (which must never be).
const (
	dedupLiveSID     = "aaaaaaaa-1111-2222-3333-444444444444"
	dedupTwinSID     = "bbbbbbbb-1111-2222-3333-444444444444"
	dedupDistinctSID = "cccccccc-1111-2222-3333-444444444444"
)

// stageDedupHome builds a hermetic fleet home holding ONE worker account with the three
// transcripts above, and redirects the fleet policy/registry env into it, so the run never
// reads the operator's real accounts or ledger.
func stageDedupHome(t *testing.T) string {
	t.Helper()
	home, regDir := t.TempDir(), t.TempDir()
	projDir := filepath.Join(home, ".claude-t3146", "projects", "proj-under-test")
	if err := os.MkdirAll(projDir, 0o755); err != nil {
		t.Fatalf("stage account dir: %v", err)
	}
	// A /loop launch turn, then a trailing USER turn: last_role=user is what splits the quiet
	// tail into STOPPED_MIDTURN (work in flight, a resume candidate) rather than STOPPED_DONE.
	write := func(sid, launch string, ageMin float64) {
		path := filepath.Join(projDir, sid+".jsonl")
		lines := []string{
			`{"type":"user","timestamp":"2026-07-07T00:00:00Z","sessionId":"` + sid +
				`","message":{"role":"user","content":[{"type":"text","text":` + jsonQuote(launch) + `}]}}`,
			`{"type":"assistant","timestamp":"2026-07-07T00:01:00Z","sessionId":"` + sid +
				`","message":{"role":"assistant","content":[{"type":"text","text":"working"}]}}`,
			`{"type":"user","timestamp":"2026-07-07T00:02:00Z","sessionId":"` + sid +
				`","message":{"role":"user","content":[{"type":"text","text":"keep going"}]}}`,
		}
		if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
			t.Fatalf("stage transcript %s: %v", sid, err)
		}
		when := time.Now().Add(-time.Duration(ageMin) * time.Minute)
		if err := os.Chtimes(path, when, when); err != nil {
			t.Fatalf("backdate %s: %v", sid, err)
		}
	}
	// Fresh mtime -> inside stopped.LiveMinutes -> DispLive: this is the owner of the lane.
	write(dedupLiveSID, "/loop --lane claude dos-dispatch-loop", 0)
	// Backdated past LiveMinutes -> a stopped row on the SAME lane in the SAME project.
	write(dedupTwinSID, "/loop --lane claude dos-dispatch-loop", 30)
	// Distinct work: a /goal wins over any lane, so this key can never collide with the loop.
	write(dedupDistinctSID, "<command-name>/goal</command-name><command-args>ship the unrelated thing</command-args>", 30)

	policy := filepath.Join(regDir, "accounts_policy.json")
	if err := os.WriteFile(policy, []byte(`{"exclude":[],"include_only":[],"notes":{},"account_profiles":{}}`), 0o644); err != nil {
		t.Fatalf("stage accounts policy: %v", err)
	}
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", filepath.Join(home, ".config"))
	t.Setenv("FLEET_POLICY_PATH", policy)
	t.Setenv("FLEET_REG_DIR", regDir)
	return home
}

// jsonQuote renders s as a JSON string literal so a staged transcript line stays valid JSON.
func jsonQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// The whole acceptance clause driven through the REAL argv surface — `fak resume dedup` as
// the dispatcher routes it, not the internal helpers. This is the leg that reds at the tip
// where the verb was never wired into runResume: an unknown verb prints usage and exits 2.
//
//	dry-run  -> lists the duplicate with its live owner + shared work-key, writes NOTHING
//	--apply  -> appends exactly one manual_override tombstone
//	re-apply -> writes nothing (idempotent)
//	always   -> the distinct-work session is NEVER tombstoned
func TestRunResumeDedupThroughArgv(t *testing.T) {
	home := stageDedupHome(t)
	ledger := filepath.Join(t.TempDir(), "reg", "resume_ledger.jsonl")

	run := func(extra ...string) (map[string]any, string) {
		t.Helper()
		var out, errb bytes.Buffer
		argv := append([]string{"dedup", "--home", home, "--ledger", ledger, "--window-h", "24"}, extra...)
		if code := runResume(&out, &errb, argv); code != 0 {
			t.Fatalf("fak resume %v exited %d\nstdout:\n%s\nstderr:\n%s", argv, code, out.String(), errb.String())
		}
		if !strings.Contains(strings.Join(extra, " "), "--json") {
			return nil, out.String()
		}
		var rec map[string]any
		if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
			t.Fatalf("decode --json record: %v\n%s", err, out.String())
		}
		return rec, out.String()
	}

	// 1. Dry-run: exactly the crashed twin, named with its live owner and shared work-key.
	rec, _ := run("--json")
	if got := rec["n_duplicates"]; got != float64(1) {
		t.Fatalf("dry-run n_duplicates = %v, want 1 (only the crashed twin)\n%v", got, rec)
	}
	if got := rec["n_written"]; got != float64(0) {
		t.Fatalf("dry-run must write nothing, n_written = %v", got)
	}
	cands, _ := rec["candidates"].([]any)
	if len(cands) != 1 {
		t.Fatalf("candidates = %v, want exactly the crashed twin", cands)
	}
	c, _ := cands[0].(map[string]any)
	if c["session"] != dedupTwinSID {
		t.Fatalf("candidate session = %v, want the crashed twin %s", c["session"], dedupTwinSID)
	}
	if c["live_owner"] != dedupLiveSID {
		t.Fatalf("candidate live_owner = %v, want the LIVE lane owner %s", c["live_owner"], dedupLiveSID)
	}
	if c["work_key"] != "loop:--lane claude" {
		t.Fatalf("candidate work_key = %v, want the shared lane key", c["work_key"])
	}
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Fatalf("dry-run must not create the ledger at %s (stat err = %v)", ledger, err)
	}
	// The human render must NAME the owner and the key — that is the operator's evidence.
	_, human := run()
	if !strings.Contains(human, "loop:--lane claude") || !strings.Contains(human, shortID(dedupLiveSID)) {
		t.Fatalf("dry-run render must name the shared work-key and the live owner:\n%s", human)
	}

	// 2. --apply: exactly one tombstone, in the shape resume_blocked() honors.
	rec, _ = run("--apply", "--json")
	if got := rec["n_written"]; got != float64(1) {
		t.Fatalf("--apply n_written = %v, want 1 tombstone", got)
	}
	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 1 {
		t.Fatalf("ledger has %d rows, want exactly one tombstone:\n%s", len(lines), raw)
	}
	var row dedupTombstoneRow
	if err := json.Unmarshal([]byte(lines[0]), &row); err != nil {
		t.Fatalf("tombstone is not one-line JSON: %v", err)
	}
	if row.Session != dedupTwinSID || !row.ManualOverride || row.LiveOwner != dedupLiveSID {
		t.Fatalf("tombstone must settle the twin and name the live owner: %+v", row)
	}
	if strings.Contains(string(raw), dedupDistinctSID) {
		t.Fatalf("a genuinely distinct session must NEVER be tombstoned:\n%s", raw)
	}

	// 3. Re-apply is idempotent: the settled fold already blocks the session.
	rec, _ = run("--apply", "--json")
	if got := rec["n_written"]; got != float64(0) {
		t.Fatalf("re-apply n_written = %v, want 0 (already tombstoned)", got)
	}
	if got, _ := os.ReadFile(ledger); len(strings.Split(strings.TrimSpace(string(got)), "\n")) != 1 {
		t.Fatalf("re-apply must not append a second tombstone:\n%s", got)
	}
}

// The tombstone plan is exactly the DUP_LIVE verdicts, joined to the live owner's session
// id — and NOTHING else: a distinct work-key, an empty work-key, and the live owner itself
// must never appear (the issue's never-tombstone acceptance fence).
func TestPlanResumeDedupFences(t *testing.T) {
	rows := []stopped.Row{
		{Disp: stopped.DispLive, Session: "live-owner", Project: "projA", WorkKey: "loop:--lane claude", AgeMin: 1},
		{Disp: stopped.DispStoppedMidturn, Session: "crashed-twin", Project: "projA", WorkKey: "loop:--lane claude", AgeMin: 30},
		{Disp: stopped.DispStoppedMidturn, Session: "distinct-work", Project: "projA", WorkKey: "issue:#42", AgeMin: 30},
		{Disp: stopped.DispStoppedMidturn, Session: "no-work-key", Project: "projA", WorkKey: "", AgeMin: 30},
		// Same work-key in a DIFFERENT project: dedup is per-project, never cross-repo.
		{Disp: stopped.DispStoppedMidturn, Session: "other-project", Project: "projB", WorkKey: "loop:--lane claude", AgeMin: 30},
	}
	d := stopped.Decide(rows, func(string) bool { return true })
	cands := planResumeDedup(d, map[string]bool{})
	if len(cands) != 1 {
		t.Fatalf("planResumeDedup returned %d candidates, want exactly the crashed twin: %+v", len(cands), cands)
	}
	c := cands[0]
	if c.Session != "crashed-twin" || c.LiveOwner != "live-owner" || c.WorkKey != "loop:--lane claude" {
		t.Fatalf("candidate = %+v, want crashed-twin owned by live-owner on the shared work-key", c)
	}
	if c.Settled {
		t.Fatalf("fresh ledger must not settle the candidate: %+v", c)
	}
	if c.Disp != string(stopped.DispStoppedMidturn) {
		t.Fatalf("the real stop-cause must ride the candidate (got %q) — dedup never masks WHY it stopped", c.Disp)
	}
}

// ledgerSettledSessions mirrors resume_blocked()'s honor point: manual_override and
// consolidate rows settle a session, a LATER rearm row lifts the settle (a rearm row
// itself carries manual_override and must not read as a settle), and malformed or
// session-less rows are skipped.
func TestLedgerSettledSessionsMirrorsResumeBlocked(t *testing.T) {
	path := filepath.Join(t.TempDir(), "resume_ledger.jsonl")
	lines := []string{
		`{"ts":"2026-07-01T00:00:00Z","session":"a","manual_override":true}`,
		`{"ts":"2026-07-01T00:01:00Z","session":"b","action":"consolidate-operator-excluded"}`,
		`{"ts":"2026-07-01T00:02:00Z","session":"c","manual_override":true}`,
		`{"ts":"2026-07-01T00:03:00Z","session":"c","phase":"rearm","manual_override":true}`,
		`{"ts":"2026-07-01T00:04:00Z","session":"d","phase":"launched","attempt":1}`,
		`not json at all`,
		`{"ts":"2026-07-01T00:05:00Z","phase":"gate_fail_open","manual_override":true}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	settled := ledgerSettledSessions(path)
	if !settled["a"] || !settled["b"] {
		t.Fatalf("manual_override / consolidate rows must settle: %v", settled)
	}
	if settled["c"] {
		t.Fatalf("a rearm AFTER the tombstone must lift the settle (resume_blocked trims to after the last rearm): %v", settled)
	}
	if settled["d"] {
		t.Fatalf("a plain launched row must not settle: %v", settled)
	}
	if len(settled) != 2 {
		t.Fatalf("settled = %v, want exactly {a,b}", settled)
	}
	if got := ledgerSettledSessions(filepath.Join(t.TempDir(), "missing.jsonl")); len(got) != 0 {
		t.Fatalf("a missing ledger must settle nothing, got %v", got)
	}
}

// --apply appends exactly one tombstone per duplicate, the row carries the shape every
// ledger reader honors (manual_override for resume_blocked, a non-launch phase for the
// rate/spacing scanners), and a re-run writes nothing (idempotent through the same
// settled fold the plan consumes).
func TestAppendDedupTombstonesIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reg", "resume_ledger.jsonl")
	now := time.Date(2026, 7, 7, 12, 0, 0, 0, time.UTC)
	cands := []dedupCandidate{{
		Session: "734355cc-dead-beef", Account: ".claude-w1", Project: "C--work-fak",
		WorkKey: "loop:--lane claude", LiveOwner: "abcd1234-live", Disp: "STOPPED_MIDTURN",
	}}

	wrote, err := appendDedupTombstones(path, cands, now)
	if err != nil || wrote != 1 {
		t.Fatalf("first apply: wrote=%d err=%v, want 1 tombstone", wrote, err)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var rows []dedupTombstoneRow
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r dedupTombstoneRow
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatalf("tombstone row is not valid one-line JSON: %v", err)
		}
		rows = append(rows, r)
	}
	if len(rows) != 1 {
		t.Fatalf("ledger has %d rows, want exactly 1 tombstone per duplicate", len(rows))
	}
	r := rows[0]
	if !r.ManualOverride || r.Session != "734355cc-dead-beef" || r.Action != dedupTombstoneAction {
		t.Fatalf("tombstone must carry manual_override+session+action (the resume_blocked honor shape): %+v", r)
	}
	if r.Phase != "skipped" || !isNonLaunchPhase(r.Phase) {
		t.Fatalf("tombstone phase %q must be in the non-launch set so it never reads as launch pressure", r.Phase)
	}
	if r.LiveOwner != "abcd1234-live" || r.WorkKey != "loop:--lane claude" {
		t.Fatalf("tombstone must name the live owner and the shared work-key: %+v", r)
	}

	// Idempotency: the settled fold now blocks the session, so a re-plan re-applies nothing.
	recands := planResumeDedup(decisionsWithDup(t), ledgerSettledSessions(path))
	if len(recands) != 1 || !recands[0].Settled {
		t.Fatalf("re-plan must mark the tombstoned session settled: %+v", recands)
	}
	wrote, err = appendDedupTombstones(path, recands, now)
	if err != nil || wrote != 0 {
		t.Fatalf("re-apply: wrote=%d err=%v, want 0 (already tombstoned)", wrote, err)
	}
}

// dedupTombstoneGoldenLine is the byte-exact ledger line --apply emits for the canonical
// duplicate. tools/fleet_resume_watchdog_test.py pins the IDENTICAL literal and feeds it to
// resume_blocked(); pinning it on both sides is what makes the cross-language honor point
// testable — rename a JSON field here and one of the two tests reds, instead of the watchdog
// silently relaunching a session an operator believes is tombstoned.
const dedupTombstoneGoldenLine = `{"ts":"2026-07-07T12:00:00Z","phase":"skipped",` +
	`"session":"734355cc-dead-beef","account":".claude-w1","project":"C--work-fak",` +
	`"action":"dedup_tombstone","manual_override":true,"reason":"duplicate of live session ` +
	`abcd1234-live owning the same work (loop:--lane claude)","work_key":"loop:--lane claude",` +
	`"live_owner":"abcd1234-live","disp":"STOPPED_MIDTURN"}`

// The wire shape is the contract with the Python watchdog, so it is pinned byte-for-byte.
func TestDedupTombstoneWireShapeIsTheWatchdogHonorShape(t *testing.T) {
	path := filepath.Join(t.TempDir(), "resume_ledger.jsonl")
	if _, err := appendDedupTombstones(path, []dedupCandidate{{
		Session: "734355cc-dead-beef", Account: ".claude-w1", Project: "C--work-fak",
		WorkKey: "loop:--lane claude", LiveOwner: "abcd1234-live", Disp: "STOPPED_MIDTURN",
	}}, time.Date(2026, 7, 7, 12, 0, 0, 0, time.UTC)); err != nil {
		t.Fatalf("apply: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(raw)); got != dedupTombstoneGoldenLine {
		t.Fatalf("tombstone wire shape drifted from the shape fleet_resume_watchdog_test.py honors\n got: %s\nwant: %s",
			got, dedupTombstoneGoldenLine)
	}
}

// decisionsWithDup rebuilds the minimal Decisions holding the same duplicate the apply
// test tombstoned, so the idempotency leg exercises the REAL plan path end to end.
func decisionsWithDup(t *testing.T) stopped.Decisions {
	t.Helper()
	return stopped.Decide([]stopped.Row{
		{Disp: stopped.DispLive, Session: "abcd1234-live", Project: "C--work-fak", WorkKey: "loop:--lane claude", AgeMin: 1},
		{Disp: stopped.DispStoppedMidturn, Session: "734355cc-dead-beef", Project: "C--work-fak", WorkKey: "loop:--lane claude", AgeMin: 30},
	}, func(string) bool { return true })
}
