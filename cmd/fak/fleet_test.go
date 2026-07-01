package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/fleetmon"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// withFleetSeams swaps the collection/kill/clock seams for a test and restores
// them, so no live fleet or process is ever touched.
func withFleetSeams(t *testing.T, procs []procguard.Proc, now time.Time, killer func(int) (bool, string)) {
	t.Helper()
	withFleetSeamsErr(t, procs, "", now, killer)
}

func withFleetSeamsErr(t *testing.T, procs []procguard.Proc, collectErr string, now time.Time, killer func(int) (bool, string)) {
	t.Helper()
	origCollect, origKill, origNow := fleetCollectRelations, fleetKillPID, fleetNow
	fleetCollectRelations = func() ([]procguard.Proc, string) { return procs, collectErr }
	if killer != nil {
		fleetKillPID = killer
	}
	fleetNow = func() time.Time { return now }
	t.Cleanup(func() {
		fleetCollectRelations, fleetKillPID, fleetNow = origCollect, origKill, origNow
	})
	// Keep the registry read hermetic — an empty reg dir yields an empty registry.
	t.Setenv("FLEET_REG_DIR", t.TempDir())
}

func writePlan(t *testing.T, plan fleetmon.RunPlan) string {
	t.Helper()
	b, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "plan.json")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func writeJSONL(t *testing.T, lines ...string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "tx.jsonl")
	if err := os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestFleetMonitorJSON(t *testing.T) {
	now := time.Now()
	tx := writeJSONL(t,
		`{"type":"assistant","timestamp":"`+now.Add(-2*time.Minute).UTC().Format(time.RFC3339)+`","message":{"role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"done, go test passes"}]}}`,
	)
	plan := fleetmon.RunPlan{RunID: "r1", Workers: []fleetmon.PlanWorker{
		{Issue: 1856, Session: "issue-1856", PID: 4242, TranscriptPath: tx},
	}}
	// The worker PID is alive in the injected snapshot.
	withFleetSeams(t, []procguard.Proc{proc4242()}, now, nil)

	var out, errb bytes.Buffer
	code := runFleetMonitor(&out, &errb, []string{"--plan", writePlan(t, plan), "--json"})
	if code != 0 {
		t.Fatalf("monitor exit %d, stderr=%s", code, errb.String())
	}
	var payload fleetmon.MonitorPayload
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if payload.Total != 1 || len(payload.Workers) != 1 {
		t.Fatalf("want 1 worker, got %+v", payload)
	}
	if payload.Workers[0].Class != fleetmon.ClassCompletedFinal {
		t.Fatalf("alive PID + final report + idle => completed-final, got %s", payload.Workers[0].Class)
	}
}

func proc4242() procguard.Proc {
	pid := 4242
	return procguard.Proc{PID: pid, Name: "claude", Cmdline: "claude -p"}
}

func TestFleetMonitorUsesChildStalenessFlagsForJanitorScan(t *testing.T) {
	now := time.Now()
	tx := writeJSONL(t,
		`{"type":"user","timestamp":"`+now.Add(-40*time.Minute).UTC().Format(time.RFC3339)+`","message":{"role":"user","content":[{"type":"tool_result","content":"still working"}]}}`,
	)
	rootAge := 3600
	childAge := 400
	ppid := 100
	procs := []procguard.Proc{
		{PID: 100, Name: "claude", Cmdline: "claude -p", Start: now.Add(-60 * time.Minute).UTC().Format(time.RFC3339), AgeSec: &rootAge},
		{PID: 200, PPID: &ppid, Name: "ls", Cmdline: "ls -la", Start: now.Add(-7 * time.Minute).UTC().Format(time.RFC3339), AgeSec: &childAge},
	}
	plan := fleetmon.RunPlan{RunID: "r1", Workers: []fleetmon.PlanWorker{
		{Issue: 2134, Session: "issue-2134", PID: 100, TranscriptPath: tx},
	}}
	withFleetSeams(t, procs, now, nil)

	var out, errb bytes.Buffer
	code := runFleetMonitor(&out, &errb, []string{"--plan", writePlan(t, plan), "--json", "--stale-child-simple", "10m"})
	if code != 0 {
		t.Fatalf("monitor exit %d, stderr=%s", code, errb.String())
	}
	var payload fleetmon.MonitorPayload
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got := payload.Workers[0].Class; got != fleetmon.ClassStaleTranscript {
		t.Fatalf("10m simple-child threshold should keep 400s ls out of stale-child; got %s (%+v)", got, payload.Workers[0])
	}
}

func TestRegistryMatchPrefersSessionRowOverFirstAccountRow(t *testing.T) {
	reg := fleetaccounts.Registry{Sessions: []fleetaccounts.Session{
		{Account: ".claude-a", Project: "unrelated", Last: "please run /login", Disp: "INFRA_AUTH", Action: "BLOCKED_AUTH"},
		{Account: ".claude-a", Project: "C:/work/fak issue-2134", Last: "issue-2134 worker is live", Disp: "LIVE", Action: "OK"},
	}}
	disp, action := registryMatch(reg, fleetmon.PlanWorker{Issue: 2134, Session: "issue-2134", Account: ".claude-a"})
	if disp != "LIVE" || action != "OK" {
		t.Fatalf("registry match should prefer the row that names the worker session, got %s/%s", disp, action)
	}
}

func TestFleetCapacityPreflightJSON(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	cfg := filepath.Join(root, "cfg")
	regDir := filepath.Join(root, "reg")
	policyPath := filepath.Join(root, "accounts_policy.json")
	mustWriteFleetFixture(t, policyPath, `{}`)
	mkClaudeFleetFixture(t, home, ".claude", "uuid-default", "default@example.test", true)
	mkClaudeFleetFixture(t, home, ".claude-gem8-acct", "uuid-gem8", "gem8@example.test", true)
	mkClaudeFleetFixture(t, home, ".claude-needslogin-acct", "uuid-needs", "needs@example.test", false)
	mustWriteFleetFixture(t, filepath.Join(regDir, "sessions.json"),
		`{"generated_utc":"2026-07-01T12:00:00Z",`+
			`"throttle":{".claude-gem8-acct":{"reset":"Jul 8, 9am","weekly":"Jul 8"}},`+
			`"auth":{},"sessions":[]}`)
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", cfg)
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv("FLEET_POLICY_PATH", policyPath)

	var out, errb bytes.Buffer
	code := runFleetCapacity(&out, &errb, []string{"--json", "--require", "2"})
	if code != 1 {
		t.Fatalf("capacity exit=%d stderr=%q stdout=%q, want under-capacity exit 1", code, errb.String(), out.String())
	}
	var rep fleetaccounts.CapacityPreflight
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if rep.TrueConcurrentCeiling != 1 || rep.FreshSeats != 1 || rep.StaleSeats != 1 || rep.BlockedSeats != 1 || rep.Verdict != "UNDER_CAPACITY" {
		t.Fatalf("capacity report = %+v, want ceiling=1 fresh/stale/blocked=1/1/1 under-capacity", rep)
	}
	states := map[string]fleetaccounts.CapacityAccount{}
	for _, acct := range rep.Accounts {
		states[acct.Account] = acct
	}
	if states[".claude"].State != fleetaccounts.CapacityFresh {
		t.Fatalf(".claude = %+v, want fresh", states[".claude"])
	}
	if states[".claude-needslogin-acct"].State != fleetaccounts.CapacityStale {
		t.Fatalf("needslogin = %+v, want stale", states[".claude-needslogin-acct"])
	}
	if got := states[".claude-gem8-acct"]; got.State != fleetaccounts.CapacityBlockedUntil || got.BlockedUntil != "Jul 8" {
		t.Fatalf("gem8 = %+v, want blocked-until Jul 8", got)
	}
}

func TestFleetJanitorDryRunThenApply(t *testing.T) {
	now := time.Now()
	rootStart := now.Add(-60 * time.Minute).UTC().Format(time.RFC3339)
	ppid := 100
	age := 400
	procs := []procguard.Proc{
		{PID: 100, Name: "claude", Cmdline: "claude -p", Start: rootStart, AgeSec: &age},
		{PID: 200, PPID: &ppid, Name: "ls", Cmdline: "ls -la", Start: now.Add(-7 * time.Minute).UTC().Format(time.RFC3339), AgeSec: &age},
	}
	plan := fleetmon.RunPlan{Workers: []fleetmon.PlanWorker{{Issue: 1, Session: "issue-1", PID: 100}}}
	planPath := writePlan(t, plan)

	// Dry-run: nothing killed.
	killed := map[int]bool{}
	withFleetSeams(t, procs, now, func(pid int) (bool, string) { killed[pid] = true; return true, "ok" })
	var out, errb bytes.Buffer
	if code := runFleetJanitor(&out, &errb, []string{"--plan", planPath, "--json"}); code != 0 {
		t.Fatalf("janitor dry-run exit %d: %s", code, errb.String())
	}
	if len(killed) != 0 {
		t.Fatalf("dry-run must not kill anything, killed=%v", killed)
	}
	if !strings.Contains(out.String(), `"stale"`) {
		t.Fatalf("json should list stale children: %s", out.String())
	}

	// Apply: the stale ls tree is terminated.
	out.Reset()
	errb.Reset()
	if code := runFleetJanitor(&out, &errb, []string{"--plan", planPath, "--json", "--apply"}); code != 0 {
		t.Fatalf("janitor apply exit %d: %s", code, errb.String())
	}
	if !killed[200] {
		t.Fatalf("apply must terminate the stale child pid 200, killed=%v", killed)
	}
	if killed[100] {
		t.Fatal("apply must NEVER kill the worker root pid 100")
	}
}

func TestFleetJanitorRecordsDecisions(t *testing.T) {
	now := time.Now()
	rootStart := now.Add(-60 * time.Minute).UTC().Format(time.RFC3339)
	ppid := 100
	age := 400
	procs := []procguard.Proc{
		{PID: 100, Name: "claude", Cmdline: "claude -p", Start: rootStart, AgeSec: &age},
		{PID: 200, PPID: &ppid, Name: "ls", Cmdline: "ls -la", Start: now.Add(-7 * time.Minute).UTC().Format(time.RFC3339), AgeSec: &age},
	}
	plan := fleetmon.RunPlan{Workers: []fleetmon.PlanWorker{{Issue: 1, Session: "issue-1", PID: 100}}}
	planPath := writePlan(t, plan)
	ledger := filepath.Join(t.TempDir(), "janitor.jsonl")

	withFleetSeams(t, procs, now, func(pid int) (bool, string) { return true, "ok" })
	var out, errb bytes.Buffer
	if code := runFleetJanitor(&out, &errb, []string{"--plan", planPath, "--apply", "--ledger", ledger}); code != 0 {
		t.Fatalf("janitor exit %d: %s", code, errb.String())
	}
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("janitor ledger not written: %v", err)
	}
	if !strings.Contains(string(data), `"action":"terminated"`) || !strings.Contains(string(data), `"root_pid":200`) {
		t.Fatalf("termination decision not recorded: %s", string(data))
	}
}

func TestFleetFoldWritesLedger(t *testing.T) {
	now := time.Now()
	tx := writeJSONL(t,
		`{"type":"assistant","timestamp":"`+now.Add(-5*time.Minute).UTC().Format(time.RFC3339)+`","message":{"role":"assistant","content":[{"type":"tool_use","name":"Read","input":{"file_path":"a.go"}}]}}`,
		`{"type":"assistant","timestamp":"`+now.Add(-1*time.Minute).UTC().Format(time.RFC3339)+`","message":{"role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"Audited, no change needed."}]}}`,
	)
	plan := fleetmon.RunPlan{RunID: "r1", Workers: []fleetmon.PlanWorker{{Issue: 1, Session: "issue-1", PID: 100, TranscriptPath: tx}}}
	withFleetSeams(t, []procguard.Proc{{PID: 100, Name: "claude"}}, now, nil)

	ledger := filepath.Join(t.TempDir(), "ledger.jsonl")
	var out, errb bytes.Buffer
	code := runFleetFold(&out, &errb, []string{"--plan", writePlan(t, plan), "--json", "--ledger", ledger, "--write"})
	if code != 0 {
		t.Fatalf("fold exit %d: %s", code, errb.String())
	}
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("ledger not written: %v", err)
	}
	rows := fleetmon.ParseLedger(string(data))
	if len(rows) != 1 || rows[0].Outcome != string(fleetmon.OutcomeReadOnlyAudit) {
		t.Fatalf("want one read-only-audit row, got %+v", rows)
	}
}

func TestFleetFoldSurfacesProcessCollectionError(t *testing.T) {
	now := time.Now()
	tx := writeJSONL(t,
		`{"type":"user","timestamp":"`+now.Add(-30*time.Minute).UTC().Format(time.RFC3339)+`","message":{"role":"user","content":[{"type":"tool_result","content":"working"}]}}`,
	)
	plan := fleetmon.RunPlan{RunID: "r1", Workers: []fleetmon.PlanWorker{{Issue: 2134, Session: "issue-2134", PID: 100, TranscriptPath: tx}}}
	withFleetSeamsErr(t, nil, "process collector unavailable", now, nil)

	var out, errb bytes.Buffer
	code := runFleetFold(&out, &errb, []string{"--plan", writePlan(t, plan), "--json"})
	if code != 0 {
		t.Fatalf("fold exit %d: %s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "process scan warning: process collector unavailable") {
		t.Fatalf("fold should surface the collection error on stderr, got %q", errb.String())
	}
	var summary fleetmon.RunLedgerSummary
	if err := json.Unmarshal(out.Bytes(), &summary); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if len(summary.Rows) != 1 {
		t.Fatalf("want one folded row, got %+v", summary)
	}
	row := summary.Rows[0]
	if row.Outcome == string(fleetmon.OutcomeCrashedNoFinal) {
		t.Fatalf("collection failure must not be reported as a worker crash: %+v", row)
	}
	if row.Outcome != string(fleetmon.OutcomeStaleIncomplete) || !strings.Contains(row.FollowUp, "process scan failed: process collector unavailable") {
		t.Fatalf("collection failure should be carried in the row, got %+v", row)
	}
}

func TestFleetReplaceRefusesHealthy(t *testing.T) {
	now := time.Now()
	withFleetSeams(t, nil, now, nil)
	plan := fleetmon.RunPlan{Workers: []fleetmon.PlanWorker{{Issue: 1, Session: "issue-1"}}}
	var out, errb bytes.Buffer
	code := runFleetReplace(&out, &errb, []string{"--plan", writePlan(t, plan), "--session", "issue-1", "--class", "healthy"})
	if code == 0 {
		t.Fatalf("replacing a healthy worker must be refused (nonzero), got 0: %s", out.String())
	}
	if !strings.Contains(out.String(), "REFUSED") {
		t.Fatalf("output should say REFUSED: %s", out.String())
	}
}

func TestFleetReplaceEligibleDeadJSON(t *testing.T) {
	now := time.Now()
	withFleetSeams(t, nil, now, nil)
	plan := fleetmon.RunPlan{RunID: "r1", Workers: []fleetmon.PlanWorker{{Issue: 1856, Session: "issue-1856", IssueURL: "https://example/1856", Area: "fleet"}}}
	var out, errb bytes.Buffer
	code := runFleetReplace(&out, &errb, []string{"--plan", writePlan(t, plan), "--session", "issue-1856", "--class", "dead", "--json"})
	if code != 0 {
		t.Fatalf("replace of a dead worker should succeed, got %d: %s", code, errb.String())
	}
	var d fleetmon.ReplaceDecision
	if err := json.Unmarshal(out.Bytes(), &d); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if !d.Eligible || d.NewSession != "issue-1856-replacement-1" {
		t.Fatalf("expected eligible replacement issue-1856-replacement-1, got %+v", d)
	}
	if d.LedgerRow == nil || d.LedgerRow.Outcome != string(fleetmon.OutcomeSuperseded) {
		t.Fatalf("expected a superseded ledger row, got %+v", d.LedgerRow)
	}
}

func TestFleetReplaceWritesSupersedingLedgerRow(t *testing.T) {
	now := time.Now()
	withFleetSeams(t, nil, now, nil)
	plan := fleetmon.RunPlan{RunID: "r1", Workers: []fleetmon.PlanWorker{{Issue: 1856, Session: "issue-1856"}}}
	ledger := filepath.Join(t.TempDir(), "run.jsonl")
	var out, errb bytes.Buffer
	code := runFleetReplace(&out, &errb, []string{"--plan", writePlan(t, plan), "--session", "issue-1856", "--class", "dead", "--ledger", ledger, "--write", "--json"})
	if code != 0 {
		t.Fatalf("replace exit %d: %s", code, errb.String())
	}
	rows := fleetmon.ParseLedger(readFile(t, ledger))
	if len(rows) != 1 || rows[0].Outcome != string(fleetmon.OutcomeSuperseded) || rows[0].SupersededBy != "issue-1856-replacement-1" {
		t.Fatalf("superseding row not written correctly: %+v", rows)
	}
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func mkClaudeFleetFixture(t *testing.T, home, account, uuid, email string, creds bool) {
	t.Helper()
	dir := filepath.Join(home, account)
	mustMkdirFleetFixture(t, filepath.Join(dir, "projects"))
	mustWriteFleetFixture(t, filepath.Join(dir, ".claude.json"),
		`{"oauthAccount":{"accountUuid":"`+uuid+`","emailAddress":"`+email+`","organizationUuid":"org-`+uuid+`","organizationType":"team"}}`)
	if creds {
		mustWriteFleetFixture(t, filepath.Join(dir, ".credentials.json"), `{}`)
	}
}

func mustMkdirFleetFixture(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}

func mustWriteFleetFixture(t *testing.T, path, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func TestFleetReplaceRefusedDoesNotWriteLedger(t *testing.T) {
	now := time.Now()
	withFleetSeams(t, nil, now, nil)
	plan := fleetmon.RunPlan{Workers: []fleetmon.PlanWorker{{Issue: 1, Session: "issue-1"}}}
	ledger := filepath.Join(t.TempDir(), "run.jsonl")
	var out, errb bytes.Buffer
	runFleetReplace(&out, &errb, []string{"--plan", writePlan(t, plan), "--session", "issue-1", "--class", "healthy", "--ledger", ledger, "--write"})
	if _, err := os.Stat(ledger); err == nil {
		t.Fatal("a refused replacement must not write a ledger row")
	}
}

func TestFleetUnknownSubcommand(t *testing.T) {
	// runFleet* handlers are dispatched by cmdFleet via dispatchSubcommands, which
	// calls os.Exit — so we exercise the handlers directly above. Here we only check
	// that a replace with no --session fails cleanly.
	var out, errb bytes.Buffer
	if code := runFleetReplace(&out, &errb, []string{}); code != 2 {
		t.Fatalf("replace with no --session should exit 2, got %d", code)
	}
}
