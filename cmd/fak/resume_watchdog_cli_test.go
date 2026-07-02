package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// The load-bearing watchdog-tick facts these pin (from tools/fleet_resume_watchdog.py):
//   - the terminal-turn signal classification feeds the shared outcome fold with the
//     same taxonomy (sessionsignals) the sweep uses — auth outranks limit outranks
//     transient;
//   - the plan/ledger readers tolerate missing and malformed files (a broken registry
//     degrades to a no-op tick, never a crash).
//
// The pre-gate screens (self-resume guard, worker policy), the probe-mode resolution,
// and the child-env strip are pinned where they live: internal/resume/watchdog_test.go.

func TestRwTerminalSignalTaxonomy(t *testing.T) {
	if s := rwTerminalSignal(""); s.Found {
		t.Fatal("empty text must report not-found")
	}
	s := rwTerminalSignal("Not logged in · Please run /login")
	if !s.Found || !s.AuthWall {
		t.Fatalf("auth wall not detected: %+v", s)
	}
	if o := resume.ClassifyOutcome(s); o != resume.OutcomeUnrecoverable {
		t.Fatalf("auth outcome = %s, want unrecoverable", o)
	}
	s = rwTerminalSignal("You've hit your session limit · resets 6am (America/Los_Angeles)")
	if !s.LimitWall {
		t.Fatalf("limit wall not detected: %+v", s)
	}
	if o := resume.ClassifyOutcome(s); o != resume.OutcomeRecoverable {
		t.Fatalf("limit outcome = %s, want recoverable", o)
	}
	s = rwTerminalSignal("API Error: Overloaded (529)")
	if !s.TransientAPIError || s.AuthWall {
		t.Fatalf("transient not detected cleanly: %+v", s)
	}
	if o := resume.ClassifyOutcome(rwTerminalSignal("all done, shipped and green")); o != resume.OutcomeProgressed {
		t.Fatalf("clean turn outcome = %s, want progressed", o)
	}
}

func TestRwLoadPlanAndHistoryTolerateBrokenFiles(t *testing.T) {
	dir := t.TempDir()
	if got := rwLoadPlan(filepath.Join(dir, "missing.json")); len(got) != 0 {
		t.Fatalf("missing plan = %v, want empty", got)
	}
	planPath := filepath.Join(dir, "resume_plan.json")
	os.WriteFile(planPath, []byte(`{"plan":[{"session":"s1","account":".claude-a","project":"P","rehomed":true}]}`), 0o644)
	plan := rwLoadPlan(planPath)
	if len(plan) != 1 || plan[0].Session != "s1" || !plan[0].Rehomed {
		t.Fatalf("plan = %+v", plan)
	}

	ledger := filepath.Join(dir, "resume_ledger.jsonl")
	os.WriteFile(ledger, []byte(
		`{"ts":"2026-07-01T10:00:00Z","session":"s1","phase":"launched"}
not json
{"ts":"2026-07-01T11:00:00Z","session":"s1","phase":"deferred"}
{"ts":"bad","session":"s2","action":"consolidate-resume-throttle-strand"}
`), 0o644)
	hist := rwLoadHistory(ledger)
	if len(hist["s1"]) != 2 {
		t.Fatalf("s1 history = %d rows, want 2", len(hist["s1"]))
	}
	// The launched row parses its timestamp; the deferred row is a non-launch.
	if got := resume.CountAttempts(hist["s1"]); got != 1 {
		t.Fatalf("s1 attempts = %d, want 1 (deferred rows are not launches)", got)
	}
	if hist["s1"][0].UnixSeconds == 0 {
		t.Fatal("launched row must carry its parsed unix timestamp")
	}
	// The operator-settled row (bad ts tolerated) blocks s2's gate forever.
	gate := resume.RetryGate(hist["s2"], resume.OutcomeRecoverable, 8)
	if !gate.Blocked {
		t.Fatal("consolidate row must block the retry gate")
	}
}

func TestRwAccountTag(t *testing.T) {
	if got := rwAccountTag(".claude-gem7"); got != "gem7" {
		t.Fatalf("tag = %q", got)
	}
	if got := rwAccountTag(".claude"); got != "default" {
		t.Fatalf("bare .claude tag = %q, want default", got)
	}
}

func TestResumeWatchdogBrokerDenyDoesNotSpawnWorker(t *testing.T) {
	regDir := t.TempDir()
	logDir := t.TempDir()
	home := t.TempDir()
	work := t.TempDir()
	configDir := filepath.Join(home, ".claude-secret")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("FLEET_CLAUDE_EXE", "claude")
	t.Setenv("ANTHROPIC_API_KEY", "sk-env-secret")
	configJSON, _ := json.Marshal(configDir)
	workJSON, _ := json.Marshal(work)
	plan := `{"plan":[{` +
		`"session":"sess-1234567890","account":".claude-secret","project":"proj",` +
		`"config_dir":` + string(configJSON) + `,` +
		`"cwd":` + string(workJSON) + `,` +
		`"disp":"STOPPED_MIDTOOL"` +
		`}]}`
	if err := os.WriteFile(filepath.Join(regDir, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}

	oldBroker := launchSpawnBroker
	oldSpawn := rwSpawnResumeLaunch
	var attempt launchBrokerAttempt
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		attempt = a
		return denyLaunchBrokerGrant(a, "unit-test-deny")
	}
	spawned := false
	rwSpawnResumeLaunch = func(claudeExe string, p resume.WatchdogPlanRow, resumeCfg, logDir string, grant launchBrokerGrant) (int, error) {
		spawned = true
		return 12345, nil
	}
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		rwSpawnResumeLaunch = oldSpawn
	})

	var out, errb bytes.Buffer
	rc := runResumeWatchdog(&out, &errb, []string{
		"--live", "--no-refresh",
		"--reg-dir", regDir,
		"--log-dir", logDir,
		"--spacing-sec", "0",
	})
	if rc != 0 {
		t.Fatalf("watchdog rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if spawned {
		t.Fatal("resume watchdog spawn seam was called after broker denial")
	}
	if attempt.Surface != "resume_watchdog" || attempt.Metadata.AgentRunID == "" ||
		!strings.HasPrefix(attempt.Metadata.PolicyDigest, "policy-sha256:") {
		t.Fatalf("broker attempt = %+v, want resume_watchdog AgentRun/PolicyDigest metadata", attempt)
	}
	got := out.String() + errb.String()
	for _, want := range []string{"DENY", "spawn broker: unit-test-deny", attempt.Metadata.AgentRunID, attempt.Metadata.PolicyDigest} {
		if !strings.Contains(got, want) {
			t.Fatalf("watchdog output missing %q:\n%s", want, got)
		}
	}
	for _, leak := range []string{"sk-env-secret", configDir} {
		if strings.Contains(got, leak) {
			t.Fatalf("watchdog output leaked %q:\n%s", leak, got)
		}
	}
	ledger, err := os.ReadFile(filepath.Join(regDir, "resume_ledger.jsonl"))
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if !strings.Contains(string(ledger), `"phase":"broker_denied"`) || strings.Contains(string(ledger), configDir) {
		t.Fatalf("broker-denied ledger = %s", ledger)
	}
}

func TestResumeWatchdogStatusJSONLaunchedNoProgressRed(t *testing.T) {
	reg := t.TempDir()
	if err := os.WriteFile(filepath.Join(reg, "resume_plan.json"), []byte(`{"plan":[
{"session":"sid-stuck","account":".claude-a","project":"P"},
{"session":"sid-2","account":".claude-a","project":"P"},
{"session":"sid-3","account":".claude-a","project":"P"},
{"session":"sid-4","account":".claude-a","project":"P"}
]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	ledger := strings.Join([]string{
		`{"ts":"2026-07-01T00:00:00Z","phase":"status","mode":"LIVE","auto_resume_depth":1}`,
		`{"ts":"2026-07-01T00:05:00Z","phase":"status","mode":"LIVE","auto_resume_depth":2}`,
		`{"ts":"2026-07-01T00:10:00Z","phase":"status","mode":"LIVE","auto_resume_depth":3}`,
		`{"ts":"2026-07-01T00:00:00Z","session":"sid-stuck","phase":"queued","mode":"LIVE"}`,
		`{"ts":"2026-07-01T00:01:00Z","session":"sid-stuck","phase":"launched","mode":"LIVE"}`,
		"",
	}, "\n")
	if err := os.WriteFile(filepath.Join(reg, "resume_ledger.jsonl"), []byte(ledger), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runResumeWatchdog(&out, &errb, []string{
		"--status", "--json", "--live", "--no-refresh", "--reg-dir", reg,
		"--silent-hours", "1", "--monotonic-ticks", "3",
	})
	if code != 3 {
		t.Fatalf("exit = %d, want red exit 3 (stderr: %s, stdout: %s)", code, errb.String(), out.String())
	}
	var rep resume.WatchdogDrainStatus
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if rep.Schema != resume.WatchdogStatusSchema || rep.Verdict != resume.WatchdogDrainRed || rep.Mode != "LIVE" {
		t.Fatalf("report header = %+v, want schema/red/LIVE", rep)
	}
	row := findWatchdogStatusRow(rep.MTTRSessions, "sid-stuck")
	if row.Status != resume.WatchdogMTTRLaunchedUnproven {
		t.Fatalf("sid-stuck row = %+v, want launched_unproven (all rows: %+v)", row, rep.MTTRSessions)
	}
	if row.ProgressWitnessedAt != 0 {
		t.Fatalf("launch alone must not be progress: %+v", row)
	}
}

func TestRwLoadWatchdogStatusEventsNormalizesLegacyRows(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "resume_ledger.jsonl")
	body := strings.Join([]string{
		`{"ts":"2026-07-01T00:00:00Z","session":"sid-legacy"}`,
		`{"ts":"2026-07-01T00:01:00Z","session":"sid-settled","action":"consolidate-operator-excluded","manual_override":true}`,
		"",
	}, "\n")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	rep := resume.FoldWatchdogStatus(resume.WatchdogStatusInput{
		Mode:    "LIVE",
		NowUnix: 2_000,
		Events:  rwLoadWatchdogStatusEvents(path),
	})
	if len(rep.MTTRSessions) != 1 {
		t.Fatalf("mttr rows = %+v, want only the phase-less legacy launch", rep.MTTRSessions)
	}
	row := rep.MTTRSessions[0]
	if row.Session != "sid-legacy" || row.Status != resume.WatchdogMTTRLaunchedUnproven {
		t.Fatalf("row = %+v, want sid-legacy launched_unproven", row)
	}
}

func TestResumeWatchdogTickRecordsDrainSamplesWithoutBurningAttempts(t *testing.T) {
	reg := t.TempDir()
	logDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(reg, "resume_plan.json"), []byte(`{"plan":[{"session":"sid-queued","account":".claude-a","project":"P"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runResumeWatchdog(&out, &errb, []string{
		"--no-refresh", "--reg-dir", reg, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	if !strings.Contains(out.String(), "TICK DRY-RUN") {
		t.Fatalf("dry-run tick did not run:\n%s", out.String())
	}
	statusLedger := rwWatchdogStatusLedger(reg)
	raw, err := os.ReadFile(statusLedger)
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if !strings.Contains(body, `"phase":"status"`) || !strings.Contains(body, `"phase":"queued"`) {
		t.Fatalf("ledger did not record status+queued breadcrumbs:\n%s", body)
	}
	hist := rwLoadHistory(filepath.Join(reg, "resume_ledger.jsonl"))
	if got := resume.CountAttempts(hist["sid-queued"]); got != 0 {
		t.Fatalf("status/queued rows burned %d attempts, want 0", got)
	}
}

func TestRwLoadPlanAcceptsUTF8BOM(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "resume_plan.json")
	body := append([]byte{0xEF, 0xBB, 0xBF}, []byte(`{"plan":[{"session":"sid-bom","account":".claude-a"}]}`)...)
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}

	rows := rwLoadPlan(path)
	if len(rows) != 1 || rows[0].Session != "sid-bom" {
		t.Fatalf("rows = %+v, want BOM-tolerant decode", rows)
	}
}

func TestResumeWatchdogTickRecordsTranscriptProgressWitness(t *testing.T) {
	reg := t.TempDir()
	logDir := t.TempDir()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	sid := "sid-progress"
	project := "C--work-fak"
	cfg := filepath.Join(home, ".claude-a")
	projDir := filepath.Join(cfg, "projects", project)
	if err := os.MkdirAll(projDir, 0o755); err != nil {
		t.Fatal(err)
	}
	transcript := strings.Join([]string{
		`{"type":"assistant","timestamp":"2026-07-01T00:00:10Z","message":{"role":"assistant","model":"claude-test","content":"done","usage":{"input_tokens":10}}}`,
		"",
	}, "\n")
	if err := os.WriteFile(filepath.Join(projDir, sid+".jsonl"), []byte(transcript), 0o644); err != nil {
		t.Fatal(err)
	}
	plan := `{"plan":[{"session":"` + sid + `","account":".claude-a","project":"` + project + `","config_dir":` +
		string(mustJSON(cfg)) + `}]}`
	if err := os.WriteFile(filepath.Join(reg, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}
	ledger := `{"ts":"2026-07-01T00:00:00Z","session":"` + sid + `","phase":"launched"}` + "\n"
	if err := os.WriteFile(filepath.Join(reg, "resume_ledger.jsonl"), []byte(ledger), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runResumeWatchdog(&out, &errb, []string{
		"--no-refresh", "--reg-dir", reg, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s stdout: %s)", code, errb.String(), out.String())
	}
	raw, err := os.ReadFile(rwWatchdogStatusLedger(reg))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if !strings.Contains(body, `"phase":"progress"`) || !strings.Contains(body, `"new_turns":1`) ||
		!strings.Contains(body, `"progress_witness_source":"transcript_real_turn_after_resume"`) {
		t.Fatalf("status ledger missing transcript progress witness:\n%s", body)
	}
}

func findWatchdogStatusRow(rows []resume.WatchdogMTTRRow, session string) resume.WatchdogMTTRRow {
	for _, row := range rows {
		if row.Session == session {
			return row
		}
	}
	return resume.WatchdogMTTRRow{}
}
