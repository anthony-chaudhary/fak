package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/resumebackoff"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
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

func TestResumeWatchdogStormSignatureKeepsUnrelatedSessionsIndependent(t *testing.T) {
	now := time.Unix(1_000, 0)
	base := resume.WatchdogPlanRow{CWD: `C:\work\fak`, Disp: "STOPPED_APIERR"}
	history := make([]resumebackoff.Event, 0, 5)
	for i := 0; i < 5; i++ {
		row := base
		row.Session = fmt.Sprintf("codex-session-%d", i)
		history = append(history, resumebackoff.Event{
			Session: row.Session, Signature: rwResumeStormSignature(row), At: now.Add(-time.Duration(i) * time.Second),
		})
	}
	candidate := base
	candidate.Session = "codex-session-5"
	decision := resumebackoff.Decide(resumebackoff.Input{
		Session: candidate.Session, Signature: rwResumeStormSignature(candidate), Now: now, History: history,
	})
	if !decision.Eligible || decision.Parked || decision.Reason != "" {
		t.Fatalf("unrelated sixth session inherited the cohort storm: %+v", decision)
	}
}

func TestResumeWatchdogStormSignatureBacksOffRepeatedLogicalSession(t *testing.T) {
	now := time.Unix(1_000, 0)
	row := resume.WatchdogPlanRow{Session: "codex-session-repeat", CWD: `C:\work\fak`, Disp: "STOPPED_APIERR"}
	signature := rwResumeStormSignature(row)
	history := []resumebackoff.Event{
		{Session: row.Session, Signature: signature, At: now.Add(-90 * time.Second)},
		{Session: row.Session, Signature: signature, At: now.Add(-30 * time.Second)},
	}
	decision := resumebackoff.Decide(resumebackoff.Input{
		Session: row.Session, Signature: rwResumeStormSignature(row), Now: now, History: history,
	})
	if decision.Eligible || decision.Reason != resumebackoff.ReasonBackoff || decision.Repeat != 2 {
		t.Fatalf("repeated crashes for one logical session escaped containment: %+v", decision)
	}
}

func TestResumeWatchdogCrashLoopQuarantinedAtBudget(t *testing.T) {
	now := time.Unix(10_000, 0)
	row := resume.WatchdogPlanRow{Session: "codex-crash-loop", CWD: `C:\work\fak`, Disp: "STOPPED_CRASH"}
	signature := rwResumeStormSignature(row)
	history := []resumebackoff.Event{
		{Session: row.Session, Signature: signature, At: now.Add(-20 * time.Minute)},
		{Session: row.Session, Signature: signature, At: now.Add(-10 * time.Minute)},
		{Session: row.Session, Signature: signature, At: now.Add(-2 * time.Minute)},
	}
	// At budget 3 with 3 repeated launches, Decision must quarantine with CRASH_LOOP_QUARANTINED.
	decision := resumebackoff.Decide(resumebackoff.Input{
		Session:         row.Session,
		Signature:       signature,
		Now:             now,
		History:         history,
		CrashLoopBudget: 3,
	})
	if decision.Eligible || !decision.Parked || !decision.Quarantined || decision.Reason != resumebackoff.ReasonCrashLoopQuarantined || decision.Repeat != 3 {
		t.Fatalf("unchanged crash loop escaped quarantine: %+v", decision)
	}

	// Witness: when the failure signature changes (e.g. repaired disposition or changed prompt/state), reset and permit one bounded attempt.
	changedRow := resume.WatchdogPlanRow{Session: "codex-crash-loop", CWD: `C:\work\fak`, Disp: "STOPPED_INTERRUPTED"}
	changedSig := rwResumeStormSignature(changedRow)
	resetDecision := resumebackoff.Decide(resumebackoff.Input{
		Session:         changedRow.Session,
		Signature:       changedSig,
		Now:             now,
		History:         history,
		CrashLoopBudget: 3,
	})
	if !resetDecision.Eligible || resetDecision.Parked || resetDecision.Quarantined {
		t.Fatalf("changed signature was not permitted a bounded attempt: %+v", resetDecision)
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
	if got := resume.CountAttempts(hist["s2"]); got != 0 {
		t.Fatalf("settled row burned %d attempts, want 0", got)
	}
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
	t.Setenv("FAK_RESUME_MAX_LIVE", "1000000")
	policy := filepath.Join(t.TempDir(), "source-policy.json")
	if err := os.WriteFile(policy, []byte(`{"default":{"max_live_resumes":1000000,"max_launches_per_window":1000000,"window_seconds":1,"min_launch_spacing_seconds":-1}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_RESUME_SOURCE_POLICY", policy)
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
	// Isolate fleet account discovery from the operator's real ~/.config (XDG_CONFIG_HOME
	// leaks live opencode accounts into rwWorkerAccounts, which would make the
	// WorkerAccounts guard non-empty and skip the test's .claude-secret row as
	// non-worker before it reaches the broker — the same isolation dispatch_tick_test.go
	// applies). An empty dir discovers no accounts, leaving the guard inert (fail-open).
	t.Setenv("FLEET_USER_HOME", filepath.Join(home, "empty-home"))
	t.Setenv("FLEET_CONFIG_HOME", filepath.Join(home, "empty-config"))
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
	if !strings.Contains(string(ledger), `"phase":"broker_denied"`) || strings.Contains(string(ledger), `"payload":"`+configDir+`"`) {
		t.Fatalf("broker-denied ledger = %s", ledger)
	}
}

func TestResumeWatchdogAuthDispositionDoesNotSpawnWorker(t *testing.T) {
	regDir := t.TempDir()
	logDir := t.TempDir()
	plan := `{"plan":[{"session":"sid-auth-1234567890","account":".claude-a","project":"P","disp":"INFRA_AUTH"}]}`
	if err := os.WriteFile(filepath.Join(regDir, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}

	oldBroker := launchSpawnBroker
	oldSpawn := rwSpawnResumeLaunch
	brokered := false
	spawned := false
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		brokered = true
		return allowLaunchBrokerGrant(a, "unit-test-allow")
	}
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
	if brokered || spawned {
		t.Fatalf("auth plan row reached launch path: brokered=%v spawned=%v", brokered, spawned)
	}
	got := out.String() + errb.String()
	if !strings.Contains(got, "requires auth/login") {
		t.Fatalf("watchdog output missing auth skip reason:\n%s", got)
	}
	ledger, err := os.ReadFile(filepath.Join(regDir, "resume_ledger.jsonl"))
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	body := string(ledger)
	for _, want := range []string{`"phase":"settled"`, `"action":"consolidate-auth-plan-row"`, `"outcome":"unrecoverable"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("auth skip ledger missing %s:\n%s", want, body)
		}
	}
	if strings.Contains(body, `"phase":"launched"`) {
		t.Fatalf("auth skip ledger recorded a launch:\n%s", body)
	}
}

// rwHoldTestEnv isolates account discovery (so the worker-policy guard stays inert) and the
// self-guard, so a drive-state test exercises exactly the operator-hold path.
func rwHoldTestEnv(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("FLEET_CLAUDE_EXE", "claude")
	t.Setenv("FLEET_USER_HOME", filepath.Join(home, "empty-home"))
	t.Setenv("FLEET_CONFIG_HOME", filepath.Join(home, "empty-config"))
	t.Setenv("CLAUDE_CODE_SESSION_ID", "") // keep the self-guard inert regardless of the ambient session
	t.Setenv("FAK_RESUME_MAX_LIVE", "1000000")
	policy := filepath.Join(t.TempDir(), "source-policy.json")
	if err := os.WriteFile(policy, []byte(`{"default":{"max_live_resumes":1000000,"max_launches_per_window":1000000,"window_seconds":1,"min_launch_spacing_seconds":-1}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_RESUME_SOURCE_POLICY", policy)
}

func TestResumeWatchdogOperatorHoldDoesNotSpawn(t *testing.T) {
	rwHoldTestEnv(t)
	regDir := t.TempDir()
	logDir := t.TempDir()
	sid := "sid-hold-1234567890"
	plan := `{"plan":[{"session":"` + sid + `","account":".claude-a","project":"P","disp":"STOPPED_MIDTOOL"}]}`
	if err := os.WriteFile(filepath.Join(regDir, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}
	// The operator stopped this session (`fak resume hold --state stopped`).
	drive := `{"ts":"2026-07-05T00:00:00Z","session":"` + sid + `","state":"stopped","via":"fak resume hold"}` + "\n"
	if err := os.WriteFile(filepath.Join(regDir, "resume_drivestate.jsonl"), []byte(drive), 0o644); err != nil {
		t.Fatal(err)
	}

	oldBroker := launchSpawnBroker
	oldSpawn := rwSpawnResumeLaunch
	brokered := false
	spawned := false
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		brokered = true
		return allowLaunchBrokerGrant(a, "unit-test-allow")
	}
	rwSpawnResumeLaunch = func(claudeExe string, p resume.WatchdogPlanRow, resumeCfg, logDir string, grant launchBrokerGrant) (int, error) {
		spawned = true
		return 12345, nil
	}
	t.Cleanup(func() { launchSpawnBroker = oldBroker; rwSpawnResumeLaunch = oldSpawn })

	var out, errb bytes.Buffer
	rc := runResumeWatchdog(&out, &errb, []string{
		"--live", "--no-refresh", "--reg-dir", regDir, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if rc != 0 {
		t.Fatalf("watchdog rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if brokered || spawned {
		t.Fatalf("operator-held session reached the launch path: brokered=%v spawned=%v", brokered, spawned)
	}
	got := out.String() + errb.String()
	if !strings.Contains(got, "SKIP") || !strings.Contains(strings.ToLower(got), "operator") {
		t.Fatalf("watchdog output missing the operator-hold skip:\n%s", got)
	}
	if ledger, _ := os.ReadFile(filepath.Join(regDir, "resume_ledger.jsonl")); strings.Contains(string(ledger), `"phase":"launched"`) {
		t.Fatalf("held session recorded a launch:\n%s", ledger)
	}
}

func TestResumeWatchdogReleaseReEnablesResume(t *testing.T) {
	rwHoldTestEnv(t)
	regDir := t.TempDir()
	logDir := t.TempDir()
	sid := "sid-rel-1234567890"
	plan := `{"plan":[{"session":"` + sid + `","account":".claude-a","project":"P","disp":"STOPPED_MIDTOOL"}]}`
	if err := os.WriteFile(filepath.Join(regDir, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}
	// Paused, then released — the later `running` row lifts the hold (fold-agnostic: a running
	// row releases a paused hold under either a sticky or a last-writer-wins fold).
	drive := `{"ts":"2026-07-05T00:00:00Z","session":"` + sid + `","state":"paused","via":"fak resume hold"}` + "\n" +
		`{"ts":"2026-07-05T00:01:00Z","session":"` + sid + `","state":"running","via":"fak resume release"}` + "\n"
	if err := os.WriteFile(filepath.Join(regDir, "resume_drivestate.jsonl"), []byte(drive), 0o644); err != nil {
		t.Fatal(err)
	}

	oldBroker := launchSpawnBroker
	oldSpawn := rwSpawnResumeLaunch
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant { return allowLaunchBrokerGrant(a, "unit-test-allow") }
	rwSpawnResumeLaunch = func(claudeExe string, p resume.WatchdogPlanRow, resumeCfg, logDir string, grant launchBrokerGrant) (int, error) {
		return 12345, nil
	}
	t.Cleanup(func() { launchSpawnBroker = oldBroker; rwSpawnResumeLaunch = oldSpawn })

	// Dry-run: a released session is resume-eligible again (WOULD RESUME), and is NOT skipped
	// as an operator hold.
	var out, errb bytes.Buffer
	rc := runResumeWatchdog(&out, &errb, []string{
		"--no-refresh", "--reg-dir", regDir, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if rc != 0 {
		t.Fatalf("watchdog rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	got := out.String() + errb.String()
	if strings.Contains(strings.ToLower(got), "operator paused") || strings.Contains(strings.ToLower(got), "operator stopped") {
		t.Fatalf("released session was still treated as an operator hold:\n%s", got)
	}
	if !strings.Contains(got, "WOULD RESUME") {
		t.Fatalf("released session should be resume-eligible (WOULD RESUME):\n%s", got)
	}
}

// #4216: a single LIVE OS relaunch writes a well-formed RelaunchResetRow to the durable
// transcript-UUID-keyed store next to its "launched" ledger row, and the read/fold path
// (rwLoadRelaunchResets -> resume.FoldRelaunchResets) recovers the latest reset for that
// session — the OS-relaunch analogue of session.ResetTransactionLog.Append (#1582). Also
// pins that the shell stamped TS at the write site (the pure #4139 constructor leaves it "").
func TestRelaunchResetWiredAtLaunchSite(t *testing.T) {
	rwHoldTestEnv(t)
	regDir := t.TempDir()
	logDir := t.TempDir()
	sid := "sid-relaunch-reset-123"
	// A plain crashed row re-homed onto a second account: PriorAccount -> RelaunchAccount is
	// the observable reset across the process boundary. rehomed is left false so the launch
	// path does not require a transcript copy — this test pins the reset append, not re-home.
	plan := `{"plan":[{"session":"` + sid + `","account":".claude-a","resume_account":".claude-b",` +
		`"project":"P","disp":"STOPPED_MIDTOOL"}]}`
	if err := os.WriteFile(filepath.Join(regDir, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}

	oldBroker := launchSpawnBroker
	oldSpawn := rwSpawnResumeLaunch
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant { return allowLaunchBrokerGrant(a, "unit-test-allow") }
	rwSpawnResumeLaunch = func(claudeExe string, p resume.WatchdogPlanRow, resumeCfg, logDir string, grant launchBrokerGrant) (int, error) {
		return 12345, nil
	}
	t.Cleanup(func() { launchSpawnBroker = oldBroker; rwSpawnResumeLaunch = oldSpawn })

	var out, errb bytes.Buffer
	rc := runResumeWatchdog(&out, &errb, []string{
		"--live", "--no-refresh", "--reg-dir", regDir, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if rc != 0 {
		t.Fatalf("watchdog rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	// Sanity: the row actually launched (else there is no reset to record).
	if led, _ := os.ReadFile(filepath.Join(regDir, "resume_ledger.jsonl")); !strings.Contains(string(led), `"phase":"launched"`) {
		t.Fatalf("expected a launched ledger row:\n%s", led)
	}

	// The store exists next to the drivestate store, and the fold recovers the reset for sid.
	resets := rwLoadRelaunchResets(regDir)
	got, ok := resets[sid]
	if !ok {
		t.Fatalf("fold recovered no relaunch reset for %s: %+v", sid, resets)
	}
	if !got.WellFormed() {
		t.Fatalf("recovered reset is not well-formed: %+v", got)
	}
	if got.TS == "" {
		t.Fatalf("reset TS was not stamped at the write site (pure constructor leaves it \"\"): %+v", got)
	}
	if got.Cause != "STOPPED_MIDTOOL" {
		t.Fatalf("reset Cause = %q, want STOPPED_MIDTOOL: %+v", got.Cause, got)
	}
	from, to, changed := got.Rehome()
	if from != ".claude-a" || to != ".claude-b" || !changed {
		t.Fatalf("reset re-home marker = (%q -> %q, changed=%v), want (.claude-a -> .claude-b, changed=true)", from, to, changed)
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
		"--silent-hours", "1000000", "--unproven-minutes", "1", "--monotonic-ticks", "0",
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
	if rep.UnprovenSeconds == 0 || !strings.Contains(strings.Join(rep.Reasons, "\n"), "unproven") {
		t.Fatalf("report did not expose launched-unproven alarm: %+v", rep)
	}
	row := findWatchdogStatusRow(rep.MTTRSessions, "sid-stuck")
	if row.Status != resume.WatchdogMTTRLaunchedUnproven {
		t.Fatalf("sid-stuck row = %+v, want launched_unproven (all rows: %+v)", row, rep.MTTRSessions)
	}
	if row.ProgressWitnessedAt != 0 {
		t.Fatalf("launch alone must not be progress: %+v", row)
	}
	if row.UnprovenSeconds == 0 {
		t.Fatalf("launched-unproven row must carry its unproven age: %+v", row)
	}
}

// The loader still reads legacy rows (the pre-phase schema): a phase-less row loads with a
// blank phase, and a manual_override/consolidate row is normalized to "settled". But post-#3801
// (internal/resume/watchdog_status.go normalizeWatchdogPhase) a blank phase folds to
// phase_unknown, NOT launched, so a phase-less legacy row is EXCLUDED from the MTTR view rather
// than minting a phantom launched_unproven row; a settled row is likewise never a launch. So
// neither legacy row produces an MTTR launch row.
//
// NOTE (#4333): #3801's remaining DoD landed — the fold and Attempt.IsLaunch now share ONE
// launch classifier (internal/resume phaseIsLaunchToken), asserted by the shared table test
// TestPhaseClassifierSharedVocabulary, so the two readers return identical verdicts for every
// NON-EMPTY phase token. The phase-less row is the one deliberate, tested divergence: those
// accounting readers count it as a real launch ON PURPOSE (TestCountAttemptsAndLastLaunch;
// TestLegacyPhaselessRowIsALaunch — "114 such rows in the production ledger ... or historical
// launch accounting silently zeroes out"), while this fold folds it to phase_unknown so it
// never mints a phantom launched_unproven MTTR row. The two readers answer different questions
// (did a launch PROVE itself, vs. did a spawn FIRE); forcing an identical empty-phase verdict
// would regress the give-up cap / LAUNCH_SPACING_FLOOR, so the split is intentional.
func TestRwLoadWatchdogStatusEventsExcludesPhaseLessLegacyRowsFromMTTR(t *testing.T) {
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

	// The loader normalizes legacy rows: phase-less passes through blank, manual_override -> settled.
	events := rwLoadWatchdogStatusEvents(path)
	if len(events) != 2 {
		t.Fatalf("loaded events = %+v, want the 2 legacy rows", events)
	}
	if events[0].Phase != "" {
		t.Fatalf("phase-less legacy row must load with a blank phase, got %+v", events[0])
	}
	if events[1].Phase != "settled" {
		t.Fatalf("manual_override row must normalize to phase \"settled\", got %+v", events[1])
	}

	rep := resume.FoldWatchdogStatus(resume.WatchdogStatusInput{
		Mode:    "LIVE",
		NowUnix: 2_000,
		Events:  events,
	})
	if len(rep.MTTRSessions) != 0 {
		t.Fatalf("mttr rows = %+v, want none: phase-less folds to phase_unknown (non-launch) post-#3801", rep.MTTRSessions)
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

// The #2368 acceptance fixture: a session with a prior successful resume (launched
// ledger row + a real turn after it), whose transcript later carries a timeout marker
// and whose LAST record is a clean tool-result (the mid-tool death shape that
// classifies as progressed), with NO live process and a long-idle transcript — the
// watchdog must NOT skip it as "already resumed once (resume took)".
func TestResumeWatchdogRevivesStaleTookLatchWhenSessionDiesAgain(t *testing.T) {
	reg := t.TempDir()
	logDir := t.TempDir()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("FLEET_CLAUDE_EXE", "claude")
	sid := "sid-relatch-46cf6bf5"
	project := "C--work-fak"
	cfg := filepath.Join(home, ".claude-a")
	projDir := filepath.Join(cfg, "projects", project)
	if err := os.MkdirAll(projDir, 0o755); err != nil {
		t.Fatal(err)
	}
	transcript := strings.Join([]string{
		// The resume took: a real model turn after the 00:00 launch…
		`{"type":"assistant","timestamp":"2026-07-01T00:10:00Z","message":{"role":"assistant","model":"claude-test","content":"working","usage":{"input_tokens":10}}}`,
		// …then the July2 evidence shape: a timeout marker NOT in terminal position…
		`{"type":"assistant","timestamp":"2026-07-01T00:20:00Z","isApiErrorMessage":true,"message":{"role":"assistant","model":"<synthetic>","content":"API Error: Request timed out"}}`,
		// …and a clean tool-result as the terminal record (mid-tool death).
		`{"type":"user","timestamp":"2026-07-01T00:21:00Z","message":{"role":"user","content":"tool result: build ok"}}`,
		"",
	}, "\n")
	path := filepath.Join(projDir, sid+".jsonl")
	if err := os.WriteFile(path, []byte(transcript), 0o644); err != nil {
		t.Fatal(err)
	}
	// The transcript has been silent far past the dead floor.
	stale := time.Now().Add(-45 * time.Minute)
	if err := os.Chtimes(path, stale, stale); err != nil {
		t.Fatal(err)
	}
	plan := `{"plan":[{"session":"` + sid + `","account":".claude-a","project":"` + project +
		`","config_dir":` + string(mustJSON(cfg)) + `,"disp":"STOPPED_MIDTOOL"}]}`
	if err := os.WriteFile(filepath.Join(reg, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}
	ledger := `{"ts":"2026-07-01T00:00:00Z","session":"` + sid + `","phase":"launched"}` + "\n"
	if err := os.WriteFile(filepath.Join(reg, "resume_ledger.jsonl"), []byte(ledger), 0o644); err != nil {
		t.Fatal(err)
	}

	oldBroker := launchSpawnBroker
	oldScan := rwCollectProcCmdlines
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		return allowLaunchBrokerGrant(a, "unit-test-allow")
	}
	// No live process holds the session id.
	rwCollectProcCmdlines = func() ([]string, bool) {
		return []string{`C:\bin\claude.exe --resume some-other-session`}, true
	}
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		rwCollectProcCmdlines = oldScan
	})

	var out, errb bytes.Buffer
	code := runResumeWatchdog(&out, &errb, []string{
		"--no-refresh", "--reg-dir", reg, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s stdout: %s)", code, errb.String(), out.String())
	}
	got := out.String()
	if strings.Contains(got, "already resumed once") {
		t.Fatalf("stale took-latch still skipped the re-dead session:\n%s", got)
	}
	if !strings.Contains(got, "REVIVE") || !strings.Contains(got, "WOULD RESUME") {
		t.Fatalf("re-dead session must be revived and (dry-run) resumed:\n%s", got)
	}

	// Same fixture with a LIVE process holding the session: the latch must hold —
	// reviving here would race two claude processes on one transcript.
	rwCollectProcCmdlines = func() ([]string, bool) {
		return []string{`C:\bin\claude.exe --resume ` + sid}, true
	}
	out.Reset()
	errb.Reset()
	code = runResumeWatchdog(&out, &errb, []string{
		"--no-refresh", "--reg-dir", reg, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	if !strings.Contains(out.String(), "already resumed once") {
		t.Fatalf("live process must keep the burn-once skip:\n%s", out.String())
	}
}

// #8722: a child that exits before writing even one post-launch transcript turn used to
// stay latched as "resume took" forever. With no transcript mtime, the old re-death proof
// remained unknown on every tick. The launch timestamp is the bounded startup grace for
// this specific unproven shape: before it expires the once-gate stays closed; after it
// expires, a readable process table proving no live driver makes the failed launch retryable.
func TestResumeWatchdogRetriesDeadUnprovenLaunchAfterGrace(t *testing.T) {
	reg := t.TempDir()
	logDir := t.TempDir()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("FLEET_CLAUDE_EXE", "claude")
	sid := "sid-unproven-8722"
	plan := `{"plan":[{"session":"` + sid + `","account":".claude-a","disp":"STOPPED_APIERR"}]}`
	if err := os.WriteFile(filepath.Join(reg, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}

	oldBroker := launchSpawnBroker
	oldScan := rwCollectProcCmdlines
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		return allowLaunchBrokerGrant(a, "unit-test-allow")
	}
	rwCollectProcCmdlines = func() ([]string, bool) { return nil, true }
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		rwCollectProcCmdlines = oldScan
	})

	writeLaunch := func(age time.Duration) {
		t.Helper()
		launched := time.Now().Add(-age).UTC().Format(time.RFC3339)
		row := `{"ts":"` + launched + `","session":"` + sid + `","phase":"launched"}` + "\n"
		if err := os.WriteFile(filepath.Join(reg, "resume_ledger.jsonl"), []byte(row), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	run := func() string {
		t.Helper()
		var out, errb bytes.Buffer
		code := runResumeWatchdog(&out, &errb, []string{
			"--no-refresh", "--reg-dir", reg, "--log-dir", logDir, "--spacing-sec", "0",
		})
		if code != 0 {
			t.Fatalf("exit = %d, want 0 (stderr: %s stdout: %s)", code, errb.String(), out.String())
		}
		return out.String()
	}

	writeLaunch(time.Minute)
	if got := run(); !strings.Contains(got, "already resumed once") || strings.Contains(got, "WOULD RESUME") {
		t.Fatalf("unproven launch inside grace must stay duplicate-safe:\n%s", got)
	}

	writeLaunch(time.Duration(resume.DeadTranscriptIdleFloorSeconds+60) * time.Second)
	if got := run(); !strings.Contains(got, "REVIVE") || !strings.Contains(got, "WOULD RESUME") {
		t.Fatalf("dead unproven launch past grace must become retryable:\n%s", got)
	}

	rwCollectProcCmdlines = func() ([]string, bool) {
		return []string{`C:\bin\claude.exe --resume ` + sid}, true
	}
	if got := run(); !strings.Contains(got, "already resumed once") || strings.Contains(got, "WOULD RESUME") {
		t.Fatalf("live unproven launch must remain duplicate-safe past grace:\n%s", got)
	}
}

// The #2367 acceptance: a targeted run consumes EXACTLY the operator's plan file even
// when the shared resume_plan.json has been regenerated by a concurrent refresh, and
// fails closed (typed reason, exit 1) when the targeted file is gone.
func TestResumeWatchdogTargetedPlanConsumesExactFile(t *testing.T) {
	reg := t.TempDir()
	logDir := t.TempDir()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("FLEET_CLAUDE_EXE", "claude")
	// The shared plan is what a concurrent fleet refresh just regenerated…
	shared := `{"plan":[{"session":"sid-replacement-aaaa","account":".claude-a"},{"session":"sid-replacement-bbbb","account":".claude-a"}]}`
	if err := os.WriteFile(filepath.Join(reg, "resume_plan.json"), []byte(shared), 0o644); err != nil {
		t.Fatal(err)
	}
	// …and the targeted file is the one-session plan the operator actually wrote.
	targeted := filepath.Join(t.TempDir(), "targeted_plan.json")
	if err := os.WriteFile(targeted, []byte(`{"plan":[{"session":"sid-target-11111111","account":".claude-a"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	oldBroker := launchSpawnBroker
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		return allowLaunchBrokerGrant(a, "unit-test-allow")
	}
	t.Cleanup(func() { launchSpawnBroker = oldBroker })

	var out, errb bytes.Buffer
	code := runResumeWatchdog(&out, &errb, []string{
		"--plan", targeted, "--reg-dir", reg, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s stdout: %s)", code, errb.String(), out.String())
	}
	got := out.String()
	if !strings.Contains(got, "sid-targ") || !strings.Contains(got, "targeted file") {
		t.Fatalf("targeted plan was not consumed:\n%s", got)
	}
	if strings.Contains(got, "sid-repl") {
		t.Fatalf("targeted run acted on the shared (raced) plan:\n%s", got)
	}

	// Missing targeted file: refuse with a typed reason instead of silently acting on
	// the shared replacement.
	out.Reset()
	errb.Reset()
	code = runResumeWatchdog(&out, &errb, []string{
		"--plan", filepath.Join(reg, "nope.json"), "--reg-dir", reg, "--log-dir", logDir,
	})
	if code != 1 {
		t.Fatalf("missing targeted plan: exit = %d, want fail-closed 1\n%s", code, out.String())
	}
	if !strings.Contains(out.String(), "PLAN RACE GUARD") || strings.Contains(out.String(), "sid-repl") {
		t.Fatalf("missing targeted plan must refuse with the typed guard, not fall back:\n%s", out.String())
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

// TestRwResumeArgvGuardFronting pins the Go port's managed-cache fronting decision to the SAME
// shape the Python reference enforces (tools/fleet_resume_watchdog_test.py:644-678, #2178/#3779):
// opted-in + fak resolvable => `fak guard <posture> --` with posture strictly BEFORE `--`;
// otherwise the bare `claude --resume` that shipped before the knob. The posture-arg SHAPING
// itself (order, auto emits nothing, malformed) is covered by guard_cache_posture_test.go.
func TestRwResumeArgvGuardFronting(t *testing.T) {
	oldAnchor := rwResumeAnchor
	rwResumeAnchor = func(string) resume.ResumeAnchor { return resume.ResumeAnchor{} }
	t.Cleanup(func() { rwResumeAnchor = oldAnchor })
	const sid = "SID"
	bare := []string{"/bin/claude", "--resume", sid, "-p", resumeWatchdogPrompt, "--dangerously-skip-permissions"}
	eq := func(got, want []string) bool { return strings.Join(got, "\x00") == strings.Join(want, "\x00") }
	has := func(xs []string, s string) bool {
		for _, x := range xs {
			if x == s {
				return true
			}
		}
		return false
	}

	// No posture configured (fak present) -> the exact bare argv, never fronted with guard.
	if got := rwResumeArgv("/bin/fak", "/bin/claude", sid, nil); !eq(got, bare) {
		t.Fatalf("nil-posture argv = %#v, want bare %#v", got, bare)
	}
	if got := rwResumeArgv("/bin/fak", "/bin/claude", sid, []string{}); !eq(got, bare) {
		t.Fatalf("empty-posture argv = %#v, want bare %#v", got, bare)
	}
	if got := rwResumeArgv("/bin/fak", "/bin/claude", sid, nil); has(got, "guard") {
		t.Fatalf("bare argv must not reference guard: %#v", got)
	}

	// Opted-in + fak resolvable -> `fak guard <posture> -- claude --resume ...`, posture strictly
	// BEFORE the `--` separator (guard parses it; the agent after `--` never sees it).
	posture := []string{"--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on"}
	want := []string{"/bin/fak", "guard", "--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on", "--",
		"/bin/claude", "--resume", sid, "-p", resumeWatchdogPrompt, "--dangerously-skip-permissions"}
	got := rwResumeArgv("/bin/fak", "/bin/claude", sid, posture)
	if !eq(got, want) {
		t.Fatalf("fronted argv = %#v, want %#v", got, want)
	}
	sep := -1
	for i, a := range got {
		if a == "--" {
			sep = i
			break
		}
	}
	if sep < 0 {
		t.Fatalf("fronted argv missing `--` separator: %#v", got)
	}
	if !has(got[:sep], "--managed-cache") || has(got[sep+1:], "--managed-cache") {
		t.Fatalf("posture flags must be strictly before `--`: %#v", got)
	}
	if !has(got[sep+1:], "--resume") {
		t.Fatalf("claude flags must be after `--`: %#v", got)
	}

	// Opted-in but fak unresolved -> cannot front; fall back to the bare direct launch (the
	// caller warns). The argv must never reference an empty fak binary.
	if got := rwResumeArgv("", "/bin/claude", sid, posture); !eq(got, bare) {
		t.Fatalf("fak-missing fallback = %#v, want bare %#v", got, bare)
	}
	if got := rwResumeArgv("   ", "/bin/claude", sid, posture); !eq(got, bare) {
		t.Fatalf("blank-fak fallback = %#v, want bare %#v", got, bare)
	}
}

// rwRecontinuePlan seeds a re-homed STOPPED_MIDTOOL plan row plus the prior guard-SessionStart
// identity row (uuid<->trace, on the OWNER account), and returns the transcript UUID. The source
// transcript is materialized so a live tick's RehomeTranscript copy succeeds (else the row is
// skipped before the launch). It is the shared fixture for the recontinue-refresh tests.
func rwRecontinuePlan(t *testing.T, regDir string) string {
	t.Helper()
	sid := "sid-recont-1234567890"
	srcConfig := t.TempDir() // owner ".claude-a" config dir (holds the transcript to re-home)
	dstConfig := t.TempDir() // ".claude-target" resume config dir
	work := t.TempDir()
	// The transcript RehomeTranscript reads: <srcCfg>/projects/<project>/<sid>.jsonl.
	proj := filepath.Join(srcConfig, "projects", "P")
	if err := os.MkdirAll(proj, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(proj, sid+".jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	srcJSON, _ := json.Marshal(srcConfig)
	dstJSON, _ := json.Marshal(dstConfig)
	workJSON, _ := json.Marshal(work)
	plan := `{"plan":[{` +
		`"session":"` + sid + `","account":".claude-a","resume_account":".claude-target","project":"P",` +
		`"config_dir":` + string(srcJSON) + `,` +
		`"resume_config_dir":` + string(dstJSON) + `,` +
		`"cwd":` + string(workJSON) + `,` +
		`"disp":"STOPPED_MIDTOOL","rehomed":true` +
		`}]}`
	if err := os.WriteFile(filepath.Join(regDir, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}
	// The prior fresh-start join (A2), recorded on the owner account before the crash.
	seed := `{"ts":"2026-07-01T00:00:00Z","uuid":"` + sid + `","trace":"trace-xyz","account":".claude-a","via":"guard-sessionstart"}` + "\n"
	if err := os.WriteFile(resume.IdentityLedgerPath(regDir), []byte(seed), 0o644); err != nil {
		t.Fatal(err)
	}
	return sid
}

// A3 (#4114): a LIVE watchdog resume of a re-homed row refreshes the identity join so the
// newest row for the UUID names the resume-target account, while carrying the prior trace
// forward — so the join stays whole and ResolveIdentity surfaces the new account.
func TestResumeIdentityRecontinueLiveRefreshesAccount(t *testing.T) {
	rwHoldTestEnv(t)
	regDir := t.TempDir()
	logDir := t.TempDir()
	sid := rwRecontinuePlan(t, regDir)

	oldBroker := launchSpawnBroker
	oldSpawn := rwSpawnResumeLaunch
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant { return allowLaunchBrokerGrant(a, "unit-test-allow") }
	rwSpawnResumeLaunch = func(claudeExe string, p resume.WatchdogPlanRow, resumeCfg, logDir string, grant launchBrokerGrant) (int, error) {
		return 12345, nil
	}
	t.Cleanup(func() { launchSpawnBroker = oldBroker; rwSpawnResumeLaunch = oldSpawn })

	var out, errb bytes.Buffer
	rc := runResumeWatchdog(&out, &errb, []string{
		"--live", "--no-refresh", "--reg-dir", regDir, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if rc != 0 {
		t.Fatalf("watchdog rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	// Sanity: the row actually launched (else there is nothing to refresh).
	if led, _ := os.ReadFile(filepath.Join(regDir, "resume_ledger.jsonl")); !strings.Contains(string(led), `"phase":"launched"`) {
		t.Fatalf("expected a launched ledger row:\n%s", led)
	}

	rows := resume.LoadIdentityRows(regDir)
	var newest resume.IdentityRow
	found := false
	for _, r := range rows {
		if strings.TrimSpace(r.UUID) == sid {
			newest, found = r, true // last write wins (append-only file order)
		}
	}
	if !found {
		t.Fatalf("no identity row for %s after live resume:\n%+v", sid, rows)
	}
	if newest.Account != ".claude-target" {
		t.Fatalf("newest row account = %q, want the resume target .claude-target", newest.Account)
	}
	if newest.Via != "resume-watchdog" {
		t.Fatalf("newest row via = %q, want resume-watchdog", newest.Via)
	}
	if newest.Trace != "trace-xyz" {
		t.Fatalf("newest row trace = %q, want the prior trace carried forward (trace-xyz)", newest.Trace)
	}
	// The refreshed row is a whole join, so the resolver surfaces the new account as latest.
	if m := resume.ResolveIdentity(rows, sid); !m.OK || m.Row.Account != ".claude-target" || m.Paired != "trace-xyz" {
		t.Fatalf("ResolveIdentity(%s) = %+v, want OK with account .claude-target paired to trace-xyz", sid, m)
	}
}

// A3 (#4114): a DRY-RUN tick records nothing — the refresh fires only on the live launch
// path, mirroring the ledger-append gating.
func TestResumeIdentityRecontinueDryRunWritesNothing(t *testing.T) {
	rwHoldTestEnv(t)
	regDir := t.TempDir()
	logDir := t.TempDir()
	_ = rwRecontinuePlan(t, regDir)

	oldBroker := launchSpawnBroker
	oldSpawn := rwSpawnResumeLaunch
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant { return allowLaunchBrokerGrant(a, "unit-test-allow") }
	spawned := false
	rwSpawnResumeLaunch = func(claudeExe string, p resume.WatchdogPlanRow, resumeCfg, logDir string, grant launchBrokerGrant) (int, error) {
		spawned = true
		return 12345, nil
	}
	t.Cleanup(func() { launchSpawnBroker = oldBroker; rwSpawnResumeLaunch = oldSpawn })

	var out, errb bytes.Buffer
	rc := runResumeWatchdog(&out, &errb, []string{
		"--no-refresh", "--reg-dir", regDir, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if rc != 0 {
		t.Fatalf("watchdog rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if spawned {
		t.Fatal("dry-run must not spawn a resume")
	}
	raw, _ := os.ReadFile(resume.IdentityLedgerPath(regDir))
	if strings.Contains(string(raw), "resume-watchdog") {
		t.Fatalf("dry-run refreshed the identity store (should be live-only):\n%s", raw)
	}
	// The seed row is the only line — the fold still resolves the pre-crash pairing untouched.
	if rows := resume.LoadIdentityRows(regDir); len(rows) != 1 || rows[0].Via != "guard-sessionstart" {
		t.Fatalf("dry-run mutated the identity store: %+v", rows)
	}
}

func TestResumeChildErrorReadbackClassifiesNewestCapture(t *testing.T) {
	dir := t.TempDir()
	sid := "019f3023-52dd-7001-b559-2818dc14ede6"
	older := filepath.Join(dir, "resume-"+shortID(sid)+"-1.log.err")
	newer := filepath.Join(dir, "resume-"+shortID(sid)+"-2.log.err")
	if err := os.WriteFile(older, []byte("transport reset"), 0o644); err != nil {
		t.Fatal(err)
	}
	time.Sleep(10 * time.Millisecond)
	if err := os.WriteFile(newer, []byte("CHILD_CRASH 400 malformed request"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, ok := rwNewestResumeChildError(dir, sid)
	if !ok || resume.ClassifyAttemptError(got) != resume.AttemptErrorMalformed400 {
		t.Fatalf("readback ok=%v text=%q class=%s", ok, got, resume.ClassifyAttemptError(got))
	}
}

func TestRecordAttemptCauseWritesConcreteReason(t *testing.T) {
	path := filepath.Join(t.TempDir(), "resume_ledger.jsonl")
	rwRecordAttemptCause(path, "LIVE", "sid", resume.AttemptErrorMalformed400)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"phase":"attempt_failed"`)) || !bytes.Contains(raw, []byte(`"reason":"MALFORMED_400"`)) {
		t.Fatalf("ledger=%s", raw)
	}
}

func writeResumeProgressTranscript(t *testing.T, home, sid string, launchUnix, turnUnix int64, model string) {
	t.Helper()
	dir := filepath.Join(home, ".claude-a", "projects", "C--work-fak")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	turn := time.Unix(turnUnix, 0).UTC().Format(time.RFC3339)
	line := fmt.Sprintf(`{"type":"assistant","timestamp":%q,"message":{"role":"assistant","model":%q,"content":"working","usage":{"input_tokens":10}}}`+"\n", turn, model)
	if err := os.WriteFile(filepath.Join(dir, sid+".jsonl"), []byte(line), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestRewitnessDroppedSessionRecordsProgress(t *testing.T) {
	home := t.TempDir()
	sid := "019f3023-52dd-7001-b559-2818dc14ede6"
	writeResumeProgressTranscript(t, home, sid, 100, 200, "claude-opus-4-8")
	status := filepath.Join(t.TempDir(), "status.jsonl")
	history := map[string][]resume.Attempt{sid: {{UnixSeconds: 100, Phase: "launched"}}}
	rwRewitnessDroppedSessions(home, status, "LIVE", nil, history, nil)
	raw, err := os.ReadFile(status)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"phase":"progress"`, `"rewitnessed_after_plan_drop":true`, `"session":"` + sid + `"`} {
		if !bytes.Contains(raw, []byte(want)) {
			t.Fatalf("status=%s missing %s", raw, want)
		}
	}
}

func TestRewitnessDroppedSessionSkipsPlannedOrAlreadyProven(t *testing.T) {
	home := t.TempDir()
	sid := "019f3023-52dd-7001-b559-2818dc14ede6"
	writeResumeProgressTranscript(t, home, sid, 100, 200, "claude-opus-4-8")
	history := map[string][]resume.Attempt{sid: {{UnixSeconds: 100, Phase: "launched"}}}
	for _, tc := range []struct {
		name   string
		plan   []resume.WatchdogPlanRow
		events []resume.WatchdogStatusEvent
	}{
		{"planned", []resume.WatchdogPlanRow{{Session: sid}}, nil},
		{"proven", nil, []resume.WatchdogStatusEvent{{Session: sid, Phase: "progress", NewTurns: 1}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status := filepath.Join(t.TempDir(), "status.jsonl")
			rwRewitnessDroppedSessions(home, status, "LIVE", tc.plan, history, tc.events)
			if _, err := os.Stat(status); !os.IsNotExist(err) {
				t.Fatalf("status unexpectedly written: %v", err)
			}
		})
	}
}

func TestResumeCarryReseedArgvAndAbsentCompatibility(t *testing.T) {
	const sid = "carry-session"
	posture := []string{"--managed-cache", "on"}
	bareFronted := rwResumeArgv("/bin/fak", "/bin/claude", sid, posture)
	carry := resume.DriveCarryRow{
		Session: sid, TurnsLeft: 7, TokensLeft: 12345, ContextTokensLeft: 4321,
		SpendMicroCentsLeft: 250000000, TimeLeftNanos: int64(90 * time.Minute),
		PaceMaxTokensPerTurn: 900, PaceMinTurnGapMs: 250,
	}
	got := rwResumeArgv("/bin/fak", "/bin/claude", sid, posture, carry)
	wantSpec := "turns=7,tokens=12345,context=4321,wall=1h30m0s,spend=2.50000000,max-tokens=900,gap=250ms"
	joined := strings.Join(got, "\x00")
	if !strings.Contains(joined, "--budget-envelope\x00"+wantSpec+"\x00--") {
		t.Fatalf("carried argv = %#v, want budget envelope %q before separator", got, wantSpec)
	}
	if strings.Join(bareFronted, "\x00") != strings.Join(rwResumeArgv("/bin/fak", "/bin/claude", sid, posture), "\x00") {
		t.Fatalf("absent carry changed argv: %#v", bareFronted)
	}
}

func TestResumeCarryReseedLoadLatestFailOpen(t *testing.T) {
	dir := t.TempDir()
	if got := rwLoadDriveCarry(dir); got != nil {
		t.Fatalf("missing ledger = %#v, want nil", got)
	}
	path := rwDriveCarryLedger(dir)
	body := "{not-json}\n" +
		`{"session":"carry-session","turns_left":9}` + "\n" +
		`{"session":"carry-session","turns_left":4}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	got := rwLoadDriveCarry(dir)
	if got["carry-session"].TurnsLeft != 4 {
		t.Fatalf("latest carry = %#v, want turns_left=4", got["carry-session"])
	}
}

func TestRwResumeArgvPrependsFreshAnchor(t *testing.T) {
	oldAnchor := rwResumeAnchor
	rwResumeAnchor = func(session string) resume.ResumeAnchor {
		curve := trajctl.ObjectiveCurve{ObjectiveID: "issue-2551", Signal: trajctl.SignalStall, Latest: .4, Delta: 0, Detail: "flat witnessed progress"}
		return resume.ResumeAnchor{Schema: resume.ResumeAnchorSchema, Session: session, ObjectiveID: curve.ObjectiveID, Objective: "prevent cascade drift", Curve: &curve, Plan: []trajctl.PlanPhase{{ID: "p1", Title: "wire anchor"}}, Present: true}
	}
	t.Cleanup(func() { rwResumeAnchor = oldAnchor })
	got := rwResumeArgv("", "/bin/claude", "SID", nil)
	promptAt := -1
	for i, arg := range got {
		if arg == "-p" {
			if promptAt >= 0 {
				t.Fatalf("duplicate -p in argv=%#v", got)
			}
			promptAt = i
		}
	}
	if promptAt < 0 || promptAt+1 >= len(got) {
		t.Fatalf("missing resume prompt in argv=%#v", got)
	}
	prompt := got[promptAt+1]
	for _, want := range []string{"fresh resume anchor", "prevent cascade drift", "STALL latest=0.40", "p1 wire anchor", resumeWatchdogPrompt} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q: %s", want, prompt)
		}
	}
}

func TestRwApplyTrajectoryWatchdogNudgesAliveStall(t *testing.T) {
	const sid, trace = "traj-sid", "traj-trace"
	oldAnchor, oldProcs := rwResumeAnchor, rwCollectProcCmdlines
	curve := trajctl.ObjectiveCurve{ObjectiveID: "issue-2559", Signal: trajctl.SignalStall, Latest: .4, Detail: "flat witnessed progress"}
	rwResumeAnchor = func(string) resume.ResumeAnchor {
		return resume.ResumeAnchor{Schema: resume.ResumeAnchorSchema, Session: sid, ObjectiveID: curve.ObjectiveID, Objective: "recover trajectory", Curve: &curve, Present: true}
	}
	rwCollectProcCmdlines = func() ([]string, bool) { return []string{"claude --resume " + sid}, true }
	t.Cleanup(func() { rwResumeAnchor, rwCollectProcCmdlines = oldAnchor, oldProcs; sessionctl.ClearObjective(trace) })
	ledger := filepath.Join(t.TempDir(), "resume.jsonl")
	handled, got := rwApplyTrajectoryWatchdog(resume.WatchdogPlanRow{Session: sid}, nil, &rwProcScan{}, map[string]string{sid: trace}, ledger, true)
	if !handled || got.Action != resume.TrajectoryNudge || sessionctl.RedirectPendingLen(trace) != 1 {
		t.Fatalf("handled=%v decision=%+v pending=%d", handled, got, sessionctl.RedirectPendingLen(trace))
	}
	raw, err := os.ReadFile(ledger)
	if err != nil || !strings.Contains(string(raw), "trajectory_nudge") || !strings.Contains(string(raw), "issue-2559") {
		t.Fatalf("ledger=%q err=%v", raw, err)
	}
}

func TestRwApplyTrajectoryWatchdogDeadFallsThroughToRevive(t *testing.T) {
	const sid = "dead-sid"
	oldAnchor, oldProcs := rwResumeAnchor, rwCollectProcCmdlines
	curve := trajctl.ObjectiveCurve{ObjectiveID: "issue-2559", Signal: trajctl.SignalStall}
	rwResumeAnchor = func(string) resume.ResumeAnchor {
		return resume.ResumeAnchor{Session: sid, ObjectiveID: curve.ObjectiveID, Objective: "recover", Curve: &curve, Present: true}
	}
	rwCollectProcCmdlines = func() ([]string, bool) { return nil, true }
	t.Cleanup(func() { rwResumeAnchor, rwCollectProcCmdlines = oldAnchor, oldProcs })
	handled, got := rwApplyTrajectoryWatchdog(resume.WatchdogPlanRow{Session: sid}, nil, &rwProcScan{}, nil, filepath.Join(t.TempDir(), "resume.jsonl"), true)
	if handled || got.Action != resume.TrajectoryReviveAnchor {
		t.Fatalf("handled=%v decision=%+v, want fallthrough revive", handled, got)
	}
}

func TestResumeWatchdogPromptTransportMovesGuardFrontedWindowsPrompt(t *testing.T) {
	argv := rwResumeArgv("fak.exe", "claude.exe", "session", []string{"--provider", "anthropic"})
	for i, arg := range argv {
		if arg == resumeWatchdogPrompt {
			argv[i] = strings.Repeat(arg, 200)
			break
		}
	}
	got, stdin, moved := guardPromptStdinTransportForOS(argv, "windows")
	if !moved || stdin == "" {
		t.Fatalf("resume prompt transport = moved %v stdin bytes %d", moved, len(stdin))
	}
	for _, arg := range got {
		if arg == stdin {
			t.Fatal("resume prompt remains on argv")
		}
	}
}

func TestResumeWatchdogDryRunLedgersSharedNextDecision(t *testing.T) {
	rwHoldTestEnv(t)
	dir := t.TempDir()
	plan := filepath.Join(dir, "plan.json")
	ledger := filepath.Join(dir, "resume_ledger.jsonl")
	row := resume.WatchdogPlanRow{Session: "session-next", CWD: dir, Account: ""}
	b, err := json.Marshal(struct {
		Plan []resume.WatchdogPlanRow `json:"plan"`
	}{Plan: []resume.WatchdogPlanRow{row}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(plan, b, 0o644); err != nil {
		t.Fatal(err)
	}

	oldSpawn := rwSpawnResumeLaunch
	rwSpawnResumeLaunch = func(string, resume.WatchdogPlanRow, string, string, launchBrokerGrant) (int, error) { return 1, nil }
	t.Cleanup(func() { rwSpawnResumeLaunch = oldSpawn })

	var out, stderr strings.Builder
	code := runResumeWatchdog(&out, &stderr, []string{"--plan", plan, "--reg-dir", dir, "--no-refresh"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	var found struct {
		Phase string                `json:"phase"`
		Next  sessionctl.NextRecord `json:"next"`
	}
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var row struct {
			Phase string                `json:"phase"`
			Next  sessionctl.NextRecord `json:"next"`
		}
		if json.Unmarshal([]byte(line), &row) == nil && row.Phase == "decision" {
			found = row
			break
		}
	}
	if found.Phase != "decision" || !found.Next.Applied || found.Next.Move.Kind != sessionctl.MoveContinue || found.Next.Move.Render != sessionctl.RenderSystemDirective {
		t.Fatalf("decision next = %+v", found)
	}
}
