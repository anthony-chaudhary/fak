package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
	resumesweep "github.com/anthony-chaudhary/fak/internal/resume/sweep"
	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionrecovery"
)

func installSessionRecoverIdentityFixture(t *testing.T) {
	t.Helper()
	regDir := t.TempDir()
	if err := os.WriteFile(resume.IdentityLedgerPath(regDir), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FLEET_REG_DIR", regDir)
}

type captureLauncher struct {
	got      []sessionrecovery.Request
	onLaunch func(int, sessionrecovery.Request)
}

func (c *captureLauncher) Launch(r sessionrecovery.Request) error {
	c.got = append(c.got, r)
	if c.onLaunch != nil {
		c.onLaunch(len(c.got), r)
	}
	return nil
}

func TestRecoveryInventoryReconcilesRuntimeSources(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("FLEET_REG_DIR", filepath.Join(home, "registry"))
	oldSources, oldRegistryPath, oldNow := recoveryInventorySources, recoveryInventoryRegistryPath, recoveryNow
	defer func() {
		recoveryInventorySources, recoveryInventoryRegistryPath, recoveryNow = oldSources, oldRegistryPath, oldNow
	}()
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	recoveryNow = func() time.Time { return now }
	recoveryInventoryRegistryPath = func() string { return filepath.Join(t.TempDir(), "missing.jsonl") }
	called := 0
	recoveryInventorySources = func(codexHome string, since time.Duration, observedAt time.Time) (sessiondiag.InventoryInput, error) {
		called++
		if codexHome != "" || since != 2*time.Hour || !observedAt.Equal(now) {
			t.Fatalf("source args home=%q since=%s now=%s", codexHome, since, observedAt)
		}
		return sessiondiag.InventoryInput{Threads: []sessiondiag.ThreadEvidence{{
			ThreadID: "91000001-0000-4000-8000-000000000001", Source: "cli", UpdatedAt: now,
		}}}, nil
	}
	report, err := recoveryInventory(2 * time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if called != 1 || len(report.Sessions) != 1 || report.Sessions[0].Thread == nil || report.Sessions[0].Thread.ID != "91000001-0000-4000-8000-000000000001" {
		t.Fatalf("called=%d report=%+v", called, report)
	}
}

func TestRecoveryInventoryReportsRuntimeSourceFailure(t *testing.T) {
	old := recoveryInventorySources
	defer func() { recoveryInventorySources = old }()
	recoveryInventorySources = func(string, time.Duration, time.Time) (sessiondiag.InventoryInput, error) {
		return sessiondiag.InventoryInput{}, errors.New("inventory unavailable")
	}
	if _, err := recoveryInventory(time.Hour); err == nil || !strings.Contains(err.Error(), "inventory unavailable") {
		t.Fatalf("err=%v", err)
	}
}

func TestSessionRecoverPreviewNeedsNoFakDevExecutable(t *testing.T) {
	installSessionRecoverIdentityFixture(t)
	if os.Getenv("FAK_RECOVERY_INSTALLED_HELPER") == "1" {
		if _, err := exec.LookPath("fak-dev"); err == nil {
			t.Fatal("fak-dev unexpectedly available on isolated PATH")
		}
		var out, er bytes.Buffer
		code := runSessionRecover(&out, &er, []string{"--json", "--journal=false", "--since", "1h"})
		if code != 0 {
			t.Fatalf("code=%d stderr=%s", code, er.String())
		}
		var summary sessionrecovery.Summary
		if err := json.Unmarshal(out.Bytes(), &summary); err != nil {
			t.Fatal(err)
		}
		if summary.Schema != sessionrecovery.SummarySchema || summary.Mode != "preview" {
			t.Fatalf("summary=%+v", summary)
		}
		return
	}
	python, err := exec.LookPath("python")
	if err != nil {
		t.Skip("Python is required by the current read-only SQLite inventory reader")
	}
	path := strings.Join([]string{filepath.Dir(os.Args[0]), filepath.Dir(python), os.Getenv("SystemRoot") + `\System32`, os.Getenv("SystemRoot")}, string(os.PathListSeparator))
	cmd := exec.Command(os.Args[0], "-test.run=^TestSessionRecoverPreviewNeedsNoFakDevExecutable$", "-test.count=1")
	cmd.Dir = t.TempDir()
	codexHome := filepath.Join(cmd.Dir, ".codex")
	if err := os.Mkdir(codexHome, 0o755); err != nil {
		t.Fatal(err)
	}
	cmd.Env = append(os.Environ(), "FAK_RECOVERY_INSTALLED_HELPER=1", "CODEX_HOME="+codexHome, "PATH="+path)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("installed-style preview: %v\n%s", err, output)
	}
}

func TestSessionRecoverIsFirstClassAlias(t *testing.T) {
	oldInv := recoveryInventory
	defer func() { recoveryInventory = oldInv }()
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		return sessionrecovery.InventoryReport{}, nil
	}
	var out, er bytes.Buffer
	if code := runSession(&out, &er, []string{"recover", "--journal=false"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, er.String())
	}
	if got := out.String(); !strings.Contains(got, "SESSION RECOVERY PREVIEW") || !strings.Contains(got, "discovered=0 selected=0") {
		t.Fatalf("output=%q", got)
	}
}

func TestSessionRecoverApplyRequiresExplicitConfirmation(t *testing.T) {
	var out, er bytes.Buffer
	if code := runSessionRecover(&out, &er, []string{"--apply", "--journal=false"}); code != 2 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	if got := er.String(); !strings.Contains(got, "requires an explicit --thread ID or --all") {
		t.Fatalf("error=%q", got)
	}
}

func TestSessionRecoverAllAndThreadAreMutuallyExclusive(t *testing.T) {
	var out, er bytes.Buffer
	if code := runSessionRecover(&out, &er, []string{"--apply", "--all", "--thread", "t1", "--journal=false"}); code != 2 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	if got := er.String(); !strings.Contains(got, "--all cannot be combined with --thread") {
		t.Fatalf("error=%q", got)
	}
}

func TestSessionRecoverPromptAndCWD(t *testing.T) {
	installSessionRecoverIdentityFixture(t)
	oldInv, oldJournal, oldLaunch, oldSleep, oldNow := recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep, recoveryNow
	defer func() {
		recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep, recoveryNow = oldInv, oldJournal, oldLaunch, oldSleep, oldNow
	}()
	recoveryNow = func() time.Time { return time.Date(2026, 8, 18, 1, 30, 0, 0, time.UTC) }
	recoveryJournalCrashes = func(string, time.Time) ([]sessionjournal.Classified, error) { return nil, nil }
	calls := 0
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		calls++
		r := sessionrecovery.Session{Thread: &sessionrecovery.Thread{ID: "t1", Source: "interactive_tui"}, LatestTurn: &sessionrecovery.Turn{Status: "inProgress", StartedAt: "2026-08-18T01:00:00Z"}}
		if calls > 1 {
			r.LatestTurn.StartedAt = "2026-08-18T02:00:00Z"
			r.ProcessTrees = []sessionrecovery.ProcessTree{{RootPID: 9, HasGuard: true}}
			r.GuardReceipt = &sessionrecovery.GuardReceipt{}
		}
		return sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{r}}, nil
	}
	cap := &captureLauncher{}
	recoveryLaunch = cap
	recoverySleep = func(time.Duration) {}
	var out, er bytes.Buffer
	code := runSessionRecover(&out, &er, []string{"--apply", "--all", "--cwd", `C:\work\fak`, "--prompt", "continue this exact task", "--receipts", t.TempDir(), "--settle", "0"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, er.String())
	}
	guardBin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{guardBin, "guard", "--", "codex", "exec", "--cd", `C:\work\fak`, "resume", "t1", "continue this exact task"}
	if len(cap.got) != 1 || !reflect.DeepEqual(cap.got[0].Argv, want) || cap.got[0].CWD != `C:\work\fak` {
		t.Fatalf("launch=%+v", cap.got)
	}
}

func TestSessionRecoverPollsUntilProductiveAndPreservesObservedCardinality(t *testing.T) {
	installSessionRecoverIdentityFixture(t)
	oldInv, oldJournal, oldLaunch, oldSleep, oldNow := recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep, recoveryNow
	defer func() {
		recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep, recoveryNow = oldInv, oldJournal, oldLaunch, oldSleep, oldNow
	}()
	recoveryJournalCrashes = func(string, time.Time) ([]sessionjournal.Classified, error) { return nil, nil }
	now := time.Unix(100, 0)
	recoveryNow = func() time.Time { return now }
	sleeps := 0
	recoverySleep = func(d time.Duration) { sleeps++; now = now.Add(d) }
	calls := 0
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		calls++
		s := sessionrecovery.Session{Thread: &sessionrecovery.Thread{ID: "t1", Source: "interactive_tui", CWD: `C:\work\fak`}, LatestTurn: &sessionrecovery.Turn{Status: "inProgress", StartedAt: "2026-08-18T01:00:00Z"}}
		if calls >= 3 {
			s.LatestTurn.StartedAt = "2026-08-18T02:00:00Z"
			s.ProcessTrees = []sessionrecovery.ProcessTree{{RootPID: 9, HasGuard: true}}
			s.GuardReceipt = &sessionrecovery.GuardReceipt{RecordedAt: "2026-08-18T02:00:01Z"}
		}
		return sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{s}}, nil
	}
	recoveryLaunch = &captureLauncher{}
	var out, er bytes.Buffer
	code := runSessionRecover(&out, &er, []string{"--apply", "--all", "--journal=false", "--receipts", t.TempDir(), "--verify-timeout", "5s", "--poll-interval", "100ms", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s output=%s", code, er.String(), out.String())
	}
	var got sessionrecovery.Summary
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if calls != 3 || sleeps != 1 || len(got.Results) != 1 || got.Results[0].Status != "productive" || got.Results[0].GuardedProcessTrees != 1 {
		t.Fatalf("calls=%d sleeps=%d summary=%+v", calls, sleeps, got)
	}
}

func TestSessionRecoverIdleShellIsNotFalseDone(t *testing.T) {
	installSessionRecoverIdentityFixture(t)
	oldInv, oldJournal, oldLaunch, oldSleep, oldNow := recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep, recoveryNow
	defer func() {
		recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep, recoveryNow = oldInv, oldJournal, oldLaunch, oldSleep, oldNow
	}()
	now := time.Date(2026, 8, 18, 1, 0, 0, 0, time.UTC)
	recoveryNow = func() time.Time { return now }
	recoverySleep = func(d time.Duration) { now = now.Add(d) }
	calls := 0
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		calls++
		s := sessionrecovery.Session{Thread: &sessionrecovery.Thread{ID: "t1", Source: "interactive_tui", CWD: `C:\work\fak`}, LatestTurn: &sessionrecovery.Turn{Status: "inProgress", StartedAt: "2026-08-18T01:00:00Z"}}
		if calls > 1 {
			s.ProcessTrees = []sessionrecovery.ProcessTree{{RootPID: 42, HasGuard: true, HasCodex: true}}
			s.GuardReceipt = &sessionrecovery.GuardReceipt{RecordedAt: "2026-08-18T01:00:01Z"}
		}
		return sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{s}}, nil
	}
	recoveryJournalCrashes = func(string, time.Time) ([]sessionjournal.Classified, error) { return nil, nil }
	recoveryLaunch = &captureLauncher{}

	var stdout, stderr bytes.Buffer
	code := runSessionRecover(&stdout, &stderr, []string{"--apply", "--thread", "t1", "--json", "--journal=false", "--receipts", t.TempDir(), "--verify-timeout", "2s", "--poll-interval", "1s"})
	if code != 1 {
		t.Fatalf("runSessionRecover code=%d, want 1 for unproven idle shell; stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var summary sessionrecovery.Summary
	if err := json.Unmarshal(stdout.Bytes(), &summary); err != nil {
		t.Fatal(err)
	}
	if summary.Counts.Active != 0 || summary.Counts.LaunchedUnproven != 1 || summary.Counts.ExactCardinality != 1 || summary.Results[0].Advanced {
		t.Fatalf("counts = %+v result=%+v, idle shell must remain unproven", summary.Counts, summary.Results[0])
	}
}

func TestSessionRecoverFailsClosedOnDuplicateGuardedTrees(t *testing.T) {
	installSessionRecoverIdentityFixture(t)
	oldInv, oldJournal, oldLaunch, oldSleep := recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep
	defer func() {
		recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep = oldInv, oldJournal, oldLaunch, oldSleep
	}()
	recoveryJournalCrashes = func(string, time.Time) ([]sessionjournal.Classified, error) { return nil, nil }
	calls := 0
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		calls++
		s := sessionrecovery.Session{Thread: &sessionrecovery.Thread{ID: "t1", Source: "interactive_tui", CWD: `C:\work\fak`}, LatestTurn: &sessionrecovery.Turn{Status: "inProgress", StartedAt: "2026-08-18T01:00:00Z"}}
		if calls > 1 {
			s.ProcessTrees = []sessionrecovery.ProcessTree{{RootPID: 9, HasGuard: true}, {RootPID: 10, HasGuard: true}}
			s.GuardReceipt = &sessionrecovery.GuardReceipt{}
		}
		return sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{s}}, nil
	}
	recoveryLaunch = &captureLauncher{}
	recoverySleep = func(time.Duration) {}
	var out, er bytes.Buffer
	code := runSessionRecover(&out, &er, []string{"--apply", "--all", "--journal=false", "--receipts", t.TempDir(), "--verify-timeout", "1s", "--json"})
	if code != 1 {
		t.Fatalf("code=%d stderr=%s output=%s", code, er.String(), out.String())
	}
	var got sessionrecovery.Summary
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Results) != 1 || got.Results[0].Status != "cardinality_failed" || got.Results[0].GuardedProcessTrees != 2 || got.Counts.Failed != 1 {
		t.Fatalf("summary=%+v", got)
	}
}

func TestSessionRecoverUsesJournalRecordedCWD(t *testing.T) {
	installSessionRecoverIdentityFixture(t)
	oldInv, oldJournal := recoveryInventory, recoveryJournalCrashes
	defer func() { recoveryInventory, recoveryJournalCrashes = oldInv, oldJournal }()
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		return sessionrecovery.InventoryReport{}, nil
	}
	journalPath := filepath.Join(t.TempDir(), "session-journal.jsonl")
	if err := sessionjournal.Append(journalPath, sessionjournal.Event{
		Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen,
		ID: "journal-thread", CWD: `D:\repos\actual tree`, PID: 0,
		TS: "2000-01-01T00:00:00Z", Boot: "boot-old",
	}); err != nil {
		t.Fatal(err)
	}
	// Classification against the host's real boot time is covered by sessionjournal.
	// This CLI witness pins the merger and recorded-CWD contract independently.
	recoveryJournalCrashes = func(path string, _ time.Time) ([]sessionjournal.Classified, error) {
		if path != journalPath {
			t.Fatalf("journal path = %q, want %q", path, journalPath)
		}
		return []sessionjournal.Classified{{Session: sessionjournal.Session{ID: "journal-thread", CWD: `D:\repos\actual tree`, Agent: "claude"}, Status: sessionjournal.StatusCrashed, Reason: "MACHINE_REBOOT"}}, nil
	}
	var out, er bytes.Buffer
	code := runSessionRecover(&out, &er, []string{"--limit", "1", "--journal-path", journalPath, "--receipts", t.TempDir(), "--prompt", "continue exactly", "--json"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, er.String())
	}
	var got sessionrecovery.Summary
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	guardBin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	wantArgv := []string{guardBin, "guard", "--", "claude", "--resume", "journal-thread", "continue exactly"}
	if len(got.Results) != 1 || got.Results[0].CWD != `D:\repos\actual tree` || got.Results[0].Source != "session_journal" || got.Results[0].Provider != sessionrecovery.ProviderClaude || !reflect.DeepEqual(got.Results[0].Argv, wantArgv) {
		t.Fatalf("summary=%+v", got)
	}
}

func TestSessionRecoverUsesCodexJournalOnlyWithExactStateRequest(t *testing.T) {
	installSessionRecoverIdentityFixture(t)
	oldInv, oldJournal := recoveryInventory, recoveryJournalCrashes
	defer func() { recoveryInventory, recoveryJournalCrashes = oldInv, oldJournal }()
	const id = "94100001-0000-4000-8000-000000000001"
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		return sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{{
			Thread: &sessionrecovery.Thread{ID: id, Source: "cli", CWD: `C:\old`}, Provider: sessionrecovery.ProviderCodex,
			Category: sessionrecovery.CategorySubstantive, Action: sessionrecovery.ActionRecover,
		}}}, nil
	}
	recoveryJournalCrashes = func(string, time.Time) ([]sessionjournal.Classified, error) {
		return []sessionjournal.Classified{{Session: sessionjournal.Session{ID: id, CWD: `D:\authoritative`, Agent: "codex"}, Status: sessionjournal.StatusCrashed, Reason: "MACHINE_REBOOT"}}, nil
	}
	var out, stderr bytes.Buffer
	if code := runSessionRecover(&out, &stderr, []string{"--json", "--receipts", t.TempDir()}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got sessionrecovery.Summary
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	guardBin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{guardBin, "guard", "--", "codex", "exec", "--cd", `D:\authoritative`, "resume", id}
	if len(got.Results) != 1 || got.Results[0].Provider != sessionrecovery.ProviderCodex || !reflect.DeepEqual(got.Results[0].Argv, want) {
		t.Fatalf("summary=%+v want=%q", got, want)
	}
}

func TestMergeClaudeRecoveryInventoryClassifiesMixedCohortAndHighRecordProbe(t *testing.T) {
	home := t.TempDir()
	now := time.Date(2026, 8, 24, 2, 0, 0, 0, time.UTC)
	write := func(sid string, rows []map[string]any) {
		t.Helper()
		path := filepath.Join(home, ".claude-a", "projects", "C--work-fak", sid+".jsonl")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		var body strings.Builder
		for _, row := range rows {
			raw, _ := json.Marshal(row)
			body.Write(raw)
			body.WriteByte('\n')
		}
		if err := os.WriteFile(path, []byte(body.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		_ = os.Chtimes(path, now, now)
	}
	record := func(role, text, id, at string, isErr bool) map[string]any {
		row := map[string]any{"uuid": id, "timestamp": at, "message": map[string]any{"role": role, "content": text}}
		if isErr {
			row["isApiErrorMessage"] = true
		}
		return row
	}
	probeID := "a1000001-0000-4000-8000-000000000001"
	probe := []map[string]any{record("user", "Reply exactly LOGIN_OK", "p0", "2026-08-24T01:00:00Z", false)}
	for i := 0; i < 30; i++ {
		probe = append(probe, record("assistant", "probe bookkeeping", "", "2026-08-24T01:00:10Z", false))
	}
	probe = append(probe, record("assistant", "Not logged in. Please run /login", "pe", "2026-08-24T01:01:00Z", true))
	write(probeID, probe)
	substantiveID := "a1000002-0000-4000-8000-000000000002"
	write(substantiveID, []map[string]any{
		record("user", "finish issue 8787", "s0", "2026-08-24T01:00:00Z", false),
		record("assistant", "API Error: Overloaded (529)", "se", "2026-08-24T01:01:00Z", true),
	})
	liveID := "a1000003-0000-4000-8000-000000000003"
	write(liveID, []map[string]any{
		record("user", "continue real work", "l0", "2026-08-24T01:00:00Z", false),
		record("assistant", "API Error: Overloaded (529)", "le", "2026-08-24T01:01:00Z", true),
	})

	report := mergeClaudeRecoveryInventory(sessionrecovery.InventoryReport{}, home, time.Hour, now, map[string]bool{liveID: true})
	byID := map[string]sessionrecovery.Session{}
	for _, row := range report.Sessions {
		byID[row.Thread.ID] = row
	}
	if byID[probeID].Category != sessionrecovery.CategoryProbe || byID[probeID].Action != sessionrecovery.ActionExcludeProbe {
		t.Fatalf("high-record probe=%+v", byID[probeID])
	}
	if byID[substantiveID].Category != sessionrecovery.CategorySubstantive || byID[substantiveID].Action != sessionrecovery.ActionRecover {
		t.Fatalf("substantive=%+v", byID[substantiveID])
	}
	if byID[liveID].Category != sessionrecovery.CategoryLive || byID[liveID].Action != sessionrecovery.ActionLeaveLive {
		t.Fatalf("live=%+v", byID[liveID])
	}
}

func TestReconcileHistoricalClaudeAuthAdmitsCohortAfterLiveCurrentAuth(t *testing.T) {
	oldProbe := recoveryClaudeAuthProbe
	defer func() { recoveryClaudeAuthProbe = oldProbe }()
	probes := 0
	recoveryClaudeAuthProbe = func(context.Context) (bool, error) {
		probes++
		return true, nil
	}
	report := sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{
		{Thread: &sessionrecovery.Thread{ID: "auth-1", Source: "claude_transcript"}, Provider: sessionrecovery.ProviderClaude, Category: sessionrecovery.CategoryIdentityBlocked, Action: sessionrecovery.ActionLoginRequired, Reason: "claude_login_required", Bucket: resumesweep.BucketAuth},
		{Thread: &sessionrecovery.Thread{ID: "auth-2", Source: "claude_transcript"}, Provider: sessionrecovery.ProviderClaude, Category: sessionrecovery.CategoryIdentityBlocked, Action: sessionrecovery.ActionLoginRequired, Reason: "claude_login_required", Bucket: resumesweep.BucketAuth},
	}}

	reconcileHistoricalClaudeAuth(&report)

	if probes != 1 {
		t.Fatalf("auth probes=%d, want one cohort-level probe", probes)
	}
	for _, row := range report.Sessions {
		if row.Category != sessionrecovery.CategorySubstantive || row.Action != sessionrecovery.ActionRecover || row.Reason != "claude_current_auth_live" {
			t.Fatalf("row=%+v, want recoverable after positive current-auth proof", row)
		}
	}
}

func TestReconcileHistoricalClaudeAuthFailsClosedOnFailedOrUnknownCurrentAuth(t *testing.T) {
	oldProbe := recoveryClaudeAuthProbe
	defer func() { recoveryClaudeAuthProbe = oldProbe }()
	for _, tc := range []struct {
		name string
		live bool
		err  error
	}{
		{name: "failed", err: errors.New("claude probe failed")},
		{name: "unknown"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			probes := 0
			recoveryClaudeAuthProbe = func(context.Context) (bool, error) {
				probes++
				return tc.live, tc.err
			}
			report := sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{{
				Thread: &sessionrecovery.Thread{ID: "auth", Source: "claude_transcript"}, Provider: sessionrecovery.ProviderClaude,
				Category: sessionrecovery.CategoryIdentityBlocked, Action: sessionrecovery.ActionLoginRequired,
				Reason: "claude_login_required", Bucket: resumesweep.BucketAuth,
			}}}

			reconcileHistoricalClaudeAuth(&report)

			row := report.Sessions[0]
			if probes != 1 || row.Category != sessionrecovery.CategoryIdentityBlocked || row.Action != sessionrecovery.ActionLoginRequired || row.Reason != "claude_login_required" {
				t.Fatalf("probes=%d row=%+v, want fail-closed login_required", probes, row)
			}
		})
	}
}

func TestReconcileHistoricalClaudeAuthSkipsProbeWithoutAuthRows(t *testing.T) {
	oldProbe := recoveryClaudeAuthProbe
	defer func() { recoveryClaudeAuthProbe = oldProbe }()
	probes := 0
	recoveryClaudeAuthProbe = func(context.Context) (bool, error) {
		probes++
		return true, nil
	}
	report := sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{{
		Thread: &sessionrecovery.Thread{ID: "substantive", Source: "claude_transcript"}, Provider: sessionrecovery.ProviderClaude,
		Category: sessionrecovery.CategorySubstantive, Action: sessionrecovery.ActionRecover,
	}}}

	reconcileHistoricalClaudeAuth(&report)

	if probes != 0 {
		t.Fatalf("auth probes=%d, want zero without a historical AUTH row", probes)
	}
}

func TestSessionRecoverHistoricalClaudeAuthUsesCurrentAuthProof(t *testing.T) {
	installSessionRecoverIdentityFixture(t)
	oldInv, oldProbe := recoveryInventory, recoveryClaudeAuthProbe
	defer func() {
		recoveryInventory, recoveryClaudeAuthProbe = oldInv, oldProbe
	}()
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	if err := os.WriteFile(resume.IdentityLedgerPath(regDir), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		return sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{{
			Thread:   &sessionrecovery.Thread{ID: "historical-auth", Source: "claude_transcript", CWD: t.TempDir()},
			Provider: sessionrecovery.ProviderClaude, Category: sessionrecovery.CategoryIdentityBlocked,
			Action: sessionrecovery.ActionLoginRequired, Reason: "claude_login_required", Bucket: resumesweep.BucketAuth,
		}}}, nil
	}
	for _, tc := range []struct {
		name         string
		live         bool
		err          error
		wantStatus   string
		wantCategory string
	}{
		{name: "live", live: true, wantStatus: "candidate", wantCategory: sessionrecovery.CategorySubstantive},
		{name: "failed", err: errors.New("probe failed"), wantStatus: "identity_blocked", wantCategory: sessionrecovery.CategoryIdentityBlocked},
		{name: "unknown", wantStatus: "identity_blocked", wantCategory: sessionrecovery.CategoryIdentityBlocked},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The recovery audit is fail-closed on its durable identity authority.
			// Recreate the empty valid ledger per case so no sibling cleanup can make
			// the auth-probe assertion depend on shared filesystem lifetime.
			if err := os.WriteFile(resume.IdentityLedgerPath(regDir), nil, 0o644); err != nil {
				t.Fatal(err)
			}
			probes := 0
			recoveryClaudeAuthProbe = func(context.Context) (bool, error) {
				probes++
				return tc.live, tc.err
			}
			var stdout, stderr bytes.Buffer
			code := runSessionRecover(&stdout, &stderr, []string{"--json", "--journal=false", "--receipts", t.TempDir()})
			if code != 0 {
				t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
			}
			var summary sessionrecovery.Summary
			if err := json.Unmarshal(stdout.Bytes(), &summary); err != nil {
				t.Fatal(err)
			}
			if probes != 1 || len(summary.Results) != 1 || summary.Results[0].Status != tc.wantStatus || summary.Results[0].Category != tc.wantCategory {
				t.Fatalf("probes=%d summary=%+v", probes, summary)
			}
		})
	}
}

func TestSessionRecoverUnifiedPreviewAndLiveWitness(t *testing.T) {
	installSessionRecoverIdentityFixture(t)
	oldInv, oldJournal, oldLaunch, oldSleep, oldNow := recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep, recoveryNow
	defer func() {
		recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep, recoveryNow = oldInv, oldJournal, oldLaunch, oldSleep, oldNow
	}()
	now := time.Date(2026, 8, 24, 1, 30, 0, 0, time.UTC)
	recoveryNow = func() time.Time { return now }
	recoverySleep = func(d time.Duration) { now = now.Add(d) }
	recoveryJournalCrashes = func(string, time.Time) ([]sessionjournal.Classified, error) { return nil, nil }
	const claudeID = "b1000001-0000-4000-8000-000000000001"
	const codexID = "b1000002-0000-4000-8000-000000000002"
	base := sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{
		{Thread: &sessionrecovery.Thread{ID: claudeID, Source: "claude_transcript", CWD: `C:\work\fak`}, Provider: sessionrecovery.ProviderClaude, Category: sessionrecovery.CategorySubstantive, Action: sessionrecovery.ActionRecover, Cursor: "a0", CursorAt: "2026-08-24T01:00:00Z"},
		{Thread: &sessionrecovery.Thread{ID: codexID, Source: "host_resurrection", CWD: `C:\work\fak`}, Provider: sessionrecovery.ProviderCodex, Category: sessionrecovery.CategorySubstantive, Action: sessionrecovery.ActionRecover, LatestTurn: &sessionrecovery.Turn{ID: "t0", Status: "inProgress", StartedAt: "2026-08-24T01:00:00Z"}},
		{Thread: &sessionrecovery.Thread{ID: "probe-row"}, Provider: sessionrecovery.ProviderClaude, Category: sessionrecovery.CategoryProbe, Action: sessionrecovery.ActionExcludeProbe, Reason: "semantic_probe"},
		{Thread: &sessionrecovery.Thread{ID: "login-row"}, Provider: sessionrecovery.ProviderClaude, Category: sessionrecovery.CategoryIdentityBlocked, Action: sessionrecovery.ActionLoginRequired, Reason: "claude_login_required"},
		{Thread: &sessionrecovery.Thread{ID: "live-row"}, Provider: sessionrecovery.ProviderCodex, Category: sessionrecovery.CategoryLive, Action: sessionrecovery.ActionLeaveLive, Reason: "live_process_tree"},
	}}
	calls := 0
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		calls++
		out := base
		out.Sessions = append([]sessionrecovery.Session(nil), base.Sessions...)
		if calls > 1 {
			out.Sessions[0].Cursor, out.Sessions[0].CursorAt = "a1", "2026-08-24T02:00:00Z"
			out.Sessions[1].LatestTurn = &sessionrecovery.Turn{ID: "t1", Status: "inProgress", StartedAt: "2026-08-24T02:00:00Z"}
		}
		return out, nil
	}
	launcher := &captureLauncher{}
	recoveryLaunch = launcher

	previewDir := t.TempDir()
	var previewOut, previewErr bytes.Buffer
	if code := runSessionRecover(&previewOut, &previewErr, []string{"--json", "--journal=false", "--limit", "10", "--receipts", previewDir}); code != 0 {
		t.Fatalf("preview code=%d stderr=%s", code, previewErr.String())
	}
	if len(launcher.got) != 0 {
		t.Fatalf("preview launched %+v", launcher.got)
	}
	var preview sessionrecovery.Summary
	if err := json.Unmarshal(previewOut.Bytes(), &preview); err != nil {
		t.Fatal(err)
	}
	if len(preview.Results) != len(base.Sessions) || preview.WitnessPath == "" {
		t.Fatalf("preview=%+v", preview)
	}
	if _, err := os.Stat(preview.WitnessPath); err != nil {
		t.Fatalf("preview witness: %v", err)
	}

	// Reset the inventory call counter so live gets the same pre-launch baseline.
	calls = 0
	liveDir := t.TempDir()
	launcher.onLaunch = func(launchNumber int, _ sessionrecovery.Request) {
		if calls != 1 {
			t.Fatalf("provider observation ran before all windows launched: calls=%d launch=%d", calls, launchNumber)
		}
		witnesses, err := filepath.Glob(filepath.Join(liveDir, "run-*.json"))
		if err != nil || len(witnesses) != 1 {
			t.Fatalf("run witness missing before launch %d: paths=%v err=%v", launchNumber, witnesses, err)
		}
	}
	var liveOut, liveErr bytes.Buffer
	if code := runSessionRecover(&liveOut, &liveErr, []string{"--live", "--all", "--json", "--journal=false", "--receipts", liveDir, "--verify-timeout", "1s"}); code != 0 {
		t.Fatalf("live code=%d stderr=%s stdout=%s", code, liveErr.String(), liveOut.String())
	}
	if len(launcher.got) != 2 {
		t.Fatalf("launches=%+v, want one per actionable session", launcher.got)
	}
	if !strings.Contains(strings.Join(launcher.got[0].Argv, " ")+" "+strings.Join(launcher.got[1].Argv, " "), "codex exec --cd C:\\work\\fak resume "+codexID) {
		t.Fatalf("missing exact codex exec resume argv: %+v", launcher.got)
	}
	var live sessionrecovery.Summary
	if err := json.Unmarshal(liveOut.Bytes(), &live); err != nil {
		t.Fatal(err)
	}
	if live.Counts.Productive != 2 || live.Counts.Probe != 1 || live.Counts.IdentityBlocked != 1 || live.Counts.Live != 1 {
		t.Fatalf("live counts=%+v results=%+v", live.Counts, live.Results)
	}
	if raw, err := os.ReadFile(live.WitnessPath); err != nil || !bytes.Contains(raw, []byte(`"advanced": true`)) {
		t.Fatalf("durable live witness err=%v body=%s", err, raw)
	}
}

func TestSessionRecoverRefusesMalformedIdentityBeforeInventoryOrLaunch(t *testing.T) {
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	if err := os.WriteFile(resume.IdentityLedgerPath(regDir), []byte("not-json\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	oldInv, oldLaunch := recoveryInventory, recoveryLaunch
	defer func() { recoveryInventory, recoveryLaunch = oldInv, oldLaunch }()
	inventoryCalls := 0
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		inventoryCalls++
		return sessionrecovery.InventoryReport{}, nil
	}
	cap := &captureLauncher{}
	recoveryLaunch = cap
	var out, stderr bytes.Buffer
	receipts := filepath.Join(t.TempDir(), "receipts")
	code := runSessionRecover(&out, &stderr, []string{"--live", "--all", "--journal=false", "--receipts", receipts})
	if code == 0 || inventoryCalls != 0 || len(cap.got) != 0 {
		t.Fatalf("code=%d inventory=%d launches=%d", code, inventoryCalls, len(cap.got))
	}
	if !strings.Contains(stderr.String(), resume.IdentityLedgerPath(regDir)) || !strings.Contains(stderr.String(), "1 invalid") {
		t.Fatalf("stderr=%q", stderr.String())
	}
	if _, err := os.Stat(receipts); !os.IsNotExist(err) {
		t.Fatalf("receipts created: %v", err)
	}
}
