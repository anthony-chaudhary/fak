package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionrecovery"
)

type captureLauncher struct{ got []sessionrecovery.Request }

func (c *captureLauncher) Launch(r sessionrecovery.Request) error {
	c.got = append(c.got, r)
	return nil
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
	oldInv, oldJournal, oldLaunch, oldSleep := recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep
	defer func() {
		recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep = oldInv, oldJournal, oldLaunch, oldSleep
	}()
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
	want := []string{guardBin, "guard", "--", "codex", "resume", "t1", "continue this exact task"}
	if len(cap.got) != 1 || !reflect.DeepEqual(cap.got[0].Argv, want) || cap.got[0].CWD != `C:\work\fak` {
		t.Fatalf("launch=%+v", cap.got)
	}
}

func TestSessionRecoverPollsUntilProductiveAndPreservesObservedCardinality(t *testing.T) {
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

func TestSessionRecoverFailsClosedOnDuplicateGuardedTrees(t *testing.T) {
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
	oldInv := recoveryInventory
	defer func() { recoveryInventory = oldInv }()
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
	wantArgv := []string{guardBin, "guard", "--", "codex", "resume", "journal-thread", "continue exactly"}
	if len(got.Results) != 1 || got.Results[0].CWD != `D:\repos\actual tree` || got.Results[0].Source != "session_journal" || !reflect.DeepEqual(got.Results[0].Argv, wantArgv) {
		t.Fatalf("summary=%+v", got)
	}
}
