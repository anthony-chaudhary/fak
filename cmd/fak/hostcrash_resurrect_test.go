package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/hostfault"
	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
)

func TestResurrectHostCrashSessionsLaunchesOnceAndPersistsReceipt(t *testing.T) {
	dir := t.TempDir()
	row := guardsessions.NewInteractiveRow("trace", "claude", 11, filepath.Join(dir, "repo"), "", "", time.Now(), []string{"claude", "--continue"})
	row.HostRecovery = true
	if err := guardsessions.Record(dir, row); err != nil {
		t.Fatal(err)
	}
	signal := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "event-1", Class: hostfault.HostCrashWTRenderAV}
	logPath := filepath.Join(dir, "host.jsonl")
	launches := 0
	launch := func(req hostresurrect.Request) (int, error) {
		launches++
		if req.CWD != row.CWD || req.ResumeHandle != row.Handle {
			t.Fatalf("request=%+v", req)
		}
		return 4242, nil
	}
	now := time.Date(2026, 7, 14, 20, 0, 0, 0, time.UTC)
	if err := hostresurrect.StoreCohort(hostresurrect.CohortPath(dir), hostresurrect.Cohort{CapturedAt: now.UTC().Format(time.RFC3339Nano), Sessions: []hostresurrect.CohortEntry{{Handle: row.Handle, PID: row.PID, StartedAt: row.StartedAt}}}); err != nil {
		t.Fatal(err)
	}
	got, _, err := resurrectHostCrashSessions(logPath, dir, []hostfault.HostCrashSignal{signal}, launch, now, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].PID != 4242 || launches != 1 {
		t.Fatalf("receipts=%+v launches=%d", got, launches)
	}
	got, _, err = resurrectHostCrashSessions(logPath, dir, []hostfault.HostCrashSignal{signal}, launch, now.Add(time.Second), true)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 || launches != 1 {
		t.Fatalf("duplicate receipts=%+v launches=%d", got, launches)
	}
}

func TestResurrectHostCrashSessionsUsesPersistedPreCrashCohort(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 7, 14, 20, 0, 0, 0, time.UTC)
	for i := 0; i < hostresurrect.MaxLaunchesPerWindow+1; i++ {
		row := guardsessions.NewInteractiveRow(fmt.Sprintf("stale-%02d", i), "claude", 100+i, dir, "", "", now.Add(-time.Hour), []string{"claude"})
		row.HostRecovery = true
		if err := guardsessions.Record(dir, row); err != nil {
			t.Fatal(err)
		}
	}
	current := guardsessions.NewInteractiveRow("current", "codex", 4242, dir, "", "", now, []string{"codex"})
	current.HostRecovery = true
	if err := guardsessions.Record(dir, current); err != nil {
		t.Fatal(err)
	}
	cohort := hostresurrect.Cohort{CapturedAt: now.Add(-time.Second).Format(time.RFC3339Nano), Sessions: []hostresurrect.CohortEntry{{Handle: current.Handle, PID: current.PID, StartedAt: current.StartedAt}}}
	if err := hostresurrect.StoreCohort(hostresurrect.CohortPath(dir), cohort); err != nil {
		t.Fatal(err)
	}

	var launched []string
	receipts, selections, err := resurrectHostCrashSessions(filepath.Join(dir, "host.jsonl"), dir, []hostfault.HostCrashSignal{{Schema: hostfault.HostCrashSignalSchema, EventID: "mixed"}}, func(req hostresurrect.Request) (int, error) {
		launched = append(launched, req.Session)
		return 99, nil
	}, now, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(receipts) != 1 || len(launched) != 1 || launched[0] != current.Handle {
		t.Fatalf("receipts=%+v launched=%v", receipts, launched)
	}
	if len(selections) != 1 || selections[0].Counts.ExcludedNotInCohort != hostresurrect.MaxLaunchesPerWindow+1 || selections[0].Counts.Selected != 1 {
		t.Fatalf("selections=%+v", selections)
	}
}
func TestResurrectHostCrashSessionsHonorsGlobalLaunchRate(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "host.jsonl")
	now := time.Now().UTC()
	for i := 0; i < hostresurrect.MaxLaunchesPerWindow; i++ {
		if err := appendHostResurrectionReceipt(hostResurrectionReceiptPath(logPath), hostResurrectionReceipt{Schema: hostresurrect.Schema, Key: "old|" + string(rune('a'+i)), EventID: "old", Session: "s", LaunchedAt: now.Format(time.RFC3339Nano)}); err != nil {
			t.Fatal(err)
		}
	}
	row := guardsessions.NewInteractiveRow("trace", "claude", 1, dir, "", "", now, []string{"claude"})
	row.HostRecovery = true
	_ = guardsessions.Record(dir, row)
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "new", Class: hostfault.HostCrashGeneric}
	if err := hostresurrect.StoreCohort(hostresurrect.CohortPath(dir), hostresurrect.Cohort{CapturedAt: now.UTC().Format(time.RFC3339Nano), Sessions: []hostresurrect.CohortEntry{{Handle: row.Handle, PID: row.PID, StartedAt: row.StartedAt}}}); err != nil {
		t.Fatal(err)
	}
	got, _, err := resurrectHostCrashSessions(logPath, dir, []hostfault.HostCrashSignal{sig}, func(hostresurrect.Request) (int, error) { t.Fatal("launch called above cap"); return 0, nil }, now, true)
	if err != nil || len(got) != 0 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}
func TestResurrectHostCrashSessionsFailedLaunchIsStillDeduped(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 7, 14, 21, 0, 0, 0, time.UTC)
	row := guardsessions.NewInteractiveRow("trace-fail", "claude", 1, dir, "", "", now, []string{"claude"})
	row.HostRecovery = true
	if err := guardsessions.Record(dir, row); err != nil {
		t.Fatal(err)
	}
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt-fail", Class: hostfault.HostCrashGeneric}
	calls := 0
	launcher := func(req hostresurrect.Request) (int, error) { calls++; return 0, errors.New("spawn failed") }
	if err := hostresurrect.StoreCohort(hostresurrect.CohortPath(dir), hostresurrect.Cohort{CapturedAt: now.UTC().Format(time.RFC3339Nano), Sessions: []hostresurrect.CohortEntry{{Handle: row.Handle, PID: row.PID, StartedAt: row.StartedAt}}}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := resurrectHostCrashSessions(filepath.Join(dir, "host.jsonl"), dir, []hostfault.HostCrashSignal{sig}, launcher, now, true); err == nil {
		t.Fatal("want launch error")
	}

	got, _, err := resurrectHostCrashSessions(filepath.Join(dir, "host.jsonl"), dir, []hostfault.HostCrashSignal{sig}, launcher, now.Add(time.Second), true)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 || calls != 1 {
		t.Fatalf("retry relaunched reserved session: receipts=%+v calls=%d", got, calls)
	}
}

func TestResurrectHostCrashSessionsDryRunDoesNotReserve(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 8, 23, 20, 0, 0, 0, time.UTC)
	row := guardsessions.NewInteractiveRow("trace", "codex", 41, dir, "", "", now.Add(-time.Minute), []string{"codex", "resume", "session-1"})
	row.HostRecovery = true
	if err := guardsessions.Record(dir, row); err != nil {
		t.Fatal(err)
	}
	cohort := hostresurrect.Cohort{CapturedAt: now.Add(-time.Second).Format(time.RFC3339Nano), Sessions: []hostresurrect.CohortEntry{{Handle: row.Handle, PID: row.PID, StartedAt: row.StartedAt}}}
	if err := hostresurrect.StoreCohort(hostresurrect.CohortPath(dir), cohort); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(dir, "host.jsonl")
	var launched []hostresurrect.Request
	receipts, selections, err := resurrectHostCrashSessions(logPath, dir, []hostfault.HostCrashSignal{{Schema: hostfault.HostCrashSignalSchema, EventID: "dry"}}, func(req hostresurrect.Request) (int, error) {
		launched = append(launched, req)
		return 0, nil
	}, now, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(receipts) != 1 || len(launched) != 1 || receipts[0].Session != row.Handle || selections[0].Counts.Selected != 1 {
		t.Fatalf("receipts=%+v launched=%+v selections=%+v", receipts, launched, selections)
	}
	if _, err := os.Stat(hostResurrectionReceiptPath(logPath)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("dry run wrote receipt ledger: %v", err)
	}
}
