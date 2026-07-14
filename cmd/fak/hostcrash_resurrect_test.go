package main

import (
	"errors"
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
	got, err := resurrectHostCrashSessions(logPath, dir, []hostfault.HostCrashSignal{signal}, launch, now)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].PID != 4242 || launches != 1 {
		t.Fatalf("receipts=%+v launches=%d", got, launches)
	}
	got, err = resurrectHostCrashSessions(logPath, dir, []hostfault.HostCrashSignal{signal}, launch, now.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 || launches != 1 {
		t.Fatalf("duplicate receipts=%+v launches=%d", got, launches)
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
	_ = guardsessions.Record(dir, row)
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "new", Class: hostfault.HostCrashGeneric}
	got, err := resurrectHostCrashSessions(logPath, dir, []hostfault.HostCrashSignal{sig}, func(hostresurrect.Request) (int, error) { t.Fatal("launch called above cap"); return 0, nil }, now)
	if err != nil || len(got) != 0 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}
func TestResurrectHostCrashSessionsFailedLaunchIsStillDeduped(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 7, 14, 21, 0, 0, 0, time.UTC)
	row := guardsessions.NewInteractiveRow("trace-fail", "claude", 1, dir, "", "", now, []string{"claude"})
	if err := guardsessions.Record(dir, row); err != nil {
		t.Fatal(err)
	}
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt-fail", Class: hostfault.HostCrashGeneric}
	calls := 0
	launcher := func(req hostresurrect.Request) (int, error) { calls++; return 0, errors.New("spawn failed") }
	if _, err := resurrectHostCrashSessions(filepath.Join(dir, "host.jsonl"), dir, []hostfault.HostCrashSignal{sig}, launcher, now); err == nil {
		t.Fatal("want launch error")
	}
	got, err := resurrectHostCrashSessions(filepath.Join(dir, "host.jsonl"), dir, []hostfault.HostCrashSignal{sig}, launcher, now.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 || calls != 1 {
		t.Fatalf("retry relaunched reserved session: receipts=%+v calls=%d", got, calls)
	}
}
