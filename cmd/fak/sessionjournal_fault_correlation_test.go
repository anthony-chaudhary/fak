package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

func TestCorrelateSessionJournalCrashesNamesTerminalCause(t *testing.T) {
	at := time.Now().UTC().Add(-time.Minute)
	rows := []sessionjournal.Classified{{
		Session: sessionjournal.Session{ID: "crashed-thread", LastSeen: at},
		Status:  sessionjournal.StatusCrashed, Reason: "MACHINE_REBOOT",
	}}
	events := []toolprocgate.ConsoleFaultEvent{
		{AtMS: at.Add(30 * time.Second).UnixMilli(), Tool: "WindowsTerminal", Class: toolprocgate.ConsoleRendererExit, Detail: "faulting_app=WindowsTerminal.exe faulting_module=Microsoft.Terminal.Control.dll exception_code=0xc0000005", Surface: string(toolprocgate.ConsoleSurfaceRenderer)},
		{AtMS: at.Add(10 * time.Minute).UnixMilli(), Tool: "pwsh", Class: toolprocgate.ConsoleRendererExit, Detail: "unrelated", Surface: string(toolprocgate.ConsoleSurfaceRenderer)},
		{AtMS: at.UnixMilli(), Tool: "pwsh", Class: toolprocgate.ConsoleHostFailFast, Detail: "duplicate dotnet surface", Surface: "host"},
	}
	got := matchEvent1000Crashes(rows, events, event1000WindowMS)
	if len(got) != 1 || got[0].SessionID != "crashed-thread" || got[0].Cause != "WINDOWS_TERMINAL_CRASH" || got[0].Source != "windows_eventlog_1000" {
		t.Fatalf("causes=%+v", got)
	}
}

func TestSessionJournalCauseNamesWinAppRuntime(t *testing.T) {
	event := toolprocgate.ConsoleFaultEvent{Tool: "WindowsTerminal", Detail: "faulting_module=Microsoft.WinAppRuntime.dll"}
	if got := event1000CauseName(event); got != "WINAPPRUNTIME_CRASH" {
		t.Fatalf("cause=%q", got)
	}
}

func TestSessionJournalReportNamesEvent1000Cause(t *testing.T) {
	oldRecords := readEvent1000Records
	defer func() { readEvent1000Records = oldRecords }()
	at := time.Now().UTC().Add(-time.Minute)
	path := filepath.Join(t.TempDir(), "journal.jsonl")
	if err := sessionjournal.Append(path, sessionjournal.Event{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "fleet-thread", PID: 0, CWD: `C:\work\fak`, TS: at.Format(time.RFC3339Nano), Boot: "old-boot"}); err != nil {
		t.Fatal(err)
	}
	readEvent1000Records = func(time.Duration) ([]winEventRecord, string) {
		return []winEventRecord{{Provider: "Application Error", ID: 1000, TimeMS: at.Add(15 * time.Second).UnixMilli(), Message: "Faulting application name: WindowsTerminal.exe, version: 1.0\nFaulting module name: Microsoft.Terminal.Control.dll\nException code: 0xc0000005\nFault offset: 0x1"}}, ""
	}
	var out, errOut bytes.Buffer
	code := runSessionJournal(&out, &errOut, []string{"report", "--path", path, "--no-guard-sessions", "--boot-time", at.Add(30 * time.Minute).Format(time.RFC3339), "--causes"})
	if code != 0 || !strings.Contains(out.String(), "WINDOWS_TERMINAL_CRASH") || !strings.Contains(out.String(), "fleet-thread") || !strings.Contains(out.String(), "0xc0000005") {
		t.Fatalf("code=%d out=%q err=%q", code, out.String(), errOut.String())
	}
}
