package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hostfault"
)

func TestHostCrashClassifiesTermServiceAndGeneric(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "events.json")
	logPath := filepath.Join(dir, "signals.jsonl")
	events := []hostfault.ApplicationError1000{
		{TimeMS: 1700000000000, App: "svchost.exe", Module: "amdxx64.dll", Exception: "0xc0000005", ProcessID: "456", ReportID: "r-amd"},
		{TimeMS: 1700000001000, App: "OpenConsole.exe", Module: "future.dll", Exception: "0xdeadbeef", ReportID: "r-generic"},
	}
	b, _ := json.Marshal(events)
	if err := os.WriteFile(fixture, b, 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errout strings.Builder
	if rc := runHostCrash(&out, &errout, []string{"--once", "--fixture", fixture, "--log", logPath}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errout.String())
	}
	for _, want := range []string{`"class":"TERMSERVICE_AMD_AV"`, `"host_pid":456`, `"class":"HOST_CRASH_GENERIC"`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %s: %s", want, out.String())
		}
	}
}
func TestHostCrashFixtureEmitsExactlyOnce(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "events.json")
	logPath := filepath.Join(dir, "signals.jsonl")
	events := []hostfault.ApplicationError1000{
		{TimeMS: 1700000000000, App: "WindowsTerminal.exe", Module: "Microsoft.Terminal.Control.dll", Exception: "0xc0000005", FaultOffset: "0x2c924", ProcessID: "0x1234", ReportID: "r1"},
		{TimeMS: 1700000001000, App: "notepad.exe", Module: "other.dll", Exception: "0xc0000005"},
	}
	b, _ := json.Marshal(events)
	if err := os.WriteFile(fixture, b, 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errout strings.Builder
	args := []string{"--once", "--fixture", fixture, "--log", logPath}
	if rc := runHostCrash(&out, &errout, args); rc != 0 {
		t.Fatalf("first rc=%d stderr=%s", rc, errout.String())
	}
	if strings.Count(strings.TrimSpace(out.String()), "\n") != 0 || !strings.Contains(out.String(), `"class":"WT_RENDER_AV"`) {
		t.Fatalf("unexpected first output: %s", out.String())
	}
	out.Reset()
	errout.Reset()
	if rc := runHostCrash(&out, &errout, args); rc != 0 {
		t.Fatalf("second rc=%d stderr=%s", rc, errout.String())
	}
	if out.Len() != 0 {
		t.Fatalf("duplicate signal emitted: %s", out.String())
	}
	rows, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(strings.TrimSpace(string(rows)), "\n") != 0 {
		t.Fatalf("want exactly one durable row: %s", rows)
	}
}
func TestHostCrashFixtureRejectsCorruptExistingLedger(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "events.json")
	logPath := filepath.Join(dir, "signals.jsonl")
	events := []hostfault.ApplicationError1000{{TimeMS: 1, App: "WindowsTerminal.exe", Module: "TerminalApp.dll", Exception: "0xc0000005"}}
	b, _ := json.Marshal(events)
	if err := os.WriteFile(fixture, b, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(logPath, []byte("not-json\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errout strings.Builder
	if rc := runHostCrash(&out, &errout, []string{"--once", "--fixture", fixture, "--log", logPath}); rc != 1 {
		t.Fatalf("rc=%d want 1; stdout=%s stderr=%s", rc, out.String(), errout.String())
	}
	if !strings.Contains(errout.String(), "parse existing signal log") {
		t.Fatalf("missing fail-closed diagnostic: %s", errout.String())
	}
}
func TestAppendNewHostCrashSignalsConcurrentExactOnce(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "signals.jsonl")
	event := hostfault.ApplicationError1000{TimeMS: 1700000000000, App: "WindowsTerminal.exe", Module: "TerminalApp.dll", Exception: "0xc0000005", ReportID: "same"}
	const writers = 12
	start := make(chan struct{})
	results := make(chan int, writers)
	errs := make(chan error, writers)
	for i := 0; i < writers; i++ {
		go func() {
			<-start
			fresh, err := appendNewHostCrashSignals(logPath, []hostfault.ApplicationError1000{event})
			if err != nil {
				errs <- err
				return
			}
			results <- len(fresh)
		}()
	}
	close(start)
	emitted := 0
	for i := 0; i < writers; i++ {
		select {
		case err := <-errs:
			t.Fatal(err)
		case n := <-results:
			emitted += n
		}
	}
	if emitted != 1 {
		t.Fatalf("concurrent emitted=%d, want exactly 1", emitted)
	}
	rows, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(strings.Fields(strings.TrimSpace(string(rows)))); got == 0 {
		t.Fatal("empty durable signal log")
	}
	if got := strings.Count(strings.TrimSpace(string(rows)), "\n") + 1; got != 1 {
		t.Fatalf("durable rows=%d, want 1: %s", got, rows)
	}
}
