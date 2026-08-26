package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hostfault"
)

func testHostSystemEvents() []hostfault.WindowsSystemEvent {
	return []hostfault.WindowsSystemEvent{
		{TimeMS: 1, Source: "Microsoft-Windows-WER-SystemErrorReporting", WindowsID: 1001, RecordID: "1", BugcheckCode: "0x4e", DumpPath: `C:\Windows\Minidump\x.dmp`},
		{TimeMS: 2, Source: "Microsoft-Windows-Kernel-Power", WindowsID: 41, RecordID: "2"},
		{TimeMS: 3, Source: "EventLog", WindowsID: 6008, RecordID: "3"},
	}
}

func TestAppendNewHostSystemIncidentsDedupesAndStaysObservational(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "host.jsonl")
	got, err := appendNewHostSystemIncidents(logPath, testHostSystemEvents())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d", len(got))
	}
	for _, incident := range got {
		if !incident.Observational {
			t.Fatalf("actuating incident %+v", incident)
		}
	}
	again, err := appendNewHostSystemIncidents(logPath, testHostSystemEvents())
	if err != nil || len(again) != 0 {
		t.Fatalf("again=%v err=%v", again, err)
	}
}

func TestAppendNewHostSystemIncidentsMixedSchemaLedger(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "host.jsonl")
	app := hostfault.ApplicationError1000{TimeMS: 10, App: "WindowsTerminal.exe", Module: "TerminalApp.dll", Exception: "0xc0000005", ReportID: "app"}
	if got, err := appendNewHostCrashSignals(logPath, []hostfault.ApplicationError1000{app}); err != nil || len(got) != 1 {
		t.Fatalf("app got=%v err=%v", got, err)
	}
	if got, err := appendNewHostSystemIncidents(logPath, testHostSystemEvents()[:1]); err != nil || len(got) != 1 {
		t.Fatalf("system got=%v err=%v", got, err)
	}
	if got, err := appendNewHostCrashSignals(logPath, []hostfault.ApplicationError1000{app}); err != nil || len(got) != 0 {
		t.Fatalf("app dedupe got=%v err=%v", got, err)
	}
	rows, err := os.ReadFile(logPath)
	if err != nil || strings.Count(strings.TrimSpace(string(rows)), "\n")+1 != 2 {
		t.Fatalf("ledger=%q err=%v", rows, err)
	}
}

func TestAppendNewHostSystemIncidentsConcurrentExactOnce(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "host.jsonl")
	const writers = 12
	start := make(chan struct{})
	results := make(chan int, writers)
	errs := make(chan error, writers)
	var group sync.WaitGroup
	for range writers {
		group.Add(1)
		go func() {
			defer group.Done()
			<-start
			fresh, err := appendNewHostSystemIncidents(logPath, testHostSystemEvents()[:1])
			if err != nil {
				errs <- err
				return
			}
			results <- len(fresh)
		}()
	}
	close(start)
	group.Wait()
	close(results)
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	emitted := 0
	for count := range results {
		emitted += count
	}
	if emitted != 1 {
		t.Fatalf("emitted=%d, want 1", emitted)
	}
}

func TestAppendNewHostSystemIncidentsBoundsCompleteLines(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "host.jsonl")
	for i := int64(1); i <= 30; i++ {
		event := hostfault.WindowsSystemEvent{TimeMS: i, Source: "EventLog", WindowsID: 6008, RecordID: string(rune('a' + i%26)), Message: strings.Repeat("x", 80)}
		if _, err := appendNewHostSystemIncidents(logPath, []hostfault.WindowsSystemEvent{event}, 700); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if int64(len(raw)) > 700 || len(raw) == 0 || raw[len(raw)-1] != '\n' {
		t.Fatalf("invalid bounded ledger bytes=%d", len(raw))
	}
	for i, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var row hostfault.SystemIncident
		if err := json.Unmarshal([]byte(line), &row); err != nil || row.Schema != hostfault.SystemIncidentSchema {
			t.Fatalf("row %d malformed: %v %q", i, err, line)
		}
	}
}

func TestHostCrashFixtureModeDoesNotQueryUnspecifiedCollector(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "system.json")
	logPath := filepath.Join(dir, "host.jsonl")
	data, _ := json.Marshal(testHostSystemEvents()[:1])
	if err := os.WriteFile(fixture, data, 0o600); err != nil {
		t.Fatal(err)
	}
	var stderr strings.Builder
	if rc := runHostCrash(&discardWriter{}, &stderr, []string{"--once", "--system-fixture", fixture, "--log", logPath}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
}

func TestHostCrashSystemFixtureRejectsMalformedLedger(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "system.json")
	logPath := filepath.Join(dir, "host.jsonl")
	data, _ := json.Marshal(testHostSystemEvents()[:1])
	if err := os.WriteFile(fixture, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(logPath, []byte("not-json\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var stderr strings.Builder
	if rc := runHostCrash(&discardWriter{}, &stderr, []string{"--once", "--system-fixture", fixture, "--log", logPath}); rc != 1 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stderr.String(), "parse existing signal log") {
		t.Fatalf("missing fail-closed diagnostic: %s", stderr.String())
	}
}
func TestHostCrashSystemIncidentsNeverResurrect(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "system.json")
	logPath := filepath.Join(dir, "host.jsonl")
	data, _ := json.Marshal(testHostSystemEvents()[:1])
	if err := os.WriteFile(fixture, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if rc := runHostCrash(&discardWriter{}, &discardWriter{}, []string{"--once", "--system-fixture", fixture, "--log", logPath, "--resurrect", "--dry-run", "--reg-dir", filepath.Join(dir, "reg")}); rc != 0 {
		t.Fatalf("rc=%d", rc)
	}
	if _, err := os.Stat(hostResurrectionReceiptPath(logPath)); !os.IsNotExist(err) {
		t.Fatalf("system incident created resurrection receipt: %v", err)
	}
}

type discardWriter struct{}

func (*discardWriter) Write(p []byte) (int, error) { return len(p), nil }
