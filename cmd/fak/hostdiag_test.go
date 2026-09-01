package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostdiag"
	"github.com/anthony-chaudhary/fak/internal/shellprov"
)

func TestHostdiagHistoricalFixtureIsRetainedUnresolved(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "events.json")
	ledger := filepath.Join(dir, "hostdiag.jsonl")
	events := []hostdiag.ResourceEvent{{TimeMS: 1787710011683, Source: "Windows Error Reporting", EventID: 1001, RecordID: "111212", Name: "RADAR_PRE_LEAK_64", ReportID: "471a1974-e0e6-426d-86aa-4ed6533cde06", App: "fak.exe"}}
	data, _ := json.Marshal(events)
	if err := os.WriteFile(fixture, data, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runHostdiag(&stdout, &stderr, []string{"correlate", "--fixture", fixture, "--ledger", ledger}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var got hostdiag.Correlation
	if err := json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &got); err != nil || got.Status != "historical_unresolved" || !got.Observational || got.ReportID != events[0].ReportID {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	if raw, err := os.ReadFile(ledger); err != nil || !strings.Contains(string(raw), hostdiag.CorrelationSchema) {
		t.Fatalf("ledger=%q err=%v", raw, err)
	}
}

func TestHostdiagFixtureIdentifiesSpanningCensus(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "events.json")
	ledger := filepath.Join(dir, "hostdiag.jsonl")
	event := hostdiag.ResourceEvent{TimeMS: 2000, Source: "Windows Error Reporting", EventID: 1001, RecordID: "2", Name: "RADAR_PRE_LEAK_64", App: "fak.exe"}
	data, _ := json.Marshal([]hostdiag.ResourceEvent{event})
	_ = os.WriteFile(fixture, data, 0o600)
	sample := hostdiag.NewProcessSample(timeUnixMilli(3000), 42, timeUnixMilli(1000), `C:\bin\fak.exe`, "sha", "rev", "guard", "g1", 10, 20, 0, 0)
	if err := appendHostdiagRow(ledger, sample, 4096); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runHostdiag(&stdout, &stderr, []string{"correlate", "--fixture", fixture, "--ledger", ledger, "--max-bytes", "4096"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var got hostdiag.Correlation
	_ = json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &got)
	if got.Status != "identified" || len(got.Candidates) != 1 || got.Candidates[0].CommandClass != "guard" {
		t.Fatalf("%+v", got)
	}
}

func TestAppendHostdiagRowBoundsCompleteLines(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hostdiag.jsonl")
	for i := 1; i <= 40; i++ {
		row := hostdiag.Correlation{Schema: hostdiag.CorrelationSchema, CorrelationID: strings.Repeat("x", 60) + string(rune(i)), TimeMS: int64(i), Status: "historical_unresolved", Observational: true}
		if err := appendHostdiagRow(path, row, 700); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) > 700 || raw[len(raw)-1] != '\n' {
		t.Fatalf("bytes=%d err=%v", len(raw), err)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("malformed retained row: %v", err)
		}
	}
}

func TestAppendHostdiagRowConcurrentComplete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hostdiag.jsonl")
	const writers = 12
	start := make(chan struct{})
	errs := make(chan error, writers)
	var group sync.WaitGroup
	for i := range writers {
		group.Add(1)
		go func() {
			defer group.Done()
			<-start
			errs <- appendHostdiagRow(path, hostdiag.Correlation{Schema: hostdiag.CorrelationSchema, CorrelationID: fmt.Sprintf("c-%d", i), TimeMS: int64(i + 1), Status: "historical_unresolved", Observational: true}, 4096)
		}()
	}
	close(start)
	group.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != writers {
		t.Fatalf("rows=%d want=%d", len(lines), writers)
	}
	for _, line := range lines {
		if !json.Valid([]byte(line)) {
			t.Fatalf("malformed row %q", line)
		}
	}
}

func timeUnixMilli(ms int64) time.Time { return time.UnixMilli(ms) }

func TestHostdiagLifecycleFixtureIsObservedWithoutRawMessage(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "events.json")
	ledger := filepath.Join(dir, "hostdiag.jsonl")
	event := hostdiag.ResourceEvent{TimeMS: 2000, Source: "User32", EventID: 1074, RecordID: "135959", Name: "HOST_RESTART_INITIATED", Message: "private user and process details"}
	data, err := json.Marshal([]hostdiag.ResourceEvent{event})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fixture, data, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runHostdiag(&stdout, &stderr, []string{"correlate", "--fixture", fixture, "--ledger", ledger}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"event_name":"HOST_RESTART_INITIATED"`) || !strings.Contains(stdout.String(), `"status":"observed"`) {
		t.Fatalf("missing lifecycle observation: %s", stdout.String())
	}
	if strings.Contains(stdout.String(), "private user and process details") || strings.Contains(stdout.String(), "owned_shell_launch") || strings.Contains(stdout.String(), `"candidates"`) {
		t.Fatalf("lifecycle row leaked raw text or attribution: %s", stdout.String())
	}
}
func TestHostdiagWindowsShellFixtureOmitsRawMessageAndAttribution(t *testing.T) {
	dir := t.TempDir()
	fixture := filepath.Join(dir, "events.json")
	ledger := filepath.Join(dir, "hostdiag.jsonl")
	event := hostdiag.ResourceEvent{
		TimeMS: 2000, Source: "Application Error", EventID: 1000, RecordID: "111218",
		Name: "WINDOWS_SHELL_PROCESS_CRASH", ReportID: "aca4a57c", App: "Explorer.EXE",
		ProcessID: 11060, ProcessStartMS: 1000, Message: "private rendered Windows event text",
		Fault: &hostdiag.ApplicationFault{AppVersion: "10.0.26100.8875", Module: "SystemTray.dll", ExceptionCode: "c0000409"},
	}
	data, err := json.Marshal([]hostdiag.ResourceEvent{event})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fixture, data, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runHostdiag(&stdout, &stderr, []string{"correlate", "--fixture", fixture, "--ledger", ledger}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"event_name":"WINDOWS_SHELL_PROCESS_CRASH"`) || !strings.Contains(stdout.String(), `"status":"historical_unresolved"`) {
		t.Fatalf("missing typed unresolved crash: %s", stdout.String())
	}
	if strings.Contains(stdout.String(), "private rendered Windows event text") || strings.Contains(stdout.String(), "owned_shell_launch") || strings.Contains(stdout.String(), `"candidates"`) {
		t.Fatalf("shell crash leaked raw text or attribution: %s", stdout.String())
	}
}

func TestHostdiagCorrelatesExactOwnedShellReceipt(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "hostdiag.jsonl")
	fixture := filepath.Join(dir, "events.json")
	provenance := filepath.Join(dir, "shell.jsonl")
	event := hostdiag.ResourceEvent{TimeMS: 2000, Source: "Application Error", EventID: 1000, RecordID: "1", Name: "POWERSHELL_PROCESS_CRASH", App: "powershell.exe", ProcessID: 42, ProcessStartMS: 1000}
	data, err := json.Marshal([]hostdiag.ResourceEvent{event})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fixture, data, 0o600); err != nil {
		t.Fatal(err)
	}
	receipt, err := shellprov.New(time.UnixMilli(1100), shellprov.Fields{ParentPID: 7, ChildPID: 42, ChildCreatedUTCMS: 1000, LaunchClass: shellprov.LaunchProbe, ShellImage: shellprov.ShellPowerShell, ShellEdition: shellprov.EditionDesktop, ShellVersion: "5.1", Outcome: shellprov.OutcomeFailed, ErrorClass: shellprov.ErrorConsoleFault})
	if err != nil {
		t.Fatal(err)
	}
	if err := shellprov.Append(provenance, receipt, 10); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runHostdiag(&stdout, &stderr, []string{"correlate", "--fixture", fixture, "--ledger", ledger, "--shell-provenance", provenance}); rc != 0 {
		t.Fatalf("runHostdiag rc=%d stderr=%s", rc, stderr.String())
	}
	var got hostdiag.Correlation
	if err := json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &got); err != nil {
		t.Fatal(err)
	}
	if got.Status != "identified" || got.OwnedLaunch == nil || got.OwnedLaunch.ChildPID != 42 || got.OwnedLaunch.ChildCreatedUTCMS != 1000 {
		t.Fatalf("correlation = %+v", got)
	}
	if strings.Contains(stdout.String(), "argv") || strings.Contains(stdout.String(), "secret") {
		t.Fatalf("correlation leaked launch content: %s", stdout.String())
	}
}

func TestLoadOwnedShellLaunchesRejectsMalformedReceipt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "shell.jsonl")
	if err := os.WriteFile(path, []byte("{\"schema\":\"foreign\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadOwnedShellLaunches(path); err == nil {
		t.Fatal("accepted foreign shell receipt")
	}
}

func TestHostdiagTrendFlagsAndPrivacySafeJSON(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "hostdiag.jsonl")
	now := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	row := hostdiag.Correlation{Schema: hostdiag.CorrelationSchema, CorrelationID: "one", TimeMS: now.Add(-time.Hour).UnixMilli(), EventName: "APP_CRASH", App: `C:\private\fak.exe`, ReportID: "sensitive-report"}
	data, _ := json.Marshal(row)
	if err := os.WriteFile(ledger, append(data, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runHostdiag(&stdout, &stderr, []string{"trend", "--ledger", ledger, "--recent", "24h", "--baseline", "48h", "--now", now.Format(time.RFC3339)}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s ledger=%s", rc, stderr.String(), data)
	}
	var got hostdiag.Trend
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil || got.Recent.Total != 1 || got.Recent.Crash != 1 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	if strings.Contains(stdout.String(), "sensitive-report") || strings.Contains(stdout.String(), "private") || strings.Contains(stdout.String(), "report_id") {
		t.Fatalf("sensitive output: %s", stdout.String())
	}
}

func TestHostdiagTrendRejectsBadFlagsAndMalformedLedger(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "hostdiag.jsonl")
	_ = os.WriteFile(ledger, []byte(`{"schema":"`+hostdiag.CorrelationSchema+`"}`+"\n"), 0o600)
	for _, args := range [][]string{
		{"trend", "--ledger", ledger, "--recent", "0s"},
		{"trend", "--ledger", ledger, "--now", "not-time"},
		{"trend", "--ledger", ledger},
	} {
		var stdout, stderr bytes.Buffer
		if rc := runHostdiag(&stdout, &stderr, args); rc == 0 {
			t.Fatalf("accepted args=%v output=%s", args, stdout.String())
		}
	}
}

func TestHostdiagMacOSResourceIncidentFixtureReplay(t *testing.T) {
	source, err := os.ReadFile(filepath.Join("..", "..", "internal", "hostdiag", "testdata", "macos-resource-incident.diag"))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	fixture := filepath.Join(dir, "private-user-private-host.diag")
	ledger := filepath.Join(dir, "hostdiag.jsonl")
	if err := os.WriteFile(fixture, source, 0o600); err != nil {
		t.Fatal(err)
	}
	event, err := hostdiag.ParseMacOSResourceIncident(filepath.Base(fixture), source)
	if err != nil {
		t.Fatal(err)
	}
	sample := hostdiag.NewProcessSample(
		time.UnixMilli(event.TimeMS+1000), 20870, time.UnixMilli(event.TimeMS-60_000),
		"/usr/local/bin/fak", "sha", "rev", "hostdiag", "session", 1, 1, 1, 1,
	)
	if err := appendHostdiagRow(ledger, sample, 1<<20); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if rc := runHostdiag(&stdout, &stderr, []string{"correlate", "--fixture", fixture, "--ledger", ledger}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var got hostdiag.Correlation
	if err := json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &got); err != nil {
		t.Fatal(err)
	}
	if got.EventName != hostdiag.MacOSResourceIncidentEventName ||
		got.Status != "historical_unresolved" ||
		!got.Observational ||
		got.Correlated ||
		len(got.Candidates) != 0 ||
		got.ResourceIncident == nil ||
		got.ResourceIncident.Artifact.Basename != "macos-resource-incident.diag" {
		t.Fatalf("correlation = %+v", got)
	}
	rawLedger, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	for _, rendered := range []string{stdout.String(), string(rawLedger)} {
		for _, forbidden := range []string{
			`"windows_event_id"`, `"application_fault"`, `"application_hang"`,
			"private-user", "private-host", "worker_loop", "flush_buffer",
		} {
			if strings.Contains(rendered, forbidden) {
				t.Fatalf("fixture replay retained %q: %s", forbidden, rendered)
			}
		}
	}
}

func TestHostdiagMacOSResourceIncidentFixtureRefusesMalformedAndOversize(t *testing.T) {
	dir := t.TempDir()
	malformed := filepath.Join(dir, "malformed.diag")
	if err := os.WriteFile(malformed, []byte("Event: disk writes\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	oversize := filepath.Join(dir, "oversize.diag")
	file, err := os.Create(oversize)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Truncate(hostdiag.MacOSDiagFixtureMaxBytes + 1); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	for _, fixture := range []string{malformed, oversize} {
		var stdout, stderr bytes.Buffer
		if rc := runHostdiag(&stdout, &stderr, []string{
			"correlate", "--fixture", fixture, "--ledger", filepath.Join(dir, "ledger.jsonl"),
		}); rc == 0 {
			t.Fatalf("accepted fixture %q: %s", fixture, stdout.String())
		}
		if stdout.Len() != 0 || !strings.Contains(stderr.String(), "events:") {
			t.Fatalf("fixture %q stdout=%q stderr=%q", fixture, stdout.String(), stderr.String())
		}
	}
}
