package hostdiag

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

func TestCorrelateHistoricalUnresolved(t *testing.T) {
	event := ResourceEvent{TimeMS: 1000, Source: "Windows Error Reporting", EventID: 1001, RecordID: "1", Name: "RADAR_PRE_LEAK_64", ReportID: "r", App: "fak.exe"}
	got, ok := Correlate(event, nil)
	if !ok || got.Status != "historical_unresolved" || !got.Observational || got.CorrelationID == "" {
		t.Fatalf("%+v ok=%v", got, ok)
	}
	again, _ := Correlate(event, nil)
	if again.CorrelationID != got.CorrelationID {
		t.Fatal("unstable identity")
	}
}

func TestCorrelateIdentifiedAndAmbiguous(t *testing.T) {
	at := time.UnixMilli(2000)
	one := NewProcessSample(at, 42, time.UnixMilli(500), `C:\bin\fak.exe`, "sha", "rev", "guard", "g1", 10, 20, 3, 4)
	event := ResourceEvent{TimeMS: 1000, EventID: 1001, Name: "RADAR_PRE_LEAK_64", App: "fak.exe"}
	got, _ := Correlate(event, []ProcessSample{one})
	if got.Status != "identified" || len(got.Candidates) != 1 || got.Candidates[0].CommandClass != "guard" {
		t.Fatalf("%+v", got)
	}
	two := one
	two.PID = 43
	got, _ = Correlate(event, []ProcessSample{one, two})
	if got.Status != "ambiguous" || len(got.Candidates) != 2 {
		t.Fatalf("%+v", got)
	}
}

func TestParseLowVirtualMemoryCulprits(t *testing.T) {
	message := `The following programs consumed the most virtual memory: pwsh.exe (48308) consumed 277440532480 bytes, pwsh.exe (43544) consumed 222151114752 bytes, and worker-host.exe (38344) consumed 210724044800 bytes.`
	want := []ResourceCulprit{{Image: "pwsh.exe", PID: 48308, Bytes: 277440532480}, {Image: "pwsh.exe", PID: 43544, Bytes: 222151114752}, {Image: "worker-host.exe", PID: 38344, Bytes: 210724044800}}
	got := ParseLowVirtualMemoryCulprits(message)
	if len(got) != len(want) {
		t.Fatalf("got=%+v want=%+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("culprit[%d]=%+v want=%+v", i, got[i], want[i])
		}
	}
}

func TestParseLowVirtualMemoryCulpritsUnknownRendering(t *testing.T) {
	if got := ParseLowVirtualMemoryCulprits("Speicherwarnung: unbekanntes lokalisiertes Format"); len(got) != 0 {
		t.Fatalf("got=%+v want empty", got)
	}
	event := ResourceEvent{TimeMS: 1000, EventID: 2004, Name: "LOW_VIRTUAL_MEMORY", Message: "unparsed rendering"}
	got, ok := Correlate(event, nil)
	if !ok || got.Status != "historical_unresolved" || len(got.Culprits) != 0 {
		t.Fatalf("event was not retained: got=%+v ok=%v", got, ok)
	}
}

func TestCorrelateRetainsLowVirtualMemoryCulprits(t *testing.T) {
	event := ResourceEvent{TimeMS: 1000, Source: "Microsoft-Windows-Resource-Exhaustion-Detector", EventID: 2004, RecordID: "9", Name: "LOW_VIRTUAL_MEMORY", Culprits: []ResourceCulprit{{Image: "pwsh.exe", PID: 48308, Bytes: 277441708032}, {Image: "pwsh.exe", PID: 43544, Bytes: 222151004160}}}
	got, ok := Correlate(event, nil)
	if !ok || got.Status != "historical_unresolved" || len(got.Culprits) != 2 || got.Culprits[0].Bytes != 277441708032 || !got.Observational {
		t.Fatalf("got=%+v ok=%v", got, ok)
	}
}

func TestCorrelateRetainsPowerShellConsoleHostCrash(t *testing.T) {
	event := ResourceEvent{TimeMS: 1000, Source: "Application Error", EventID: 1000, RecordID: "110382", Name: "POWERSHELL_PROCESS_CRASH", App: "pwsh.exe", Fault: &ApplicationFault{AppVersion: "7.6.5.500", Module: "Microsoft.PowerShell.ConsoleHost.dll", ModuleVersion: "7.6.5.500", ExceptionCode: "80131623", FaultOffset: "000000000004d072"}}
	got, ok := Correlate(event, nil)
	if !ok || got.Status != "historical_unresolved" || got.Fault == nil || got.Fault.Module != "Microsoft.PowerShell.ConsoleHost.dll" || got.Fault.ExceptionCode != "80131623" || !got.Observational {
		t.Fatalf("got=%+v ok=%v", got, ok)
	}
}

func TestCorrelateRetainsBoundedGenericApplicationCrash(t *testing.T) {
	event := ResourceEvent{
		TimeMS: 1777214146000, Source: "Application Error", EventID: 1000, RecordID: "111576",
		Name: "WINDOWS_APPLICATION_PROCESS_CRASH", ReportID: "3aa7443d-7c82-4099-88c8-823bd032a4ea", App: "msiexec.exe",
		Fault: &ApplicationFault{AppVersion: "5.0.26100.8875", Module: "RPCRT4.dll", ModuleVersion: "10.0.26100.8875", ExceptionCode: "c0000005", FaultOffset: "0000000000016043"},
	}
	sample := ProcessSample{Schema: CensusSchema, PID: 7, ProcessStartMS: event.TimeMS - 1000, SampledAtMS: event.TimeMS + 1000, Executable: `C:\bin\fak.exe`}
	launch := OwnedShellLaunch{TimestampUTCMS: event.TimeMS - 100, ChildPID: 42, ChildCreatedUTCMS: event.TimeMS - 1000, Outcome: "failed"}
	got, ok := CorrelateWithOwnedLaunches(event, []ProcessSample{sample}, []OwnedShellLaunch{launch})
	if !ok || got.Status != "historical_unresolved" || !got.Observational || len(got.Candidates) != 0 || got.OwnedLaunch != nil {
		t.Fatalf("correlation = %+v, ok=%v", got, ok)
	}
	if got.EventName != event.Name || got.App != "msiexec.exe" || got.ReportID != event.ReportID || got.Fault == nil || *got.Fault != *event.Fault {
		t.Fatalf("bounded application fault not retained: %+v", got)
	}
	if !strings.Contains(got.Reason, "not attributed to fak") {
		t.Fatalf("reason = %q", got.Reason)
	}
	payload, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"message", "command_line", "argv", "owned_shell_launch", "candidates"} {
		if strings.Contains(string(payload), `"`+forbidden+`"`) {
			t.Fatalf("generic application crash retained %q in %s", forbidden, payload)
		}
	}
}

func TestCorrelateRejectsMalformedGenericApplicationCrash(t *testing.T) {
	valid := ResourceEvent{TimeMS: 1, Source: "Application Error", EventID: 1000, Name: "WINDOWS_APPLICATION_PROCESS_CRASH", App: "msiexec.exe", Fault: &ApplicationFault{Module: "RPCRT4.dll", ExceptionCode: "c0000005"}}
	tests := map[string]ResourceEvent{
		"missing time":   func() ResourceEvent { event := valid; event.TimeMS = 0; return event }(),
		"wrong source":   func() ResourceEvent { event := valid; event.Source = "Other"; return event }(),
		"wrong event ID": func() ResourceEvent { event := valid; event.EventID = 1001; return event }(),
		"missing app":    func() ResourceEvent { event := valid; event.App = ""; return event }(),
		"missing fault":  func() ResourceEvent { event := valid; event.Fault = nil; return event }(),
		"missing module": func() ResourceEvent {
			event := valid
			fault := *event.Fault
			fault.Module = ""
			event.Fault = &fault
			return event
		}(),
		"missing exception": func() ResourceEvent {
			event := valid
			fault := *event.Fault
			fault.ExceptionCode = ""
			event.Fault = &fault
			return event
		}(),
		"specialized app label": func() ResourceEvent { event := valid; event.App = "pwsh.exe"; return event }(),
	}
	for name, event := range tests {
		t.Run(name, func(t *testing.T) {
			if got, ok := Correlate(event, nil); ok {
				t.Fatalf("accepted malformed generic application crash: %+v", got)
			}
		})
	}
}

func TestCorrelateDoesNotAttributeOtherProcessEventsToFak(t *testing.T) {
	sample := NewProcessSample(time.UnixMilli(2000), 42, time.UnixMilli(500), `C:\bin\fak.exe`, "sha", "rev", "guard", "g1", 10, 20, 3, 4)
	for _, event := range []ResourceEvent{
		{TimeMS: 1000, EventID: 1000, Name: "POWERSHELL_PROCESS_CRASH", App: "pwsh.exe"},
		{TimeMS: 1000, EventID: 2004, Name: "LOW_VIRTUAL_MEMORY", Culprits: []ResourceCulprit{{Image: "pwsh.exe", PID: 42, Bytes: 100}}},
		{TimeMS: 1000, EventID: 1014, Name: "RESOURCE_EXHAUSTION_1014"},
	} {
		got, ok := Correlate(event, []ProcessSample{sample})
		if !ok || got.Status != "historical_unresolved" || len(got.Candidates) != 0 {
			t.Fatalf("event=%+v got=%+v ok=%v", event, got, ok)
		}
	}
}

func TestCorrelateLowVirtualMemoryRequiresNamedFakPID(t *testing.T) {
	sample := NewProcessSample(time.UnixMilli(2000), 42, time.UnixMilli(500), `C:\bin\fak.exe`, "sha", "rev", "guard", "g1", 10, 20, 3, 4)
	event := ResourceEvent{TimeMS: 1000, EventID: 2004, Name: "LOW_VIRTUAL_MEMORY", Culprits: []ResourceCulprit{{Image: "fak.exe", PID: 42, Bytes: 100}}}
	got, ok := Correlate(event, []ProcessSample{sample})
	if !ok || got.Status != "identified" || len(got.Candidates) != 1 || got.Candidates[0].PID != 42 {
		t.Fatalf("got=%+v ok=%v", got, ok)
	}
}

func TestCorrelateRejectsUnrelated(t *testing.T) {
	for _, event := range []ResourceEvent{{TimeMS: 1, EventID: 1000, Name: "POWERSHELL_PROCESS_CRASH", App: "other.exe"}, {TimeMS: 1, EventID: 1001, Name: "APPCRASH", App: "fak.exe"}, {TimeMS: 1, EventID: 1001, Name: "RADAR_PRE_LEAK_64", App: "other.exe"}, {Name: "RADAR_PRE_LEAK_64", App: "fak.exe"}} {
		if _, ok := Correlate(event, nil); ok {
			t.Fatalf("accepted %+v", event)
		}
	}
}

func TestClassifyCommandUsesClosedAllowlist(t *testing.T) {
	tests := []struct {
		commandLine string
		want        string
	}{
		{`C:\bin\fak.exe guard --api-key secret`, "guard"},
		{`C:\bin\fak.exe model inspect --token secret`, "model"},
		{`C:\bin\fak.exe bench run --credential secret`, "bench"},
		{`C:\bin\fak.exe test ./internal/hostdiag`, "test"},
		{`C:\bin\fak.exe validate --mine internal/hostdiag`, "validate"},
		{`C:\bin\fak.exe hostdiag census --ledger private.jsonl`, "hostdiag"},
		{`C:\bin\fak.exe unusual --token secret`, "other"},
		{`C:\bin\fak.exe api-key secret`, "other"},
		{`C:\bin\fak.exe secret-token`, "other"},
	}
	for _, test := range tests {
		t.Run(test.want+"/"+test.commandLine, func(t *testing.T) {
			if got := ClassifyCommand(test.commandLine); got != test.want {
				t.Fatalf("ClassifyCommand(%q) = %q, want %q", test.commandLine, got, test.want)
			}
		})
	}
}

func TestProcessSampleJSONRoundTripOmitsCommandLine(t *testing.T) {
	commandLine := `C:\bin\fak.exe model inspect --api-key census-secret`
	sample := NewProcessSample(time.UnixMilli(2000), 42, time.UnixMilli(1000), `C:\bin\fak.exe`, "sha", "rev", ClassifyCommand(commandLine), "session", 10, 20, 3, 4)

	payload, err := json.Marshal(sample)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip ProcessSample
	if err := json.Unmarshal(payload, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if roundTrip.CommandClass != "model" {
		t.Fatalf("command class = %q, want model", roundTrip.CommandClass)
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(payload, &fields); err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"argv", "command_line", "commandLine"} {
		if _, ok := fields[forbidden]; ok {
			t.Fatalf("privacy-sensitive field %q persisted in %s", forbidden, payload)
		}
	}
	if strings.Contains(string(payload), "census-secret") || strings.Contains(string(payload), "--api-key") {
		t.Fatalf("command arguments persisted in %s", payload)
	}
}

func TestHostLifecycleSignalsRemainObservedAndUnattributed(t *testing.T) {
	tests := []ResourceEvent{
		{TimeMS: 1000, Source: "User32", EventID: 1074, RecordID: "135959", Name: "HOST_RESTART_INITIATED"},
		{TimeMS: 2000, Source: "EventLog", EventID: 6008, RecordID: "135960", Name: "HOST_UNEXPECTED_SHUTDOWN"},
		{TimeMS: 3000, Source: "Microsoft-Windows-Kernel-Power", EventID: 41, RecordID: "135961", Name: "HOST_UNCLEAN_RESTART"},
	}
	sample := ProcessSample{Schema: CensusSchema, PID: 7, ProcessStartMS: 500, SampledAtMS: 4000, Executable: `C:\bin\fak.exe`}
	launch := OwnedShellLaunch{TimestampUTCMS: 900, ParentPID: 7, ChildPID: 42, ChildCreatedUTCMS: 800, Outcome: "failed"}
	for _, event := range tests {
		got, ok := CorrelateWithOwnedLaunches(event, []ProcessSample{sample}, []OwnedShellLaunch{launch})
		if !ok || got.Status != "observed" || got.OwnedLaunch != nil || len(got.Candidates) != 0 {
			t.Fatalf("%s correlation = %+v, ok=%v", event.Name, got, ok)
		}
		if !got.Observational || !strings.Contains(got.Reason, "without attributing cause") {
			t.Fatalf("%s reason = %q observational=%v", event.Name, got.Reason, got.Observational)
		}
	}
}

func TestHostLifecycleRejectsProviderAndIDLookalikes(t *testing.T) {
	tests := []ResourceEvent{
		{TimeMS: 1, Source: "Other", EventID: 1074, Name: "HOST_RESTART_INITIATED"},
		{TimeMS: 1, Source: "User32", EventID: 6008, Name: "HOST_UNEXPECTED_SHUTDOWN"},
		{TimeMS: 1, Source: "Microsoft-Windows-Kernel-Power", EventID: 42, Name: "HOST_UNCLEAN_RESTART"},
	}
	for _, event := range tests {
		if _, ok := Correlate(event, nil); ok {
			t.Fatalf("accepted lifecycle lookalike: %+v", event)
		}
	}
}
func TestWindowsShellCrashRemainsUnattributed(t *testing.T) {
	event := ResourceEvent{
		TimeMS: 2000, Source: "Application Error", EventID: 1000, RecordID: "111218",
		Name: "WINDOWS_SHELL_PROCESS_CRASH", ReportID: "aca4a57c", App: "Explorer.EXE",
		ProcessID: 11060, ProcessStartMS: 1000,
		Fault: &ApplicationFault{AppVersion: "10.0.26100.8875", Module: "SystemTray.dll", ModuleVersion: "2605.22002.100.0", ExceptionCode: "c0000409", FaultOffset: "000000000018253f"},
	}
	sample := ProcessSample{Schema: CensusSchema, PID: 7, ProcessStartMS: 500, SampledAtMS: 3000, Executable: `C:\bin\fak.exe`}
	launch := OwnedShellLaunch{TimestampUTCMS: 1900, ParentPID: 7, ChildPID: 11060, ChildCreatedUTCMS: 1000, Outcome: "failed"}
	got, ok := CorrelateWithOwnedLaunches(event, []ProcessSample{sample}, []OwnedShellLaunch{launch})
	if !ok || got.Status != "historical_unresolved" || got.OwnedLaunch != nil || len(got.Candidates) != 0 {
		t.Fatalf("correlation = %+v, ok=%v", got, ok)
	}
	if got.EventName != "WINDOWS_SHELL_PROCESS_CRASH" || got.Fault == nil || got.Fault.Module != "SystemTray.dll" {
		t.Fatalf("typed shell fault not retained: %+v", got)
	}
	if !strings.Contains(got.Reason, "not attributed to fak") {
		t.Fatalf("reason = %q", got.Reason)
	}
}

func TestWindowsShellCrashRejectsOtherApplications(t *testing.T) {
	_, ok := Correlate(ResourceEvent{TimeMS: 1, EventID: 1000, Name: "WINDOWS_SHELL_PROCESS_CRASH", App: "other.exe"}, nil)
	if ok {
		t.Fatal("accepted non-Explorer Windows shell crash")
	}
}
func TestPowerShellCrashIdentifiesExactOwnedLaunch(t *testing.T) {
	event := ResourceEvent{TimeMS: 2000, Source: "Application Error", EventID: 1000, Name: "POWERSHELL_PROCESS_CRASH", App: "powershell.exe", ProcessID: 42, ProcessStartMS: 1000}
	launch := OwnedShellLaunch{ParentPID: 7, ChildPID: 42, ChildCreatedUTCMS: 1000, LaunchID: "sha256:test", LaunchClass: "probe", ShellImage: "powershell", ShellEdition: "desktop", ShellVersion: "5.1", Outcome: "failed", ErrorClass: "console_fault"}
	got, ok := CorrelateWithOwnedLaunches(event, nil, []OwnedShellLaunch{launch})
	if !ok || got.Status != "identified" || got.OwnedLaunch == nil {
		t.Fatalf("correlation = %+v, ok=%v", got, ok)
	}
	if got.OwnedLaunch.ChildPID != 42 || got.OwnedLaunch.ChildCreatedUTCMS != 1000 || got.OwnedLaunch.ParentPID != 7 {
		t.Fatalf("owned launch = %+v", got.OwnedLaunch)
	}
}

func TestPowerShellCrashPrefersNewestTerminalReceipt(t *testing.T) {
	event := ResourceEvent{TimeMS: 2000, Source: "Application Error", EventID: 1000, Name: "POWERSHELL_PROCESS_CRASH", App: "powershell.exe", ProcessID: 42, ProcessStartMS: 1000}
	started := OwnedShellLaunch{TimestampUTCMS: 1100, ParentPID: 7, ChildPID: 42, ChildCreatedUTCMS: 1000, LaunchID: "sha256:test", Outcome: "started", ErrorClass: "none"}
	terminal := started
	terminal.TimestampUTCMS = 1900
	terminal.Outcome = "failed"
	terminal.ErrorClass = "console_fault"
	got, ok := CorrelateWithOwnedLaunches(event, nil, []OwnedShellLaunch{terminal, started})
	if !ok || got.OwnedLaunch == nil || got.OwnedLaunch.Outcome != "failed" || got.OwnedLaunch.ErrorClass != "console_fault" {
		t.Fatalf("correlation = %+v, ok=%v", got, ok)
	}
}
func TestPowerShellCrashRejectsPIDReuse(t *testing.T) {
	event := ResourceEvent{TimeMS: 3000, Source: "Application Error", EventID: 1000, Name: "POWERSHELL_PROCESS_CRASH", App: "powershell.exe", ProcessID: 42, ProcessStartMS: 2000}
	launch := OwnedShellLaunch{ParentPID: 7, ChildPID: 42, ChildCreatedUTCMS: 1000, LaunchID: "sha256:old", LaunchClass: "probe", ShellImage: "powershell", ShellEdition: "desktop", ShellVersion: "5.1", Outcome: "succeeded", ErrorClass: "none"}
	got, ok := CorrelateWithOwnedLaunches(event, nil, []OwnedShellLaunch{launch})
	if !ok || got.Status != "historical_unresolved" || got.OwnedLaunch != nil {
		t.Fatalf("PID-reused correlation = %+v, ok=%v", got, ok)
	}
	if !strings.Contains(got.Reason, "exact PID and process creation time") {
		t.Fatalf("reason = %q", got.Reason)
	}
}

func TestCorrelateRetainsBoundedApplicationHang(t *testing.T) {
	event := ResourceEvent{
		TimeMS:         1720000000000,
		Source:         "Application Hang",
		EventID:        1002,
		RecordID:       "103681",
		Name:           "WINDOWS_APPLICATION_HANG",
		ReportID:       "report-123",
		App:            "chrome.exe",
		Hang:           &ApplicationHang{AppVersion: "151.0.7922.109", Class: "Cross-process"},
		Message:        `raw message X:\private\chrome.exe --private-flag`,
		ProcessID:      42,
		ProcessStartMS: 1719999999000,
	}
	sample := NewProcessSample(time.UnixMilli(event.TimeMS), 42, time.UnixMilli(event.TimeMS-1000), "fak.exe", "sha", "rev", "hostdiag", "session", 1, 1, 1, 1)
	launch := OwnedShellLaunch{ChildPID: 42, ChildCreatedUTCMS: event.ProcessStartMS, LaunchID: "owned"}

	got, ok := CorrelateWithOwnedLaunches(event, []ProcessSample{sample}, []OwnedShellLaunch{launch})
	if !ok {
		t.Fatal("application hang was rejected")
	}
	if got.EventName != "WINDOWS_APPLICATION_HANG" || got.Status != "historical_unresolved" || !got.Observational {
		t.Fatalf("unexpected correlation: %+v", got)
	}
	if len(got.Candidates) != 0 || got.OwnedLaunch != nil {
		t.Fatalf("application hang was attributed: candidates=%+v owned=%+v", got.Candidates, got.OwnedLaunch)
	}
	if got.Hang == nil || got.Hang.AppVersion != "151.0.7922.109" || got.Hang.Class != "Cross-process" { //boundarylint:ignore CHANGE_DETECTOR_TEST the fixture pins the parsed Chrome version and crash class as the parser contract
		t.Fatalf("bounded hang identity not retained: %+v", got.Hang)
	}
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"raw message", `X:\\private`, "private-flag", "process_id", "process_start_ms", "command_line"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("correlation leaked %q: %s", forbidden, encoded)
		}
	}
}

func TestCorrelateRejectsMalformedApplicationHang(t *testing.T) {
	valid := ResourceEvent{TimeMS: 1720000000000, Source: "Application Hang", EventID: 1002, RecordID: "103612", Name: "WINDOWS_APPLICATION_HANG", ReportID: "report-123", App: "explorer.exe", Hang: &ApplicationHang{AppVersion: "10.0.26100.8875", Class: "Unknown"}}
	tests := map[string]func(*ResourceEvent){
		"provider":   func(event *ResourceEvent) { event.Source = "Application Error" },
		"event ID":   func(event *ResourceEvent) { event.EventID = 1000 },
		"app":        func(event *ResourceEvent) { event.App = "" },
		"version":    func(event *ResourceEvent) { event.Hang.AppVersion = "" },
		"report ID":  func(event *ResourceEvent) { event.ReportID = "" },
		"hang class": func(event *ResourceEvent) { event.Hang.Class = "Novel class" },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			event := valid
			hang := *valid.Hang
			event.Hang = &hang
			mutate(&event)
			if _, ok := Correlate(event, nil); ok {
				t.Fatalf("malformed application hang accepted: %+v", event)
			}
		})
	}
}

func TestCorrelateMacOSResourceIncidentRemainsUnattributed(t *testing.T) {
	data, err := os.ReadFile("testdata/macos-resource-incident.diag")
	if err != nil {
		t.Fatal(err)
	}
	event, err := ParseMacOSResourceIncident("macos-resource-incident.diag", data)
	if err != nil {
		t.Fatal(err)
	}
	sample := NewProcessSample(
		time.UnixMilli(event.TimeMS+1000), 20870, time.UnixMilli(event.TimeMS-60_000),
		"/usr/local/bin/fak", "sha", "rev", "hostdiag", "session", 1, 1, 1, 1,
	)
	got, ok := Correlate(event, []ProcessSample{sample})
	if !ok {
		t.Fatal("macOS resource incident was rejected")
	}
	if got.Status != "historical_unresolved" || !got.Observational || got.Correlated ||
		len(got.Candidates) != 0 || got.OwnedLaunch != nil {
		t.Fatalf("correlation = %+v", got)
	}
	if got.ResourceIncident == nil || got.Fault != nil || got.Hang != nil ||
		got.ResourceIncident.PID != 20870 || got.ResourceIncident.SampledStackEnd != "write(2)" {
		t.Fatalf("typed incident = %+v", got)
	}
	if !strings.Contains(got.Reason, "causal attribution remain unknown") {
		t.Fatalf("reason = %q", got.Reason)
	}
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{`"windows_event_id"`, `"application_fault"`, `"application_hang"`, `"candidates"`} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("resource incident emitted %q: %s", forbidden, encoded)
		}
	}
	trend, err := SummarizeTrend(
		bytes.NewReader(append(encoded, '\n')),
		time.UnixMilli(event.TimeMS).Add(time.Hour), 24*time.Hour, 24*time.Hour,
	)
	if err != nil {
		t.Fatal(err)
	}
	if trend.Recent.Total != 1 || trend.Recent.Crash != 0 || trend.Recent.Hang != 0 {
		t.Fatalf("resource incident was classified as a crash or hang: %+v", trend.Recent)
	}
}

func TestMacOSResourceIncidentCorrelationIdentityIncludesTypeAndArtifactDigest(t *testing.T) {
	data, err := os.ReadFile("testdata/macos-resource-incident.diag")
	if err != nil {
		t.Fatal(err)
	}
	event, err := ParseMacOSResourceIncident("macos-resource-incident.diag", data)
	if err != nil {
		t.Fatal(err)
	}
	first, ok := Correlate(event, nil)
	if !ok {
		t.Fatal("macOS resource incident was rejected")
	}
	again, _ := Correlate(event, nil)
	if first.CorrelationID == "" || again.CorrelationID != first.CorrelationID {
		t.Fatalf("unstable identity: first=%q again=%q", first.CorrelationID, again.CorrelationID)
	}
	changedDigest := event
	changedDigestIncident := *event.ResourceIncident
	changedDigest.ResourceIncident = &changedDigestIncident
	changedDigest.ResourceIncident.Artifact.SHA256 = "sha256:" + strings.Repeat("a", 64)
	second, ok := Correlate(changedDigest, nil)
	if !ok || second.CorrelationID == first.CorrelationID {
		t.Fatalf("artifact digest missing from identity: first=%q second=%q ok=%v", first.CorrelationID, second.CorrelationID, ok)
	}
	changedType := event
	changedTypeIncident := *event.ResourceIncident
	changedType.ResourceIncident = &changedTypeIncident
	changedType.ResourceIncident.IncidentType = "other_resource_incident"
	if correlationIdentityKey(changedType, MacOSResourceIncidentEventName) == correlationIdentityKey(event, MacOSResourceIncidentEventName) {
		t.Fatal("incident type missing from stable identity key")
	}
}
