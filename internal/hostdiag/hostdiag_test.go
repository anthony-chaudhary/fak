package hostdiag

import (
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

func TestClassifyCommandDoesNotRetainArguments(t *testing.T) {
	if got := ClassifyCommand(`C:\bin\fak.exe guard --api-key secret`); got != "guard" {
		t.Fatalf("got %q", got)
	}
	if got := ClassifyCommand(`C:\bin\fak.exe unusual --token secret`); got != "other" {
		t.Fatalf("got %q", got)
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
