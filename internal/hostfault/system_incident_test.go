package hostfault

import (
	"strings"
	"testing"
)

func TestClassifyWindowsSystemEvent(t *testing.T) {
	cases := []struct {
		id            int
		source, class string
	}{
		{1001, "Microsoft-Windows-WER-SystemErrorReporting", "bugcheck"},
		{41, "Microsoft-Windows-Kernel-Power", "unclean_restart"},
		{6008, "EventLog", "unexpected_shutdown"},
	}
	for _, tc := range cases {
		t.Run(tc.class, func(t *testing.T) {
			event := WindowsSystemEvent{TimeMS: 1720960000000, Source: tc.source, WindowsID: tc.id, RecordID: "42", BugcheckCode: "0x0000004e", Parameters: []string{" 0x6 "}, DumpPath: `C:\WINDOWS\Minidump\x.dmp`}
			got, ok := ClassifyWindowsSystemEvent(event)
			if !ok || got.Class != tc.class || !got.Observational || got.Schema != SystemIncidentSchema || got.EventID == "" {
				t.Fatalf("%+v ok=%v", got, ok)
			}
			if got.TimeUTC != "2024-07-14T12:26:40Z" || len(got.Parameters) != 1 || got.Parameters[0] != "0x6" {
				t.Fatalf("normalization failed: %+v", got)
			}
			again, _ := ClassifyWindowsSystemEvent(event)
			if got.EventID != again.EventID {
				t.Fatal("unstable id")
			}
		})
	}
}

func TestClassifyWindowsSystemEventIdentityIncludesPayload(t *testing.T) {
	base := WindowsSystemEvent{TimeMS: 1, Source: "EventLog", WindowsID: 6008}
	first, _ := ClassifyWindowsSystemEvent(base)
	base.Message = "a distinct event at the same timestamp"
	second, _ := ClassifyWindowsSystemEvent(base)
	if first.EventID == second.EventID {
		t.Fatal("distinct events collapsed")
	}
}

func TestClassifyWindowsSystemEventRejectsMalformedOrUnrelated(t *testing.T) {
	for _, event := range []WindowsSystemEvent{
		{TimeMS: 1, Source: "Other", WindowsID: 41},
		{Source: "Microsoft-Windows-Kernel-Power", WindowsID: 41},
		{TimeMS: 1, Source: "Microsoft-Windows-Kernel-Power", WindowsID: 1001},
	} {
		if _, ok := ClassifyWindowsSystemEvent(event); ok {
			t.Fatalf("accepted %+v", event)
		}
	}
}

func TestClassifyWindowsSystemEventBoundsMessage(t *testing.T) {
	got, ok := ClassifyWindowsSystemEvent(WindowsSystemEvent{TimeMS: 1, Source: "EventLog", WindowsID: 6008, Message: strings.Repeat("x", maxSystemMessageBytes+10)})
	if !ok || len(got.Message) != maxSystemMessageBytes {
		t.Fatalf("ok=%v message bytes=%d", ok, len(got.Message))
	}
}
