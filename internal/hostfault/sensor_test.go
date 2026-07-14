package hostfault

import "testing"

func TestClassifyApplicationErrorTerminalCrashes(t *testing.T) {
	tests := []struct {
		name  string
		event ApplicationError1000
		want  HostCrashClass
	}{
		{"WT render", ApplicationError1000{TimeMS: 1, App: "WindowsTerminal.exe", Module: "Microsoft.Terminal.Control.dll", Exception: "c0000005", ProcessID: "0x123"}, HostCrashWTRenderAV},
		{"TermService AMD", ApplicationError1000{TimeMS: 2, App: "svchost.exe", Module: "amdxx64.dll", Exception: "0xc0000005", ProcessID: "456"}, HostCrashTermServiceAMDAV},
		{"unknown terminal crash fails closed", ApplicationError1000{TimeMS: 3, App: "OpenConsole.exe", Module: "new.dll", Exception: "0xdeadbeef"}, HostCrashGeneric},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ClassifyApplicationError(tt.event)
			if !ok {
				t.Fatal("terminal-family event was dropped")
			}
			if got.Class != tt.want {
				t.Fatalf("class=%s want %s", got.Class, tt.want)
			}
			if got.Schema != HostCrashSignalSchema || got.EventID == "" {
				t.Fatalf("incomplete signal: %+v", got)
			}
		})
	}
}

func TestClassifyApplicationErrorParsesDecimalAndHexPID(t *testing.T) {
	decimal, _ := ClassifyApplicationError(ApplicationError1000{App: "OpenConsole.exe", ProcessID: "456"})
	hex, _ := ClassifyApplicationError(ApplicationError1000{App: "OpenConsole.exe", ProcessID: "0x1c8"})
	if decimal.HostPID != 456 || hex.HostPID != 456 {
		t.Fatalf("pid parsing decimal=%d hex=%d", decimal.HostPID, hex.HostPID)
	}
}
func TestClassifyApplicationErrorIgnoresUnrelatedApp(t *testing.T) {
	if _, ok := ClassifyApplicationError(ApplicationError1000{App: "notepad.exe", Module: "x.dll"}); ok {
		t.Fatal("unrelated crash classified")
	}
}

func TestClassifyApplicationErrorStableIdentity(t *testing.T) {
	e := ApplicationError1000{TimeMS: 1, App: "WindowsTerminal.exe", Module: "TerminalApp.dll", Exception: "c0000005", ReportID: "r"}
	a, _ := ClassifyApplicationError(e)
	b, _ := ClassifyApplicationError(e)
	if a.EventID != b.EventID {
		t.Fatalf("unstable event id: %s != %s", a.EventID, b.EventID)
	}
}
