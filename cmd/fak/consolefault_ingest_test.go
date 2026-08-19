package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// TestConsoleFaultEventsFromWinRecords pins the live-producer mapping against
// the real .NET Runtime Event 1026 shapes this host emits: the CONSOLE_INPUT_LOST
// PSReadLine crash, the #2170 CONSOLE_HOST_FAILFAST output crash, and an
// unrelated .NET crash that must be dropped fail-closed (never fabricated into a
// console-fault row).
func TestConsoleFaultEventsFromWinRecords(t *testing.T) {
	const inputCrash = `Application: pwsh.exe
CoreCLR Version: 10.0.0
.NET Version: 10.0.0
Description: The process was terminated due to an unhandled exception.
Exception Info: System.InvalidOperationException: Cannot read keys when either application does not have a console or when console input has been redirected.
   at System.ConsolePal.ReadKey(Boolean intercept)
   at Microsoft.PowerShell.PSConsoleReadLine.ReadKeyThreadProc()`
	const outputFailFast = `Application: pwsh.exe
Exception Info: System.Management.Automation.Host.HostException
   at Microsoft.PowerShell.ConsoleHostRawUserInterface.get_CursorPosition()
   ... GetConsoleScreenBufferInfo failed`
	const unrelated = `Application: myapp.exe
Exception Info: System.NullReferenceException
   at MyApp.Foo()`

	recs := []winEventRecord{
		{Provider: ".NET Runtime", ID: 1026, TimeMS: 1_700_000_000_000, Message: inputCrash},
		{Provider: ".NET Runtime", ID: 1026, TimeMS: 0, Message: outputFailFast},
		{Provider: ".NET Runtime", ID: 1026, TimeMS: 1_700_000_000_000, Message: unrelated},
	}

	got := consoleFaultEventsFromWinRecords(recs, 42)
	if len(got) != 2 {
		t.Fatalf("want 2 classified rows (unrelated dropped fail-closed), got %d", len(got))
	}
	if got[0].Class != toolprocgate.ConsoleInputLost {
		t.Errorf("row0 class = %q, want CONSOLE_INPUT_LOST", got[0].Class)
	}
	if got[0].Tool != "pwsh" {
		t.Errorf("row0 tool = %q, want pwsh", got[0].Tool)
	}
	if got[0].AtMS != 1_700_000_000_000 {
		t.Errorf("row0 at_ms = %d, want the record time", got[0].AtMS)
	}
	if got[1].Class != toolprocgate.ConsoleHostFailFast {
		t.Errorf("row1 class = %q, want CONSOLE_HOST_FAILFAST", got[1].Class)
	}
	if got[1].AtMS != 42 {
		t.Errorf("row1 at_ms = %d, want the fallback (record time was 0)", got[1].AtMS)
	}
	// Every emitted row must satisfy the durable schema so it round-trips
	// through AppendConsoleFaultEvent / ParseConsoleFaultEvents.
	for i, ev := range got {
		if err := toolprocgate.ValidateConsoleFaultEvent(ev); err != nil {
			t.Errorf("row%d invalid: %v", i, err)
		}
		if ev.Surface != string(toolprocgate.ConsoleSurfaceStderr) {
			t.Errorf("row%d surface = %q, want stderr", i, ev.Surface)
		}
	}
}

// werBanner is a representative WER Application Error (Event 1000) FailFast
// banner: a pwsh.exe __fastfail whose faulting module is KERNELBASE.dll (where
// RaiseFailFastException lives — NOT ConsoleHost.dll), exception code
// 0xc0000409. This is the 1026-less crash shape #3513 targets.
const werBanner = `Faulting application name: pwsh.exe, version: 7.6.3.500, time stamp: 0x66abc123
Faulting module name: KERNELBASE.dll, version: 10.0.26100.1234, time stamp: 0x11223344
Exception code: 0xc0000409
Fault offset: 0x000000000005a3f0
Faulting process id: 0x1a2b
Faulting application path: C:\Program Files\PowerShell\7\pwsh.exe
Faulting module path: C:\Windows\System32\KERNELBASE.dll
Report Id: 12345678-1234-1234-1234-1234567890ab`

// TestParseWERFields pins the structured-field extraction against both the
// native multi-line banner and a whitespace-flattened one (the shape a naive
// consumer might hand us). App/module basenames and the exception code each
// reduce to their first whitespace token, so a flattened banner still parses.
func TestParseWERFields(t *testing.T) {
	app, module, code := parseWERFields(werBanner)
	if app != "pwsh.exe" || module != "KERNELBASE.dll" || code != "0xc0000409" {
		t.Fatalf("multi-line parse = (%q,%q,%q), want (pwsh.exe, KERNELBASE.dll, 0xc0000409)", app, module, code)
	}
	app, module, code = parseWERFields(flattenWhitespace(werBanner))
	if app != "pwsh.exe" || module != "KERNELBASE.dll" || code != "0xc0000409" {
		t.Fatalf("flattened parse = (%q,%q,%q), want (pwsh.exe, KERNELBASE.dll, 0xc0000409)", app, module, code)
	}
}

// TestConsoleFaultEventsFromWERRecords covers the 1026-less WER FailFast path:
// an un-paired console-host FailFast becomes a CONSOLE_HOST_FAILFAST row, while
// a non-console app and a non-FailFast code are both dropped fail-closed.
func TestConsoleFaultEventsFromWERRecords(t *testing.T) {
	const nonConsole = `Faulting application name: myapp.exe, version: 1.0.0.0
Faulting module name: myapp.exe, version: 1.0.0.0
Exception code: 0xc0000409`
	const consoleAV = `Faulting application name: pwsh.exe, version: 7.6.3.500
Faulting module name: ntdll.dll, version: 10.0.26100.1
Exception code: 0xc0000005`

	recs := []winEventRecord{
		{Provider: "Application Error", ID: 1000, TimeMS: 1_700_000_000_000, Message: werBanner},
		{Provider: "Application Error", ID: 1000, TimeMS: 1_700_000_000_000, Message: nonConsole},
		{Provider: "Application Error", ID: 1000, TimeMS: 1_700_000_000_000, Message: consoleAV},
	}
	got := consoleFaultEventsFromWinRecords(recs, 42)
	if len(got) != 1 {
		t.Fatalf("want 1 classified WER row (non-console + access-violation dropped), got %d", len(got))
	}
	ev := got[0]
	if ev.Class != toolprocgate.ConsoleHostFailFast {
		t.Errorf("class = %q, want CONSOLE_HOST_FAILFAST", ev.Class)
	}
	if ev.Tool != "pwsh" {
		t.Errorf("tool = %q, want pwsh", ev.Tool)
	}
	if ev.AtMS != 1_700_000_000_000 {
		t.Errorf("at_ms = %d, want the record time", ev.AtMS)
	}
	if ev.Surface != "" {
		t.Errorf("surface = %q, want empty (a WER banner does not witness a stdio surface)", ev.Surface)
	}
	if !strings.Contains(ev.Detail, "code=0xc0000409") || !strings.Contains(ev.Detail, "module=KERNELBASE.dll") {
		t.Errorf("detail = %q, want the parsed structured fields", ev.Detail)
	}
	if err := toolprocgate.ValidateConsoleFaultEvent(ev); err != nil {
		t.Errorf("row invalid: %v", err)
	}
}

// TestConsoleFaultWERDedup pins the "1026-less" contract: a WER FailFast that
// pairs in time with a .NET 1026 console fault for the SAME tool is the same
// crash and is dropped, so the fold is not double-counted — but an un-paired
// WER FailFast (different tool, or outside the window) is kept.
func TestConsoleFaultIngestProbeIncludesWindowsTerminalRendererExit(t *testing.T) {
	if !strings.Contains(consoleFaultIngestPS, `Faulting application name:\s*WindowsTerminal\.exe`) {
		t.Fatal("Windows event probe drops Windows Terminal renderer exits before Go can classify them")
	}
}

func TestConsoleFaultEventsCaptureWindowsTerminalRendererCrash(t *testing.T) {
	const rendererCrash = `Faulting application name: WindowsTerminal.exe, version: 1.24.2607.10001
Faulting module name: TerminalApp.dll, version: 1.24.2607.10001
Exception code: 0xc0000005`
	const terminalFailFast = `Faulting application name: WindowsTerminal.exe, version: 1.24.2607.10001
Faulting module name: TerminalApp.dll, version: 1.24.2607.10001
Exception code: 0xc000041d`

	got := consoleFaultEventsFromWinRecords([]winEventRecord{
		{Provider: "Application Error", ID: 1000, TimeMS: 1_700_000_000_000, Message: rendererCrash},
		{Provider: "Application Error", ID: 1000, TimeMS: 1_700_000_007_000, Message: terminalFailFast},
	}, 42)
	if len(got) != 2 {
		t.Fatalf("rows = %d, want 2 Windows Terminal renderer crashes: %#v", len(got), got)
	}
	for i, ev := range got {
		if ev.Class != toolprocgate.ConsoleRendererExit {
			t.Errorf("row %d class = %q, want %q", i, ev.Class, toolprocgate.ConsoleRendererExit)
		}
		if ev.Surface != string(toolprocgate.ConsoleSurfaceRenderer) {
			t.Errorf("row %d surface = %q, want %q", i, ev.Surface, toolprocgate.ConsoleSurfaceRenderer)
		}
		if ev.Tool != "windowsterminal" {
			t.Errorf("row %d tool = %q, want windowsterminal", i, ev.Tool)
		}
	}
}

func TestConsoleFaultWERDedup(t *testing.T) {
	const inputCrash = `Application: pwsh.exe
Exception Info: System.InvalidOperationException: Cannot read keys when either application does not have a console or when console input has been redirected.
   at Microsoft.PowerShell.PSConsoleReadLine.ReadKeyThreadProc()`

	base := int64(1_700_000_000_000)
	recs := []winEventRecord{
		// A pwsh crash logged BOTH a 1026 managed stack and a WER 1000 banner
		// ~2s apart: one crash, must fold to ONE row (the 1026 wins).
		{Provider: ".NET Runtime", ID: 1026, TimeMS: base, Message: inputCrash},
		{Provider: "Application Error", ID: 1000, TimeMS: base + 2_000, Message: werBanner},
		// A second, genuinely 1026-less pwsh FailFast far outside the window.
		{Provider: "Application Error", ID: 1000, TimeMS: base + 3_600_000, Message: werBanner},
	}
	got := consoleFaultEventsFromWinRecords(recs, 1)
	if len(got) != 2 {
		t.Fatalf("want 2 rows (paired WER deduped, un-paired WER kept), got %d", len(got))
	}
	if got[0].Class != toolprocgate.ConsoleInputLost {
		t.Errorf("row0 class = %q, want CONSOLE_INPUT_LOST (the 1026 fault)", got[0].Class)
	}
	if got[1].Class != toolprocgate.ConsoleHostFailFast || got[1].AtMS != base+3_600_000 {
		t.Errorf("row1 = (%q, %d), want the un-paired WER FailFast at base+1h", got[1].Class, got[1].AtMS)
	}
}

// TestFlattenWhitespace confirms a multi-line dump collapses to one greppable
// JSONL line.
func TestFlattenWhitespace(t *testing.T) {
	in := "line one\n   at Foo()\n\tat Bar()"
	want := "line one at Foo() at Bar()"
	if got := flattenWhitespace(in); got != want {
		t.Errorf("flattenWhitespace = %q, want %q", got, want)
	}
}
