package toolprocgate

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// tpgConsoleCrashEnv re-purposes the re-exec'd test binary as the crashing
// child shell (the procbind_realtree_test.go idiom). Not set => the helper is
// inert in the normal suite run.
const tpgConsoleCrashEnv = "FAK_TPG_CONSOLE_CRASH_CHILD"

// TestHelperConsoleCrashChild is a REAL OS child that dies the way the
// 2026-07-01 pwsh.exe FailFast died (#2170): it emits normal tool output on
// stdout, then writes the .NET console-host crash banner to stderr — the
// HostException plus the Win32 0xE9 "No process is on the other end of the
// pipe." message — and hard-exits with code 233 (0xE9) mid-stream. It is NOT a
// test; the env guard turns the re-exec'd binary into the child.
func TestHelperConsoleCrashChild(t *testing.T) {
	if os.Getenv(tpgConsoleCrashEnv) != "1" {
		return
	}
	fmt.Println("tool-output-line-1")
	fmt.Println("tool-output-line-2")
	fmt.Fprintln(os.Stderr, "Process terminated. System.Management.Automation.Host.HostException: No process is on the other end of the pipe.")
	fmt.Fprintln(os.Stderr, "   at Microsoft.PowerShell.ConsoleHost.Start(String bannerText, String helpText, String[] args)")
	os.Exit(233) // 0xE9
}

// TestConsolePipeFailureParentSurvivesAndRecordsChildFailure is the #2170
// row-1 regression witness: a child shell's console host crashes mid-tool
// (a REAL process death, not a mock), and the parent-side supervisor
//
//   - keeps running (the fold still answers, no panic, no shared fate),
//   - records the crash as a STRUCTURED child failure carrying the typed
//     console-fault class instead of prose,
//   - leaves the sibling session's live proc untouched, and
//   - yields a fault row that round-trips the fail-closed JSONL parser, so
//     the crash class is durably searchable (#2170 row 4) instead of living
//     only in Event Viewer.
func TestConsolePipeFailureParentSurvivesAndRecordsChildFailure(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	sup := NewSupervisor(toolproc.Config{})
	// The sibling: a live long-runner in the same fleet that must survive the
	// child crash untouched.
	if err := sup.Spawn("sibling", "bg_tail", "s-sibling", 0, 0, 1_000, nil); err != nil {
		t.Fatalf("spawn sibling: %v", err)
	}
	if err := sup.Spawn("pwsh-child", "PowerShell", "s-crash", 0, 0, 1_000, nil); err != nil {
		t.Fatalf("spawn pwsh-child: %v", err)
	}

	// Launch the real crashing child and drain its pipes like an embedder does.
	child := exec.Command(os.Args[0], "-test.run=^TestHelperConsoleCrashChild$", "-test.v")
	child.Env = append(os.Environ(), tpgConsoleCrashEnv+"=1")
	stdout, err := child.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	var stderrBuf bytes.Buffer
	child.Stderr = &stderrBuf
	if err := child.Start(); err != nil {
		t.Fatalf("start child: %v", err)
	}
	// The drain ends abruptly when the child's console surface dies.
	if _, err := io.Copy(io.Discard, stdout); err != nil && !errors.Is(err, io.EOF) {
		t.Logf("stdout drain ended with: %v (an abrupt pipe error is the expected crash shape)", err)
	}
	waitErr := child.Wait()
	var exitErr *exec.ExitError
	if !errors.As(waitErr, &exitErr) {
		t.Fatalf("child must die abnormally, got wait err %v", waitErr)
	}
	if code := exitErr.ExitCode(); code != 233 {
		t.Fatalf("child must exit with the 0xE9 crash code 233, got %d", code)
	}

	// Classify the captured crash banner: this is the exact Event-Viewer
	// signature from the 2026-07-01 audit, and it must map to a TYPED class.
	class, ok := ClassifyConsoleFault(stderrBuf.String())
	if !ok {
		t.Fatalf("ClassifyConsoleFault must recognize the pwsh FailFast banner, got no class; banner:\n%s", stderrBuf.String())
	}
	if class != ConsoleHostFailFast {
		t.Fatalf("want %s, got %s", ConsoleHostFailFast, class)
	}

	// Record the structured child failure on the supervisor.
	ev, err := sup.ExitConsoleFault("pwsh-child", 2_000, class, ConsoleSurfaceStderr, stderrBuf.String())
	if err != nil {
		t.Fatalf("ExitConsoleFault: %v", err)
	}
	if ev.Class != ConsoleHostFailFast || ev.CallID != "pwsh-child" || ev.Tool != "PowerShell" || ev.Session != "s-crash" {
		t.Fatalf("fault event must carry the typed class and the child identity, got %+v", ev)
	}
	if !strings.Contains(ev.Detail, "No process is on the other end of the pipe") {
		t.Fatalf("fault detail must keep the searchable crash string, got %q", ev.Detail)
	}

	// THE PARENT SURVIVES: the fold still answers, the crashed child is a
	// terminal error — a child-session failure, not a supervisor failure —
	// and the sibling is still RUNNING.
	tab, err := sup.Table(3_000)
	if err != nil {
		t.Fatalf("parent table fold after child crash: %v", err)
	}
	states := map[string]toolproc.Proc{}
	for _, p := range tab.Procs {
		states[p.CallID] = p
	}
	if got := states["pwsh-child"]; got.State != toolproc.StateDone || got.ExitStatus != "error" {
		t.Fatalf("crashed child must fold to a structured DONE/error, got %+v", got)
	}
	if got := states["sibling"]; got.State != toolproc.StateRunning {
		t.Fatalf("sibling must survive the child console crash, got %+v", got)
	}
	// A subsequent enforcement tick must not punish the sibling for the crash.
	rep, err := sup.Tick(3_000)
	if err != nil {
		t.Fatalf("parent tick after child crash: %v", err)
	}
	for _, act := range rep.Actions {
		if act.CallID == "sibling" && (act.Advice == toolproc.AdviceKill || act.Advice == toolproc.AdviceReap) {
			t.Fatalf("sibling must not be killed by the child's crash, got %+v", act)
		}
	}

	// SEARCHABLE (#2170 row 4): the fault row round-trips the fail-closed
	// JSONL parser and the fold counts it by class.
	var sink bytes.Buffer
	if err := AppendConsoleFaultEvent(&sink, ev); err != nil {
		t.Fatalf("append fault event: %v", err)
	}
	parsed, err := ParseConsoleFaultEvents(&sink)
	if err != nil {
		t.Fatalf("parse fault events: %v", err)
	}
	if len(parsed) != 1 || parsed[0].Class != ConsoleHostFailFast {
		t.Fatalf("round-trip must keep the typed class, got %+v", parsed)
	}
	repFold := ConsoleFaultReportFromEvents(parsed)
	if repFold.Counts.ByClass[string(ConsoleHostFailFast)] != 1 {
		t.Fatalf("fold must count the crash class, got %+v", repFold.Counts)
	}
}

// TestClassifyConsoleFault pins the closed signature vocabulary to the crash
// classes #2170 names: the pwsh HostException FailFast, the 0xE9 / broken /
// closing pipe family, the lost console handle, and the 2026-07-08 PSReadLine
// lost-input (ReadKey on a redirected/absent console) sibling. A plain tool
// error must NOT classify — fail-closed, no guessing.
func TestClassifyConsoleFault(t *testing.T) {
	cases := []struct {
		name   string
		detail string
		want   ConsoleFaultClass
		ok     bool
	}{
		{"pwsh HostException", "System.Management.Automation.Host.HostException: something", ConsoleHostFailFast, true},
		{"ConsoleHost frame", "at Microsoft.PowerShell.ConsoleHost.Start(...)", ConsoleHostFailFast, true},
		{"0xE9 no-process pipe", "Win32 error: No process is on the other end of the pipe.", ConsolePipeLost, true},
		{"pipe ended (109)", "write error: The pipe has been ended.", ConsolePipeLost, true},
		{"pipe closing (232)", "The pipe is being closed.", ConsolePipeLost, true},
		{"posix EPIPE", "write |1: broken pipe", ConsolePipeLost, true},
		{"lost console handle", "read console: The handle is invalid.", ConsoleHandleLost, true},
		{"failfast+pipe classifies as host crash", "Process terminated. System.Management.Automation.Host.HostException: No process is on the other end of the pipe.", ConsoleHostFailFast, true},
		// 2026-06-29 (recurring through 2026-07-08) signature: PSReadLine's
		// key-reader thread throws InvalidOperationException on a
		// redirected/absent console. Message rung and thread rung, each alone.
		{"psreadline readkey message", "Unhandled exception. System.InvalidOperationException: Cannot read keys when either application does not have a console or when console input has been redirected. Try Console.Read.", ConsoleInputLost, true},
		{"psreadline readkey stack", "   at Microsoft.PowerShell.PSConsoleReadLine.ReadKeyThreadProc()", ConsoleInputLost, true},
		// Mixed-signal ordering — the precedence contract is the whole point of
		// the check placement (host -> input -> pipe -> handle), so pin it.
		{"host token pre-empts readkey message", "System.Management.Automation.Host.HostException while unwinding; Cannot read keys when either application does not have a console.", ConsoleHostFailFast, true},
		{"readkey thread pre-empts pipe cause", "at Microsoft.PowerShell.PSConsoleReadLine.ReadKeyThreadProc(); No process is on the other end of the pipe.", ConsoleInputLost, true},
		// #2170 mis-route regression: the output-buffer FailFast scraped WITHOUT
		// the HostException type token must still route to host, not pipe, on the
		// strength of the console-output-API mechanism token.
		{"output API pre-empts pipe message", "get_CursorPosition -> GetConsoleScreenBufferInfo failed; No process is on the other end of the pipe.", ConsoleHostFailFast, true},
		{"plain tool error", "exit status 1: compilation failed", "", false},
		{"empty", "", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ClassifyConsoleFault(tc.detail)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("ClassifyConsoleFault(%q) = (%q, %v), want (%q, %v)", tc.detail, got, ok, tc.want, tc.ok)
			}
		})
	}
}

// TestClassifyConsoleFaultWER pins the structured-field WER path (#3513): a
// console-host/shell faulting app + the __fastfail code 0xc0000409 folds to the
// coarse CONSOLE_HOST_FAILFAST; everything else fails closed. The module is
// deliberately irrelevant to the verdict (a FailFast most often surfaces in
// KERNELBASE.dll, not ConsoleHost.dll), so it is not a parameter here.
func TestClassifyConsoleFaultWER(t *testing.T) {
	cases := []struct {
		name string
		app  string
		code string
		want ConsoleFaultClass
		ok   bool
	}{
		{"pwsh failfast", "pwsh.exe", "0xc0000409", ConsoleHostFailFast, true},
		{"winps failfast", "powershell.exe", "0xc0000409", ConsoleHostFailFast, true},
		{"conhost failfast", "conhost.exe", "0xc0000409", ConsoleHostFailFast, true},
		{"openconsole failfast", "OpenConsole.exe", "0xc0000409", ConsoleHostFailFast, true},
		{"cmd failfast", "cmd.exe", "0xc0000409", ConsoleHostFailFast, true},
		{"case-insensitive app + code", "PWSH.EXE", "0xC0000409", ConsoleHostFailFast, true},
		// Fail-closed: the FailFast code alone is NOT console-specific — a
		// non-console app carrying it stays a plain app crash.
		{"non-console app failfast dropped", "myapp.exe", "0xc0000409", "", false},
		// Fail-closed: a console app with a NON-FailFast code (e.g. an access
		// violation, the WindowsTerminal WinAppRuntime class) is not this class.
		{"console app access-violation dropped", "pwsh.exe", "0xc0000005", "", false},
		// Fail-closed: the operator's OUTER terminal is not a child console fault.
		{"outer terminal dropped", "WindowsTerminal.exe", "0xc0000409", "", false},
		{"empty app", "", "0xc0000409", "", false},
		{"empty code", "pwsh.exe", "", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ClassifyConsoleFaultWER(tc.app, tc.code)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("ClassifyConsoleFaultWER(%q, %q) = (%q, %v), want (%q, %v)", tc.app, tc.code, got, ok, tc.want, tc.ok)
			}
		})
	}
}

// TestClassifyDrainError covers the drain-side shapes: an EOF while the call
// is still live is a PTY/console EOF fault; an EOF after a clean exit is not a
// fault at all; a pipe error classifies by its signature.
func TestClassifyDrainError(t *testing.T) {
	cases := []struct {
		name     string
		err      error
		callLive bool
		want     ConsoleFaultClass
		ok       bool
	}{
		{"eof while live", io.EOF, true, ConsolePTYEOF, true},
		{"unexpected eof while live", io.ErrUnexpectedEOF, true, ConsolePTYEOF, true},
		{"eof after clean exit", io.EOF, false, "", false},
		{"broken pipe", errors.New("write |1: broken pipe"), true, ConsolePipeLost, true},
		{"nil error", nil, true, "", false},
		{"plain error not console", errors.New("context deadline exceeded"), true, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ClassifyDrainError(tc.err, tc.callLive)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("ClassifyDrainError(%v, %v) = (%q, %v), want (%q, %v)", tc.err, tc.callLive, got, ok, tc.want, tc.ok)
			}
		})
	}
}

// TestExitConsoleFaultFailsClosed: the vocabulary is CLOSED — an unknown class
// is refused before anything is journaled, and an unknown call is refused the
// same way a bare Exit would refuse it.
func TestExitConsoleFaultFailsClosed(t *testing.T) {
	sup := NewSupervisor(toolproc.Config{})
	if err := sup.Spawn("c1", "PowerShell", "s1", 0, 0, 1_000, nil); err != nil {
		t.Fatalf("spawn: %v", err)
	}
	if _, err := sup.ExitConsoleFault("c1", 2_000, ConsoleFaultClass("MADE_UP"), ConsoleSurfaceStdout, "x"); err == nil {
		t.Fatal("unknown fault class must be refused")
	}
	if _, err := sup.ExitConsoleFault("ghost", 2_000, ConsolePipeLost, ConsoleSurfaceStdout, "x"); err == nil {
		t.Fatal("unknown call must be refused")
	}
	// The refused calls must not have journaled a terminal state for c1.
	tab, err := sup.Table(3_000)
	if err != nil {
		t.Fatalf("table: %v", err)
	}
	if len(tab.Procs) != 1 || tab.Procs[0].State != toolproc.StateRunning {
		t.Fatalf("refused fault must leave the proc running, got %+v", tab.Procs)
	}
}

// TestExitConsoleFaultBoundsDetail: the durable row is byte-bounded — a
// multi-megabyte crash spew cannot become the memory/storage leak.
func TestExitConsoleFaultBoundsDetail(t *testing.T) {
	sup := NewSupervisor(toolproc.Config{})
	if err := sup.Spawn("c1", "PowerShell", "s1", 0, 0, 1_000, nil); err != nil {
		t.Fatalf("spawn: %v", err)
	}
	huge := strings.Repeat("No process is on the other end of the pipe. ", 100)
	ev, err := sup.ExitConsoleFault("c1", 2_000, ConsolePipeLost, ConsoleSurfaceStderr, huge)
	if err != nil {
		t.Fatalf("ExitConsoleFault: %v", err)
	}
	if len(ev.Detail) > consoleFaultDetailLimit {
		t.Fatalf("detail must be bounded to %d bytes, got %d", consoleFaultDetailLimit, len(ev.Detail))
	}
	if !strings.Contains(ev.Detail, "No process is on the other end of the pipe") {
		t.Fatalf("bounded detail must keep the leading searchable signature, got %q", ev.Detail)
	}
}

// TestParseConsoleFaultEventsFailClosed: the JSONL surface refuses unknown
// fields and unknown classes, so a fabricated or drifted row can never enter
// an operator report as a legitimate crash record.
func TestParseConsoleFaultEventsFailClosed(t *testing.T) {
	if _, err := ParseConsoleFaultEvents(strings.NewReader(
		`{"class":"CONSOLE_PIPE_LOST","at_unix_ms":1,"call_id":"c1","payload":"raw-bytes"}` + "\n")); err == nil {
		t.Fatal("unknown field must be refused")
	}
	if _, err := ParseConsoleFaultEvents(strings.NewReader(
		`{"class":"NOT_A_CLASS","at_unix_ms":1,"call_id":"c1"}` + "\n")); err == nil {
		t.Fatal("unknown class must be refused")
	}
	if _, err := ParseConsoleFaultEvents(strings.NewReader(
		`{"class":"CONSOLE_PIPE_LOST","call_id":"c1"}` + "\n")); err == nil {
		t.Fatal("missing timestamp must be refused")
	}
	// Comments and blank lines are journal furniture, not rows.
	evs, err := ParseConsoleFaultEvents(strings.NewReader(
		"# comment\n\n" + `{"class":"CONSOLE_PIPE_LOST","at_unix_ms":1,"call_id":"c1","surface":"stdout"}` + "\n"))
	if err != nil {
		t.Fatalf("valid row must parse: %v", err)
	}
	if len(evs) != 1 || evs[0].Class != ConsolePipeLost {
		t.Fatalf("want one CONSOLE_PIPE_LOST row, got %+v", evs)
	}
}

// TestConsoleFaultReportCounts: the report fold is the searchable summary an
// operator (or `fak toolproc`) keys on — counts by class, surface, and tool.
func TestConsoleFaultReportCounts(t *testing.T) {
	events := []ConsoleFaultEvent{
		{Class: ConsolePipeLost, AtMS: 1, CallID: "a", Tool: "PowerShell", Surface: string(ConsoleSurfaceStdout)},
		{Class: ConsolePipeLost, AtMS: 2, CallID: "b", Tool: "Bash", Surface: string(ConsoleSurfaceStderr)},
		{Class: ConsoleHostFailFast, AtMS: 3, CallID: "c", Tool: "PowerShell", Surface: string(ConsoleSurfaceStderr)},
	}
	rep := ConsoleFaultReportFromEvents(events)
	if rep.Rows != 3 {
		t.Fatalf("want 3 rows, got %d", rep.Rows)
	}
	if rep.Counts.ByClass[string(ConsolePipeLost)] != 2 || rep.Counts.ByClass[string(ConsoleHostFailFast)] != 1 {
		t.Fatalf("class counts wrong: %+v", rep.Counts.ByClass)
	}
	if rep.Counts.ByTool["PowerShell"] != 2 {
		t.Fatalf("tool counts wrong: %+v", rep.Counts.ByTool)
	}
	if rep.Counts.BySurface["stderr"] != 2 {
		t.Fatalf("surface counts wrong: %+v", rep.Counts.BySurface)
	}
	var out bytes.Buffer
	RenderConsoleFaultReport(&out, rep)
	if !strings.Contains(out.String(), string(ConsoleHostFailFast)) {
		t.Fatalf("rendered report must name the class, got:\n%s", out.String())
	}
}
