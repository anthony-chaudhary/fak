package toolprocgate

// Console-host fault boundary (#2170): a terminal, shell, PTY, or TUI render
// surface is a FAULT BOUNDARY. When a child's console surface crashes — the
// 2026-07-01 class was a pwsh.exe FailFast throwing
// System.Management.Automation.Host.HostException with Win32 0xE9 "No process
// is on the other end of the pipe." — the parent supervisor must fold it to a
// STRUCTURED child failure and keep the kernel plus unrelated agents alive.
// Before this leaf the crash class lived only in Windows Event Viewer: nothing
// in-process could NAME it, so forensics meant manual registry correlation.
//
// This file gives the class a closed vocabulary, a signature classifier for
// the drain/exit path, a Supervisor seam that records the fault as a normal
// terminal exit (sibling procs untouched), and a fail-closed JSONL row +
// report fold in the LeakEvent idiom, so the class is durably searchable.

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/toolproc"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// ConsoleFaultEventSchema stamps one durable console-fault row.
const ConsoleFaultEventSchema = "fak.toolprocgate.console-fault-event.v1"

// ConsoleFaultReportSchema stamps the folded operator report.
const ConsoleFaultReportSchema = "fak.toolprocgate.console-fault-report.v1"

// consoleFaultDetailLimit bounds the recorded crash detail: enough to keep the
// leading searchable signature, never enough to become the storage leak.
const consoleFaultDetailLimit = 512

// ConsoleFaultClass is the CLOSED vocabulary of child terminal/shell/PTY/
// console-host crash classes. Parse and record fail closed: an unknown class
// is refused, never folded into an operator report as legitimate.
type ConsoleFaultClass string

const (
	// ConsoleHostFailFast: the child's managed console host crashed the
	// process — the pwsh/.NET FailFast class from the 2026-07-01 audit
	// (HostException through Microsoft.PowerShell.ConsoleHost).
	ConsoleHostFailFast ConsoleFaultClass = "CONSOLE_HOST_FAILFAST"
	// ConsolePipeLost: the console stdio pipe went away under the child —
	// Win32 0xE9 "No process is on the other end of the pipe.", 109 "The pipe
	// has been ended.", 232 "The pipe is being closed.", or POSIX EPIPE.
	ConsolePipeLost ConsoleFaultClass = "CONSOLE_PIPE_LOST"
	// ConsoleHandleLost: the console handle itself became invalid (the
	// classic Windows lost-console error on a detached/killed conhost).
	ConsoleHandleLost ConsoleFaultClass = "CONSOLE_HANDLE_LOST"
	// ConsoleInputLost: the child's console INPUT surface vanished under an
	// interactive console host — it tried to read a key from a console that
	// was detached or whose stdin was redirected. Witnessed on this host as a
	// recurring class (same pwsh.exe v7.6.3.500 / .NET 10 faulting module
	// Microsoft.PowerShell.ConsoleHost.dll): the full managed stack was
	// captured 2026-06-29 (.NET Event 1026) and the same crash recurred through
	// 2026-07-08 (WER Event 1000). PSReadLine's key-reader thread
	// (ReadKeyThreadProc -> Console.ReadKey) throws System.InvalidOperationException
	// "Cannot read keys when either application does not have a console or when
	// console input has been redirected." — the console *input*-path sibling of
	// the #2170 ConsoleHostFailFast (which crashes on the *output* buffer via
	// GetConsoleScreenBufferInfo). Distinct remediation: never drive an
	// interactive REPL with a redirected/absent stdin (-NonInteractive) or
	// sever the console (see RESEARCH-windows-handles-terminal-limits note).
	ConsoleInputLost ConsoleFaultClass = "CONSOLE_INPUT_LOST"
	// ConsolePTYEOF: the PTY/pipe drain hit EOF while the call was still
	// live — the surface vanished without a terminal exit observation.
	ConsolePTYEOF ConsoleFaultClass = "CONSOLE_PTY_EOF"
	// ConsoleRendererExit: a TUI render surface exited unexpectedly. No
	// string signature — the embedder that owns the renderer asserts it
	// directly (the #2170 render-pane witness's class).
	ConsoleRendererExit ConsoleFaultClass = "CONSOLE_RENDERER_EXIT"
)

// ConsoleSurface names the child-owned surface the fault was observed on.
// Bounded free text in the row (mirrors LeakEvent.SourceChannel); constants
// pin the common cases.
type ConsoleSurface string

const (
	ConsoleSurfaceStdout   ConsoleSurface = "stdout"
	ConsoleSurfaceStderr   ConsoleSurface = "stderr"
	ConsoleSurfacePTY      ConsoleSurface = "pty"
	ConsoleSurfaceRenderer ConsoleSurface = "renderer"
)

// validConsoleFaultClass is the closed-set membership check.
func validConsoleFaultClass(c ConsoleFaultClass) bool {
	switch c {
	case ConsoleHostFailFast, ConsolePipeLost, ConsoleHandleLost, ConsoleInputLost, ConsolePTYEOF, ConsoleRendererExit:
		return true
	}
	return false
}

// consoleHostSignatures identify a managed console-host crash. Checked FIRST:
// a FailFast banner usually also carries the pipe message that CAUSED it, and
// the crash (not the cause) is the class an operator searches for. The console
// OUTPUT-buffer mechanism tokens (GetConsoleScreenBufferInfo / get_CursorPosition,
// the #2170 output path) live here too — ahead of the pipe signatures — so the
// flagship output FailFast is not mis-routed to CONSOLE_PIPE_LOST by the "No
// process is on the other end of the pipe" message it carries when its detail
// is scraped without the HostException type token.
//
// Deliberately NOT included: the generic "0xc0000409" (__fastfail /
// STATUS_STACK_BUFFER_OVERRUN), "environment.failfast", and "unknown hard error"
// tokens. They are real for #2170 but NOT console-specific — any non-console
// __fastfail carries them, so a bare substring here would over-match a plain
// tool crash. They need co-occurrence context this single-substring classifier
// cannot express; leave them to a future WER ingester with structured fields.
var consoleHostSignatures = []string{
	"system.management.automation.host.hostexception",
	"microsoft.powershell.consolehost",
	"getconsolescreenbufferinfo", // #2170 output-buffer API: GetConsoleScreenBufferInfo -> 0xE9 -> HostException -> FailFast
	"get_cursorposition",         // ConsoleHostRawUserInterface.get_CursorPosition — the caller that triggers the 0xE9 read
}

// consolePipeSignatures identify the lost-stdio-pipe family.
var consolePipeSignatures = []string{
	"no process is on the other end of the pipe", // Win32 0xE9 / ERROR_PIPE_NOT_CONNECTED
	"the pipe has been ended",                    // Win32 109 / ERROR_BROKEN_PIPE
	"the pipe is being closed",                   // Win32 232 / ERROR_NO_DATA
	"broken pipe",                                // POSIX EPIPE
}

// consoleHandleSignatures identify a lost console handle.
var consoleHandleSignatures = []string{
	"the handle is invalid", // Win32 6 / ERROR_INVALID_HANDLE on a dead conhost
}

// consoleInputSignatures identify the lost console-INPUT crash: an interactive
// host's key reader tried to read from a console that is gone or whose stdin is
// redirected. Two rungs, by design: the message string is the precise .NET
// ConsolePal text but is ENGLISH-LOCALE-dependent (a localized runtime
// translates it); the non-localized, durable rung is the crashing-THREAD token
// "readkeythreadproc". That thread token is deliberately narrower than the bare
// module name "psconsolereadline" — PSReadLine frames appear on the stack of
// nearly every interactive pwsh crash (including the #2170 *output*-buffer
// FailFast), so a bare module token, checked ahead of the pipe/handle causes,
// would swallow a pipe/handle crash that merely has a PSReadLine frame. Like the
// host FailFast, this IS the crash (not its cause), so it is checked ahead of
// the pipe/handle *cause* signatures.
var consoleInputSignatures = []string{
	"cannot read keys when either application does not have a console", // .NET InvalidOperationException message (English-locale-dependent rung)
	"readkeythreadproc", // Microsoft.PowerShell.PSConsoleReadLine.ReadKeyThreadProc — the crashing reader thread (durable, non-localized rung)
}

// ClassifyConsoleFault maps observed child-failure text onto the closed class.
// Feed it the child's DRAINED STDERR or .NET-runtime unhandled-exception text
// (the managed stack) — NOT a Windows WER "Event 1000 / Faulting module name:"
// banner: that banner names Microsoft.PowerShell.ConsoleHost.dll as the module
// even for an INPUT crash, so the bare host token would then mis-route it to
// CONSOLE_HOST_FAILFAST. (The stderr / .NET-runtime form of the input crash
// traverses PSReadLine's ReadKeyThreadProc, which does not carry the ConsoleHost
// namespace token, so it classifies correctly.) ok=false means "not a console
// fault" — a plain tool error stays a plain tool error; the classifier never
// guesses.
func ClassifyConsoleFault(detail string) (ConsoleFaultClass, bool) {
	d := strings.ToLower(detail)
	if d == "" {
		return "", false
	}
	for _, sig := range consoleHostSignatures {
		if strings.Contains(d, sig) {
			return ConsoleHostFailFast, true
		}
	}
	for _, sig := range consoleInputSignatures {
		if strings.Contains(d, sig) {
			return ConsoleInputLost, true
		}
	}
	for _, sig := range consolePipeSignatures {
		if strings.Contains(d, sig) {
			return ConsolePipeLost, true
		}
	}
	for _, sig := range consoleHandleSignatures {
		if strings.Contains(d, sig) {
			return ConsoleHandleLost, true
		}
	}
	return "", false
}

// werFailFastCode is the NTSTATUS a Windows Error Reporting Application-Error
// (Event 1000) banner carries for a __fastfail / Environment.FailFast
// termination: STATUS_STACK_BUFFER_OVERRUN. It is the WER-side signature of the
// #2170 crash class — the pipe error (Win32 0xE9) throws a HostException the
// runtime converts into a FailFast, and the process dies with this code. An
// UNHANDLED managed exception instead dies with the CLR SEH code 0xe0434352 AND
// logs a paired .NET Runtime Event 1026 (already covered by ClassifyConsoleFault
// on its managed stack). Keying the WER path on the FailFast code alone is thus
// exactly the "1026-less" class #3513 targets — not a re-catch of the 1026 path.
const werFailFastCode = "0xc0000409"

// werConsoleHostApps is the CLOSED set of faulting-application names whose WER
// FailFast we treat as a child console-host fault. The generic 0xc0000409 is NOT
// console-specific on its own — any __fastfail carries it — so it becomes a
// console fault ONLY in co-occurrence with a console-host/shell faulting app.
// That co-occurrence is precisely the structured-field context the
// single-substring ClassifyConsoleFault could not express (see its comment, and
// the "leave them to a future WER ingester with structured fields" note on
// consoleHostSignatures). Deliberately excludes the operator's OUTER terminal
// (WindowsTerminal.exe / wt.exe): the console-fault class names a CHILD console
// surface, not the host UI the operator is sitting in front of.
var werConsoleHostApps = map[string]bool{
	"pwsh.exe":        true, // PowerShell 7.x — the #2170 witnessed shell
	"powershell.exe":  true, // Windows PowerShell 5.1
	"conhost.exe":     true, // the classic console host for a child shell
	"openconsole.exe": true, // the modern console host (ConPTY)
	"cmd.exe":         true, // the legacy shell
}

// ClassifyConsoleFaultWER maps a Windows Error Reporting Application-Error
// (Event 1000) banner's STRUCTURED FIELDS onto the closed console-fault class.
// It is the deliberate structured-field counterpart to ClassifyConsoleFault:
// the WER banner names Microsoft.PowerShell.ConsoleHost.dll as the faulting
// module even for an INPUT crash, so feeding its free text to the single-
// substring classifier would mis-route it (its own doc comment says so). From
// the banner alone the input/output mechanism is indistinguishable, so a
// recognized console-host FailFast folds to the COARSE CONSOLE_HOST_FAILFAST —
// an honest "a child console host hard-terminated", never an over-claimed
// input-vs-output verdict the fields cannot support. Fail-closed: a non-console
// faulting app, or any exception code that is not the FailFast code, returns
// ok=false (a plain app crash stays a plain app crash; the classifier never
// guesses). The faulting module is intentionally NOT a gate — a FailFast most
// often surfaces in KERNELBASE.dll/ntdll.dll (where RaiseFailFastException
// lives), not in ConsoleHost.dll, so gating on the module would MISS real
// FailFasts; the module is carried in the row Detail for forensics instead.
func ClassifyConsoleFaultWER(faultingApp, exceptionCode string) (ConsoleFaultClass, bool) {
	app := strings.ToLower(strings.TrimSpace(faultingApp))
	code := strings.ToLower(strings.TrimSpace(exceptionCode))
	if app == "" || code == "" {
		return "", false
	}
	if !werConsoleHostApps[app] {
		return "", false
	}
	if code != werFailFastCode {
		return "", false
	}
	return ConsoleHostFailFast, true
}

// ClassifyDrainError classifies the error a child-output drain returned.
// callLive reports whether the call was still live (no terminal exit observed)
// when the drain ended: an EOF on a live call means the console/PTY surface
// vanished (a fault); an EOF after a clean exit is just the end of output.
func ClassifyDrainError(err error, callLive bool) (ConsoleFaultClass, bool) {
	if err == nil {
		return "", false
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		if callLive {
			return ConsolePTYEOF, true
		}
		return "", false
	}
	return ClassifyConsoleFault(err.Error())
}

// ConsoleFaultEvent is one append-only, byte-bounded crash record. It carries
// identity and the typed class — never payload bytes — so the row is safe to
// keep durable and cheap to search.
type ConsoleFaultEvent struct {
	Schema  string            `json:"schema,omitempty"`
	Class   ConsoleFaultClass `json:"class"`
	AtMS    int64             `json:"at_unix_ms"`
	CallID  string            `json:"call_id,omitempty"`
	Tool    string            `json:"tool,omitempty"`
	Session string            `json:"session,omitempty"`
	Surface string            `json:"surface,omitempty"`
	// Detail keeps the leading crash signature (bounded to
	// consoleFaultDetailLimit bytes) so the exact Event-Viewer string stays
	// greppable in the fak surface.
	Detail string `json:"detail,omitempty"`
}

// ValidateConsoleFaultEvent enforces the closed vocabulary and the required
// fields on one row.
func ValidateConsoleFaultEvent(ev ConsoleFaultEvent) error {
	if !validConsoleFaultClass(ev.Class) {
		return fmt.Errorf("toolprocgate console fault: unknown class %q", ev.Class)
	}
	if ev.AtMS <= 0 {
		return fmt.Errorf("toolprocgate console fault: at_unix_ms must be positive")
	}
	if len(ev.Detail) > consoleFaultDetailLimit {
		return fmt.Errorf("toolprocgate console fault: detail exceeds %d bytes", consoleFaultDetailLimit)
	}
	return nil
}

// boundConsoleFaultDetail truncates crash detail to the durable bound,
// keeping the head — the crash banner leads with the searchable signature.
func boundConsoleFaultDetail(s string) string {
	if len(s) <= consoleFaultDetailLimit {
		return s
	}
	return s[:consoleFaultDetailLimit]
}

// BoundConsoleFaultDetail is the exported bound for callers that build a row
// directly — an event-log ingester classifying real OS crashes — instead of
// through ExitConsoleFault. Keeps the head so the leading signature survives
// and the row satisfies ValidateConsoleFaultEvent's detail limit.
func BoundConsoleFaultDetail(s string) string { return boundConsoleFaultDetail(s) }

// ExitConsoleFault records a child console/shell/PTY/renderer fault as a
// STRUCTURED child failure: the journal gets a normal exit(status=error) — so
// the parent's table stays consistent, enforcement stays scoped to the dead
// call, and sibling procs are untouched — and the typed fault row is returned
// for the embedder's durable sink. Fail-closed: an unknown class or an unknown
// call is refused before anything is journaled.
func (s *Supervisor) ExitConsoleFault(callID string, nowMS int64, class ConsoleFaultClass, surface ConsoleSurface, detail string) (ConsoleFaultEvent, error) {
	if !validConsoleFaultClass(class) {
		return ConsoleFaultEvent{}, fmt.Errorf("toolprocgate: unknown console fault class %q", class)
	}
	// Snapshot the child's identity from its spawn row before the exit.
	var tool, session string
	s.mu.Lock()
	for _, ev := range s.events {
		if ev.Kind == toolproc.EvSpawn && ev.CallID == callID {
			tool, session = ev.Tool, ev.Session
			break
		}
	}
	s.mu.Unlock()
	if err := s.Exit(callID, nowMS, "error"); err != nil {
		return ConsoleFaultEvent{}, err
	}
	ev := ConsoleFaultEvent{
		Schema:  ConsoleFaultEventSchema,
		Class:   class,
		AtMS:    nowMS,
		CallID:  callID,
		Tool:    tool,
		Session: session,
		Surface: string(surface),
		Detail:  boundConsoleFaultDetail(detail),
	}
	// Remember the fault so AdmitSpawn can contain the blast radius of the NEXT
	// spawn (bounded ring: a storm cannot grow this without limit).
	s.mu.Lock()
	s.recentFaults = append(s.recentFaults, ev)
	if over := len(s.recentFaults) - recentFaultRingCap; over > 0 {
		s.recentFaults = s.recentFaults[over:]
	}
	s.mu.Unlock()
	return ev, nil
}

// AdmitSpawn is the containment gate: BEFORE the embedder launches an agent onto
// a console surface, it asks whether that spawn would let a terminal crash
// cascade. The verdict folds the console faults this supervisor has recorded
// (ExitConsoleFault) against pol, so a surface stuck in a re-crash loop, a
// surface already at its blast-radius cap, or a host in a cross-session fault
// storm is refused/held instead of admitted. A ContainAdmit verdict is the only
// green light; every other verdict is a protective refusal the embedder honors
// by placing the agent elsewhere or holding. Pure w.r.t. the recorded history —
// it reads the fault ring and req, never a clock of its own.
func (s *Supervisor) AdmitSpawn(req ContainmentRequest, pol ContainmentPolicy) ContainmentDecision {
	s.mu.Lock()
	faults := make([]ConsoleFaultEvent, len(s.recentFaults))
	copy(faults, s.recentFaults)
	s.mu.Unlock()
	return DecideContainment(pol, faults, req)
}

// AppendConsoleFaultEvent writes one validated row as a JSONL line.
func AppendConsoleFaultEvent(w io.Writer, ev ConsoleFaultEvent) error {
	if ev.Schema == "" {
		ev.Schema = ConsoleFaultEventSchema
	}
	return jsonlledger.AppendValidated(w, ev, ValidateConsoleFaultEvent)
}

// ParseConsoleFaultEvents reads console-fault JSONL fail-closed: unknown
// fields and unknown classes are refused, so a drifted or fabricated row can
// never enter an operator report as a legitimate crash record. Blank lines and
// `#` comments are journal furniture, not rows.
func ParseConsoleFaultEvents(r io.Reader) ([]ConsoleFaultEvent, error) {
	var out []ConsoleFaultEvent
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" || strings.HasPrefix(raw, "#") {
			continue
		}
		dec := json.NewDecoder(strings.NewReader(raw))
		dec.DisallowUnknownFields()
		var ev ConsoleFaultEvent
		if err := dec.Decode(&ev); err != nil {
			return nil, fmt.Errorf("toolprocgate console fault: line %d: %v", line, err)
		}
		if err := ValidateConsoleFaultEvent(ev); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		out = append(out, ev)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// ConsoleFaultCounts is the searchable attention summary.
type ConsoleFaultCounts struct {
	ByClass   map[string]int `json:"by_class"`
	BySurface map[string]int `json:"by_surface"`
	ByTool    map[string]int `json:"by_tool"`
	BySession map[string]int `json:"by_session"`
}

// ConsoleFaultReport is the folded operator view: how many child console
// surfaces died, in which class, on which tools — the durable answer to the
// question the 2026-07-01 audit had to reconstruct from Event Viewer.
type ConsoleFaultReport struct {
	Schema    string              `json:"schema"`
	Rows      int                 `json:"rows"`
	Counts    ConsoleFaultCounts  `json:"counts"`
	EventRows []ConsoleFaultEvent `json:"event_rows"`
}

// ConsoleFaultReportFromEvents folds rows into the report.
func ConsoleFaultReportFromEvents(events []ConsoleFaultEvent) ConsoleFaultReport {
	rep := ConsoleFaultReport{
		Schema: ConsoleFaultReportSchema,
		Rows:   len(events),
		Counts: ConsoleFaultCounts{
			ByClass:   map[string]int{},
			BySurface: map[string]int{},
			ByTool:    map[string]int{},
			BySession: map[string]int{},
		},
		EventRows: append([]ConsoleFaultEvent(nil), events...),
	}
	for _, ev := range events {
		rep.Counts.ByClass[string(ev.Class)]++
		if ev.Surface != "" {
			rep.Counts.BySurface[ev.Surface]++
		}
		if ev.Tool != "" {
			rep.Counts.ByTool[ev.Tool]++
		}
		if ev.Session != "" {
			rep.Counts.BySession[ev.Session]++
		}
	}
	return rep
}

// RenderConsoleFaultReport writes the operator text view: one summary line,
// then class counts sorted for stable output, then one line per row.
func RenderConsoleFaultReport(w io.Writer, rep ConsoleFaultReport) {
	fmt.Fprintf(w, "console faults: %d row(s)\n", rep.Rows)
	classes := make([]string, 0, len(rep.Counts.ByClass))
	for c := range rep.Counts.ByClass {
		classes = append(classes, c)
	}
	sort.Strings(classes)
	for _, c := range classes {
		fmt.Fprintf(w, "  %-24s %d\n", c, rep.Counts.ByClass[c])
	}
	for _, ev := range rep.EventRows {
		fmt.Fprintf(w, "  %s call=%s tool=%s session=%s surface=%s\n",
			ev.Class, ev.CallID, ev.Tool, ev.Session, ev.Surface)
	}
}
