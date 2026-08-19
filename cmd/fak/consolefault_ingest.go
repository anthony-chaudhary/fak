package main

// consolefault_ingest.go — the LIVE PRODUCER the #2170 audit found missing.
//
// internal/toolprocgate shipped the console-fault vocabulary, the classifier,
// the row schema, and the operator fold — but nothing fed real crashes into it,
// so `fak toolproc console-faults` could only ever fold a journal a supervisor
// embedder wrote by hand. This file closes the loop for the classes that
// recur on this host: the pwsh / .NET console-host crash lives in the Windows
// Application event log as either a .NET Runtime unhandled-exception dump
// (Event 1026) carrying the managed stack, OR — when the crash is a FailFast
// that logs no managed stack — a WER Application Error banner (Event 1000)
// carrying structured fields (#3513). We read both, run each through the SAME
// classifiers the unit tests pin (ClassifyConsoleFault for the managed stack,
// ClassifyConsoleFaultWER for the WER banner), and emit durable ConsoleFaultEvent
// rows — so the class becomes searchable from the fak surface instead of
// Windows Event Viewer only.
//
// The mapper here is pure and OS-independent (testable without an event log);
// the OS read lives in consolefault_ingest_windows.go behind a build tag, with
// a refusal stub for other platforms.

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// winEventRecord is one parsed Windows Application-log record relevant to the
// console-host fault class. Two shapes flow through here, dispatched by Provider
// and ID: the .NET Runtime Event 1026 dump whose Message carries the managed
// exception + stack that ClassifyConsoleFault reads, and the WER Application
// Error Event 1000 banner whose STRUCTURED FIELDS ClassifyConsoleFaultWER reads
// (the 1026-less FailFast class from #3513).
type winEventRecord struct {
	Provider string `json:"provider"`
	ID       int    `json:"id"`
	TimeMS   int64  `json:"time_ms"`
	Message  string `json:"message"`
}

// werDedupWindowMS bounds how close in time a WER Event-1000 FailFast must be to
// a .NET Event-1026 console fault (for the SAME tool) to be judged the same
// crash and dropped. A single crash logs both records within the same second;
// WER can lag the .NET Runtime log slightly, so the window is generous. This is
// what makes the WER path additive — it contributes only the "1026-less" crashes
// #3513 targets, never a second row for a crash the 1026 path already named.
const werDedupWindowMS = 10_000

// consoleFaultEventsFromWinRecords maps parsed event-log records onto durable
// ConsoleFaultEvent rows. It runs in two passes over the mixed record set:
//
//	Pass 1 (.NET Runtime Event 1026) feeds the managed exception text to
//	ClassifyConsoleFault — the input/output/pipe/handle mechanism is legible on
//	the managed stack. Only the .NET-runtime managed text is fed, honoring the
//	narrowed ClassifyConsoleFault contract (a WER banner must NOT be fed there).
//
//	Pass 2 (WER Application Error Event 1000) parses the banner's structured
//	fields and classifies via ClassifyConsoleFaultWER — the 1026-less FailFast
//	class. A WER FailFast that pairs with a Pass-1 fault for the same tool within
//	werDedupWindowMS is the SAME crash and is dropped, so the fold is not
//	double-counted.
//
// Fail-closed throughout: a crash neither classifier recognizes is DROPPED,
// never folded as a fabricated crash record.
func consoleFaultEventsFromWinRecords(recs []winEventRecord, fallbackMS int64) []toolprocgate.ConsoleFaultEvent {
	out := make([]toolprocgate.ConsoleFaultEvent, 0, len(recs))
	// Times of the Pass-1 console faults, keyed by tool, so a Pass-2 WER
	// FailFast that pairs with one can be recognized as the same crash.
	faultTimes := map[string][]int64{}

	// Pass 1: .NET Runtime Event 1026 — the managed-stack path (unchanged).
	for _, r := range recs {
		if !isDotNetRuntime1026(r) {
			continue
		}
		class, ok := toolprocgate.ClassifyConsoleFault(r.Message)
		if !ok {
			continue
		}
		at := recAtMS(r, fallbackMS)
		tool := toolFromMessage(r.Message)
		out = append(out, toolprocgate.ConsoleFaultEvent{
			Class:   class,
			AtMS:    at,
			Tool:    tool,
			Surface: string(toolprocgate.ConsoleSurfaceStderr),
			Detail:  toolprocgate.BoundConsoleFaultDetail(flattenWhitespace(r.Message)),
		})
		faultTimes[tool] = append(faultTimes[tool], at)
	}

	// Pass 2: WER Application Error Event 1000 — the 1026-less FailFast path.
	for _, r := range recs {
		if !isWERAppError1000(r) {
			continue
		}
		app, module, code := parseWERFields(r.Message)
		class, ok := toolprocgate.ClassifyConsoleFaultWER(app, code)
		surface := ""
		if !ok && strings.EqualFold(strings.TrimSpace(app), "WindowsTerminal.exe") && strings.TrimSpace(code) != "" {
			class, ok = toolprocgate.ConsoleRendererExit, true
			surface = string(toolprocgate.ConsoleSurfaceRenderer)
		}
		if !ok {
			continue
		}
		tool := toolFromAppName(app)
		at := recAtMS(r, fallbackMS)
		if pairedWith1026(faultTimes[tool], at, werDedupWindowMS) {
			continue // same crash the 1026 path already named — do not double-count.
		}
		out = append(out, toolprocgate.ConsoleFaultEvent{
			Class: class,
			AtMS:  at,
			Tool:  tool,
			// A child FailFast has no attributable stdio surface. A Windows
			// Terminal process exit does: the renderer-owned visible surface.
			Surface: surface,
			Detail:  toolprocgate.BoundConsoleFaultDetail(werDetail(app, module, code)),
		})
	}
	return out
}

// isDotNetRuntime1026 recognizes the .NET Runtime unhandled-exception dump.
func isDotNetRuntime1026(r winEventRecord) bool {
	return r.ID == 1026 && strings.EqualFold(strings.TrimSpace(r.Provider), ".NET Runtime")
}

// isWERAppError1000 recognizes the WER Application Error crash banner.
func isWERAppError1000(r winEventRecord) bool {
	return r.ID == 1000 && strings.EqualFold(strings.TrimSpace(r.Provider), "Application Error")
}

// recAtMS returns the record time, falling back to the ingest wall-clock when
// the record carried no usable timestamp.
func recAtMS(r winEventRecord, fallbackMS int64) int64 {
	if r.TimeMS > 0 {
		return r.TimeMS
	}
	return fallbackMS
}

// toolFromMessage derives the tool identity from a .NET dump message: any
// PowerShell edition folds to "pwsh" (matching the WER path's normalization),
// otherwise unknown.
func toolFromMessage(message string) string {
	low := strings.ToLower(message)
	if strings.Contains(low, "pwsh") || strings.Contains(low, "powershell") {
		return "pwsh"
	}
	return ""
}

// toolFromAppName derives the tool identity from a WER faulting-application
// name (e.g. "pwsh.exe" -> "pwsh"). Both PowerShell editions fold to "pwsh" so
// the WER row's tool key matches the .NET path's — required for the dedup to
// pair a 1000 FailFast with its 1026 fault.
func toolFromAppName(app string) string {
	a := strings.ToLower(strings.TrimSpace(app))
	a = strings.TrimSuffix(a, ".exe")
	if a == "powershell" {
		return "pwsh"
	}
	return a
}

// parseWERFields pulls the three structured fields ClassifyConsoleFaultWER needs
// (plus the module, for Detail) out of a WER Event-1000 banner. It tolerates
// both the native multi-line banner and a whitespace-flattened one: each value
// is taken up to the first comma/newline, then reduced to its first whitespace
// token (an app/module basename or an exception code never contains a space).
func parseWERFields(message string) (app, module, code string) {
	app = werFieldValue(message, "faulting application name:")
	module = werFieldValue(message, "faulting module name:")
	code = werFieldValue(message, "exception code:")
	return app, module, code
}

// werFieldValue extracts the value following a "Label:" field in a WER banner,
// case-insensitively. Returns "" when the label is absent.
func werFieldValue(text, label string) string {
	idx := strings.Index(strings.ToLower(text), label)
	if idx < 0 {
		return ""
	}
	rest := text[idx+len(label):]
	for i, r := range rest {
		if r == ',' || r == '\n' || r == '\r' {
			rest = rest[:i]
			break
		}
	}
	if f := strings.Fields(rest); len(f) > 0 {
		return f[0]
	}
	return ""
}

// pairedWith1026 reports whether any recorded Pass-1 fault time is within
// windowMS of at — i.e. the WER FailFast is the same crash.
func pairedWith1026(times []int64, at, windowMS int64) bool {
	for _, t := range times {
		d := at - t
		if d < 0 {
			d = -d
		}
		if d <= windowMS {
			return true
		}
	}
	return false
}

// werDetail builds the greppable one-line Detail for a WER FailFast row from the
// parsed structured fields.
func werDetail(app, module, code string) string {
	parts := make([]string, 0, 3)
	if app != "" {
		parts = append(parts, "app="+app)
	}
	if module != "" {
		parts = append(parts, "module="+module)
	}
	if code != "" {
		parts = append(parts, "code="+code)
	}
	return strings.TrimSpace("WER Event 1000 FailFast " + strings.Join(parts, " "))
}

// flattenWhitespace collapses a multi-line event message to a single greppable
// line so each JSONL row stays one physical line in the journal.
func flattenWhitespace(s string) string { return strings.Join(strings.Fields(s), " ") }

// writeConsoleFaultSnapshot writes the ingested rows as the current event-log
// PROJECTION (truncate, not append): re-running ingest is idempotent because the
// file mirrors the OS log's current window rather than accumulating duplicates.
// This is deliberately a separate file from the append-only hook journal.
func writeConsoleFaultSnapshot(path string, events []toolprocgate.ConsoleFaultEvent) error {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, ev := range events {
		if err := toolprocgate.AppendConsoleFaultEvent(f, ev); err != nil {
			return err
		}
	}
	return nil
}

// runConsoleFaultIngest is the live wiring for `fak toolproc console-faults
// --ingest`: read this host's Windows event log for console-host crashes,
// classify each, write the durable snapshot, and fold it to the operator view.
func runConsoleFaultIngest(stdout, stderr io.Writer, outPath string, since time.Duration, nowMS int64, asJSON bool) int {
	recs, errMsg := gatherWinConsoleFaultRecords(since)
	if errMsg != "" {
		fmt.Fprintf(stderr, "fak toolproc console-faults: ingest: %s\n", errMsg)
		return 1
	}
	events := consoleFaultEventsFromWinRecords(recs, nowMS)
	if err := writeConsoleFaultSnapshot(outPath, events); err != nil {
		fmt.Fprintf(stderr, "fak toolproc console-faults: ingest: write %s: %v\n", outPath, err)
		return 1
	}
	report := toolprocgate.ConsoleFaultReportFromEvents(events)
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak toolproc console-faults")
	}
	toolprocgate.RenderConsoleFaultReport(stdout, report)
	fmt.Fprintf(stdout, "ingested %d console-fault row(s) from the Windows event log -> %s\n", len(events), outPath)
	return 0
}
