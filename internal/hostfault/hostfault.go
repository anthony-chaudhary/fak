package hostfault

// Host-fault boundary (#2170 sibling): the console-fault vocabulary in
// internal/toolprocgate names the class where a CHILD's terminal/shell/PTY/TUI
// surface crashes. This package names the sibling class the same #2170 host
// audit surfaced but which is deliberately NOT a console fault — HOST-level
// failures that destabilize the fleet without being one of fak's child tool
// processes crashing:
//
//   - a Windows Update package install that FAILED because a version of the
//     package was running (WindowsUpdateClient Event 20, e.g. 0x80073D02),
//   - the Windows Update ORCHESTRATOR worker itself faulting (WER 1001 with the
//     MoUpdateOrchestrator / MoUsoCoreWorker identity),
//   - a GPU driver LIVE-KERNEL / TDR watchdog event (WER 1001 LiveKernelEvent,
//     video-TDR bucket 141 / vendor watchdog dumps),
//   - an app-termination (hang) failure dump (WER 1001 AppTermFailureEvent).
//
// These live in the Windows event log across several providers; before this
// leaf nothing in-process NAMED them, so correlating "my agents died" with "the
// host was updating / the GPU reset" meant manual Event Viewer correlation.
//
// Same idiom as toolprocgate.ConsoleFaultEvent: a closed class vocabulary, a
// fail-closed classifier over parsed event-log records, a byte-bounded durable
// JSONL row, and a folded operator report. Kept in its OWN package (not
// toolprocgate) because a host fault is NOT a child-tool-process gate event;
// folding it into the console-fault closed set would corrupt exactly the
// terminal/shell/PTY/TUI boundary that vocabulary exists to protect.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// HostFaultEventSchema stamps one durable host-fault row.
const HostFaultEventSchema = "fak.hostfault.host-fault-event.v1"

// HostFaultReportSchema stamps the folded operator report.
const HostFaultReportSchema = "fak.hostfault.host-fault-report.v1"

// hostFaultDetailLimit bounds the recorded detail: enough to keep the leading
// searchable signature, never enough to become the storage leak (mirrors the
// console-fault bound).
const hostFaultDetailLimit = 512

// HostFaultClass is the CLOSED vocabulary of host-level fault classes. Parse and
// record fail closed: an unknown class is refused, never folded into an operator
// report as legitimate.
type HostFaultClass string

const (
	// HostWUInstallFailure: a Windows Update package install failed. The
	// witnessed class is WindowsUpdateClient Event 20 with 0x80073D02 ("a
	// version required by the update is currently running") — a Store/MSIX
	// package that could not update because it was open.
	HostWUInstallFailure HostFaultClass = "WU_INSTALL_FAILURE"
	// HostWUOrchestratorFault: the Windows Update ORCHESTRATOR worker faulted —
	// WER 1001 whose faulting identity is the MoUpdateOrchestrator /
	// MoUsoCoreWorker update-scan worker (SearchForAllUpdatesWithUpdateOptions).
	HostWUOrchestratorFault HostFaultClass = "WU_ORCHESTRATOR_FAULT"
	// HostGPULiveKernel: a GPU driver live-kernel / TDR watchdog event — WER
	// 1001 LiveKernelEvent carrying a GPU signal (video-TDR bucket 141, a vendor
	// watchdog dump under \LiveKernelReports). The dominant host-instability
	// class on the audited host (AMD watchdog / VIDEO_TDR_ERROR).
	HostGPULiveKernel HostFaultClass = "GPU_LIVEKERNEL"
	// HostAppTermFailure: an application-termination (hang) failure dump — WER
	// 1001 AppTermFailureEvent, an app killed for not responding.
	HostAppTermFailure HostFaultClass = "APP_TERM_FAILURE"
)

// validHostFaultClass is the closed-set membership check.
func validHostFaultClass(c HostFaultClass) bool {
	switch c {
	case HostWUInstallFailure, HostWUOrchestratorFault, HostGPULiveKernel, HostAppTermFailure:
		return true
	}
	return false
}

// WinFaultRecord is one parsed Windows event-log record fed to the classifier.
// Provider+ID pin the source; App is the faulting-app / package / bucket
// identity; Message is the full event text the signature rungs read. It is a
// pure value — the classifier does no I/O, so classification is unit-testable
// without an event log.
type WinFaultRecord struct {
	Provider string
	ID       int
	TimeMS   int64
	App      string
	Message  string
}

// gpuLiveKernelSignals gate the GPU_LIVEKERNEL class: a LiveKernelEvent is only
// folded as a GPU fault when it carries a GPU/video/driver token. A bare
// LiveKernelEvent with no GPU signal is DROPPED (fail-closed) rather than
// mis-attributed to the GPU — a non-GPU live-kernel event is a different class
// this leaf does not yet name.
var gpuLiveKernelSignals = []string{
	"livekernelreports", // \WINDOWS\LiveKernelReports\AMD_WATCHDOG\... attached-file path
	"amd_watchdog",
	"amd_report",
	"video_tdr",
	"dxgkrnl",  // the DirectX graphics kernel
	"nvlddmkm", // NVIDIA display driver
	"igdkmd",   // Intel display driver
	" tdr",     // timeout detection & recovery (space-guarded to avoid substring noise)
	"amd",
	"nvidia",
}

func hasGPULiveKernelSignal(msg, app string) bool {
	for _, sig := range gpuLiveKernelSignals {
		if strings.Contains(msg, sig) || strings.Contains(app, sig) {
			return true
		}
	}
	return false
}

// ClassifyHostFault maps one parsed event-log record onto the closed host-fault
// class. Fail-closed: a record that is not one of the four witnessed host-fault
// classes returns ok=false and is DROPPED by callers, never fabricated into a
// row. Ordering is by specificity — the provider/id-pinned WU install failure
// first, then the identity-pinned orchestrator fault (which is itself a WER 1001
// and would otherwise be swallowed by the generic app-term/GPU message rungs),
// then the GPU-signalled live-kernel event, then the generic app-termination
// dump.
func ClassifyHostFault(r WinFaultRecord) (HostFaultClass, bool) {
	prov := strings.ToLower(r.Provider)
	msg := strings.ToLower(r.Message)
	app := strings.ToLower(r.App)

	// Windows Update install failure — provider + event id pinned.
	if strings.Contains(prov, "windowsupdateclient") && r.ID == 20 {
		return HostWUInstallFailure, true
	}
	// Windows Update orchestrator worker fault — identity pinned (more specific
	// than the generic WER 1001 message rungs below, so checked first).
	if containsAny(app, "moupdateorchestrator", "mousocoreworker") ||
		containsAny(msg, "moupdateorchestrator", "mousocoreworker") {
		return HostWUOrchestratorFault, true
	}
	// GPU driver live-kernel / TDR — only when a GPU signal co-occurs.
	if strings.Contains(msg, "livekernelevent") && hasGPULiveKernelSignal(msg, app) {
		return HostGPULiveKernel, true
	}
	// Application-termination (hang) failure dump.
	if strings.Contains(msg, "apptermfailureevent") {
		return HostAppTermFailure, true
	}
	return "", false
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// HostFaultEvent is one append-only, byte-bounded host-fault record. It carries
// the typed class and bounded identity — never a dump payload — so the row is
// safe to keep durable and cheap to search.
type HostFaultEvent struct {
	Schema string         `json:"schema,omitempty"`
	Class  HostFaultClass `json:"class"`
	AtMS   int64          `json:"at_unix_ms"`
	// Source names the event-log origin, e.g. "WindowsUpdateClient/20" or
	// "Windows Error Reporting/1001".
	Source string `json:"source,omitempty"`
	// App is the faulting app / package / bucket identity.
	App string `json:"app,omitempty"`
	// Code is the class-specific code when one is present: the WU error hex
	// (0x80073D02), the TDR/live-kernel bucket, etc.
	Code string `json:"code,omitempty"`
	// Detail keeps the leading event signature (bounded to hostFaultDetailLimit
	// bytes) so the exact Event-Viewer string stays greppable.
	Detail string `json:"detail,omitempty"`
}

// ValidateHostFaultEvent enforces the closed vocabulary and required fields.
func ValidateHostFaultEvent(ev HostFaultEvent) error {
	if !validHostFaultClass(ev.Class) {
		return fmt.Errorf("hostfault: unknown class %q", ev.Class)
	}
	if ev.AtMS <= 0 {
		return fmt.Errorf("hostfault: at_unix_ms must be positive")
	}
	if len(ev.Detail) > hostFaultDetailLimit {
		return fmt.Errorf("hostfault: detail exceeds %d bytes", hostFaultDetailLimit)
	}
	return nil
}

// BoundHostFaultDetail truncates detail to the durable bound, keeping the head —
// the event banner leads with the searchable signature — so the row satisfies
// ValidateHostFaultEvent's detail limit.
func BoundHostFaultDetail(s string) string {
	if len(s) <= hostFaultDetailLimit {
		return s
	}
	return s[:hostFaultDetailLimit]
}

// AppendHostFaultEvent writes one validated row as a JSONL line.
func AppendHostFaultEvent(w io.Writer, ev HostFaultEvent) error {
	if ev.Schema == "" {
		ev.Schema = HostFaultEventSchema
	}
	return jsonlledger.AppendValidated(w, ev, ValidateHostFaultEvent)
}

// ParseHostFaultEvents reads host-fault JSONL fail-closed: unknown fields and
// unknown classes are refused, so a drifted or fabricated row can never enter an
// operator report as a legitimate host fault. Blank lines and `#` comments are
// journal furniture, not rows.
func ParseHostFaultEvents(r io.Reader) ([]HostFaultEvent, error) {
	var out []HostFaultEvent
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
		var ev HostFaultEvent
		if err := dec.Decode(&ev); err != nil {
			return nil, fmt.Errorf("hostfault: line %d: %v", line, err)
		}
		if err := ValidateHostFaultEvent(ev); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		out = append(out, ev)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// HostFaultCounts is the searchable attention summary.
type HostFaultCounts struct {
	ByClass  map[string]int `json:"by_class"`
	BySource map[string]int `json:"by_source"`
	ByApp    map[string]int `json:"by_app"`
}

// HostFaultReport is the folded operator view: how many host faults, in which
// class, from which source and app — the durable answer to "was the host
// updating / did the GPU reset while my agents died?"
type HostFaultReport struct {
	Schema    string           `json:"schema"`
	Rows      int              `json:"rows"`
	Counts    HostFaultCounts  `json:"counts"`
	EventRows []HostFaultEvent `json:"event_rows"`
}

// HostFaultReportFromEvents folds rows into the report.
func HostFaultReportFromEvents(events []HostFaultEvent) HostFaultReport {
	rep := HostFaultReport{
		Schema: HostFaultReportSchema,
		Rows:   len(events),
		Counts: HostFaultCounts{
			ByClass:  map[string]int{},
			BySource: map[string]int{},
			ByApp:    map[string]int{},
		},
		EventRows: append([]HostFaultEvent(nil), events...),
	}
	for _, ev := range events {
		rep.Counts.ByClass[string(ev.Class)]++
		if ev.Source != "" {
			rep.Counts.BySource[ev.Source]++
		}
		if ev.App != "" {
			rep.Counts.ByApp[ev.App]++
		}
	}
	return rep
}

// RenderHostFaultReport writes the operator text view: one summary line, then
// class counts sorted for stable output, then one line per row.
func RenderHostFaultReport(w io.Writer, rep HostFaultReport) {
	fmt.Fprintf(w, "host faults: %d row(s)\n", rep.Rows)
	classes := make([]string, 0, len(rep.Counts.ByClass))
	for c := range rep.Counts.ByClass {
		classes = append(classes, c)
	}
	sort.Strings(classes)
	for _, c := range classes {
		fmt.Fprintf(w, "  %-24s %d\n", c, rep.Counts.ByClass[c])
	}
	for _, ev := range rep.EventRows {
		fmt.Fprintf(w, "  %s source=%s app=%s code=%s\n",
			ev.Class, ev.Source, ev.App, ev.Code)
	}
}
