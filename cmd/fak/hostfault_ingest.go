package main

// hostfault_ingest.go — the LIVE PRODUCER for the #2170 sibling class: HOST
// faults that destabilize the fleet without being one of fak's child tool
// processes crashing (a Windows Update install failure, the update orchestrator
// worker faulting, a GPU driver live-kernel / TDR watchdog, an app-termination
// hang). internal/hostfault ships the closed vocabulary, classifier, row schema,
// and operator fold; this file feeds it from THIS host's Windows event log so
// the classes become searchable from the fak surface instead of Event Viewer
// only — the same shape as consolefault_ingest.go, a different (broader) event
// surface.
//
// The mapper here is pure and OS-independent (testable without an event log);
// the OS read lives in hostfault_ingest_windows.go behind a build tag, with a
// refusal stub for other platforms.

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostfault"
)

var (
	// hexCodeRE pulls the leading Win32/HRESULT hex code from a message
	// (a Windows Update install failure carries "error 0x80073D02: <pkg>").
	hexCodeRE = regexp.MustCompile(`0x[0-9A-Fa-f]{4,}`)
	// liveKernelCatRE pulls the LiveKernelReports category folder
	// (AMD_WATCHDOG, AMD_REPORT_UM, ...) from an attached-file path — the most
	// specific GPU-fault code a WER 1001 live-kernel event carries.
	liveKernelCatRE = regexp.MustCompile(`(?i)LiveKernelReports[\\/]+([A-Za-z0-9_]+)`)
)

// extractHexCode returns the first hex code in s, or "".
func extractHexCode(s string) string { return hexCodeRE.FindString(s) }

// extractLiveKernelCategory returns the LiveKernelReports category folder in s
// (upper-cased for a stable code), or "".
func extractLiveKernelCategory(s string) string {
	m := liveKernelCatRE.FindStringSubmatch(s)
	if len(m) < 2 {
		return ""
	}
	return strings.ToUpper(m[1])
}

// extractWUPackage returns the package identity a WindowsUpdateClient Event 20
// names after its error code (".. error 0x80073D02: <pkg>."), or "".
func extractWUPackage(s string) string {
	i := strings.LastIndex(s, ": ")
	if i < 0 {
		return ""
	}
	return strings.TrimRight(strings.TrimSpace(s[i+2:]), ".")
}

// hostFaultSource names the event-log origin of a record as "<provider>/<id>",
// dropping the noisy "Microsoft-Windows-" provider prefix.
func hostFaultSource(r hostfault.WinFaultRecord) string {
	prov := strings.TrimPrefix(r.Provider, "Microsoft-Windows-")
	if prov == "" {
		return ""
	}
	return fmt.Sprintf("%s/%d", prov, r.ID)
}

// hostFaultEventsFromWinRecords maps parsed event-log records onto durable
// HostFaultEvent rows through ClassifyHostFault. Fail-closed: a record the
// classifier does not recognize as one of the four host-fault classes is
// DROPPED, never folded as a fabricated fault. Class-specific fields (the WU
// error code + package, the live-kernel category) are derived here from the
// message the reader captured.
func hostFaultEventsFromWinRecords(recs []hostfault.WinFaultRecord, fallbackMS int64) []hostfault.HostFaultEvent {
	out := make([]hostfault.HostFaultEvent, 0, len(recs))
	for _, r := range recs {
		class, ok := hostfault.ClassifyHostFault(r)
		if !ok {
			continue
		}
		at := r.TimeMS
		if at <= 0 {
			at = fallbackMS
		}
		app := r.App
		code := ""
		switch class {
		case hostfault.HostWUInstallFailure:
			code = extractHexCode(r.Message)
			if app == "" {
				app = extractWUPackage(r.Message)
			}
		case hostfault.HostGPULiveKernel:
			code = extractLiveKernelCategory(r.Message)
		}
		out = append(out, hostfault.HostFaultEvent{
			Class:  class,
			AtMS:   at,
			Source: hostFaultSource(r),
			App:    app,
			Code:   code,
			Detail: hostfault.BoundHostFaultDetail(flattenWhitespace(r.Message)),
		})
	}
	return out
}

// writeHostFaultSnapshot writes the ingested rows as the current event-log
// PROJECTION (truncate, not append): re-running ingest is idempotent because the
// file mirrors the OS log's current window rather than accumulating duplicates.
func writeHostFaultSnapshot(path string, events []hostfault.HostFaultEvent) error {
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
		if err := hostfault.AppendHostFaultEvent(f, ev); err != nil {
			return err
		}
	}
	return nil
}

// runHostFaultIngest is the live wiring for `fak toolproc host-faults --ingest`:
// read this host's Windows event log for the host-fault classes, classify each,
// write the durable snapshot, and fold it to the operator view. maxPerSource
// bounds the per-source scan so a high-volume class (GPU live-kernel events run
// to thousands) cannot blow the read up; the bound is stated in the output, not
// applied silently.
func runHostFaultIngest(stdout, stderr io.Writer, outPath string, since time.Duration, maxPerSource int, nowMS int64, asJSON bool) int {
	recs, errMsg := gatherWinHostFaultRecords(since, maxPerSource)
	if errMsg != "" {
		fmt.Fprintf(stderr, "fak toolproc host-faults: ingest: %s\n", errMsg)
		return 1
	}
	events := hostFaultEventsFromWinRecords(recs, nowMS)
	if err := writeHostFaultSnapshot(outPath, events); err != nil {
		fmt.Fprintf(stderr, "fak toolproc host-faults: ingest: write %s: %v\n", outPath, err)
		return 1
	}
	report := hostfault.HostFaultReportFromEvents(events)
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak toolproc host-faults")
	}
	hostfault.RenderHostFaultReport(stdout, report)
	fmt.Fprintf(stdout, "ingested %d host-fault row(s) from the Windows event log "+
		"(WindowsUpdateClient/20 + WER/1001, last %d day(s), max %d/source) -> %s\n",
		len(events), int(since.Hours()/24), maxPerSource, outPath)
	return 0
}
