package main

// hostshare — systematic reader + regression gate for host-side overhead in
// small-batch decode. It scans receipt/profile-note JSON (perf-receipt-v2
// receipts, device-busy fractions, GPU busy percents, ms/token splits) under a
// directory and reports each reading's host_share — the host-phase share of
// decode step time. A reading with no recorded device-busy data prints UNKNOWN
// (never a fabricated zero), and the most recent reading gates the exit code:
// above --threshold (default 0.30) the run exits 3, flagging a launch-bound
// regime before it shows up as a tok/s regression.
//
// Exit codes: 0 clean or unknown, 1 scan/IO failure, 2 usage, 3 threshold
// exceeded on the most recent reading.

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/hostshare"
)

const (
	hostShareUsage = "usage: fak-dev hostshare [--receipts DIR] [--latest N] [--threshold F] [--model S] [--hardware S] [--json]"
	// hostShareSchema versions the --json document.
	hostShareSchema = "fak.dev.hostshare.v1"
	// hostShareDefaultDir is where committed benchmark receipts live.
	hostShareDefaultDir = "docs/benchmarks/receipts"
	// hostShareDefaultThreshold is the launch-bound alarm line: more than 30%
	// of decode step time spent off-device is host-overhead-dominated.
	hostShareDefaultThreshold = 0.30
	// hostShareGateExit is the non-zero exit for a threshold breach.
	hostShareGateExit = 3
)

// hostShareDoc is the --json output.
type hostShareDoc struct {
	Schema    string              `json:"schema"`
	Receipts  string              `json:"receipts"`
	Files     int                 `json:"files_scanned"`
	Evidence  int                 `json:"evidence_free_files"`
	Malformed []string            `json:"malformed_files,omitempty"`
	Threshold float64             `json:"threshold"`
	Readings  []hostshare.Reading `json:"readings"`
	Gated     *hostshare.Reading  `json:"gated,omitempty"`
	Exceeded  bool                `json:"exceeded"`
	Verdict   string              `json:"verdict"`
}

func runHostShare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hostshare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	receiptsDir := fs.String("receipts", hostShareDefaultDir, "directory of receipt/profile-note JSON to scan recursively (or a single receipt file)")
	latest := fs.Int("latest", 0, "show at most the N most recent readings (0 = all)")
	threshold := fs.Float64("threshold", hostShareDefaultThreshold, "host_share above which the most recent reading is LAUNCH-BOUND (exit 3)")
	modelFilter := fs.String("model", "", "case-insensitive substring filter on the model identity")
	hardwareFilter := fs.String("hardware", "", "case-insensitive substring filter on the hardware identity")
	asJSON := fs.Bool("json", false, "emit the answer as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *threshold < 0 || *threshold > 1 {
		fmt.Fprintln(stderr, "fak-dev hostshare: --threshold must be in [0,1]")
		fmt.Fprintln(stderr, hostShareUsage)
		return 2
	}

	rep, err := hostshare.ScanReport([]string{*receiptsDir})
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(stderr, "fak-dev hostshare: receipts dir %q does not exist (pass --receipts DIR)\n", *receiptsDir)
			return 1
		}
		fmt.Fprintf(stderr, "fak-dev hostshare: %v\n", err)
		return 1
	}

	readings := rep.Readings
	if *modelFilter != "" || *hardwareFilter != "" {
		filtered := make([]hostshare.Reading, 0, len(readings))
		for _, r := range readings {
			if *modelFilter != "" && !strings.Contains(strings.ToLower(r.Model), strings.ToLower(*modelFilter)) {
				continue
			}
			if *hardwareFilter != "" && !strings.Contains(strings.ToLower(r.Hardware), strings.ToLower(*hardwareFilter)) {
				continue
			}
			filtered = append(filtered, r)
		}
		readings = filtered
	}
	if *latest > 0 && len(readings) > *latest {
		readings = readings[len(readings)-*latest:]
	}

	gated, exceeded, ok := hostshare.Evaluate(readings, *threshold)
	doc := hostShareDoc{
		Schema: hostShareSchema, Receipts: *receiptsDir,
		Files: rep.FilesScanned, Evidence: rep.EvidenceFree, Malformed: rep.MalformedFiles,
		Threshold: *threshold, Readings: readings, Exceeded: exceeded,
	}
	if ok {
		doc.Gated = &gated
		doc.Verdict = hostshare.Verdict(gated.HostShare, *threshold)
	} else {
		doc.Verdict = "NO-DATA"
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(doc); err != nil {
			fmt.Fprintf(stderr, "fak-dev hostshare: encode: %v\n", err)
			return 1
		}
		if exceeded {
			return hostShareGateExit
		}
		return 0
	}

	if len(readings) == 0 {
		fmt.Fprintf(stdout, "hostshare: no decode-step readings under %s (%d files scanned, %d without host-share evidence)\n",
			*receiptsDir, rep.FilesScanned, rep.EvidenceFree)
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "RECEIPT\tMODEL\tHARDWARE\tHOST_SHARE\tVERDICT\n")
	for _, r := range readings {
		share := "UNKNOWN"
		if r.HostShare != nil {
			share = fmt.Sprintf("%.3f", *r.HostShare)
		}
		model, hardware := dash(r.Model), dash(r.Hardware)
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", r.Path, model, hardware, share, hostshare.Verdict(r.HostShare, *threshold))
	}
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(stderr, "fak-dev hostshare: %v\n", err)
		return 1
	}
	switch {
	case !ok:
		fmt.Fprintf(stdout, "gate: NO-DATA no readable readings after filters under %s\n", *receiptsDir)
	case exceeded:
		fmt.Fprintf(stdout, "gate: FAIL most recent %s host_share=%.3f exceeds threshold %.2f -> LAUNCH-BOUND\n",
			gated.Path, *gated.HostShare, *threshold)
	default:
		verdict := hostshare.Verdict(gated.HostShare, *threshold)
		if verdict == "UNKNOWN" {
			fmt.Fprintf(stdout, "gate: NO-SIGNAL most recent %s records no device-busy data (UNKNOWN)\n", gated.Path)
		} else {
			fmt.Fprintf(stdout, "gate: PASS most recent %s host_share=%.3f <= threshold %.2f\n",
				gated.Path, *gated.HostShare, *threshold)
		}
	}
	if exceeded {
		return hostShareGateExit
	}
	return 0
}

func dash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
