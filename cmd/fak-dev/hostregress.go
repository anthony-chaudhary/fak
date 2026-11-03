package main

// hostregress — cross-receipt regression trend gate for host-side overhead in
// small-batch decode. It orders every decode-step reading under a receipts dir
// (internal/hostshare), pins a baseline from the oldest K known readings
// (median, K default 3) or an explicit --baseline-file value, and reports the
// latest-vs-baseline relative delta plus each reading's overhead regime
// (internal/regime). The gate fails (exit 3) when the latest reading regresses
// >=50% RELATIVE vs baseline (0.13 -> 0.20 is +54%, fail; 0.13 -> 0.15 is
// +15%, pass) or flips COMPUTE_BOUND -> LAUNCH_BOUND while baseline held
// COMPUTE_BOUND. UNKNOWN readings never fail: they print UNKNOWN and skip.
// Fewer than K+1 known readings prints INSUFFICIENT_DATA and exits 0 with a
// warning — a trend needs both a floor and a new point.
//
// This replaces the ad-hoc method of noticing months later, in a prose note,
// that decode got host-bound: the same committed receipts now carry a
// deterministic landing-time verdict with a machine-checkable exit code.
//
// Exit codes: 0 pass / unknown / insufficient data, 1 scan/IO failure, 2
// usage, 3 regression detected.

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/hostshare"
	"github.com/anthony-chaudhary/fak/internal/regime"
)

const (
	hostRegressUsage = "usage: fak-dev hostregress [--receipts DIR] [--latest N] [--model S] [--hardware S] [--baseline-k K] [--baseline-file F] [--delta-threshold F] [--json]"
	// hostRegressSchema versions the --json document.
	hostRegressSchema = "fak.dev.hostregress.v1"
	// hostRegressDefaultDir is where committed benchmark receipts live.
	hostRegressDefaultDir = "docs/benchmarks/receipts"
	// hostRegressDefaultBaselineK sizes the baseline window: the median of
	// the oldest K known readings.
	hostRegressDefaultBaselineK = 3
	// hostRegressDefaultDeltaThreshold is the relative-regression line: the
	// latest host_share at or above baseline*(1+0.50) fails the gate.
	hostRegressDefaultDeltaThreshold = 0.50
)

// hostRegressRow is one ordered series entry in the report table.
type hostRegressRow struct {
	Path      string        `json:"path"`
	Model     string        `json:"model,omitempty"`
	Hardware  string        `json:"hardware,omitempty"`
	HostShare *float64      `json:"host_share"`
	Source    string        `json:"source,omitempty"`
	Regime    regime.Regime `json:"regime"`
	Role      string        `json:"role,omitempty"` // "baseline" | "latest"
	Delta     *float64      `json:"delta_vs_baseline,omitempty"`
}

// hostRegressBase is the pinned or derived baseline.
type hostRegressBase struct {
	Value     float64       `json:"value"`
	Source    string        `json:"source"` // "median-oldest-K" | "pinned-file"
	Paths     []string      `json:"paths,omitempty"`
	Regime    regime.Regime `json:"regime"`
	RegimeFor int           `json:"regime_for,omitempty"` // readings behind the majority verdict
}

// hostRegressDoc is the --json output.
type hostRegressDoc struct {
	Schema    string           `json:"schema"`
	Receipts  string           `json:"receipts"`
	Files     int              `json:"files_scanned"`
	Evidence  int              `json:"evidence_free_files"`
	Malformed []string         `json:"malformed_files,omitempty"`
	BaselineK int              `json:"baseline_k"`
	Threshold float64          `json:"delta_threshold"`
	Baseline  *hostRegressBase `json:"baseline,omitempty"`
	Latest    *hostRegressRow  `json:"latest,omitempty"`
	Delta     *float64         `json:"delta_vs_baseline,omitempty"`
	Readings  []hostRegressRow `json:"readings"`
	Reasons   []string         `json:"reasons,omitempty"`
	Verdict   string           `json:"verdict"`
}

func runHostRegress(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hostregress", flag.ContinueOnError)
	fs.SetOutput(stderr)
	receiptsDir := fs.String("receipts", hostRegressDefaultDir, "directory of receipt/profile-note JSON to scan recursively (or a single receipt file)")
	latestN := fs.Int("latest", 0, "consider at most the N most recent readings (0 = all)")
	modelFilter := fs.String("model", "", "case-insensitive substring filter on the model identity")
	hardwareFilter := fs.String("hardware", "", "case-insensitive substring filter on the hardware identity")
	baselineK := fs.Int("baseline-k", hostRegressDefaultBaselineK, "baseline window: median of the oldest K known readings")
	baselineFile := fs.String("baseline-file", "", "pin the baseline host_share explicitly (bare number or JSON object with host_share); disables the K-window and its regime-flip check")
	threshold := fs.Float64("delta-threshold", hostRegressDefaultDeltaThreshold, "relative regression that fails the gate (0.50 = +50%)")
	asJSON := fs.Bool("json", false, "emit the answer as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *baselineK < 1 {
		fmt.Fprintln(stderr, "fak-dev hostregress: --baseline-k must be >= 1")
		fmt.Fprintln(stderr, hostRegressUsage)
		return 2
	}
	if *threshold < 0 || math.IsNaN(*threshold) {
		fmt.Fprintln(stderr, "fak-dev hostregress: --delta-threshold must be a non-negative number")
		fmt.Fprintln(stderr, hostRegressUsage)
		return 2
	}

	rep, err := hostshare.ScanReport([]string{*receiptsDir})
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(stderr, "fak-dev hostregress: receipts dir %q does not exist (pass --receipts DIR)\n", *receiptsDir)
			return 1
		}
		fmt.Fprintf(stderr, "fak-dev hostregress: %v\n", err)
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
	if *latestN > 0 && len(readings) > *latestN {
		readings = readings[len(readings)-*latestN:]
	}

	rows := make([]hostRegressRow, len(readings))
	for i, r := range readings {
		v := classifyReading(r)
		rows[i] = hostRegressRow{
			Path:      r.Path,
			Model:     r.Model,
			Hardware:  r.Hardware,
			HostShare: r.HostShare,
			Source:    r.Source,
			Regime:    v.Regime,
		}
	}

	doc := hostRegressDoc{
		Schema: hostRegressSchema, Receipts: *receiptsDir,
		Files: rep.FilesScanned, Evidence: rep.EvidenceFree, Malformed: rep.MalformedFiles,
		BaselineK: *baselineK, Threshold: *threshold, Readings: rows,
	}

	var knownIdx []int
	for i, row := range rows {
		if row.HostShare != nil {
			knownIdx = append(knownIdx, i)
		}
	}

	if len(rows) == 0 {
		doc.Verdict = "INSUFFICIENT_DATA"
		if *asJSON {
			encodeHostRegress(stdout, stderr, &doc)
			return 0
		}
		printHostRegressTable(stdout, &doc)
		fmt.Fprintf(stdout, "gate: INSUFFICIENT_DATA 0 known reading(s), need %d (baseline-k=%d)\n",
			*baselineK+1, *baselineK)
		return 0
	}

	// The most recent reading decides the gate; if it carries no device-busy
	// signal there is nothing to regress, whatever the history holds.
	latestIdx := len(rows) - 1
	latest := &rows[latestIdx]
	if latest.HostShare == nil {
		doc.Verdict = "UNKNOWN"
		doc.Latest = latest
		if *asJSON {
			encodeHostRegress(stdout, stderr, &doc)
			return 0
		}
		printHostRegressTable(stdout, &doc)
		fmt.Fprintf(stdout, "gate: UNKNOWN latest %s records no device-busy data (UNKNOWN); skipped\n", latest.Path)
		return 0
	}

	// A pinned baseline replaces the K-window: one known latest reading
	// suffices, and no baseline regime exists to flip against.
	required := *baselineK + 1
	if *baselineFile != "" {
		required = 1
	}
	if len(knownIdx) < required {
		doc.Verdict = "INSUFFICIENT_DATA"
		warn := fmt.Sprintf("gate: INSUFFICIENT_DATA %d known reading(s), need %d (baseline-k=%d%s)",
			len(knownIdx), required, *baselineK, "; pin via --baseline-file to lower the bar to 1")
		if *baselineFile != "" {
			warn = fmt.Sprintf("gate: INSUFFICIENT_DATA %d known reading(s), need 1 even with a pinned baseline", len(knownIdx))
		}
		if *asJSON {
			encodeHostRegress(stdout, stderr, &doc)
			return 0
		}
		printHostRegressTable(stdout, &doc)
		fmt.Fprintln(stdout, warn)
		return 0
	}

	base, baseErr := resolveBaseline(rows, knownIdx, *baselineK, *baselineFile)
	if baseErr != "" {
		fmt.Fprintln(stderr, "fak-dev hostregress: "+baseErr)
		fmt.Fprintln(stderr, hostRegressUsage)
		return 2
	}
	doc.Baseline = base

	delta := relativeDelta(*latest.HostShare, base.Value)
	d := delta
	doc.Delta = &d
	latest.Delta = &d
	for _, i := range knownIdx {
		if i == latestIdx {
			continue
		}
		vd := relativeDelta(*rows[i].HostShare, base.Value)
		rows[i].Delta = &vd
	}
	// The K-window roles only exist without a pin; a pinned baseline has no
	// readings behind it.
	if *baselineFile == "" {
		for _, i := range knownIdx[:min(len(knownIdx), *baselineK)] {
			if rows[i].Role == "" {
				rows[i].Role = "baseline"
			}
		}
	}
	latest.Role = "latest"
	doc.Latest = latest

	var reasons []string
	if delta >= *threshold {
		reasons = append(reasons, fmt.Sprintf("delta-regression (%s >= +%s)", formatDelta(delta), formatPercent(*threshold)))
	}
	if latest.Regime == regime.RegimeLaunchBound && base.Regime == regime.RegimeComputeBound {
		reasons = append(reasons, "regime-flip (COMPUTE_BOUND -> LAUNCH_BOUND)")
	}

	if len(reasons) > 0 {
		doc.Reasons = reasons
		doc.Verdict = "FAIL"
	} else {
		doc.Verdict = "PASS"
	}

	if *asJSON {
		code := encodeHostRegress(stdout, stderr, &doc)
		if code != 0 {
			return code
		}
		if doc.Verdict == "FAIL" {
			return hostShareGateExit
		}
		return 0
	}

	printHostRegressTable(stdout, &doc)
	switch doc.Verdict {
	case "FAIL":
		fmt.Fprintf(stdout, "gate: FAIL latest %s host_share=%.3f vs baseline %.3f (%s): %s\n",
			latest.Path, *latest.HostShare, base.Value, formatDelta(delta), strings.Join(reasons, "; "))
	default:
		fmt.Fprintf(stdout, "gate: PASS latest %s host_share=%.3f vs baseline %.3f (%s) below +%s; regime %s -> %s\n",
			latest.Path, *latest.HostShare, base.Value, formatDelta(delta), formatPercent(*threshold), base.Regime, latest.Regime)
	}
	if doc.Verdict == "FAIL" {
		return hostShareGateExit
	}
	return 0
}

// resolveBaseline derives the baseline value and regime. With a pinned file the
// value comes from the file and the regime stays UNKNOWN (a scalar pin carries
// no profile signals, so the flip check is disabled by construction).
func resolveBaseline(rows []hostRegressRow, knownIdx []int, k int, pinnedPath string) (*hostRegressBase, string) {
	if pinnedPath != "" {
		v, err := parseBaselineFile(pinnedPath)
		if err != nil {
			return nil, err.Error()
		}
		return &hostRegressBase{Value: v, Source: "pinned-file", Regime: regime.RegimeUnknown}, ""
	}
	window := knownIdx[:k]
	vals := make([]float64, 0, len(window))
	regimes := make([]regime.Regime, 0, len(window))
	paths := make([]string, 0, len(window))
	for _, i := range window {
		vals = append(vals, *rows[i].HostShare)
		regimes = append(regimes, rows[i].Regime)
		paths = append(paths, rows[i].Path)
	}
	r, n := majorityRegime(regimes)
	return &hostRegressBase{Value: median(vals), Source: fmt.Sprintf("median-oldest-%d", k), Paths: paths, Regime: r, RegimeFor: n}, ""
}

// parseBaselineFile reads a pinned baseline host_share: either a bare number
// ("0.13") or a JSON object carrying a numeric host_share field.
func parseBaselineFile(path string) (float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	trimmed := bytes.TrimSpace(data)
	if f, err := strconv.ParseFloat(string(trimmed), 64); err == nil {
		if f < 0 || f > 1 || math.IsNaN(f) || math.IsInf(f, 0) {
			return 0, fmt.Errorf("baseline file %s: value %v outside [0,1]", path, f)
		}
		return f, nil
	}
	var rec map[string]any
	if err := json.Unmarshal(trimmed, &rec); err != nil {
		return 0, fmt.Errorf("baseline file %s: must be a bare number or a JSON object with host_share", path)
	}
	v, ok := rec["host_share"].(float64)
	if !ok || v < 0 || v > 1 || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("baseline file %s: missing numeric host_share in [0,1]", path)
	}
	return v, nil
}

// classifyReading decodes the receipt file behind one reading and returns its
// regime verdict. Unreadable or undecodable files classify UNKNOWN (never a
// fabricated verdict): multi-line JSONL receipts are one such case, since the
// classifier expects a single JSON document.
func classifyReading(r hostshare.Reading) regime.Verdict {
	data, err := os.ReadFile(r.Path)
	if err != nil {
		return regime.Verdict{Regime: regime.RegimeUnknown, Evidence: []string{"file unreadable: " + err.Error()}}
	}
	p, err := regime.DecodeProfile(data)
	if err != nil {
		return regime.Verdict{Regime: regime.RegimeUnknown, Evidence: []string{"undecodable profile: " + err.Error()}}
	}
	return regime.Classify(p)
}

// majorityRegime returns the regime held by a strict majority (>half) of the
// window, counting UNKNOWN readings against every candidate. Ties are
// UNKNOWN. Candidates are visited in fixed physical order, so the result is
// deterministic.
func majorityRegime(rs []regime.Regime) (regime.Regime, int) {
	counts := map[regime.Regime]int{}
	for _, r := range rs {
		counts[r]++
	}
	for _, cand := range []regime.Regime{regime.RegimeLaunchBound, regime.RegimeComputeBound, regime.RegimeMemBound} {
		if counts[cand]*2 > len(rs) {
			return cand, counts[cand]
		}
	}
	return regime.RegimeUnknown, 0
}

// median returns the middle value (even count: mean of the two middles).
func median(vals []float64) float64 {
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// relativeDelta returns (latest-baseline)/baseline. A zero baseline cannot be
// regressed by an equal reading (delta 0) but any positive host share against
// it is unbounded relative growth (+Inf, which fails any threshold).
func relativeDelta(latest, baseline float64) float64 {
	if baseline <= 0 {
		if latest > 0 {
			return math.Inf(1)
		}
		return 0
	}
	return (latest - baseline) / baseline
}

func formatDelta(d float64) string {
	if math.IsInf(d, 1) {
		return "+inf"
	}
	return fmt.Sprintf("%+.1f%%", d*100)
}

func formatPercent(p float64) string {
	return fmt.Sprintf("%.0f%%", p*100)
}

func printHostRegressTable(stdout io.Writer, doc *hostRegressDoc) {
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	src := "-"
	if doc.Baseline != nil {
		src = fmt.Sprintf("%.3f (%s)", doc.Baseline.Value, doc.Baseline.Source)
	}
	fmt.Fprintf(tw, "HOSTREGRESS\tbaseline=%s\tthreshold=+%s\treceipts=%s\n", src, formatPercent(doc.Threshold), doc.Receipts)
	fmt.Fprintf(tw, "ROLE\tHOST_SHARE\tREGIME\tDELTA-VS-BASELINE\tRECEIPT\n")
	for _, r := range doc.Readings {
		share, delta := "UNKNOWN", "-"
		if r.HostShare != nil {
			share = fmt.Sprintf("%.3f", *r.HostShare)
		}
		if r.Delta != nil {
			delta = formatDelta(*r.Delta)
		}
		role := "-"
		if r.Role != "" {
			role = r.Role
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", role, share, r.Regime, delta, r.Path)
	}
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(stdout, "(table flush failed: %v)\n", err)
	}
}

func encodeHostRegress(stdout, stderr io.Writer, doc *hostRegressDoc) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(doc); err != nil {
		fmt.Fprintf(stderr, "fak-dev hostregress: encode: %v\n", err)
		return 1
	}
	return 0
}
