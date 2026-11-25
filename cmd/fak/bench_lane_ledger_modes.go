package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak bench lane-ledger sites and fill: two static views beside the per-launch partition. sites lists the
// CUDA launch sites under internal/compute and flags partial warps and symbolic thread counts. fill
// partitions the lane slots of a caller's execution-unit width. Neither one measures anything.

const (
	laneLaunchSitesSchema = "fak.lane-launch-sites/1"
	laneLaunchSitesUnit   = "launch_site_rows"
	laneFillSchema        = "fak.lane-fill/1"
)

var laneLaunchSitesCaveats = []string{
	"static inventory of launch sites as written in internal/compute source: rows are distinct launch texts, not runtime launch counts",
	"partial_warp is a static fact of threads_per_block modulo warp_size, not a measured lane occupancy",
	"partial_lanes is the occupied lane count of the last warp, so warp_size minus partial_lanes lanes of that warp are idle",
	"unjudged rows have a symbolic thread count (threads_per_block 0): they are neither clean nor judged",
	"threads_per_block values are baked into the inventory by a generator outside this repo; tests check the literal rows only",
	"warp_size is a caller input when --warp is given; otherwise it is the repo NVArch.WarpSize (see warp_provenance)",
}

var laneFillCaveats = []string{
	"model-only partition: lane slots of one pass; no clock, no measured time, and no stall cause is named",
	"unit width is a caller input: the provenance field records the --width-source text as given, and no vendor width is asserted",
	"family, width, items and group size are caller inputs describing the launch shape",
	"masked tail runs the last group whole with the lanes past the last item idle; scalar tail runs each remainder item alone on one lane",
}

// laneFillScalarGroupsCaveat is added only when a scalar tail leaves a remainder: its items are not a group, so
// groups times units_per_group times width does not reconcile to capacity.
const laneFillScalarGroupsCaveat = "tail scalar: groups is ceil(items/group) and counts the remainder as a group; the remainder items occupy scalar_tail slots, so groups*units_per_group*width is not capacity"

// nvidiaWarpSize returns the warp width every registered NVIDIA row agrees on. It reports false when the
// rows disagree, so no default is invented.
func nvidiaWarpSize() (int, bool) {
	rows := compute.KnownNVArches()
	if len(rows) == 0 {
		return 0, false
	}
	warp := rows[0].WarpSize
	for _, r := range rows[1:] {
		if r.WarpSize != warp {
			return 0, false
		}
	}
	return warp, true
}

// writeLaneDocJSON encodes a verb's document with the same indentation as the default partition.
func writeLaneDocJSON(stdout, stderr io.Writer, verb string, doc any) int {
	out, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak bench lane-ledger %s: encode report: %v\n", verb, err)
		return 1
	}
	fmt.Fprintf(stdout, "%s\n", out)
	return 0
}

// ---- sites ----

type launchSitesOptions struct {
	warp           int
	warpProvenance string
	json           bool
}

type launchSiteJSON struct {
	File            string `json:"file"`
	Kernel          string `json:"kernel"`
	ThreadsExpr     string `json:"threads_expr"`
	ThreadsPerBlock int    `json:"threads_per_block"`
	GridExpr        string `json:"grid_expr"`
	SmemExpr        string `json:"smem_expr"`
}

type launchSiteFindingJSON struct {
	Kind         string         `json:"kind"`
	PartialLanes int            `json:"partial_lanes"`
	Site         launchSiteJSON `json:"site"`
}

type launchSitesReport struct {
	Schema         string                  `json:"schema"`
	Unit           string                  `json:"unit"`
	WarpSize       int                     `json:"warp_size"`
	WarpProvenance string                  `json:"warp_provenance"`
	Rows           int                     `json:"rows"`
	Counts         map[string]int          `json:"counts"`
	Findings       []launchSiteFindingJSON `json:"findings"`
	Sites          []launchSiteJSON        `json:"sites"`
	Caveats        []string                `json:"caveats"`
}

func runLaneLedgerSites(stdout, stderr io.Writer, args []string) int {
	opt, err := parseLaunchSitesArgs(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		return 2
	}
	if opt.warp <= 0 {
		fmt.Fprintf(stderr, "fak bench lane-ledger sites: --warp must be a positive warp width (got %d)\n", opt.warp)
		return 2
	}
	report := buildLaunchSitesReport(opt)
	if opt.json {
		return writeLaneDocJSON(stdout, stderr, "sites", report)
	}
	writeLaunchSitesTable(stdout, report)
	return 0
}

func parseLaunchSitesArgs(args []string, stderr io.Writer) (launchSitesOptions, error) {
	fs := flag.NewFlagSet("bench lane-ledger sites", flag.ContinueOnError)
	fs.SetOutput(stderr)
	defaultWarp, sharedRows := nvidiaWarpSize()
	var opt launchSitesOptions
	fs.IntVar(&opt.warp, "warp", defaultWarp, "warp width the partial-warp check uses (default: the NVArch.WarpSize shared by all registered NVIDIA rows)")
	fs.BoolVar(&opt.json, "json", false, "emit the "+laneLaunchSitesSchema+" JSON document")
	if err := fs.Parse(args); err != nil {
		return opt, err
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak bench lane-ledger sites: unexpected argument %q\n", fs.Arg(0))
		return opt, errors.New("unexpected argument")
	}
	warpSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "warp" {
			warpSet = true
		}
	})
	switch {
	case warpSet:
		opt.warpProvenance = "caller input: --warp"
	case sharedRows:
		opt.warpProvenance = fmt.Sprintf("repo NVArch.WarpSize, shared by all %d registered NVIDIA rows", len(compute.KnownNVArches()))
	default:
		opt.warpProvenance = "unset: registered NVIDIA rows disagree on WarpSize; pass --warp"
	}
	return opt, nil
}

func launchSiteJSONOf(s compute.LaunchSite) launchSiteJSON {
	return launchSiteJSON{
		File: s.File, Kernel: s.Kernel, ThreadsExpr: s.ThreadsExpr, ThreadsPerBlock: s.ThreadsPerBlock,
		GridExpr: s.GridExpr, SmemExpr: s.SmemExpr,
	}
}

func buildLaunchSitesReport(opt launchSitesOptions) launchSitesReport {
	sites := compute.LaunchSites()
	report := launchSitesReport{
		Schema:         laneLaunchSitesSchema,
		Unit:           laneLaunchSitesUnit,
		WarpSize:       opt.warp,
		WarpProvenance: opt.warpProvenance,
		Rows:           len(sites),
		Counts: map[string]int{
			compute.LaunchFindingPartialWarp: 0,
			compute.LaunchFindingUnjudged:    0,
		},
		Findings: []launchSiteFindingJSON{},
		Sites:    make([]launchSiteJSON, 0, len(sites)),
		Caveats:  laneLaunchSitesCaveats,
	}
	for _, s := range sites {
		report.Sites = append(report.Sites, launchSiteJSONOf(s))
	}
	for _, f := range compute.LaunchSiteFindings(opt.warp) {
		report.Counts[f.Kind]++
		report.Findings = append(report.Findings, launchSiteFindingJSON{
			Kind: f.Kind, PartialLanes: f.PartialLanes, Site: launchSiteJSONOf(f.Site),
		})
	}
	return report
}

func writeLaunchSitesTable(w io.Writer, r launchSitesReport) {
	fmt.Fprintf(w, "fak bench lane-ledger sites  warp_size=%d rows=%d partial_warp=%d unjudged=%d\n",
		r.WarpSize, r.Rows, r.Counts[compute.LaunchFindingPartialWarp], r.Counts[compute.LaunchFindingUnjudged])
	fmt.Fprintf(w, "warp_provenance: %s\n\n", r.WarpProvenance)

	fmt.Fprintln(w, "findings:")
	if len(r.Findings) == 0 {
		fmt.Fprintln(w, "  none")
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	if len(r.Findings) > 0 {
		fmt.Fprintln(tw, "  KIND\tPARTIAL-LANES\tFILE\tKERNEL\tTHREADS\tGRID\tSMEM")
	}
	for _, f := range r.Findings {
		s := f.Site
		fmt.Fprintf(tw, "  %s\t%d\t%s\t%s\t%s\t%s\t%s\n",
			f.Kind, f.PartialLanes, s.File, s.Kernel, s.ThreadsExpr, s.GridExpr, s.SmemExpr)
	}
	tw.Flush()

	fmt.Fprintf(w, "\ninventory (%d rows; threads_per_block 0 = symbolic):\n", r.Rows)
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "FILE\tKERNEL\tTHREADS\tTPB\tGRID\tSMEM")
	for _, s := range r.Sites {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\t%s\n",
			s.File, s.Kernel, s.ThreadsExpr, s.ThreadsPerBlock, s.GridExpr, s.SmemExpr)
	}
	tw.Flush()

	fmt.Fprintln(w, "\ncaveats:")
	for _, c := range r.Caveats {
		fmt.Fprintf(w, "  - %s\n", c)
	}
}

// ---- fill ----

type laneFillOptions struct {
	family      string
	width       int
	widthSource string
	group       int
	items       int64
	tail        string
	kernel      string
	json        bool
}

type laneFillReport struct {
	Schema        string                    `json:"schema"`
	Unit          string                    `json:"unit"`
	Provenance    string                    `json:"provenance"`
	Family        string                    `json:"family"`
	Width         int                       `json:"width"`
	Kernel        string                    `json:"kernel"`
	TailMode      compute.TailMode          `json:"tail_mode"`
	GroupSize     int                       `json:"group_size"`
	Items         int64                     `json:"items"`
	Groups        int64                     `json:"groups"`
	UnitsPerGroup int64                     `json:"units_per_group"`
	Capacity      compute.LaneSlots         `json:"capacity"`
	Active        compute.LaneSlots         `json:"active"`
	PartialUnit   compute.LaneSlots         `json:"partial_unit"`
	TailIdle      compute.LaneSlots         `json:"tail_idle"`
	ScalarTail    compute.LaneSlots         `json:"scalar_tail"`
	Conserved     bool                      `json:"conserved"`
	Findings      []compute.LaneFillFinding `json:"findings"`
	Caveats       []string                  `json:"caveats"`
}

func runLaneLedgerFill(stdout, stderr io.Writer, args []string) int {
	opt, err := parseLaneFillArgs(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		return 2
	}
	if err := opt.validate(); err != nil {
		fmt.Fprintf(stderr, "fak bench lane-ledger fill: %v\n", err)
		return 2
	}
	report, err := buildLaneFillReport(opt)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench lane-ledger fill: %v\n", err)
		return 2
	}
	if opt.json {
		return writeLaneDocJSON(stdout, stderr, "fill", report)
	}
	writeLaneFillTable(stdout, report)
	return 0
}

func parseLaneFillArgs(args []string, stderr io.Writer) (laneFillOptions, error) {
	fs := flag.NewFlagSet("bench lane-ledger fill", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var opt laneFillOptions
	fs.StringVar(&opt.family, "family", "", "execution-unit family label, e.g. wavefront or vulkan_subgroup (required)")
	fs.IntVar(&opt.width, "width", 0, "lanes per unit (required; a caller input, see --width-source)")
	fs.StringVar(&opt.widthSource, "width-source", "", "where the width came from; recorded as the provenance (required)")
	fs.IntVar(&opt.group, "group", 0, "work items per group (required)")
	fs.Int64Var(&opt.items, "items", 0, "work items in the launch (required)")
	fs.StringVar(&opt.tail, "tail", string(compute.TailMasked), "tail mode: masked or scalar")
	fs.StringVar(&opt.kernel, "kernel", "kernel", "kernel label recorded in the report")
	fs.BoolVar(&opt.json, "json", false, "emit the "+laneFillSchema+" JSON document")
	if err := fs.Parse(args); err != nil {
		return opt, err
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak bench lane-ledger fill: unexpected argument %q\n", fs.Arg(0))
		return opt, errors.New("unexpected argument")
	}
	return opt, nil
}

func (opt laneFillOptions) validate() error {
	switch {
	case strings.TrimSpace(opt.widthSource) == "":
		return errors.New("--width-source is required: name where the width came from, since a caller-supplied width must carry its source")
	case opt.family == "":
		return errors.New("--family is required (an execution-unit family label)")
	case opt.width <= 0:
		return fmt.Errorf("--width is required and must be positive (got %d)", opt.width)
	case opt.group <= 0:
		return fmt.Errorf("--group is required and must be positive (got %d)", opt.group)
	case opt.items <= 0:
		return fmt.Errorf("--items is required and must be positive (got %d)", opt.items)
	case opt.tail != string(compute.TailMasked) && opt.tail != string(compute.TailScalar):
		return fmt.Errorf("--tail must be %q or %q (got %q)", compute.TailMasked, compute.TailScalar, opt.tail)
	}
	return nil
}

func buildLaneFillReport(opt laneFillOptions) (laneFillReport, error) {
	unit, err := compute.CallerLaneUnit(opt.family, opt.width, opt.widthSource)
	if err != nil {
		return laneFillReport{}, err
	}
	fill, err := unit.Fill(compute.LaneFillLaunch{
		Kernel:    opt.kernel,
		Items:     opt.items,
		GroupSize: opt.group,
		TailMode:  compute.TailMode(opt.tail),
	})
	if err != nil {
		return laneFillReport{}, err
	}
	caveats := append([]string(nil), laneFillCaveats...)
	if fill.TailMode == compute.TailScalar && fill.Items%int64(fill.GroupSize) != 0 {
		caveats = append(caveats, laneFillScalarGroupsCaveat)
	}
	return laneFillReport{
		Schema:        laneFillSchema,
		Unit:          compute.LaneFillUnit,
		Provenance:    fill.Provenance,
		Family:        fill.Family,
		Width:         fill.Width,
		Kernel:        fill.Kernel,
		TailMode:      fill.TailMode,
		GroupSize:     fill.GroupSize,
		Items:         fill.Items,
		Groups:        fill.Groups,
		UnitsPerGroup: fill.UnitsPerGroup,
		Capacity:      fill.Capacity,
		Active:        fill.Active,
		PartialUnit:   fill.PartialUnit,
		TailIdle:      fill.TailIdle,
		ScalarTail:    fill.ScalarTail,
		Conserved:     fill.Conserved(),
		Findings:      orEmptySlice(fill.Findings()),
		Caveats:       caveats,
	}, nil
}

func writeLaneFillTable(w io.Writer, r laneFillReport) {
	fmt.Fprintf(w, "fak bench lane-ledger fill  kernel=%s family=%s width=%d tail=%s group=%d items=%d\n",
		r.Kernel, r.Family, r.Width, r.TailMode, r.GroupSize, r.Items)
	fmt.Fprintf(w, "provenance: %s\n", r.Provenance)
	fmt.Fprintf(w, "partition (lane slots): groups=%d units_per_group=%d capacity=%d active=%d partial_unit=%d tail_idle=%d scalar_tail=%d conserved=%t\n\n",
		r.Groups, r.UnitsPerGroup, r.Capacity, r.Active, r.PartialUnit, r.TailIdle, r.ScalarTail, r.Conserved)

	fmt.Fprintln(w, "findings (lane slots):")
	if len(r.Findings) == 0 {
		fmt.Fprintln(w, "  none")
	}
	for _, f := range r.Findings {
		fmt.Fprintf(w, "  %-14s %12d  %.4f  %s\n", f.Name, f.LaneSlots, f.ShareOfCapacity, f.Trigger)
	}

	fmt.Fprintln(w, "\ncaveats:")
	for _, c := range r.Caveats {
		fmt.Fprintf(w, "  - %s\n", c)
	}
}
