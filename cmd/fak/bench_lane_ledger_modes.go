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
	"the inventory is generated from the CUDA sources by a test-only generator (FAK_LAUNCH_SITES_UPDATE=1); threads_resolution says how each thread count was judged",
	"runtime and template rows carry no thread count: they are unjudged for partial warps and never counted as clean",
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
	File              string `json:"file"`
	Kernel            string `json:"kernel"`
	ThreadsExpr       string `json:"threads_expr"`
	ThreadsPerBlock   int    `json:"threads_per_block"`
	ThreadsResolution string `json:"threads_resolution"`
	GridExpr          string `json:"grid_expr"`
	SmemExpr          string `json:"smem_expr"`
}

type launchSiteFindingJSON struct {
	Kind         string         `json:"kind"`
	PartialLanes int            `json:"partial_lanes"`
	Site         launchSiteJSON `json:"site"`
}

type launchSitesReport struct {
	Schema           string                  `json:"schema"`
	Unit             string                  `json:"unit"`
	WarpSize         int                     `json:"warp_size"`
	WarpProvenance   string                  `json:"warp_provenance"`
	Rows             int                     `json:"rows"`
	Counts           map[string]int          `json:"counts"`
	ResolutionCounts map[string]int          `json:"resolution_counts"`
	Findings         []launchSiteFindingJSON `json:"findings"`
	Sites            []launchSiteJSON        `json:"sites"`
	Caveats          []string                `json:"caveats"`
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
		ThreadsResolution: s.Resolution, GridExpr: s.GridExpr, SmemExpr: s.SmemExpr,
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
		ResolutionCounts: map[string]int{
			compute.LaunchGenResLiteral:  0,
			compute.LaunchGenResChain:    0,
			compute.LaunchGenResRuntime:  0,
			compute.LaunchGenResTemplate: 0,
		},
		Findings: []launchSiteFindingJSON{},
		Sites:    make([]launchSiteJSON, 0, len(sites)),
		Caveats:  laneLaunchSitesCaveats,
	}
	for _, s := range sites {
		report.Sites = append(report.Sites, launchSiteJSONOf(s))
		report.ResolutionCounts[s.Resolution]++
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
	fmt.Fprintf(w, "warp_provenance: %s\n", r.WarpProvenance)
	fmt.Fprintf(w, "resolution: literal=%d resolved-chain=%d runtime=%d template=%d\n\n",
		r.ResolutionCounts[compute.LaunchGenResLiteral], r.ResolutionCounts[compute.LaunchGenResChain],
		r.ResolutionCounts[compute.LaunchGenResRuntime], r.ResolutionCounts[compute.LaunchGenResTemplate])

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
	fmt.Fprintln(tw, "FILE\tKERNEL\tTHREADS\tTPB\tRES\tGRID\tSMEM")
	for _, s := range r.Sites {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\t%s\t%s\n",
			s.File, s.Kernel, s.ThreadsExpr, s.ThreadsPerBlock, s.ThreadsResolution, s.GridExpr, s.SmemExpr)
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

// ---- sweep ----

const (
	benchLaneSweepSchema = "fak.lane-ptxas-sweep/1"
	benchLaneSweepUnit   = "sweep_rows"
	benchLaneDefaultTop  = 10
	benchLaneRefused     = "refused"
)

var benchLaneSweepCaveats = []string{
	"measured input: registers and static shared memory of each entry come from the caller-supplied ptxas log; the arch row is checked against the log's sm_ target",
	"model-only: lane-slot-waves, register-file figures and the limiter are the occupancy model applied to that input; no clock, no measured occupancy and no time",
	"dynamic shared memory is not in the sweep: smem_dynamic_unknown marks a launch whose dynamic shared-memory expression is not the literal 0, so only its static bytes are priced",
	"the join is by qualified base name and by the literal template arguments (true, false, integers): sites and entries that do not join are counted as unjoined and never priced",
	"rows are ranked by resource_capped_lane_slot_waves, largest first; a refused row carries zero and sorts last",
}

const benchLaneGridDefaultCaveat = "grid_blocks defaults to the SM count (one block per SM): a probe-shape assumption, not the launch's grid, since the inventory's grid expressions are symbolic; pass --grid to price another grid"

type benchLaneSweepOptions struct {
	arch, deviceProps, ptxasLog string
	sms, grid, top              int
	json                        bool
	set                         map[string]bool
}

type benchLaneJoinStats struct {
	SiteCount          int     `json:"site_count"`
	JoinedSiteCount    int     `json:"joined_site_count"`
	JoinRate           float64 `json:"join_rate"`
	AmbiguousSiteCount int     `json:"ambiguous_site_count"`
	EntryCount         int     `json:"entry_count"`
	JoinedEntryCount   int     `json:"joined_entry_count"`
	UnjoinedSiteCount  int     `json:"unjoined_site_count"`
	UnjoinedEntryCount int     `json:"unjoined_entry_count"`
}

type benchLaneSweepDoc struct {
	Schema      string                    `json:"schema"`
	Unit        string                    `json:"unit"`
	Arch        string                    `json:"arch"`
	DeviceSMs   int                       `json:"device_sms"`
	GridBlocks  int                       `json:"grid_blocks"`
	GridSource  string                    `json:"grid_source"`
	Provenance  string                    `json:"provenance"`
	DeviceCheck *compute.DevicePropsCheck `json:"device_props,omitempty"`
	Parse       compute.PtxasCounts       `json:"parse"`
	ArchsSeen   []string                  `json:"archs_seen"`
	Join        benchLaneJoinStats        `json:"join"`
	Limiters    map[string]int            `json:"limiter_histogram"`
	RowsTotal   int                       `json:"rows_total"`
	RowsShown   int                       `json:"rows_shown"`
	Rows        []compute.PtxasSweepRow   `json:"rows"`
	Caveats     []string                  `json:"caveats"`
}

func benchLaneRunSweep(stdout, stderr io.Writer, args []string) int {
	opt, err := parseBenchLaneSweepArgs(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		return 2
	}
	if err := opt.validate(); err != nil {
		fmt.Fprintf(stderr, "fak bench lane-ledger sweep: %v\n", err)
		return 2
	}
	doc, err := benchLaneBuildSweep(opt, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench lane-ledger sweep: %v\n", err)
		return 2
	}
	if opt.json {
		return writeLaneDocJSON(stdout, stderr, "sweep", doc)
	}
	benchLaneWriteSweepTable(stdout, doc)
	return 0
}

func parseBenchLaneSweepArgs(args []string, stderr io.Writer) (benchLaneSweepOptions, error) {
	fs := flag.NewFlagSet("bench lane-ledger sweep", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var opt benchLaneSweepOptions
	benchLaneHardwareFlags(fs, &opt.arch, &opt.sms, &opt.deviceProps)
	fs.StringVar(&opt.ptxasLog, "ptxas-log", "", "nvcc -Xptxas -v output: each entry's registers and static shared memory (required)")
	fs.IntVar(&opt.grid, "grid", 0, "blocks in the probe launch (default: the SM count)")
	fs.IntVar(&opt.top, "top", benchLaneDefaultTop, "ranked rows to show; rows_total counts every joined pair")
	fs.BoolVar(&opt.json, "json", false, "emit the "+benchLaneSweepSchema+" JSON document")
	if err := fs.Parse(args); err != nil {
		return opt, err
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak bench lane-ledger sweep: unexpected argument %q\n", fs.Arg(0))
		return opt, errors.New("unexpected argument")
	}
	opt.set = benchLaneSetFlags(fs)
	return opt, nil
}

func (opt benchLaneSweepOptions) validate() error {
	switch {
	case opt.ptxasLog == "":
		return errors.New("--ptxas-log is required: the nvcc -Xptxas -v output that gives each entry's registers")
	case opt.top <= 0:
		return fmt.Errorf("--top must be positive (got %d)", opt.top)
	case opt.set["grid"] && opt.grid <= 0:
		return fmt.Errorf("--grid must be positive (got %d); omit it to use the SM count", opt.grid)
	}
	return nil
}

func benchLaneBuildSweep(opt benchLaneSweepOptions, stderr io.Writer) (benchLaneSweepDoc, error) {
	hw, err := benchLaneHardwareReporting(opt.arch, opt.sms, opt.deviceProps, stderr)
	if err != nil {
		return benchLaneSweepDoc{}, err
	}
	arch, ok := laneLedgerArchByLabel(hw.Arch)
	if !ok {
		return benchLaneSweepDoc{}, fmt.Errorf("unknown --arch %q", hw.Arch)
	}
	report, err := benchLaneReadFile("ptxas-log", opt.ptxasLog, compute.ParsePtxasLog)
	if err != nil {
		return benchLaneSweepDoc{}, err
	}
	join := compute.PtxasJoinSites(report, compute.LaunchSites())
	grid, gridSource, caveats := benchLaneSweepGrid(opt, hw.SMs)
	rows, err := arch.PtxasSweep(join, hw.SMs, grid)
	if err != nil {
		return benchLaneSweepDoc{}, err
	}
	shown := compute.RankPtxasSweepRows(rows)
	if len(shown) > opt.top {
		shown = shown[:opt.top]
	}
	if hw.Check != nil {
		caveats = append(caveats, benchLaneDevicePropsCaveat)
	}
	if report.Counts.FailedUnits > 0 {
		caveats = append(caveats, fmt.Sprintf("%d of %d ptxas units exited non-zero: their entries may be partial, and any site they miss is unjoined rather than priced", report.Counts.FailedUnits, report.Counts.Units))
	}
	return benchLaneSweepDoc{
		Schema: benchLaneSweepSchema, Unit: benchLaneSweepUnit,
		Arch: hw.Arch, DeviceSMs: hw.SMs, GridBlocks: grid, GridSource: gridSource,
		Provenance: join.Provenance, DeviceCheck: hw.Check,
		Parse: report.Counts, ArchsSeen: orEmptySlice(report.ArchsSeen),
		Join:      benchLaneJoinStatsOf(join),
		Limiters:  benchLaneLimiterHistogram(rows),
		RowsTotal: len(rows), RowsShown: len(shown), Rows: orEmptySlice(shown),
		Caveats: append(append([]string(nil), benchLaneSweepCaveats...), caveats...),
	}, nil
}

// benchLaneSweepGrid returns the probe grid and its provenance. The default is stated as an assumption, never
// as the launch's own grid.
func benchLaneSweepGrid(opt benchLaneSweepOptions, sms int) (int, string, []string) {
	if opt.set["grid"] {
		return opt.grid, "caller input: --grid", nil
	}
	return sms, "default: the device SM count, one block per SM", []string{benchLaneGridDefaultCaveat}
}

func benchLaneJoinStatsOf(j compute.PtxasJoin) benchLaneJoinStats {
	return benchLaneJoinStats{
		SiteCount: j.SiteCount, JoinedSiteCount: j.JoinedSiteCount, JoinRate: j.JoinRate,
		AmbiguousSiteCount: j.AmbiguousSiteCount, EntryCount: j.EntryCount, JoinedEntryCount: j.JoinedEntryCount,
		UnjoinedSiteCount: len(j.UnjoinedSites), UnjoinedEntryCount: len(j.UnjoinedEntries),
	}
}

// benchLaneLimiterHistogram counts the rows by the resource that bounds their blocks per SM. A refused row has
// no limiter and is counted apart, so no row is dropped from the histogram.
func benchLaneLimiterHistogram(rows []compute.PtxasSweepRow) map[string]int {
	h := map[string]int{}
	for _, r := range rows {
		key := r.Limiter
		if r.Refusal != "" {
			key = benchLaneRefused
		}
		h[key]++
	}
	return h
}

func benchLaneHistogramText(h map[string]int) string {
	if len(h) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(h))
	for _, k := range benchLaneSortedKeys(h) {
		parts = append(parts, fmt.Sprintf("%s=%d", k, h[k]))
	}
	return strings.Join(parts, " ")
}

func benchLaneWriteSweepTable(w io.Writer, d benchLaneSweepDoc) {
	fmt.Fprintf(w, "fak bench lane-ledger sweep  arch=%s device_sms=%d grid_blocks=%d rows=%d shown=%d\n",
		d.Arch, d.DeviceSMs, d.GridBlocks, d.RowsTotal, d.RowsShown)
	fmt.Fprintf(w, "grid: %s\n", d.GridSource)
	if d.DeviceCheck != nil {
		benchLaneWriteDeviceCheck(w, *d.DeviceCheck)
	}
	fmt.Fprintf(w, "provenance: %s\n", d.Provenance)
	p, j := d.Parse, d.Join
	fmt.Fprintf(w, "parse: entries=%d incomplete=%d undemangled=%d unparsed_clauses=%d callee_blocks=%d tool_error_lines=%d archs=%s\n",
		p.Entries, p.Incomplete, p.Undemangled, p.UnparsedClauses, p.CalleeBlocks, p.ToolErrorLines, strings.Join(d.ArchsSeen, ","))
	fmt.Fprintf(w, "join: sites=%d joined=%d rate=%.4f ambiguous=%d unjoined_sites=%d | entries=%d joined=%d unjoined_entries=%d\n",
		j.SiteCount, j.JoinedSiteCount, j.JoinRate, j.AmbiguousSiteCount, j.UnjoinedSiteCount,
		j.EntryCount, j.JoinedEntryCount, j.UnjoinedEntryCount)
	fmt.Fprintf(w, "limiter histogram (rows): %s\n\n", benchLaneHistogramText(d.Limiters))
	fmt.Fprintln(w, "ranked rows (resource-capped lane-slot-waves, largest first):")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "KERNEL\tTPB\tREGS\tSMEM\tBLOCKS/SM\tLIMITER\tRES-CAP-LSW\tACTIVE-LSW\tCAP-LSW\tFLAGS")
	for _, r := range d.Rows {
		if r.Refusal != "" {
			fmt.Fprintf(tw, "%s\t%d\t-\t-\t-\t%s\t-\t-\t-\trefused: %s\n", r.Kernel, r.ThreadsPerBlock, benchLaneRefused, r.Refusal)
			continue
		}
		fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%s\n",
			r.Kernel, r.ThreadsPerBlock, r.RegsPerThread, r.StaticSmemBytes, r.BlocksPerSM, r.Limiter,
			r.ResourceCappedLaneSlotWaves, r.ActiveLaneSlotWaves, r.LaneWaveCapacity, benchLaneSweepFlags(r))
	}
	tw.Flush()
	fmt.Fprintln(w, "\ncaveats:")
	for _, c := range d.Caveats {
		fmt.Fprintf(w, "  - %s\n", c)
	}
}

func benchLaneSweepFlags(r compute.PtxasSweepRow) string {
	var flags []string
	if r.SmemDynamicUnknown {
		flags = append(flags, "smem_dynamic_unknown")
	}
	if r.Ambiguous {
		flags = append(flags, "ambiguous")
	}
	if len(flags) == 0 {
		return "-"
	}
	return strings.Join(flags, ",")
}

// ---- observed ----

const (
	benchLaneObservedSchema = "fak.lane-observed/1"
	benchLaneObservedUnit   = "ns"
	benchLaneKeyMatching    = "unqualified kernel name: the namespace is stripped from the inventory and ptxas names"
)

var benchLaneObservedCaveats = []string{
	"measured input: observed time per launch shape comes from --profile as device event-pair time, so the launch gap is inside it; registers come from the ptxas log and threads from the launch inventory",
	"model-only: each launch shape's lane-slot-wave partition and its cell shares; a row's time is split across cells in proportion to lane-slot-waves, which is a model assumption",
	"a kernel name whose sources disagree, or whose dynamic shared-memory expression is not the literal 0, is left out of the join: its rows are unmodelled and inputs.conflicts names it",
	"matching is by unqualified name, so one profile name can cover several instantiations; when they disagree on registers the name is left out rather than guessed",
	"unmodelled rows keep their time share and name a reason; their time is not spread over the modelled cells",
}

type benchLaneObservedOptions struct {
	arch, deviceProps, profile, ptxasLog string
	sms, top                             int
	json                                 bool
	set                                  map[string]bool
}

// benchLaneObservedInputSummary counts what the observed join was keyed with, and names every input that was
// left out and why.
type benchLaneObservedInputSummary struct {
	ProfileRows    int                    `json:"profile_rows"`
	PtxasEntries   int                    `json:"ptxas_entries"`
	KeyedRegs      int                    `json:"keyed_regs"`
	KeyedThreads   int                    `json:"keyed_threads"`
	KeyedSmemBytes int                    `json:"keyed_smem_bytes"`
	KeyMatching    string                 `json:"key_matching"`
	Conflicts      []benchLaneKeyConflict `json:"conflicts"`
}

type benchLaneObservedDoc struct {
	Schema        string                        `json:"schema"`
	Unit          string                        `json:"unit"`
	Arch          string                        `json:"arch"`
	DeviceSMs     int                           `json:"device_sms"`
	DeviceCheck   *compute.DevicePropsCheck     `json:"device_props,omitempty"`
	Inputs        benchLaneObservedInputSummary `json:"inputs"`
	Totals        compute.ObservedTotals        `json:"totals"`
	RowsTotal     int                           `json:"rows_total"`
	TopModelled   []compute.ObservedRow         `json:"top_modelled"`
	TopUnmodelled []compute.ObservedRow         `json:"top_unmodelled"`
	Caveats       []string                      `json:"caveats"`
}

func benchLaneRunObserved(stdout, stderr io.Writer, args []string) int {
	opt, err := parseBenchLaneObservedArgs(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		return 2
	}
	if err := opt.validate(); err != nil {
		fmt.Fprintf(stderr, "fak bench lane-ledger observed: %v\n", err)
		return 2
	}
	doc, err := benchLaneBuildObserved(opt, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench lane-ledger observed: %v\n", err)
		return 2
	}
	if opt.json {
		return writeLaneDocJSON(stdout, stderr, "observed", doc)
	}
	benchLaneWriteObservedTable(stdout, doc)
	return 0
}

func parseBenchLaneObservedArgs(args []string, stderr io.Writer) (benchLaneObservedOptions, error) {
	fs := flag.NewFlagSet("bench lane-ledger observed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var opt benchLaneObservedOptions
	benchLaneHardwareFlags(fs, &opt.arch, &opt.sms, &opt.deviceProps)
	fs.StringVar(&opt.profile, "profile", "", "per-launch-shape observed time: a JSON array of {name, gx, gy, calls, total_ns} (required)")
	fs.StringVar(&opt.ptxasLog, "ptxas-log", "", "nvcc -Xptxas -v output: each entry's registers and static shared memory (required)")
	fs.IntVar(&opt.top, "top", benchLaneDefaultTop, "rows to show in each of the modelled and unmodelled lists")
	fs.BoolVar(&opt.json, "json", false, "emit the "+benchLaneObservedSchema+" JSON document")
	if err := fs.Parse(args); err != nil {
		return opt, err
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak bench lane-ledger observed: unexpected argument %q\n", fs.Arg(0))
		return opt, errors.New("unexpected argument")
	}
	opt.set = benchLaneSetFlags(fs)
	return opt, nil
}

func (opt benchLaneObservedOptions) validate() error {
	switch {
	case opt.profile == "":
		return errors.New("--profile is required: the observed per-launch-shape time")
	case opt.ptxasLog == "":
		return errors.New("--ptxas-log is required: the nvcc -Xptxas -v output that gives each entry's registers")
	case opt.top <= 0:
		return fmt.Errorf("--top must be positive (got %d)", opt.top)
	}
	return nil
}

func benchLaneBuildObserved(opt benchLaneObservedOptions, stderr io.Writer) (benchLaneObservedDoc, error) {
	hw, err := benchLaneHardwareReporting(opt.arch, opt.sms, opt.deviceProps, stderr)
	if err != nil {
		return benchLaneObservedDoc{}, err
	}
	arch, ok := laneLedgerArchByLabel(hw.Arch)
	if !ok {
		return benchLaneObservedDoc{}, fmt.Errorf("unknown --arch %q", hw.Arch)
	}
	profile, err := benchLaneReadFile("profile", opt.profile, compute.ParseObservedProfile)
	if err != nil {
		return benchLaneObservedDoc{}, err
	}
	report, err := benchLaneReadFile("ptxas-log", opt.ptxasLog, compute.ParsePtxasLog)
	if err != nil {
		return benchLaneObservedDoc{}, err
	}
	in, conflicts := benchLaneObservedInputs(report, compute.LaunchSites(), arch, hw.SMs)
	obs, err := compute.JoinObserved(profile, in)
	if err != nil {
		return benchLaneObservedDoc{}, err
	}
	modelled, err := obs.Top(opt.top)
	if err != nil {
		return benchLaneObservedDoc{}, err
	}
	caveats := append([]string{compute.ObservedCaveat}, benchLaneObservedCaveats...)
	if hw.Check != nil {
		caveats = append(caveats, benchLaneDevicePropsCaveat)
	}
	return benchLaneObservedDoc{
		Schema: benchLaneObservedSchema, Unit: benchLaneObservedUnit,
		Arch: obs.Arch, DeviceSMs: obs.DeviceSMs, DeviceCheck: hw.Check,
		Inputs: benchLaneObservedInputSummary{
			ProfileRows: len(profile), PtxasEntries: len(report.Entries),
			KeyedRegs: len(in.Regs), KeyedThreads: len(in.Threads), KeyedSmemBytes: len(in.SmemBytes),
			KeyMatching: benchLaneKeyMatching, Conflicts: orEmptySlice(conflicts),
		},
		Totals:        obs.Totals,
		RowsTotal:     len(obs.Rows),
		TopModelled:   orEmptySlice(modelled),
		TopUnmodelled: benchLaneUnmodelledRows(obs.Rows, opt.top),
		Caveats:       caveats,
	}, nil
}

// benchLaneUnmodelledRows returns the first n rows that name a reason, in the report's rank order.
func benchLaneUnmodelledRows(rows []compute.ObservedRow, n int) []compute.ObservedRow {
	out := []compute.ObservedRow{}
	for _, r := range rows {
		if r.Reason == "" {
			continue
		}
		if len(out) == n {
			break
		}
		out = append(out, r)
	}
	return out
}

// benchLaneUnmodelledReasons is every reason the join can name, in a fixed order, so the table lists zeros too.
var benchLaneUnmodelledReasons = []string{
	compute.ObservedReasonRegsUnmeasured,
	compute.ObservedReasonThreadsUnjudged,
	compute.ObservedReasonSmemUnjudged,
	compute.ObservedReasonUnlaunchable,
	compute.ObservedReasonOverflow,
}

func benchLaneWriteObservedTable(w io.Writer, d benchLaneObservedDoc) {
	t := d.Totals
	fmt.Fprintf(w, "fak bench lane-ledger observed  arch=%s device_sms=%d rows=%d modelled=%d unmodelled=%d\n",
		d.Arch, d.DeviceSMs, d.RowsTotal, t.ModelledRows, t.UnmodelledRows)
	if d.DeviceCheck != nil {
		benchLaneWriteDeviceCheck(w, *d.DeviceCheck)
	}
	in := d.Inputs
	fmt.Fprintf(w, "inputs: profile_rows=%d ptxas_entries=%d keyed regs=%d threads=%d smem=%d conflicts=%d (match: %s)\n",
		in.ProfileRows, in.PtxasEntries, in.KeyedRegs, in.KeyedThreads, in.KeyedSmemBytes, len(in.Conflicts), in.KeyMatching)
	fmt.Fprintf(w, "coverage: modelled_share=%.4f modelled_ns=%d unmodelled_ns=%d total_ns=%d\n",
		t.ModelledShare, t.ModelledNS, t.UnmodelledNS, t.TotalNS)
	fmt.Fprint(w, "unmodelled ns by reason:")
	for _, r := range benchLaneUnmodelledReasons {
		fmt.Fprintf(w, " %s=%d", r, t.UnmodelledNSByReason[r])
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "time_weighted_idle_ns (prioritisation heuristic, not occupancy or speedup): %d\n", t.TimeWeightedIdleNS)
	fmt.Fprint(w, "cell time ns (modelled rows):")
	for _, c := range compute.AllLaneCells() {
		fmt.Fprintf(w, " %s=%d", c.String(), t.CellTimeNS[c.String()])
	}
	fmt.Fprint(w, "\n\n")

	fmt.Fprintln(w, "top modelled rows by time_weighted_idle_ns:")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "KERNEL\tGRID\tTOTAL-NS\tSHARE\tLIMITER\tBLOCKS/SM\tACTIVE-SHARE\tIDLE-NS")
	for _, r := range d.TopModelled {
		m := r.Model
		fmt.Fprintf(tw, "%s\t%dx%d\t%d\t%.4f\t%s\t%d\t%.4f\t%d\n",
			r.Kernel, r.GridX, r.GridY, r.TotalNS, r.TimeShare, m.Limiter, m.BlocksPerSM, m.ActiveShare, m.TimeWeightedIdleNS)
	}
	tw.Flush()

	fmt.Fprintln(w, "\ntop unmodelled rows by time:")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "KERNEL\tGRID\tTOTAL-NS\tSHARE\tREASON")
	for _, r := range d.TopUnmodelled {
		fmt.Fprintf(tw, "%s\t%dx%d\t%d\t%.4f\t%s\n", r.Kernel, r.GridX, r.GridY, r.TotalNS, r.TimeShare, r.Reason)
	}
	tw.Flush()

	if len(in.Conflicts) > 0 {
		fmt.Fprintln(w, "\ninput conflicts (left out of the join):")
		for _, c := range in.Conflicts {
			fmt.Fprintf(w, "  %s %s: %s\n", c.Name, c.Field, c.Reason)
		}
	}
	fmt.Fprintln(w, "\ncaveats:")
	for _, c := range d.Caveats {
		fmt.Fprintf(w, "  - %s\n", c)
	}
}
