package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak bench lane-ledger: the exact lane-slot-wave partition of one decode launch on one NVIDIA
// compute capability, plus the register-file footprint of the same launch. The partition is a model
// only. The optional measured window joins caller-supplied cycles to it in lane-slot-cycles. The
// sibling verbs sites and fill live in bench_lane_ledger_modes.go.

const (
	laneLedgerSchema       = "fak.lane-ledger/1"
	laneLedgerUnit         = "lane-slot-waves"
	laneLedgerMeasuredUnit = "lane-slot-cycles"
)

// laneLedgerCaveats ships with every report so the partition cannot be read as a timing.
var laneLedgerCaveats = []string{
	"model-only partition: no clock and no measured time; counts are lane-slot-waves, not seconds or throughput",
	"placement assumes one block per SM first, before any SM takes a second block",
	"waves are static (no block backfill): wave_tail is an upper bound on what a dynamic scheduler would lose",
	"register allocation unit, warp allocation granularity and shared-memory allocation unit are occupancy-calculator values, not vendor-published",
	"the SM count is a caller input (--sms): a SKU property, not an ISA constant",
}

// laneLedgerMeasuredCaveats labels the measured section: its window and counts are caller inputs, and
// the two residuals are named without a cause.
var laneLedgerMeasuredCaveats = []string{
	"caller inputs: cycles_per_sm, wave_weights, active_lane_cycles and the tolerance are read from a measurement; the model supplies only the cell shares",
	"cells are the model's lane-slot-waves times the cycles of the wave that carries them, in lane-slot-cycles",
	"unattributed is window capacity that no modelled wave covers; the model names no cause for it",
	"unexplained_inactive (model calls more lanes active than measured) and unexplained_active (fewer) are the active residual outside --tolerance, present only when the verdict is not consistent; neither is attributed to a stall cause",
}

// laneLedgerEqualWavesCaveat is added only when no --wave-weights were given.
const laneLedgerEqualWavesCaveat = "no --wave-weights: equal-length waves are assumed, so per-wave timing is a model assumption, not a measurement"

// measuredFlagNames are the flags that describe a measured window. Each one is refused unless
// --cycles-per-sm is also given, so none is silently ignored.
var measuredFlagNames = []string{"wave-weights", "active-lane-cycles", "tolerance", "exact"}

type laneLedgerOptions struct {
	arch    string
	sms     int
	threads int
	regs    int
	smem    int
	grid    int
	kernel  string
	json    bool

	cyclesPerSM      int64
	waveWeights      string
	activeLaneCycles int64
	tolerance        float64
	exact            bool
	// set names every flag the caller passed explicitly, so a zero value can be told from an absent one.
	set map[string]bool
}

type laneLedgerLaunch struct {
	Kernel            string `json:"kernel"`
	DeviceSMs         int    `json:"device_sms"`
	ThreadsPerBlock   int    `json:"threads_per_block"`
	RegsPerThread     int    `json:"regs_per_thread"`
	SmemBytesPerBlock int    `json:"smem_bytes_per_block"`
	GridBlocks        int    `json:"grid_blocks"`
}

type laneLedgerSummary struct {
	Capacity           int64            `json:"capacity"`
	Cells              map[string]int64 `json:"cells"`
	Waves              int              `json:"waves"`
	BlocksPerSM        int              `json:"blocks_per_sm"`
	Limiter            string           `json:"limiter"`
	BlockSlotsUnfilled int64            `json:"block_slots_unfilled"`
	Conserved          bool             `json:"conserved"`
}

// laneLedgerMeasured is the measured window in lane-slot-cycles. Verdict and the active fields are empty or
// zero unless the caller supplied active_lane_cycles.
type laneLedgerMeasured struct {
	Unit               string                      `json:"unit"`
	CyclesPerSM        int64                       `json:"cycles_per_sm"`
	CapacityLaneCycles int64                       `json:"capacity_lane_cycles"`
	Cells              map[string]int64            `json:"cells"`
	Unattributed       int64                       `json:"unattributed"`
	AssumedEqualWaves  bool                        `json:"assumed_equal_waves"`
	ActiveSupplied     bool                        `json:"active_supplied"`
	ModelActive        int64                       `json:"model_active"`
	MeasuredActive     int64                       `json:"measured_active"`
	ActiveShortfall    int64                       `json:"active_shortfall"`
	Verdict            string                      `json:"verdict"`
	Findings           []laneLedgerMeasuredFinding `json:"findings"`
	Caveats            []string                    `json:"caveats"`
}

type laneLedgerMeasuredFinding struct {
	Name            string  `json:"name"`
	Cell            string  `json:"cell"`
	LaneSlotCycles  int64   `json:"lane_slot_cycles"`
	ShareOfCapacity float64 `json:"share_of_capacity"`
	Additive        bool    `json:"additive"`
}

type laneLedgerReport struct {
	Schema            string                       `json:"schema"`
	Unit              string                       `json:"unit"`
	Arch              string                       `json:"arch"`
	Provenance        string                       `json:"provenance"`
	Launch            laneLedgerLaunch             `json:"launch"`
	Ledger            laneLedgerSummary            `json:"ledger"`
	Findings          []compute.GapFinding         `json:"findings"`
	RegisterFootprint compute.RegisterFootprint    `json:"register_footprint"`
	RegisterFindings  []compute.RegisterGapFinding `json:"register_findings"`
	Caveats           []string                     `json:"caveats"`
	Measured          *laneLedgerMeasured          `json:"measured,omitempty"`
}

// runBenchLaneLedger is the testable entry point: args follow the "lane-ledger" verb. A leading "sites" or
// "fill" selects the static inventory or the lane-fill model; anything else is the per-launch partition.
func runBenchLaneLedger(stdout, stderr io.Writer, args []string) int {
	if len(args) > 0 {
		switch args[0] {
		case "sites":
			return runLaneLedgerSites(stdout, stderr, args[1:])
		case "fill":
			return runLaneLedgerFill(stdout, stderr, args[1:])
		}
	}
	opt, err := parseLaneLedgerArgs(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		return 2
	}
	if err := opt.validate(); err != nil {
		fmt.Fprintf(stderr, "fak bench lane-ledger: %v\n", err)
		return 2
	}
	report, err := buildLaneLedgerReport(opt)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench lane-ledger: %v\n", err)
		return 2
	}
	if opt.json {
		out, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak bench lane-ledger: encode report: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "%s\n", out)
		return 0
	}
	writeLaneLedgerTable(stdout, report)
	return 0
}

func parseLaneLedgerArgs(args []string, stderr io.Writer) (laneLedgerOptions, error) {
	fs := flag.NewFlagSet("bench lane-ledger", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var opt laneLedgerOptions
	fs.StringVar(&opt.arch, "arch", "", "compute capability: "+laneLedgerArchLabels()+" (required)")
	fs.IntVar(&opt.sms, "sms", 0, "device SM count of the SKU (required; a caller input with no default)")
	fs.IntVar(&opt.threads, "threads", 0, "threads per block (required)")
	fs.IntVar(&opt.regs, "regs", 0, "registers per thread from a measurement (required)")
	fs.IntVar(&opt.smem, "smem", 0, "dynamic shared memory bytes per block")
	fs.IntVar(&opt.grid, "grid", 0, "blocks in the launch (required)")
	fs.StringVar(&opt.kernel, "kernel", "kernel", "kernel label recorded in the report")
	fs.BoolVar(&opt.json, "json", false, "emit the "+laneLedgerSchema+" JSON document")
	fs.Int64Var(&opt.cyclesPerSM, "cycles-per-sm", 0, "measured window length in cycles on one SM; enables the measured section (a caller input)")
	fs.StringVar(&opt.waveWeights, "wave-weights", "", "comma list of each modelled wave's cycles, in wave order (needs --cycles-per-sm)")
	fs.Int64Var(&opt.activeLaneCycles, "active-lane-cycles", 0, "measured integral of occupied lanes over the window (needs --cycles-per-sm)")
	fs.Float64Var(&opt.tolerance, "tolerance", 0, "accepted active-lane gap as a fraction of window capacity, in [0,1] (needs --cycles-per-sm)")
	fs.BoolVar(&opt.exact, "exact", false, "chooses the error text for a --wave-weights sum above --cycles-per-sm; no computed value changes (needs --cycles-per-sm)")
	if err := fs.Parse(args); err != nil {
		return opt, err
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak bench lane-ledger: unexpected argument %q\n", fs.Arg(0))
		return opt, errors.New("unexpected argument")
	}
	opt.set = map[string]bool{}
	fs.Visit(func(f *flag.Flag) { opt.set[f.Name] = true })
	return opt, nil
}

func (opt laneLedgerOptions) validate() error {
	if opt.arch == "" {
		return fmt.Errorf("--arch is required (one of %s)", laneLedgerArchLabels())
	}
	if _, ok := laneLedgerArchByLabel(opt.arch); !ok {
		return fmt.Errorf("unknown --arch %q (supported: %s)", opt.arch, laneLedgerArchLabels())
	}
	if opt.sms <= 0 {
		return errors.New("--sms is required and must be positive: the device SM count is a SKU property the caller supplies, with no default")
	}
	if opt.threads <= 0 {
		return errors.New("--threads is required and must be positive (threads per block)")
	}
	if opt.regs <= 0 {
		return fmt.Errorf("--regs is required and must be positive (got %d): registers per thread must come from a measurement such as the compiler's or profiler's per-kernel register count", opt.regs)
	}
	if opt.grid <= 0 {
		return errors.New("--grid is required and must be positive (blocks in the launch)")
	}
	if opt.smem < 0 {
		return fmt.Errorf("--smem must not be negative (got %d)", opt.smem)
	}
	return opt.validateMeasuredFlags()
}

// validateMeasuredFlags refuses a measured-window flag given without --cycles-per-sm, since it would
// otherwise be dropped without a word.
func (opt laneLedgerOptions) validateMeasuredFlags() error {
	if opt.set["cycles-per-sm"] {
		return nil
	}
	for _, name := range measuredFlagNames {
		if opt.set[name] {
			return fmt.Errorf("--%s describes a measured window and needs --cycles-per-sm", name)
		}
	}
	return nil
}

// measuredWindow turns the measured flags into the caller's window. A nil WaveWeights means equal waves.
func (opt laneLedgerOptions) measuredWindow() (compute.MeasuredWindow, error) {
	w := compute.MeasuredWindow{
		CyclesPerSM:      opt.cyclesPerSM,
		ActiveLaneCycles: opt.activeLaneCycles,
		ActiveSupplied:   opt.set["active-lane-cycles"],
		ActiveTolerance:  opt.tolerance,
		Exact:            opt.exact,
	}
	if opt.set["wave-weights"] {
		weights, err := parseWaveWeights(opt.waveWeights)
		if err != nil {
			return w, err
		}
		w.WaveWeights = weights
	}
	return w, nil
}

// parseWaveWeights reads a comma list of integer cycle counts. The ranges are checked by the compute
// package, which refuses a non-positive or oversized weight with its own sentinel.
func parseWaveWeights(s string) ([]int64, error) {
	parts := strings.Split(s, ",")
	weights := make([]int64, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.ParseInt(strings.TrimSpace(p), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("--wave-weights: %q is not an integer cycle count", p)
		}
		weights = append(weights, n)
	}
	return weights, nil
}

func buildLaneLedgerReport(opt laneLedgerOptions) (laneLedgerReport, error) {
	arch, ok := laneLedgerArchByLabel(opt.arch)
	if !ok {
		return laneLedgerReport{}, fmt.Errorf("unknown --arch %q", opt.arch)
	}
	launch := compute.DecodeLaunch{
		Kernel:            opt.kernel,
		ThreadsPerBlock:   opt.threads,
		RegsPerThread:     opt.regs,
		SmemBytesPerBlock: opt.smem,
		GridBlocks:        opt.grid,
	}
	ledger, err := arch.BuildLaneLedger(launch, opt.sms)
	if err != nil {
		return laneLedgerReport{}, err
	}
	footprint, err := arch.RegisterFootprintOf(launch)
	if err != nil {
		return laneLedgerReport{}, err
	}
	report := laneLedgerReport{
		Schema:     laneLedgerSchema,
		Unit:       laneLedgerUnit,
		Arch:       ledger.Arch,
		Provenance: ledger.Provenance,
		Launch: laneLedgerLaunch{
			Kernel:            launch.Kernel,
			DeviceSMs:         ledger.DeviceSMs,
			ThreadsPerBlock:   launch.ThreadsPerBlock,
			RegsPerThread:     launch.RegsPerThread,
			SmemBytesPerBlock: launch.SmemBytesPerBlock,
			GridBlocks:        launch.GridBlocks,
		},
		Ledger:            laneLedgerSummaryOf(ledger),
		Findings:          orEmptySlice(ledger.Findings()),
		RegisterFootprint: footprint,
		RegisterFindings:  orEmptySlice(footprint.Findings(ledger.Limiter)),
		Caveats:           laneLedgerCaveats,
	}
	if opt.set["cycles-per-sm"] {
		measured, err := laneLedgerMeasuredOf(ledger, opt)
		if err != nil {
			return laneLedgerReport{}, err
		}
		report.Measured = &measured
	}
	return report, nil
}

// laneLedgerMeasuredOf joins the measured window to the ledger and copies the result into the wire shape.
func laneLedgerMeasuredOf(x compute.LaneLedger, opt laneLedgerOptions) (laneLedgerMeasured, error) {
	window, err := opt.measuredWindow()
	if err != nil {
		return laneLedgerMeasured{}, err
	}
	joined, err := x.JoinMeasured(window)
	if err != nil {
		return laneLedgerMeasured{}, err
	}
	cells := make(map[string]int64, len(compute.AllLaneCells()))
	for _, c := range compute.AllLaneCells() {
		cells[c.String()] = joined.Cells[c]
	}
	caveats := append([]string(nil), laneLedgerMeasuredCaveats...)
	if joined.AssumedEqualWaves {
		caveats = append(caveats, laneLedgerEqualWavesCaveat)
	}
	joinedFindings := joined.Findings()
	findings := make([]laneLedgerMeasuredFinding, 0, len(joinedFindings))
	for _, f := range joinedFindings {
		findings = append(findings, laneLedgerMeasuredFinding{
			Name: f.Name, Cell: f.Cell, LaneSlotCycles: f.LaneSlotCycles,
			ShareOfCapacity: f.ShareOfCapacity, Additive: f.Additive,
		})
	}
	return laneLedgerMeasured{
		Unit:               laneLedgerMeasuredUnit,
		CyclesPerSM:        joined.CyclesPerSM,
		CapacityLaneCycles: joined.CapacityLaneCycles,
		Cells:              cells,
		Unattributed:       joined.Unattributed,
		AssumedEqualWaves:  joined.AssumedEqualWaves,
		ActiveSupplied:     joined.ActiveSupplied,
		ModelActive:        joined.ModelActive,
		MeasuredActive:     joined.MeasuredActive,
		ActiveShortfall:    joined.ActiveShortfall,
		Verdict:            joined.Verdict,
		Findings:           findings,
		Caveats:            caveats,
	}, nil
}

func laneLedgerSummaryOf(x compute.LaneLedger) laneLedgerSummary {
	return laneLedgerSummary{
		Capacity:           x.Capacity,
		Cells:              x.CellCounts(),
		Waves:              x.Waves,
		BlocksPerSM:        x.BlocksPerSM,
		Limiter:            x.Limiter,
		BlockSlotsUnfilled: x.BlockSlotsUnfilled,
		Conserved:          x.Conserved(),
	}
}

// laneLedgerArchByLabel resolves a repo arch label such as "sm_120" to its registered row.
func laneLedgerArchByLabel(label string) (compute.NVArch, bool) {
	for _, a := range compute.KnownNVArches() {
		if a.Label == label {
			return a, true
		}
	}
	return compute.NVArch{}, false
}

func laneLedgerArchLabels() string {
	var labels []string
	for _, a := range compute.KnownNVArches() {
		labels = append(labels, a.Label)
	}
	return strings.Join(labels, ", ")
}

func orEmptySlice[T any](s []T) []T {
	if s == nil {
		return []T{}
	}
	return s
}

func laneLedgerShare(n, capacity int64) float64 {
	if capacity <= 0 {
		return 0
	}
	return float64(n) / float64(capacity)
}

func writeLaneLedgerTable(w io.Writer, r laneLedgerReport) {
	fmt.Fprintf(w, "fak bench lane-ledger  arch=%s kernel=%s device_sms=%d\n", r.Arch, r.Launch.Kernel, r.Launch.DeviceSMs)
	fmt.Fprintf(w, "launch: threads=%d regs=%d smem=%d grid=%d\n",
		r.Launch.ThreadsPerBlock, r.Launch.RegsPerThread, r.Launch.SmemBytesPerBlock, r.Launch.GridBlocks)
	fmt.Fprintf(w, "ledger: blocks/SM=%d limiter=%s waves=%d capacity=%d block_slots_unfilled=%d conserved=%t\n\n",
		r.Ledger.BlocksPerSM, r.Ledger.Limiter, r.Ledger.Waves, r.Ledger.Capacity,
		r.Ledger.BlockSlotsUnfilled, r.Ledger.Conserved)

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "CELL\tLANE-SLOT-WAVES\tSHARE")
	for _, c := range compute.AllLaneCells() {
		n := r.Ledger.Cells[c.String()]
		fmt.Fprintf(tw, "%s\t%d\t%.4f\n", c.String(), n, laneLedgerShare(n, r.Ledger.Capacity))
	}
	tw.Flush()

	fmt.Fprintln(w, "\nfindings (lane-slot-waves):")
	if len(r.Findings) == 0 {
		fmt.Fprintln(w, "  none")
	}
	for _, f := range r.Findings {
		fmt.Fprintf(w, "  %-20s %12d  %.4f  %s\n", f.Name, f.LaneSlotWaves, f.ShareOfCapacity, f.Trigger)
	}

	fp := r.RegisterFootprint
	fmt.Fprintf(w, "\nregister file (registers per SM, %d total): useful=%d rounding=%d granularity=%d headroom=%d remainder=%d\n",
		fp.RegsPerSM, fp.Useful, fp.RoundingWaste, fp.GranularityWaste, fp.HeadroomUnused, fp.Remainder)
	fmt.Fprintln(w, "register findings (registers per SM):")
	if len(r.RegisterFindings) == 0 {
		fmt.Fprintln(w, "  none")
	}
	for _, f := range r.RegisterFindings {
		fmt.Fprintf(w, "  %-20s %12d  %.4f  %s\n", f.Name, f.RegistersPerSM, f.ShareOfRegsPerSM, f.Trigger)
	}

	fmt.Fprintln(w, "\ncaveats:")
	for _, c := range r.Caveats {
		fmt.Fprintf(w, "  - %s\n", c)
	}
	if r.Measured != nil {
		writeLaneLedgerMeasured(w, *r.Measured)
	}
}

func writeLaneLedgerMeasured(w io.Writer, m laneLedgerMeasured) {
	fmt.Fprintf(w, "\nmodel cells over measured window (%s): cycles_per_sm=%d capacity=%d assumed_equal_waves=%t\n",
		m.Unit, m.CyclesPerSM, m.CapacityLaneCycles, m.AssumedEqualWaves)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "CELL\tLANE-SLOT-CYCLES\tSHARE")
	for _, c := range compute.AllLaneCells() {
		n := m.Cells[c.String()]
		fmt.Fprintf(tw, "%s\t%d\t%.4f\n", c.String(), n, laneLedgerShare(n, m.CapacityLaneCycles))
	}
	fmt.Fprintf(tw, "unattributed\t%d\t%.4f\n", m.Unattributed, laneLedgerShare(m.Unattributed, m.CapacityLaneCycles))
	tw.Flush()

	if m.ActiveSupplied {
		fmt.Fprintf(w, "active: model=%d measured=%d shortfall=%d verdict=%s\n",
			m.ModelActive, m.MeasuredActive, m.ActiveShortfall, m.Verdict)
	} else {
		fmt.Fprintln(w, "active: not supplied (--active-lane-cycles); no verdict")
	}
	fmt.Fprintf(w, "model-apportioned findings (%s):\n", m.Unit)
	if len(m.Findings) == 0 {
		fmt.Fprintln(w, "  none")
	}
	for _, f := range m.Findings {
		fmt.Fprintf(w, "  %-20s %16d  %.4f  additive=%t\n", f.Name, f.LaneSlotCycles, f.ShareOfCapacity, f.Additive)
	}
	fmt.Fprintln(w, "measured caveats:")
	for _, c := range m.Caveats {
		fmt.Fprintf(w, "  - %s\n", c)
	}
}
