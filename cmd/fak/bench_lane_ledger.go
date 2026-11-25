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

// fak bench lane-ledger: the exact lane-slot-wave partition of one decode launch on one NVIDIA
// compute capability, plus the register-file footprint of the same launch. It is a model only:
// no clock and no measured time.

const (
	laneLedgerSchema = "fak.lane-ledger/1"
	laneLedgerUnit   = "lane-slot-waves"
)

// laneLedgerCaveats ships with every report so the partition cannot be read as a timing.
var laneLedgerCaveats = []string{
	"model-only partition: no clock and no measured time; counts are lane-slot-waves, not seconds or throughput",
	"placement assumes one block per SM first, before any SM takes a second block",
	"waves are static (no block backfill): wave_tail is an upper bound on what a dynamic scheduler would lose",
	"register allocation unit, warp allocation granularity and shared-memory allocation unit are occupancy-calculator values, not vendor-published",
	"the SM count is a caller input (--sms): a SKU property, not an ISA constant",
}

type laneLedgerOptions struct {
	arch    string
	sms     int
	threads int
	regs    int
	smem    int
	grid    int
	kernel  string
	json    bool
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
}

// runBenchLaneLedger is the testable entry point: args follow the "lane-ledger" verb.
func runBenchLaneLedger(stdout, stderr io.Writer, args []string) int {
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
	if err := fs.Parse(args); err != nil {
		return opt, err
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak bench lane-ledger: unexpected argument %q\n", fs.Arg(0))
		return opt, errors.New("unexpected argument")
	}
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
	return nil
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
	return laneLedgerReport{
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
}
