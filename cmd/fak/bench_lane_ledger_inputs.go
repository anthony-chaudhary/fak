package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak bench lane-ledger --device-props, sweep and observed read caller-supplied files: a CUDA device probe, an
// nvcc ptxas log and a timestamp profile. This file holds the readers, the arch and SM resolution, and the
// per-kernel inputs the observed join needs. The verbs are in bench_lane_ledger_modes.go.

var (
	benchLaneErrNoHardware   = errors.New("no device properties file, and the arch and SM count are not both given")
	benchLaneErrArchDisagree = errors.New("--arch disagrees with the compute capability in the device properties file")
	benchLaneErrSMsDisagree  = errors.New("--sms disagrees with the SM count in the device properties file")
	benchLaneErrRefuted      = errors.New("the device properties file refutes the NVArch row: a checkable field disagrees")
)

// benchLaneHardware is the arch and SM count a verb prices against. Check is set only when a device
// properties file supplied them; it then holds the field-by-field comparison with the arch row.
type benchLaneHardware struct {
	Arch  string
	SMs   int
	Check *compute.DevicePropsCheck
}

// benchLaneResolveHardware takes the arch and SM count from the flags, or from the device properties file
// when propsPath is given. A flag that disagrees with the file is refused, and so is a refuted arch row. An
// incomplete check is returned with its verdict and is not treated as a pass.
func benchLaneResolveHardware(archFlag string, smsFlag int, propsPath string) (benchLaneHardware, error) {
	if propsPath == "" {
		return benchLaneFlagHardware(archFlag, smsFlag)
	}
	props, err := benchLaneReadFile("device-props", propsPath, compute.ParseDeviceProps)
	if err != nil {
		return benchLaneHardware{}, err
	}
	row, err := props.Arch()
	if err != nil {
		return benchLaneHardware{}, fmt.Errorf("--device-props %s: %w", filepath.Base(propsPath), err)
	}
	devSMs, err := props.DeviceSMs()
	if err != nil {
		return benchLaneHardware{}, fmt.Errorf("--device-props %s: %w", filepath.Base(propsPath), err)
	}
	check := props.CheckArch(row)
	hw := benchLaneHardware{Arch: row.Label, SMs: devSMs, Check: &check}
	switch {
	case archFlag != "" && archFlag != row.Label:
		return hw, fmt.Errorf("%w: --arch %s, file reports %s", benchLaneErrArchDisagree, archFlag, row.Label)
	case smsFlag != 0 && smsFlag != devSMs:
		return hw, fmt.Errorf("%w: --sms %d, file reports %d", benchLaneErrSMsDisagree, smsFlag, devSMs)
	case check.Verdict == compute.DevicePropsRefuted:
		return hw, fmt.Errorf("%w: arch %s", benchLaneErrRefuted, row.Label)
	}
	return hw, nil
}

func benchLaneFlagHardware(archFlag string, smsFlag int) (benchLaneHardware, error) {
	if archFlag == "" {
		return benchLaneHardware{}, fmt.Errorf("%w: --arch is missing", benchLaneErrNoHardware)
	}
	if _, ok := laneLedgerArchByLabel(archFlag); !ok {
		return benchLaneHardware{}, fmt.Errorf("unknown --arch %q (supported: %s)", archFlag, laneLedgerArchLabels())
	}
	if smsFlag <= 0 {
		return benchLaneHardware{}, fmt.Errorf("%w: --sms must be positive (got %d); the SM count is a caller input when no device properties file is given",
			benchLaneErrNoHardware, smsFlag)
	}
	return benchLaneHardware{Arch: archFlag, SMs: smsFlag}, nil
}

// benchLaneHardwareReporting resolves the hardware and, on refusal, prints the device check first so the
// caller sees which field disagreed.
func benchLaneHardwareReporting(archFlag string, smsFlag int, propsPath string, stderr io.Writer) (benchLaneHardware, error) {
	hw, err := benchLaneResolveHardware(archFlag, smsFlag, propsPath)
	if err != nil && hw.Check != nil {
		benchLaneWriteDeviceCheck(stderr, *hw.Check)
	}
	return hw, err
}

// benchLaneApplyDeviceProps fills the default partition's arch and SM count from --device-props.
func benchLaneApplyDeviceProps(opt *laneLedgerOptions, stderr io.Writer) error {
	hw, err := benchLaneHardwareReporting(opt.arch, opt.sms, opt.deviceProps, stderr)
	if err != nil {
		return err
	}
	opt.arch, opt.sms, opt.deviceCheck = hw.Arch, hw.SMs, hw.Check
	return nil
}

// benchLaneHardwareFlags registers the arch, SM count and probe-file flags that the sweep and observed verbs share.
func benchLaneHardwareFlags(fs *flag.FlagSet, arch *string, sms *int, propsPath *string) {
	fs.StringVar(arch, "arch", "", "compute capability: "+laneLedgerArchLabels()+" (required unless --device-props)")
	fs.IntVar(sms, "sms", 0, "device SM count of the SKU (required unless --device-props)")
	fs.StringVar(propsPath, "device-props", "", "CUDA device probe JSON: supplies --sms and --arch, checks the arch row field by field, and refuses a disagreement")
}

// benchLaneSetFlags names every flag the caller passed explicitly, so a zero value can be told from an absent one.
func benchLaneSetFlags(fs *flag.FlagSet) map[string]bool {
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	return set
}

// benchLaneReadFile opens a caller-supplied file and parses it, naming the flag and the file on error.
func benchLaneReadFile[T any](flagName, path string, parse func(io.Reader) (T, error)) (T, error) {
	var zero T
	f, err := os.Open(path)
	if err != nil {
		return zero, fmt.Errorf("--%s: %w", flagName, err)
	}
	defer f.Close()
	v, err := parse(f)
	if err != nil {
		return zero, fmt.Errorf("--%s %s: %w", flagName, filepath.Base(path), err)
	}
	return v, nil
}

// benchLaneWriteDeviceCheck prints the arch check: the verdict, the capability comparison, and one row per
// checkable field. Unverifiable granularities are listed beside them and never change the verdict.
func benchLaneWriteDeviceCheck(w io.Writer, c compute.DevicePropsCheck) {
	fmt.Fprintf(w, "device props check: arch=%s capability=%s verdict=%s\n", c.Arch, c.Capability, c.Verdict)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  FIELD\tSOURCE\tSTATUS\tDEVICE\tROW")
	for _, f := range c.Fields {
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\t%d\n", f.Field, f.Source, f.Status, benchLaneFieldValue(f.Device), f.Row)
	}
	for _, f := range c.Unverifiable {
		fmt.Fprintf(tw, "  %s\t%s\t%s\t-\t%d\n", f.Field, f.Source, f.Status, f.Row)
	}
	tw.Flush()
}

func benchLaneFieldValue(f compute.DevicePropsField) string {
	if !f.Present {
		return "-"
	}
	return strconv.FormatInt(f.Value, 10)
}

// benchLaneKeyName reduces a qualified kernel name to the name the timestamp profile reports: the part after the
// last namespace separator. Two sources that reduce to one key are compared before either is used.
func benchLaneKeyName(qualified string) string {
	if i := strings.LastIndex(qualified, "::"); i >= 0 {
		return qualified[i+2:]
	}
	return qualified
}

// benchLaneSiteBase is the kernel name of a launch site before any template argument list.
func benchLaneSiteBase(kernel string) string {
	base, _, _ := strings.Cut(kernel, "<")
	return strings.TrimSpace(base)
}

// benchLaneValues is the set of values one per-kernel input takes across every source that names the kernel.
// A source with no measured value marks the field unknown, which keeps the field from being keyed.
type benchLaneValues struct {
	seen    map[int]bool
	unknown bool
}

func (v *benchLaneValues) add(n int, known bool) {
	if v.seen == nil {
		v.seen = map[int]bool{}
	}
	if !known {
		v.unknown = true
		return
	}
	v.seen[n] = true
}

// benchLaneNoSource marks a field that no source reported at all: an absent input, not a conflict.
const benchLaneNoSource = "no source"

// resolve returns the single value every source agrees on, or the reason it cannot be keyed.
func (v benchLaneValues) resolve() (int, string) {
	switch {
	case v.unknown:
		return 0, "a source has no measured value"
	case len(v.seen) == 0:
		return 0, benchLaneNoSource
	case len(v.seen) > 1:
		return 0, "sources disagree"
	}
	for n := range v.seen {
		return n, ""
	}
	return 0, benchLaneNoSource
}

// benchLaneSource collects the threads, registers and shared memory that name one profile key.
type benchLaneSource struct {
	regs, threads, smem benchLaneValues
}

// benchLaneKeyConflict names a kernel input that was left out because its sources disagree or have no
// measured value. The row is then reported as unmodelled, never priced from a guess.
type benchLaneKeyConflict struct {
	Name   string `json:"name"`
	Field  string `json:"field"`
	Reason string `json:"reason"`
}

// benchLaneObservedInputs keys the observed join by kernel name. Threads come from the launch inventory,
// registers from the ptxas entries, and shared memory from ptxas static bytes, but only for launches whose
// dynamic shared-memory expression is the literal 0. Matching is by the unqualified name, so a profile row
// reaches an entry the qualified join in the sweep does not connect.
func benchLaneObservedInputs(report compute.PtxasReport, sites []compute.LaunchSite, a compute.NVArch, deviceSMs int) (compute.ObservedInputs, []benchLaneKeyConflict) {
	sources := map[string]*benchLaneSource{}
	source := func(name string) *benchLaneSource {
		s, ok := sources[name]
		if !ok {
			s = &benchLaneSource{}
			sources[name] = s
		}
		return s
	}
	for _, s := range sites {
		src := source(benchLaneKeyName(benchLaneSiteBase(s.Kernel)))
		src.threads.add(s.ThreadsPerBlock, s.ThreadsPerBlock > 0)
		if s.SmemExpr != "0" {
			src.smem.add(0, false)
		}
	}
	for _, e := range report.Entries {
		if e.BaseName == "" {
			continue
		}
		src := source(benchLaneKeyName(e.BaseName))
		src.regs.add(e.Registers, e.HasRegisters && e.Registers > 0)
		src.smem.add(int(e.SmemBytes), true)
	}
	in := compute.ObservedInputs{
		Arch: a, DeviceSMs: deviceSMs,
		Regs: map[string]int{}, Threads: map[string]int{}, SmemBytes: map[string]int{},
	}
	var conflicts []benchLaneKeyConflict
	for _, name := range benchLaneSortedKeys(sources) {
		src := sources[name]
		fields := []struct {
			field string
			v     benchLaneValues
			dst   map[string]int
		}{
			{"regs", src.regs, in.Regs},
			{"threads", src.threads, in.Threads},
			{"smem_bytes", src.smem, in.SmemBytes},
		}
		for _, f := range fields {
			n, reason := f.v.resolve()
			switch {
			case reason == "":
				f.dst[name] = n
			case reason != benchLaneNoSource:
				conflicts = append(conflicts, benchLaneKeyConflict{Name: name, Field: f.field, Reason: reason})
			}
		}
	}
	return in, conflicts
}

func benchLaneSortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
