package main

// `fak fleet res` — the fleet accountant (#6557, rung 1 of epic #6552).
//
// `fak guard` already meters ONE session's harness slice (internal/harnessres) and banks
// it to .fak/nightrun/harness-resources.jsonl. What nothing could answer from inside fak
// was the fleet question: how much of this box do all the live seats, guards and MCP
// brokers cost TOGETHER. The measurement that opened #6552 (87 processes, 12.90 GiB) was
// taken with Get-Process from a shell — i.e. the fleet's own footprint was invisible to
// the thing that spawned it, and every later rung of that epic is justified by a delta in
// exactly this number.
//
// This verb only READS. It kills nothing, reaps nothing, throttles nothing. It gathers
// the host's pid/ppid/cmdline census (internal/procguard, which already owns the per-GOOS
// collector and its fail-open rules), hands it to harnessres to select the fak-owned
// subtree, classify it, read each PID's resources through the same per-platform readers
// the single-session path uses, and fold one rollup — then renders it and banks a row.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// --- injectable seams (overridden in tests so no live host is walked) --- //

// fleetResCensus returns the host's process census as harnessres rows. Production goes
// through procguard.CollectRelations — the same collector `fak fleet monitor/janitor`
// use, so the two verbs cannot disagree about what is running and neither re-derives the
// `ps` dialect / native-snapshot handling procguard already got right (#5537).
var fleetResCensus = func() ([]harnessres.ProcRef, string) {
	procs, collectErr := procguard.CollectRelations()
	out := make([]harnessres.ProcRef, 0, len(procs))
	for _, p := range procs {
		ref := harnessres.ProcRef{PID: p.PID, Name: p.Name, Cmdline: p.Cmdline}
		if p.PPID != nil {
			ref.PPID = *p.PPID
		}
		if p.AgeSec != nil {
			ref.AgeSec, ref.HaveAge = *p.AgeSec, true
		}
		out = append(out, ref)
	}
	return out, collectErr
}

// fleetResHost supplies the box context (RAM) the rollup is reported as a fraction of.
//
// It goes through the SAME reader the guard's per-session host block uses —
// compute.HostSystemMemoryInfo, over internal/compute/hostmem_{linux,darwin,windows}.go —
// so a fleet row and a session row divide by the same denominator. internal/harnessres is
// a stdlib-only leaf that imports nothing internal, which is why the reading has to arrive
// through a provider seam here rather than from inside the package (#2053).
//
// Fail-soft: a platform with no reader reports known=false, so the whole block stays
// absent and the rollup renders host fractions as n/a rather than fabricating them; a host
// that knows its total but not its available bytes returns compute.FreeUnknown (-1), so
// only that one axis drops out.
var fleetResHost = func() (harnessres.Host, bool) {
	total, avail, known := compute.HostSystemMemoryInfo()
	if !known || total <= 0 {
		return harnessres.Host{}, false
	}
	h := harnessres.Host{TotalRAMBytes: uint64(total), HaveRAM: true}
	if avail > 0 {
		h.AvailRAMBytes = uint64(avail)
	}
	return h, true
}

// fleetResNow is the clock seam for the banked row's timestamp.
var fleetResNow = time.Now

func runFleetRes(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleet res", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the rollup as one machine-readable JSON row instead of the text table")
	ledger := fs.String("ledger", "", "append the rollup row here instead of <repo>/"+harnessres.DefaultLedgerRel)
	noLedger := fs.Bool("no-ledger", false, "render only; do not append a rollup row to the ledger")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak fleet res: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	census, collectErr := fleetResCensus()
	// A census that returned rows AND an error is still a census (procguard keeps partial
	// output on a non-zero collector exit), so the error is a warning here and only a
	// census with NO rows is a failure. The opposite reading is how an unreadable host
	// starts rendering as an empty fleet (#5537).
	if len(census) == 0 {
		if collectErr == "" {
			collectErr = "no processes returned"
		}
		fmt.Fprintf(stderr, "fak fleet res: process census failed: %s\n", collectErr)
		return 1
	}
	if collectErr != "" {
		fmt.Fprintf(stderr, "fak fleet res: warning: partial process census: %s\n", collectErr)
	}

	host, _ := fleetResHost()
	rollup := harnessres.FoldFleet(
		harnessres.SampleFleet(harnessres.WalkFleet(census)),
		host,
		runtime.NumCPU(),
	)

	row, err := rollup.MarshalLedgerRow(fleetResNow())
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet res: encode rollup: %v\n", err)
		return 1
	}
	if *asJSON {
		fmt.Fprintf(stdout, "%s\n", row)
	} else {
		fmt.Fprint(stdout, rollup.Report())
	}

	// Bank the row by DEFAULT rather than behind a --write flag, matching what the guard
	// already does with its per-session row: the ledger is the gitignored runtime one, and
	// a wave-over-wave comparison only exists if looking at the fleet also records it. Use
	// --no-ledger for a pure read. A write failure is reported and never changes the exit
	// code — the measurement is the deliverable, banking it is best-effort.
	if !*noLedger {
		if err := appendFleetResRow(fleetResLedgerPath(*ledger), row); err != nil {
			fmt.Fprintf(stderr, "fak fleet res: warning: ledger append failed: %v\n", err)
		}
	}
	return 0
}

// fleetResLedgerPath resolves the ledger target: an explicit --ledger wins, otherwise the
// repo-relative runtime ledger the per-session rows already land in.
func fleetResLedgerPath(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return nightrunLedgerPath(harnessres.DefaultLedgerRel)
}

// appendFleetResRow appends one JSONL rollup row, creating the parent directory if
// needed. It shares the ledger FILE with the per-session rows and is distinguished by its
// own schema tag (harnessres.FleetLedgerSchema), so a reader that filters on the session
// schema is unaffected and both grains stay comparable in one place.
func appendFleetResRow(path string, row []byte) error {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		_ = os.MkdirAll(dir, 0o755)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(row, '\n'))
	return err
}
