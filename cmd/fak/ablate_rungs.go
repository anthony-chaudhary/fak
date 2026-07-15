package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

// ablate_rungs.go — the RUNG axis of `fak ablate`. Where `--sweep` ablates cache LEVERS
// (ablate.KnownFeatures: vdso + env-gated cache knobs) over a bench.Trace, `--rungs`
// ablates adjudicator RUNGS (grammar / ifc-sink / monitor / …) over a turnbench.Trace via
// turnbench.RunLeverFlip: replay the frozen trace through the full-chain baseline plus one
// masked kernel per rung, and diff the kernel counters for EXACT per-rung causal
// attribution. This is the operator surface the rung half of epic #607 was missing (#3972);
// the harness (leverflip.go) shipped tested but CLI-unreachable. turnbench.Trace ≠
// bench.Trace, so the two modes load different codecs and never cross.

// rungsFlag is the optional-value flag behind `--rungs`. A bare `--rungs` selects the FULL
// per-rung sweep (every registered rung + the vDSO lever); `--rungs=grammar,ifc-sink`
// restricts it to the named rungs. It implements IsBoolFlag so the bare form parses AND
// never swallows the following flag — a plain string flag would read `--rungs --trace FILE`
// as rungs="--trace". Trailing bare names after it (`--trace FILE --rungs grammar`) are
// folded in from fs.Args() too, so both `--rungs=grammar` and a trailing `--rungs grammar`
// select the same single rung.
type rungsFlag struct {
	set   bool
	names []string
}

func (f *rungsFlag) IsBoolFlag() bool { return true }
func (f *rungsFlag) String() string   { return strings.Join(f.names, ",") }

// Set records that --rungs was given. Go passes "true" for the bare bool form (full sweep,
// no names) and the attached string for `--rungs=grammar,ifc-sink` (a name subset).
func (f *rungsFlag) Set(v string) error {
	f.set = true
	if v != "" && v != "true" {
		f.names = append(f.names, splitCommaList(v)...)
	}
	return nil
}

// runAblateRungs is the `--rungs` mode: load a TURNBENCH trace (an explicit --trace wins;
// otherwise resolve --suite under testdata/turntax), replay it through the full adjudicator
// chain plus one masked kernel per rung (turnbench.RunLeverFlip), and emit the per-rung
// attribution table — mirroring the arm-table --json/--out UX. Exit 0 ok, 1 a load/run
// error.
func runAblateRungs(stdout, stderr io.Writer, fs *flag.FlagSet, rungs *rungsFlag, tracePath, suite, outPath string, asJSON bool) int {
	// A rung sweep replays a turnbench trace, so an explicit --trace wins. Otherwise resolve
	// --suite under testdata/turntax (NOT tau2). The ablate default suite is the tau2
	// "tau2-smoke", meaningless here, so when --suite was left at its default we substitute
	// the canonical turntax suite rather than resolve a tau2 name under turntax.
	path := tracePath
	if path == "" {
		s := suite
		if !flagWasSet(fs, "suite") {
			s = "turntax-airline"
		}
		path = resolveSuite(turnTaxDir(), s)
	}
	t, err := turnbench.LoadTrace(path)
	if err != nil {
		fmt.Fprintln(stderr, "fak ablate:", err)
		return 1
	}

	// levers: the attached-value names plus any trailing bare names (fs.Args()). Empty =
	// the full sweep RunLeverFlip defaults to (every named rung + vdso). An unknown name is
	// reported Present=false by the harness, never dropped.
	levers := append(append([]string{}, rungs.names...), fs.Args()...)

	rep, err := turnbench.RunLeverFlip(ctx(), t, levers...)
	if err != nil {
		fmt.Fprintln(stderr, "fak ablate:", err)
		return 1
	}

	if outPath != "" {
		if err := os.WriteFile(outPath, rep.JSON(), 0o644); err != nil {
			fmt.Fprintln(stderr, "fak ablate:", err)
			return 1
		}
	}
	if asJSON {
		_, _ = stdout.Write(rep.JSON())
		return 0
	}
	printAblateRungs(stdout, rep)
	if outPath != "" {
		fmt.Fprintf(stdout, "report written : %s\n", outPath)
	}
	return 0
}

// printAblateRungs renders the per-rung attribution table for a human: one row per flipped
// rung with how it was realized (chain-mask / vdso-off), whether it was present in the
// chain, the outcome-counter delta its removal caused, and a one-line witness — the rung
// analogue of printAblation's arm table.
func printAblateRungs(w io.Writer, rep *turnbench.LeverFlipReport) {
	fmt.Fprintf(w, "== fak ablate --rungs: %s ==\n", rep.Provenance.SliceID)
	fmt.Fprintf(w, "workload hash  : %s\n", rep.Provenance.WorkloadHash)
	b := rep.Baseline
	fmt.Fprintf(w, "baseline (full chain, vdso on): calls %d  submits %d  vdso_hits %d  engine_calls %d  denies %d  transforms %d  quarantines %d\n",
		rep.Calls, b.Submits, b.VDSOHits, b.EngineCalls, b.Denies, b.Transforms, b.Quarantines)
	fmt.Fprintf(w, "rungs replayed : %d   (each is one model-free replay off the SAME trace; deltas are apples-to-apples)\n\n",
		rep.LeversReplayed)

	fmt.Fprintf(w, "%-17s %-11s %-8s %8s %10s %12s %8s %10s %8s\n",
		"rung", "realized", "present", "transf", "vdso_hits", "quarantines", "denies", "eng_calls", "changed")
	for i := range rep.Levers {
		l := &rep.Levers[i]
		d := l.Delta
		fmt.Fprintf(w, "%-17s %-11s %-8v %+8d %+10d %+12d %+8d %+10d %8v\n",
			l.Lever, l.Realization, l.Present, d.Transforms, d.VDSOHits, d.Quarantines, d.Denies, d.EngineCalls, l.Changed)
	}

	fmt.Fprintln(w, "\nwitness (what removing each rung proved on this trace):")
	for i := range rep.Levers {
		l := &rep.Levers[i]
		fmt.Fprintf(w, "  %-17s %s\n", l.Lever, l.Witness)
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, rep.CheapnessNote)
}

// ablateRungCatalog returns the rung levers a full `--rungs` sweep flips, in chain order:
// every uniquely-named registered adjudicator rung (abi.RungNames) plus the vDSO fast-path
// lever appended — the SAME set turnbench.RunLeverFlip's defaultLevers builds. It backs the
// rung section of `fak ablate --list`, so the printed menu can never drift from what a
// sweep actually runs.
func ablateRungCatalog() []string {
	seen := map[string]bool{}
	out := []string{}
	for _, n := range abi.RungNames(abi.Adjudicators()) {
		if n == "" || seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	if !seen[turnbench.VDSOLever] {
		out = append(out, turnbench.VDSOLever)
	}
	return out
}
