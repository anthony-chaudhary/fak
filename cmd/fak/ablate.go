package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/bench"
)

// fak ablate — the SELF-ABLATION sweep. Where `fak bench` ablates ONE feature (the
// vDSO) over a frozen trace in a fixed 2-arm A/B, `fak ablate` generalizes it to an
// N-ARM matrix: replay the same frozen tool-call trace under each feature
// configuration and print one row per arm, so the cost/benefit of a fak feature is
// read straight off the kernel counters. Every arm replays the SAME trace, so one
// workload hash binds them (the identical-workload guard) — the deltas are
// apples-to-apples. This is the deterministic, $0, no-model core of the harness
// (epic #607); the live + cross-agent (pure fak vs Claude Code) halves are later rungs.
//
//	fak ablate --sweep vdso                         (built-in tau2-smoke trace)
//	fak ablate --trace FILE --sweep vdso --json     (emit the AblationReport JSON)
//	fak ablate --suite NAME --out ablation.json
//
// Exit 0 ok, 1 a load/run error, 2 a usage error.
func cmdAblate(argv []string) { os.Exit(runAblate(os.Stdout, os.Stderr, argv)) }

// runAblate is the testable core: it returns the process exit code instead of calling
// os.Exit, and takes its streams explicitly (mirrors runRoute).
func runAblate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("ablate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	suite := fs.String("suite", "tau2-smoke", "trace suite under testdata/tau2")
	tracePath := fs.String("trace", "", "explicit trace path (overrides --suite)")
	sweep := fs.String("sweep", "vdso", "comma list of runtime features to sweep (known: "+strings.Join(ablate.KnownFeatures, ",")+")")
	baseline := fs.String("baseline", "all-off", "arm id used as the delta reference in the table")
	out := fs.String("out", "", "write the AblationReport JSON to this path")
	asJSON := fs.Bool("json", false, "emit the AblationReport JSON to stdout (no table)")
	engine := fs.String("engine", "mock", "engine id (the offline mock by default)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	path := *tracePath
	if path == "" {
		path = resolveSuite(traceDir(), *suite)
	}
	t, err := bench.LoadTrace(path)
	if err != nil {
		fmt.Fprintln(stderr, "fak ablate:", err)
		return 1
	}

	configs, err := ablate.BuildSweep(splitCommaList(*sweep))
	if err != nil {
		fmt.Fprintln(stderr, "fak ablate:", err)
		return 2
	}

	rep, err := ablate.Sweep(ctx(), t, *engine, *engine+"-offline", configs, *baseline)
	if err != nil {
		fmt.Fprintln(stderr, "fak ablate:", err)
		return 1
	}

	if *out != "" {
		if err := os.WriteFile(*out, rep.JSON(), 0o644); err != nil {
			fmt.Fprintln(stderr, "fak ablate:", err)
			return 1
		}
	}
	if *asJSON {
		_, _ = stdout.Write(rep.JSON())
		return 0
	}
	printAblation(stdout, rep)
	if *out != "" {
		fmt.Fprintf(stdout, "report written : %s\n", *out)
	}
	return 0
}

// splitCommaList splits "a,b ,c" into ["a","b","c"], dropping empties.
func splitCommaList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// printAblation renders the N-arm table for a human: one row per arm with the kernel
// counters, then deltas vs the baseline arm.
func printAblation(w io.Writer, rep *ablate.Report) {
	fmt.Fprintf(w, "== fak ablate: %s ==\n", rep.Provenance.SliceID)
	fmt.Fprintf(w, "workload hash  : %s\n", rep.WorkloadHash)
	fmt.Fprintf(w, "arms           : %d   baseline: %s   (same trace each arm; deltas are apples-to-apples)\n\n",
		len(rep.Runs), rep.Baseline)

	fmt.Fprintf(w, "%-10s %-12s %6s %9s %7s %7s %9s %9s %8s\n",
		"arm", "features", "calls", "vdso_hits", "denies", "quar", "p50_ns", "tokens", "wall_s")
	for i := range rep.Runs {
		r := &rep.Runs[i]
		fmt.Fprintf(w, "%-10s %-12s %6d %9d %7d %7d %9d %9d %8.3f\n",
			r.ArmID, featStr(r.Features), r.Arm.Calls, r.Arm.VDSOHits, r.Arm.Denies,
			r.Arm.Quarantines, r.Arm.P50Ns, r.Tokens(), r.WallSeconds)
	}

	base := rep.ArmByID(rep.Baseline)
	if base == nil {
		return
	}
	fmt.Fprintf(w, "\ndeltas vs %s (per arm):\n", rep.Baseline)
	for i := range rep.Runs {
		r := &rep.Runs[i]
		if r.ArmID == rep.Baseline {
			continue
		}
		fmt.Fprintf(w, "  %-10s vdso_hits %+d   denies %+d   quar %+d   p50_ns %+d   tokens %+d\n",
			r.ArmID,
			r.Arm.VDSOHits-base.Arm.VDSOHits,
			r.Arm.Denies-base.Arm.Denies,
			r.Arm.Quarantines-base.Arm.Quarantines,
			r.Arm.P50Ns-base.Arm.P50Ns,
			r.Tokens()-base.Tokens())
	}
}

// featStr renders an arm's descriptor as "k=v k=v" in sorted key order.
func featStr(d map[string]string) string {
	keys := sortedKeys(d)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+d[k])
	}
	return strings.Join(parts, " ")
}
