package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/maputil"
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

// ablateArmRunner is the runner runAblate hands to ablate.SweepViaSubprocess for the
// env-gated (rung-2) path. Production binds the real subprocess re-exec; a test swaps a
// fake so the cmd suite never spawns the real fak binary.
var ablateArmRunner = ablate.ExecArmRunner

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

	features := splitCommaList(*sweep)

	path := *tracePath
	if path == "" {
		path = resolveSuite(traceDir(), *suite)
	}
	t, err := bench.LoadTrace(path)
	if err != nil {
		fmt.Fprintln(stderr, "fak ablate:", err)
		return 1
	}

	// BuildSweep validates the sweep spec (an unknown token fails loud → usage exit 2) and
	// builds the arm matrix shared by both rungs.
	configs, err := ablate.BuildSweep(features)
	if err != nil {
		fmt.Fprintln(stderr, "fak ablate:", err)
		return 2
	}

	// Route by rung. A vdso-only sweep flips the one runtime knob IN-PROCESS (no spawn).
	// Any env-gated feature is read once at process start, so its arms must re-exec a child
	// carrying the arm's FAK_* env (rung 2). A MIXED sweep goes wholly through the subprocess
	// path — each child still applies vdso in-process AND reads its own env — so the vdso and
	// the env arms land in one report under the same identical-workload guard.
	if anyEnvGated(features) {
		bin, err := os.Executable()
		if err != nil {
			fmt.Fprintln(stderr, "fak ablate: resolve fak binary for arm re-exec:", err)
			return 1
		}
		rep, dropped, err := ablate.SweepViaSubprocess(ctx(), bin, t, *engine, *engine+"-offline", configs, *baseline, ablateArmRunner)
		if err != nil {
			fmt.Fprintln(stderr, "fak ablate:", err)
			return 1
		}
		// A dropped child is a logged hole with a reason, never a silent gap.
		for _, d := range dropped {
			fmt.Fprintf(stderr, "fak ablate: arm %q dropped: %s\n", d.ArmID, d.Reason)
		}
		return emitAblation(stdout, stderr, rep, *out, *asJSON)
	}

	rep, err := ablate.Sweep(ctx(), t, *engine, *engine+"-offline", configs, *baseline)
	if err != nil {
		fmt.Fprintln(stderr, "fak ablate:", err)
		return 1
	}
	return emitAblation(stdout, stderr, rep, *out, *asJSON)
}

// anyEnvGated reports whether the sweep contains a feature the kernel reads at process
// start (so the run must take the rung-2 subprocess path). A pure-vdso sweep stays
// in-process.
func anyEnvGated(features []string) bool {
	for _, f := range features {
		if ablate.EnvGated(f) {
			return true
		}
	}
	return false
}

// emitAblation writes the assembled report to the requested sinks — an --out file, --json
// to stdout, or the human table — and returns the process exit code. Shared by the
// in-process and subprocess rungs so both render identically.
func emitAblation(stdout, stderr io.Writer, rep *ablate.Report, outPath string, asJSON bool) int {
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
	printAblation(stdout, rep)
	if outPath != "" {
		fmt.Fprintf(stdout, "report written : %s\n", outPath)
	}
	return 0
}

// cmdAblateArm is the HIDDEN child verb the rung-2 re-exec drives: it reads one arm request
// from stdin, runs that single arm in THIS process (whose FAK_* env the parent set at
// spawn), and writes the resulting AblationRun to stdout. It is not in usage() — an
// internal seam of `fak ablate`'s subprocess fan-out, reachable only via the re-exec.
func cmdAblateArm(argv []string) { os.Exit(runAblateArm(os.Stdin, os.Stdout, os.Stderr, argv)) }

// runAblateArm is the testable core of the child verb: it drives ablate.RunArmMode with the
// production trace codec and returns the process exit code.
func runAblateArm(stdin io.Reader, stdout, stderr io.Writer, _ []string) int {
	if err := ablate.RunArmMode(ctx(), stdin, stdout, ablate.UnmarshalTrace); err != nil {
		fmt.Fprintln(stderr, "fak ablate-arm:", err)
		return 1
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

	fmt.Fprintf(w, "%-10s %-12s %6s %9s %7s %7s %9s %9s %14s %9s %8s\n",
		"arm", "features", "calls", "vdso_hits", "denies", "quar", "p50_ns", "tokens", "provider_tokeq", "fak_tokeq", "wall_s")
	for i := range rep.Runs {
		r := &rep.Runs[i]
		fmt.Fprintf(w, "%-10s %-12s %6d %9d %7d %7d %9d %9d %14s %9s %8.3f\n",
			r.ArmID, featStr(r.Features), r.Arm.Calls, r.Arm.VDSOHits, r.Arm.Denies,
			r.Arm.Quarantines, r.Arm.P50Ns, r.Tokens(),
			formatAblationTokenEquiv(r.ProviderTokenEquiv()), formatAblationTokenEquiv(r.FakTokenEquiv()),
			r.WallSeconds)
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
		fmt.Fprintf(w, "  %-10s vdso_hits %+d   denies %+d   quar %+d   p50_ns %+d   tokens %+d   provider_tokeq %s   fak_tokeq %s\n",
			r.ArmID,
			r.Arm.VDSOHits-base.Arm.VDSOHits,
			r.Arm.Denies-base.Arm.Denies,
			r.Arm.Quarantines-base.Arm.Quarantines,
			r.Arm.P50Ns-base.Arm.P50Ns,
			r.Tokens()-base.Tokens(),
			formatAblationTokenEquivDelta(r.ProviderTokenEquiv()-base.ProviderTokenEquiv()),
			formatAblationTokenEquivDelta(r.FakTokenEquiv()-base.FakTokenEquiv()))
	}
}

func formatAblationTokenEquiv(v float64) string {
	return gateway.HumanTokenEquiv(v)
}

func formatAblationTokenEquivDelta(v float64) string {
	if v > 0 {
		return "+" + gateway.HumanTokenEquiv(v)
	}
	return gateway.HumanTokenEquiv(v)
}

// featStr renders an arm's descriptor as "k=v k=v" in sorted key order.
func featStr(d map[string]string) string {
	keys := maputil.SortedKeys(d)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+d[k])
	}
	return strings.Join(parts, " ")
}
