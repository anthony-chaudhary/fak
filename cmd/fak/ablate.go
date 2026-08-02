package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/bench"
	enginepkg "github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardtrace"
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
//	fak ablate --report experiments/ablate/tau2-smoke-vdso-ablation.json   (re-render a saved run)
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
	verbFlagUsage(fs, "ablate") // #2232: overview verb -> deep help above the flag dump
	suite := fs.String("suite", "tau2-smoke", "trace suite under testdata/tau2")
	tracePath := fs.String("trace", "", "explicit trace path (overrides --suite)")
	fromSession := fs.String("from-session", "", "captured fak guard replay fixture to replay as a session-backed cassette")
	sweep := fs.String("sweep", "vdso", "comma list of runtime features to sweep (known: "+strings.Join(ablate.KnownFeatures(), ",")+")")
	baseline := fs.String("baseline", "all-off", "arm id used as the delta reference in the table")
	sweepPlan := fs.String("sweep-plan", "", "bounded planner: main or pairwise")
	top := fs.Int("top", 0, "top-K main-effect movers used by --sweep-plan pairwise")
	out := fs.String("out", "", "write the AblationReport JSON to this path")
	asJSON := fs.Bool("json", false, "emit the AblationReport JSON to stdout (no table)")
	list := fs.Bool("list", false, "print the sweepable cache-lever catalog (owner/plane/fidelity/env) and exit; with --json emit the FeatureCard array + presets")
	var rungs rungsFlag
	fs.Var(&rungs, "rungs", "rung-attribution mode: per-rung lever-flip over a turnbench trace (bare --rungs = full sweep; --rungs=grammar,ifc-sink for a subset)")
	report := fs.String("report", "", "load a saved AblationReport JSON (e.g. from experiments/ablate/) and re-render its table/deltas — no trace, engine, or replay")
	engineIDFlag := fs.String("engine", "mock", "engine id (the offline mock by default)")
	models := fs.String("models", "", "model-strength axis: comma list of tiers to grade each feature across (known: "+strings.Join(ablate.KnownModelTiers(), ",")+"); omitted keeps the legacy single-tier output")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	// --models is the MODEL-STRENGTH axis (#4412): grade every feature arm's marginal
	// delta across a weak -> strong ladder so a scaffold the model still NEEDS separates
	// from one it has OUTGROWN. Parsed UP FRONT — before any trace load or replay — so a
	// typo costs a usage error rather than a whole sweep. Empty means the axis never
	// runs, which is what preserves the legacy output.
	tiers, tierErr := ablate.ParseModelTiers(*models)
	if tierErr != nil {
		fmt.Fprintln(stderr, "fak ablate:", tierErr)
		return 2
	}

	// --list is the "see what I can try" surface behind FeatureCatalog(): it needs no
	// trace, no engine, and no replay — it prints the static cache-lever catalog straight
	// from internal/ablate/catalog.go (the single source of truth a live arm also reads).
	if *list {
		printAblateCatalog(stdout, *asJSON)
		return 0
	}

	// --report is the READ counterpart to --out: it loads a saved AblationReport JSON (a
	// prior sweep, e.g. a committed experiments/ablate/*.json artifact) and re-renders its
	// table + baseline deltas with no trace, engine, or replay — so an ablation captured to
	// an experiment file stays re-analyzable without re-running the whole sweep. It
	// short-circuits before any sweep, exactly like --list. When --baseline is given
	// explicitly it re-selects the delta reference arm on the loaded report (view the same
	// experiment against a different reference without re-running); otherwise the report's
	// own stored baseline stands. --json / --out re-emit the (re-baselined) report.
	if *report != "" {
		rep, err := ablate.LoadReport(*report)
		if err != nil {
			return ablateFail(stderr, err)
		}
		if flagWasSet(fs, "baseline") {
			rep.Baseline = *baseline
		}
		// The identical-workload guard (+ baseline-membership check) is what keeps the
		// deltas apples-to-apples; apply it to a loaded report too, so a tampered or
		// mismatched artifact fails loud rather than rendering misleading deltas.
		if err := rep.Validate(); err != nil {
			return ablateFail(stderr, err)
		}
		return emitAblation(stdout, stderr, rep, *out, *asJSON, tiers)
	}

	// --rungs is the RUNG axis (issue #3972): rather than sweep cache levers, ablate the
	// adjudicator chain one rung at a time over a turnbench trace (turnbench.RunLeverFlip).
	// It replaces the whole --sweep path (a rung is a chain adjudicator, not an
	// ablate.KnownFeatures cache lever) and short-circuits before any BuildSweep, exactly
	// like --list / --report.
	if rungs.set {
		return runAblateRungs(stdout, stderr, fs, &rungs, *tracePath, *suite, *out, *asJSON)
	}

	features := splitCommaList(*sweep)

	engineID := *engineIDFlag
	engineModel := *engineIDFlag + "-offline"
	var t *bench.Trace
	if *fromSession != "" {
		if *tracePath != "" {
			fmt.Fprintln(stderr, "fak ablate: --from-session and --trace are mutually exclusive")
			return 2
		}
		sessionTrace, cas, sessionEngineID, err := guardtrace.LoadSessionTrace(*fromSession)
		if err != nil {
			return ablateFail(stderr, err)
		}
		abi.RegisterEngine(sessionEngineID, enginepkg.NewCassetteEngine(cas))
		t = sessionTrace
		engineID = sessionEngineID
		engineModel = "session-cassette"
	} else {
		path := *tracePath
		if path == "" {
			path = resolveSuite(traceDir(), *suite)
		}
		var err error
		t, err = bench.LoadTrace(path)
		if err != nil {
			return ablateFail(stderr, err)
		}
	}

	// Build the legacy full lattice or a bounded registry-driven plan. Pairwise first
	// measures main effects, then runs only interactions among the measured top-K movers.
	var configs []ablate.FeatureConfig
	var err error
	if *sweepPlan == "" {
		configs, err = ablate.BuildSweep(features)
	} else if *sweepPlan == ablate.SweepPlanMain {
		configs, err = ablate.BuildSweepPlan(*sweepPlan, *top, nil)
		features = ablate.KnownFeatures()
		*baseline = "all-off"
	} else if *sweepPlan == ablate.SweepPlanPairwise {
		mainConfigs, planErr := ablate.BuildSweepPlan(ablate.SweepPlanMain, *top, nil)
		if planErr != nil {
			err = planErr
		} else {
			mainReport, runErr := runAblatePlan(t, engineID, engineModel, mainConfigs, "all-off")
			if runErr != nil {
				err = runErr
			} else {
				configs, err = ablate.BuildSweepPlan(*sweepPlan, *top, ablate.RankMainEffects(mainReport))
			}
		}
		features = ablate.KnownFeatures()
		*baseline = "all-off"
	} else {
		configs, err = ablate.BuildSweepPlan(*sweepPlan, *top, nil)
	}
	if err != nil {
		fmt.Fprintln(stderr, "fak ablate:", err)
		return 2
	}
	// Route by the EXPANDED feature set: a preset (e.g. @wire-cache) resolves to its
	// levers, so an env-gated member correctly selects the rung-2 subprocess path
	// instead of the raw preset token reading as non-env-gated. BuildSweep already
	// validated the spec above, so expansion cannot error here.
	routeFeatures, _ := ablate.ExpandPresets(features)
	if *fromSession != "" && anyEnvGated(routeFeatures) {
		fmt.Fprintln(stderr, "fak ablate: --from-session currently supports in-process sweeps only; use --sweep vdso or a bench trace for env-gated sweeps")
		return 2
	}

	// Route by rung. A vdso-only sweep flips the one runtime knob IN-PROCESS (no spawn).
	// Any env-gated feature is read once at process start, so its arms must re-exec a child
	// carrying the arm's FAK_* env (rung 2). A MIXED sweep goes wholly through the subprocess
	// path — each child still applies vdso in-process AND reads its own env — so the vdso and
	// the env arms land in one report under the same identical-workload guard.
	if anyEnvGated(routeFeatures) {
		bin, err := os.Executable()
		if err != nil {
			fmt.Fprintln(stderr, "fak ablate: resolve fak binary for arm re-exec:", err)
			return 1
		}
		rep, dropped, err := ablate.SweepViaSubprocess(ctx(), bin, t, engineID, engineModel, configs, *baseline, ablateArmRunner)
		if err != nil {
			return ablateFail(stderr, err)
		}
		// A dropped child is a logged hole with a reason, never a silent gap.
		for _, d := range dropped {
			fmt.Fprintf(stderr, "fak ablate: arm %q dropped: %s\n", d.ArmID, d.Reason)
		}
		return emitAblation(stdout, stderr, rep, *out, *asJSON, tiers)
	}

	rep, err := ablate.Sweep(ctx(), t, engineID, engineModel, configs, *baseline)
	if err != nil {
		return ablateFail(stderr, err)
	}
	return emitAblation(stdout, stderr, rep, *out, *asJSON, tiers)
}

// ablateFail reports one `fak ablate` runtime failure and yields its exit code. A report that
// will not load or will not validate, a session/bench trace that will not load, a sweep that
// errored -- all of them are the same kind of event to an operator (the sweep did not happen),
// so they are surfaced under the same prefix and exit 1 alike. The exit-2 sites above are the
// argument-validation refusals (a bad --models ladder, a bad sweep plan, a flag pair that
// cannot both hold); several print in this same bare form, so the exit code -- not the
// wording -- is what separates "you asked for the impossible" from "the sweep broke".
func ablateFail(stderr io.Writer, err error) int {
	fmt.Fprintln(stderr, "fak ablate:", err)
	return 1
}

func runAblatePlan(t *bench.Trace, engineID, engineModel string, configs []ablate.FeatureConfig, baseline string) (*ablate.Report, error) {
	if anyEnvGated(ablate.KnownFeatures()) {
		bin, err := os.Executable()
		if err != nil {
			return nil, err
		}
		rep, _, err := ablate.SweepViaSubprocess(ctx(), bin, t, engineID, engineModel, configs, baseline, ablateArmRunner)
		return rep, err
	}
	return ablate.Sweep(ctx(), t, engineID, engineModel, configs, baseline)
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

// ablateTierScorer is the TierScorer runAblate hands the model-strength axis (#4412).
// Production binds the STUB: no live per-tier measurement is wired yet, so a real run
// reports UNMEASURED rather than a fabricated grade. A test swaps in a fake with fixed
// per-tier outcomes — the same injection idiom as ablateArmRunner above.
var ablateTierScorer ablate.TierScorer = ablate.StubTierScorer{}

// emitAblation writes the assembled report to the requested sinks — an --out file, --json
// to stdout, or the human table — and returns the process exit code. Shared by the
// in-process and subprocess rungs so both render identically.
//
// tiers is the --models ladder: non-empty runs the model-strength axis over the report
// BEFORE any sink sees it, so the file, the JSON, and the table all carry the same
// verdicts. Empty skips the axis entirely, leaving the legacy output untouched.
func emitAblation(stdout, stderr io.Writer, rep *ablate.Report, outPath string, asJSON bool, tiers []string) int {
	if err := ablate.AnnotateModelStrength(ctx(), rep, tiers, ablateTierScorer, ablate.StrengthParams{}); err != nil {
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

// printAblation renders the N-arm table for a human: one row per arm with the kernel
// counters, then deltas vs the baseline arm.
func printAblation(w io.Writer, rep *ablate.Report) {
	fmt.Fprintf(w, "== fak ablate: %s ==\n", rep.Provenance.SliceID)
	fmt.Fprintf(w, "workload hash  : %s\n", rep.WorkloadHash)
	fmt.Fprintf(w, "arms           : %d   baseline: %s   (same trace each arm; deltas are apples-to-apples)\n\n",
		len(rep.Runs), rep.Baseline)

	fmt.Fprintf(w, "%-10s %-12s %-36s %6s %9s %7s %7s %9s %9s %14s %9s %15s %8s\n",
		"arm", "features", "cache_effects", "calls", "vdso_hits", "denies", "quar", "p50_ns", "tokens", "provider_tokeq", "fak_tokeq", "prefix_mismatch", "wall_s")
	for i := range rep.Runs {
		r := &rep.Runs[i]
		fmt.Fprintf(w, "%-10s %-12s %-36s %6d %9d %7d %7d %9d %9d %14s %9s %15d %8.3f\n",
			r.ArmID, featStr(r.Features), cacheEffectStr(r.CacheEffects), r.Arm.Calls, r.Arm.VDSOHits, r.Arm.Denies,
			r.Arm.Quarantines, r.Arm.P50Ns, r.Tokens(),
			formatAblationTokenEquiv(r.ProviderTokenEquiv()), formatAblationTokenEquiv(r.FakTokenEquiv()),
			r.PrefixIntegrity.PrefixMismatch, r.WallSeconds)
	}
	if len(rep.Dropped) > 0 {
		fmt.Fprintf(w, "\ndropped arms (subprocess/reporting holes):\n")
		for _, d := range rep.Dropped {
			fmt.Fprintf(w, "  %-10s %s\n", d.ArmID, droppedArmSummary(d))
		}
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
		fmt.Fprintf(w, "  %-10s vdso_hits %+d   denies %+d   quar %+d   p50_ns %+d   tokens %+d   provider_tokeq %s   fak_tokeq %s   prefix_mismatch %+d\n",
			r.ArmID,
			r.Arm.VDSOHits-base.Arm.VDSOHits,
			r.Arm.Denies-base.Arm.Denies,
			r.Arm.Quarantines-base.Arm.Quarantines,
			r.Arm.P50Ns-base.Arm.P50Ns,
			r.Tokens()-base.Tokens(),
			formatAblationTokenEquivDelta(r.ProviderTokenEquiv()-base.ProviderTokenEquiv()),
			formatAblationTokenEquivDelta(r.FakTokenEquiv()-base.FakTokenEquiv()),
			int64(r.PrefixIntegrity.PrefixMismatch)-int64(base.PrefixIntegrity.PrefixMismatch))
	}
	printModelStrength(w, rep)
}

// printModelStrength renders the #4412 model-strength axis: one block per feature arm
// carrying its per-tier marginal-delta trajectory and the verdict folded from it. It
// prints NOTHING when no arm carries a card, which is what keeps a run without
// --models byte-identical to the legacy table.
func printModelStrength(w io.Writer, rep *ablate.Report) {
	graded := 0
	for i := range rep.Runs {
		if rep.Runs[i].ModelStrength != nil {
			graded++
		}
	}
	if graded == 0 {
		return
	}
	fmt.Fprintf(w, "\nmodel-strength axis (verdict per feature, classified on the strongest tier):\n")
	for i := range rep.Runs {
		ms := rep.Runs[i].ModelStrength
		if ms == nil {
			continue
		}
		fmt.Fprintf(w, "  %-10s %-12s %s\n", rep.Runs[i].ArmID, ms.Verdict, modelStrengthTrajectory(ms))
		fmt.Fprintf(w, "             %s\n", ms.Reason)
	}
}

// modelStrengthTrajectory renders the per-tier deltas as `weak=+0.2000 strong=-0.0500`.
// An unmeasured rung prints "unmeasured" rather than a 0.0000 delta — a rung nobody
// scored must not read as a rung that scored zero.
func modelStrengthTrajectory(ms *ablate.ModelStrength) string {
	parts := make([]string, 0, len(ms.Tiers))
	for _, td := range ms.Tiers {
		if !td.Measured {
			parts = append(parts, td.Tier+"=unmeasured")
			continue
		}
		parts = append(parts, fmt.Sprintf("%s=%+.4f", td.Tier, td.Delta))
	}
	return strings.Join(parts, " ")
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

func droppedArmSummary(d ablate.DroppedArm) string {
	parts := []string{}
	if strings.TrimSpace(d.Stage) != "" {
		parts = append(parts, "stage="+d.Stage)
	}
	if d.ExitCode != nil {
		parts = append(parts, fmt.Sprintf("exit=%d", *d.ExitCode))
	}
	if d.DurationSeconds > 0 {
		parts = append(parts, fmt.Sprintf("duration=%.3fs", d.DurationSeconds))
	}
	if strings.TrimSpace(d.StderrTail) != "" {
		parts = append(parts, "stderr="+d.StderrTail)
	}
	if strings.TrimSpace(d.StdoutTail) != "" {
		parts = append(parts, "stdout="+d.StdoutTail)
	}
	if strings.TrimSpace(d.ExpectedWorkloadHash) != "" || strings.TrimSpace(d.ActualWorkloadHash) != "" {
		parts = append(parts, fmt.Sprintf("hash=%s/%s", cachevalueEmptyDash(d.ActualWorkloadHash), cachevalueEmptyDash(d.ExpectedWorkloadHash)))
	}
	parts = append(parts, d.Reason)
	return strings.Join(parts, " ")
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

func cacheEffectStr(effects []ablate.CacheEffect) string {
	if len(effects) == 0 {
		return "-"
	}
	parts := make([]string, 0, len(effects))
	for _, e := range effects {
		parts = append(parts, e.Feature+"="+e.Status+"/"+e.Fidelity+"/"+e.Owner)
	}
	return strings.Join(parts, " ")
}
