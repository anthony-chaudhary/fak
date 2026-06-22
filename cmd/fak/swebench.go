package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/swebench"
)

// fak swebench — run SWE-bench Verified as a fak-native benchmark whose results
// are directly comparable to the external Benchmark tool ("bench") that runs the
// same task set against an SGLang endpoint, on the metrics fak is built to move:
// KV-cache reuse / prefill elimination (the value stack), turns + tokens (the
// turn-tax), in-process adjudication cost, and resolve-rate + safety.
//
// Subcommands (more land as the harness fills in):
//
//	describe — load the instance set + derived geometry; print the deterministic
//	           prefill-token work-elimination (the value-stack floor) at a worker
//	           sweep. Fully offline; needs no model, GPU, or network. RUNNABLE NOW.
func cmdSwebench(argv []string) {
	if len(argv) == 0 {
		swebenchUsage()
		os.Exit(2)
	}
	sub := argv[0]
	rest := argv[1:]
	switch sub {
	case "describe":
		cmdSwebenchDescribe(rest)
	case "run":
		cmdSwebenchRun(rest)
	case "eval":
		cmdSwebenchEval(rest)
	case "compare":
		cmdSwebenchCompare(rest)
	case "compare-runners":
		cmdSwebenchCompareRunners(rest)
	case "-h", "--help", "help":
		swebenchUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak swebench: unknown subcommand %q\n", sub)
		swebenchUsage()
		os.Exit(2)
	}
}

func swebenchUsage() {
	fmt.Fprint(os.Stderr, `fak swebench — SWE-bench Verified, comparable to the external bench harness

usage:
  fak swebench describe [--difficulty FILE | --dataset FILE] [--workers 1,2,4,8] [--out FILE]
        Load the SWE-bench Verified instance set and its derived agent-workload
        geometry, then print the DETERMINISTIC prefill-token work-elimination
        (the value-stack floor) across a worker sweep. --difficulty reads bench's
        swebench_verified_difficulty.json (all 500 ids + official buckets, fully
        offline); --dataset reads a full princeton-nlp/SWE-bench_Verified export
        (JSONL or JSON array) for real problem-statement token geometry.

  fak swebench run --agent AGENT [--filter FILTER] [--output DIR]
        [--max-steps N] [--timeout DURATION] [--gateway ADDR] [--model MODEL] [--allow-exec]
        Run an agent on SWE-bench instances and generate predictions.json.
        AGENT: mock (dummy patches), fleet (gateway coding agent → fak serve), deepswe (R2E-Gym baseline).
        FILTER: smoke (~5 instances), l3 (~50), full (all 500).
        fleet drives a read/edit loop against a running 'fak serve' (point --gateway at it;
        --allow-exec additionally enables the shell tool — sandbox/container only).

  fak swebench eval --predictions preds.json [--run-id ID] [--max-workers N] [--out FILE]
        Grade a predictions file into the SWE-bench Verified resolve-rate via the
        OFFICIAL harness (the identical path bench grades with). Runs locally when
        Docker + the swebench module are present; otherwise prints an honestly
        GATED result plus the exact command to run on a Docker box (the DGX).

  fak swebench compare [--difficulty FILE | --dataset FILE] [--workers 1,2,4,8]
        [--predictions preds.json] [--bench-result results_RUNID.json]
        [--with-adjudication] [--out FILE] [--md FILE]
        THE comparison: fak's four headline metric families keyed to bench's own
        vocabulary, beside a bench results_*.json when given. --predictions folds
        the resolve-rate; --with-adjudication folds the in-process vs spawn-per-hook
        gate; --md writes the house-style markdown table.

  fak swebench compare-runners --runners RUNNERS [--filter FILTER] [--output DIR]
        [--max-steps N] [--timeout DURATION] [--gateway ADDR] [--model MODEL]
        Head-to-head comparison between multiple agent runners (fleet, deepswe, mock).
        Generates comparison.json + comparison.md with resolve rates and per-runner stats.

the metrics most relevant to us, on the real SWE-bench Verified set:
  A/C  net prefill work-elimination vs the naive re-prefill-every-turn harness
  B/C  cross-worker prefix reuse vs a tuned single-tenant per-worker KV
  A/B  the turn-tax (re-prefill vs KV persistence), independent of workers
`)
}

func cmdSwebenchDescribe(argv []string) {
	fs := flag.NewFlagSet("swebench describe", flag.ExitOnError)
	difficulty := fs.String("difficulty", "", "bench difficulty map (swebench_verified_difficulty.json) — all 500 ids + buckets, offline")
	dataset := fs.String("dataset", "", "full SWE-bench Verified dataset (JSONL or JSON array) for real problem-statement geometry")
	workersArg := fs.String("workers", "1,2,4,8", "comma-separated worker counts to sweep (the bench mini-workers-sweep axis)")
	limit := fs.Int("limit", 0, "cap to the first N instances (0 = all)")
	out := fs.String("out", "", "write the Summary JSON here (default: stdout JSON + human table on stderr)")
	_ = fs.Parse(argv)

	// Resolve the source. Prefer an explicit flag; if neither is given, fall back
	// to FAK_SWEBENCH_DIFFICULTY / FAK_SWEBENCH_DATASET when set — otherwise
	// require a flag.
	d, srcDesc, err := loadSwebenchSource(*difficulty, *dataset)
	must(err)
	if *limit > 0 {
		d = d.Limit(*limit)
	}

	workers := parseIntList(*workersArg)
	if len(workers) == 0 {
		workers = []int{1, 2, 4, 8}
	}

	gm := swebench.DefaultGeometryModel()
	s := swebench.Describe(d, gm, workers)

	if *out != "" {
		must(os.WriteFile(*out, jsonIndent(s), 0o644))
	} else {
		fmt.Println(string(jsonIndent(s)))
	}

	// Human-readable headline on stderr (so stdout stays clean JSON when piped).
	printSwebenchSummary(os.Stderr, s, srcDesc, *out)
}

// loadSwebenchSource loads the instance set from whichever source is given,
// overlaying the difficulty buckets when both a full dataset and the difficulty
// map are available.
func loadSwebenchSource(difficulty, dataset string) (*swebench.Dataset, string, error) {
	// Ergonomic default via env, never a baked-in machine path: a hardcoded
	// developer-home default used to live here, which was a dead default on every
	// other machine and leaked an operator path into tracked source (issue #180,
	// PUBLIC-SCRUB-POLICY.md). Honor FAK_SWEBENCH_DIFFICULTY / FAK_SWEBENCH_DATASET
	// when nothing explicit is passed; otherwise require the flags.
	if difficulty == "" && dataset == "" {
		if env := os.Getenv("FAK_SWEBENCH_DIFFICULTY"); env != "" {
			difficulty = env
		} else if env := os.Getenv("FAK_SWEBENCH_DATASET"); env != "" {
			dataset = env
		} else {
			return nil, "", fmt.Errorf("pass --difficulty FILE or --dataset FILE (or set FAK_SWEBENCH_DIFFICULTY / FAK_SWEBENCH_DATASET)")
		}
	}

	if dataset != "" {
		d, err := swebench.LoadDataset(dataset)
		if err != nil {
			return nil, "", err
		}
		desc := fmt.Sprintf("dataset %s (%d instances, real problem-statement geometry)", dataset, d.Len())
		if difficulty != "" {
			if dd, _, err := swebench.LoadDifficulty(difficulty); err == nil {
				n := d.MergeDifficulty(dd)
				desc += fmt.Sprintf(" + %d difficulty annotations", n)
			}
		}
		return d, desc, nil
	}

	d, meta, err := swebench.LoadDifficulty(difficulty)
	if err != nil {
		return nil, "", err
	}
	desc := fmt.Sprintf("difficulty map %s (%d instances from %s, bucket-derived geometry)",
		difficulty, d.Len(), meta.SourceDataset)
	return d, desc, nil
}

func printSwebenchSummary(w *os.File, s swebench.Summary, src, out string) {
	fmt.Fprintf(w, "\n== fak swebench describe ==\n")
	fmt.Fprintf(w, "source        : %s\n", src)
	fmt.Fprintf(w, "instances     : %d\n", s.Instances)
	fmt.Fprintf(w, "difficulty    : %s\n", sortedCountsSWE(s.DifficultyDist))
	fmt.Fprintf(w, "geometry src  : %s\n", sortedCountsSWE(s.GeometrySources))
	fmt.Fprintf(w, "turns         : min %d  median %d  max %d  (total %d round-trips)\n",
		s.TurnsMin, s.TurnsMedian, s.TurnsMax, s.TotalTurns)
	fmt.Fprintf(w, "\nprefill-token work-elimination (deterministic floor, no model):\n")
	fmt.Fprintf(w, "  %-8s %16s %16s %16s   %8s %8s %8s\n", "workers", "A naive", "B per-agent", "C fak", "A/C", "B/C", "A/B")
	for _, p := range s.Prefill {
		fmt.Fprintf(w, "  %-8d %16d %16d %16d   %7.1fx %7.2fx %7.1fx\n",
			p.Workers, p.A, p.B, p.C, p.AOverC, p.BOverC, p.AOverB)
	}
	fmt.Fprintf(w, "\n  A/C = net prefill work-elimination vs the naive re-prefill-every-turn harness\n")
	fmt.Fprintf(w, "  B/C = cross-worker prefix reuse (the value stack; bites at workers>1)\n")
	fmt.Fprintf(w, "  A/B = the turn-tax (re-prefill vs KV persistence), worker-independent\n")
	if out != "" {
		fmt.Fprintf(w, "\nSummary JSON written: %s\n", out)
	}
}

func sortedCountsSWE(m map[string]int) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, m[k]))
	}
	return strings.Join(parts, "  ")
}

// fak swebench eval — grade a predictions file into the resolve-rate via the
// official harness, honestly gated when this box can't run it.
func cmdSwebenchEval(argv []string) {
	fs := flag.NewFlagSet("swebench eval", flag.ExitOnError)
	preds := fs.String("predictions", "", "path to preds.json (required)")
	runID := fs.String("run-id", "fak-swebench", "harness run id (names logs/run_evaluation/<run_id>)")
	maxWorkers := fs.Int("max-workers", 4, "harness parallelism")
	dataset := fs.String("dataset-name", "princeton-nlp/SWE-bench_Verified", "HF dataset name the harness grades against")
	python := fs.String("python", "", "python interpreter (default: detected python3/python)")
	out := fs.String("out", "", "write the EvalResult JSON here (default: stdout)")
	_ = fs.Parse(argv)
	if *preds == "" {
		fmt.Fprintln(os.Stderr, "fak swebench eval: --predictions is required")
		os.Exit(2)
	}

	res, err := swebench.RunEval(swebench.EvalConfig{
		PredictionsPath: *preds, RunID: *runID, MaxWorkers: *maxWorkers,
		DatasetName: *dataset, Python: *python,
	})
	must(err)

	if *out != "" {
		must(os.WriteFile(*out, jsonIndent(res), 0o644))
	} else {
		fmt.Println(string(jsonIndent(res)))
	}
	fmt.Fprintf(os.Stderr, "\n== fak swebench eval ==\n")
	if res.Available {
		fmt.Fprintf(os.Stderr, "RESOLVED %d / %d  (%.1f%% pass rate)\n", res.Resolved, res.Total, res.ResolveRatePct)
		if res.ReportPath != "" {
			fmt.Fprintf(os.Stderr, "report: %s\n", res.ReportPath)
		}
	} else {
		fmt.Fprintf(os.Stderr, "GATED on this box: %s\n", res.Reason)
		fmt.Fprintf(os.Stderr, "run on a Docker box (the DGX):\n  %s\n", res.Command)
	}
}

// fak swebench compare — the four-family fak<->bench comparison.
func cmdSwebenchCompare(argv []string) {
	fs := flag.NewFlagSet("swebench compare", flag.ExitOnError)
	difficulty := fs.String("difficulty", "", "bench difficulty map (offline geometry source)")
	dataset := fs.String("dataset", "", "full SWE-bench Verified dataset (real problem-statement geometry)")
	workersArg := fs.String("workers", "1,2,4,8", "worker sweep (mirrors bench --sweep-workers)")
	limit := fs.Int("limit", 0, "cap to the first N instances (0 = all)")
	preds := fs.String("predictions", "", "preds.json to fold the resolve-rate (optional)")
	runID := fs.String("run-id", "fak-swebench", "harness run id for the resolve grade")
	benchResult := fs.String("bench-result", "", "a bench results_<run_id>.json for a true side-by-side (optional)")
	withAdj := fs.Bool("with-adjudication", false, "measure the in-process vs spawn-per-hook adjudication gate inline")
	out := fs.String("out", "", "write the Comparison JSON here (default: stdout)")
	md := fs.String("md", "", "write the house-style markdown report here (optional)")
	_ = fs.Parse(argv)

	d, srcDesc, err := loadSwebenchSource(*difficulty, *dataset)
	must(err)
	if *limit > 0 {
		d = d.Limit(*limit)
	}
	workers := parseIntList(*workersArg)

	in := swebench.CompareInputs{
		Dataset:     d,
		Geometry:    swebench.DefaultGeometryModel(),
		Workers:     workers,
		BenchResult: *benchResult,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
	}

	// Resolve-rate (gated) if predictions were supplied.
	if *preds != "" {
		res, err := swebench.RunEval(swebench.EvalConfig{PredictionsPath: *preds, RunID: *runID})
		must(err)
		in.Eval = &res
	}

	// Adjudication gate (fak-native family 3) measured inline if requested.
	if *withAdj {
		if adj, ok := measureAdjudication(); ok {
			in.Adjudication = &adj
		} else {
			fmt.Fprintln(os.Stderr, "fak swebench compare: adjudication gate unavailable (tau2-smoke trace missing); family 3 stays gated")
		}
	}

	c := swebench.BuildComparison(in)

	if *out != "" {
		must(os.WriteFile(*out, jsonIndent(c), 0o644))
	} else {
		fmt.Println(string(jsonIndent(c)))
	}
	if *md != "" {
		must(os.WriteFile(*md, []byte(swebench.RenderMarkdown(c)), 0o644))
	}

	fmt.Fprintf(os.Stderr, "\n== fak swebench compare ==\nsource: %s\n", srcDesc)
	for _, f := range c.Families {
		fmt.Fprintf(os.Stderr, "  %-30s %-11s %s\n", f.Name, "["+f.Kind+"]", f.Provenance)
	}
	if c.Bench != nil && c.Bench.Present {
		fmt.Fprintf(os.Stderr, "bench side: %s (schema v%d, hit-ratio %.1f%%)\n", c.Bench.ProfileName, c.Bench.SchemaVersion, c.Bench.TokenHitRatioPct)
	}
	if *md != "" {
		fmt.Fprintf(os.Stderr, "markdown: %s\n", *md)
	}
}

// measureAdjudication runs the in-process vs spawn-per-hook adjudication A/B over
// the committed tau2-smoke trace (the same gate `fak bench` reports) and returns
// it as an AdjCost. Returns ok=false if the trace can't be found.
func measureAdjudication() (swebench.AdjCost, bool) {
	path := filepath.Join(traceDir(), "tau2-smoke.json")
	t, err := bench.LoadTrace(path)
	if err != nil {
		return swebench.AdjCost{}, false
	}
	opt := bench.Options{EngineID: "mock", EngineModel: "mock-offline", BaselineN: 30}
	if bin, err := os.Executable(); err == nil {
		opt.BinPath = bin
	}
	rep, err := bench.Run(ctx(), t, opt)
	if err != nil {
		return swebench.AdjCost{}, false
	}
	adj := swebench.AdjCost{
		InProcessP50Ns: rep.On.P50Ns,
		SpawnHookP50Ns: rep.Baseline.P50Ns,
	}
	if rep.On.P50Ns > 0 && rep.Baseline.P50Ns > 0 {
		adj.SpeedupX = float64(rep.Baseline.P50Ns) / float64(rep.On.P50Ns)
	}
	return adj, true
}

// parseIntList parses "1,2,4,8" into []int, skipping non-numeric tokens.
func parseIntList(s string) []int {
	var out []int
	for _, tok := range strings.Split(s, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		n := 0
		ok := true
		for _, r := range tok {
			if r < '0' || r > '9' {
				ok = false
				break
			}
			n = n*10 + int(r-'0')
		}
		if ok && n > 0 {
			out = append(out, n)
		}
	}
	return out
}

// cmdSwebenchRun runs an agent on SWE-bench instances and generates predictions.json
func cmdSwebenchRun(argv []string) {
	fs := flag.NewFlagSet("swebench run", flag.ExitOnError)
	filter := fs.String("filter", "smoke", "instance filter: smoke (~5 instances), l3 (~50 instances), full (all 500)")
	limit := fs.Int("limit", 0, "cap to the first N instances (0 = all from filter)")
	agent := fs.String("agent", "mock", "agent type: mock, fleet, or deepswe")
	output := fs.String("output", "", "output directory for predictions.json and results")
	predsOnly := fs.Bool("preds-only", false, "only generate predictions.json, skip evaluation")
	difficulty := fs.String("difficulty", "", "bench difficulty map (optional, for better geometry)")
	dataset := fs.String("dataset", "", "full dataset path (optional, for real problem statements)")
	maxSteps := fs.Int("max-steps", 50, "max agent steps per instance")
	timeout := fs.Duration("timeout", 10*time.Minute, "per-instance timeout (0 = no limit)")
	gateway := fs.String("gateway", "localhost:8080", "fleet gateway address (for fleet agent)")
	model := fs.String("model", "", "model id to request from the gateway (fleet) or endpoint/path (deepswe)")
	allowExec := fs.Bool("allow-exec", false, "allow the fleet agent's shell (run) tool — use ONLY in a sandboxed/containerized run")
	_ = fs.Parse(argv)

	// Map agent string to RunnerType
	var rt swebench.RunnerType
	switch *agent {
	case "mock":
		rt = swebench.RunnerMock
	case "fleet":
		rt = swebench.RunnerFleet
	case "deepswe":
		rt = swebench.RunnerDeepSWE
	default:
		fmt.Fprintf(os.Stderr, "fak swebench run: unknown agent %q (use mock, fleet, or deepswe)\n", *agent)
		os.Exit(2)
	}

	cfg := swebench.RunConfig{
		Runner:      rt,
		Filter:      *filter,
		Limit:       *limit,
		MaxSteps:    *maxSteps,
		Timeout:     *timeout,
		OutputDir:   *output,
		DatasetPath: *dataset,
		Difficulty:  *difficulty,
		GatewayAddr: *gateway,
		Model:       *model,
		AllowExec:   *allowExec,
	}
	if rt == swebench.RunnerFleet {
		cfg.Planner = newFleetPlanner(*gateway, *model)
	}

	ctx := context.Background()
	res, err := swebench.Run(ctx, cfg)
	must(err)

	fmt.Fprintf(os.Stderr, "\n== fak swebench run ==\n")
	fmt.Fprintf(os.Stderr, "runner       : %s\n", res.Meta.Runner)
	fmt.Fprintf(os.Stderr, "instances    : %d total, %d done, %d failed, %d skipped\n",
		res.Meta.TotalInstances, res.Meta.DoneInstances, res.Meta.Failed, res.Meta.Skipped)
	fmt.Fprintf(os.Stderr, "elapsed      : %.1fs\n", res.Elapsed.Seconds())
	fmt.Fprintf(os.Stderr, "predictions  : %s\n", res.PredictionsPath)
	fmt.Fprintf(os.Stderr, "meta         : %s\n", res.MetaPath)

	// Run evaluation unless preds-only
	if !*predsOnly {
		runID := fmt.Sprintf("fak-swebench-%s", time.Now().Format("20060102T150405Z"))
		evalRes, err := swebench.RunEval(swebench.EvalConfig{
			PredictionsPath: res.PredictionsPath,
			RunID:           runID,
			MaxWorkers:      4,
		})
		must(err)

		if evalRes.Available {
			fmt.Fprintf(os.Stderr, "\n== eval results ==\n")
			fmt.Fprintf(os.Stderr, "RESOLVED     %d / %d  (%.1f%% pass rate)\n",
				evalRes.Resolved, evalRes.Total, evalRes.ResolveRatePct)
			if evalRes.ReportPath != "" {
				fmt.Fprintf(os.Stderr, "report       : %s\n", evalRes.ReportPath)
			}
		} else {
			fmt.Fprintf(os.Stderr, "\n== eval gated ==\n")
			fmt.Fprintf(os.Stderr, "%s\n", evalRes.Reason)
			fmt.Fprintf(os.Stderr, "run on DGX:\n  %s\n", evalRes.Command)
		}
	}
}

// cmdSwebenchCompareRunners runs a side-by-side comparison between fleet and DeepSWE baselines.
func cmdSwebenchCompareRunners(argv []string) {
	fs := flag.NewFlagSet("swebench compare", flag.ExitOnError)
	filter := fs.String("filter", "smoke", "instance filter: smoke (~5), l3 (~50), full (all 500)")
	limit := fs.Int("limit", 0, "cap to the first N instances (0 = all from filter)")
	runners := fs.String("runners", "fleet,deepswe", "comma-separated runners to compare (fleet, deepswe, mock)")
	output := fs.String("output", "", "output directory for comparison results")
	difficulty := fs.String("difficulty", "", "bench difficulty map (optional)")
	dataset := fs.String("dataset", "", "full dataset path (optional)")
	maxSteps := fs.Int("max-steps", 50, "max agent steps per instance")
	timeout := fs.Duration("timeout", 10*time.Minute, "per-instance timeout")
	gateway := fs.String("gateway", "localhost:8080", "fleet gateway address")
	model := fs.String("model", "", "model endpoint for deepswe")
	_ = fs.Parse(argv)

	// Parse runners list.
	runnerStrs := strings.Split(*runners, ",")
	runnerTypes := make([]swebench.RunnerType, 0, len(runnerStrs))
	for _, r := range runnerStrs {
		r = strings.TrimSpace(r)
		switch r {
		case "fleet":
			runnerTypes = append(runnerTypes, swebench.RunnerFleet)
		case "deepswe":
			runnerTypes = append(runnerTypes, swebench.RunnerDeepSWE)
		case "mock":
			runnerTypes = append(runnerTypes, swebench.RunnerMock)
		default:
			fmt.Fprintf(os.Stderr, "fak swebench compare: unknown runner %q (use fleet, deepswe, or mock)\n", r)
			os.Exit(2)
		}
	}
	if len(runnerTypes) == 0 {
		runnerTypes = []swebench.RunnerType{swebench.RunnerFleet, swebench.RunnerDeepSWE}
	}

	cfg := swebench.CompareConfig{
		Runners:      runnerTypes,
		Filter:       *filter,
		Limit:        *limit,
		MaxSteps:     *maxSteps,
		Timeout:      *timeout,
		OutputDir:    *output,
		DatasetPath:  *dataset,
		Difficulty:   *difficulty,
		FleetGateway: *gateway,
		DeepSWEModel: *model,
	}
	for _, rt := range runnerTypes {
		if rt == swebench.RunnerFleet {
			cfg.FleetPlanner = newFleetPlanner(*gateway, *model)
			break
		}
	}

	ctx := context.Background()
	cr, err := swebench.RunComparison(ctx, cfg)
	must(err)

	fmt.Fprintf(os.Stderr, "\n== fak swebench compare ==\n")
	fmt.Fprintf(os.Stderr, "instances    : %d\n", cr.Summary.TotalInstances)
	fmt.Fprintf(os.Stderr, "runners      : %s\n", strings.Join(cr.Summary.Runners, ", "))
	fmt.Fprintf(os.Stderr, "headline     : %s\n", cr.Summary.Headline)

	// Print per-runner resolve rates.
	if len(cr.Summary.ResolveRates) > 0 {
		fmt.Fprintf(os.Stderr, "\nresolve rates:\n")
		for _, rt := range cr.Summary.Runners {
			if rate, ok := cr.Summary.ResolveRates[rt]; ok {
				fmt.Fprintf(os.Stderr, "  %-12s %.1f%%\n", rt, rate)
			} else {
				fmt.Fprintf(os.Stderr, "  %-12s (gated)\n", rt)
			}
		}
	}

	fmt.Fprintf(os.Stderr, "\noutput       : %s/comparison.json + %s/comparison.md\n", cfg.OutputDir, cfg.OutputDir)
}
