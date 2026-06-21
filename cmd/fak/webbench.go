// fak webbench — frontier web/browser agent benchmarks as a fak-native benchmark.
// Measures the value of fak's session value stack on multi-turn web automation tasks.
// Subcommands:
//
//	describe — load the web task set + derived geometry; print the deterministic
//	           prefill-token work-elimination (the value-stack floor) at a worker
//	           sweep. Fully offline; needs no model, GPU, or network. RUNNABLE NOW.
//
//	eval     — grade predictions into the task success-rate via the official harness
//	           (when available). Gated when this box lacks the runtime.
//
//	compare  — the full comparison: fak's metric families keyed to external benchmarks,
//	           with optional side-by-side against a benchmark results file.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/webbench"
)

func cmdWebbench(argv []string) {
	if len(argv) == 0 {
		webbenchUsage()
		os.Exit(2)
	}
	sub := argv[0]
	rest := argv[1:]
	switch sub {
	case "describe":
		cmdWebbenchDescribe(rest)
	case "eval":
		cmdWebbenchEval(rest)
	case "compare":
		cmdWebbenchCompare(rest)
	case "-h", "--help", "help":
		webbenchUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak webbench: unknown subcommand %q\n", sub)
		webbenchUsage()
		os.Exit(2)
	}
}

func webbenchUsage() {
	fmt.Fprint(os.Stderr, `fak webbench — Frontier Web Agent Benchmarks as a fak-native benchmark

usage:
  fak webbench describe [--dataset FILE] [--workers 1,2,4,8] [--out FILE]
        Load the web task instance set and its derived agent-workload geometry,
        then print the DETERMINISTIC prefill-token work-elimination (the value-stack
        floor) across a worker sweep. Fully offline; needs no model, GPU, or network.

  fak webbench eval --predictions preds.json [--run-id ID] [--max-workers N] [--out FILE]
        Grade a predictions file into the task success-rate via the official harness
        (when available). Gated when this box lacks the browser runtime.

  fak webbench compare [--dataset FILE] [--workers 1,2,4,8] [--predictions preds.json]
        [--bench-result FILE] [--out FILE] [--md FILE]
        THE comparison: fak's headline metric families keyed to external benchmarks,
        with optional side-by-side against a benchmark results file.

The metrics most relevant to web agents:
  A/C  net prefill work-elimination vs the naive re-prefill-every-turn harness
  B/C  cross-worker prefix reuse vs a tuned single-tenant per-worker KV
  A/B  the turn-tax (re-prefill vs KV persistence), independent of workers
`)
}

func cmdWebbenchDescribe(argv []string) {
	fs := flag.NewFlagSet("webbench describe", flag.ExitOnError)
	dataset := fs.String("dataset", "", "path to web task dataset (JSONL or JSON array)")
	workersArg := fs.String("workers", "1,2,4,8", "comma-separated worker counts to sweep")
	limit := fs.Int("limit", 0, "cap to the first N instances (0 = all)")
	out := fs.String("out", "", "write the Summary JSON here (default: stdout JSON + human table on stderr)")
	_ = fs.Parse(argv)

	if *dataset == "" {
		fmt.Fprintln(os.Stderr, "fak webbench describe: --dataset is required")
		os.Exit(2)
	}

	d, err := webbench.LoadDataset(*dataset)
	must(err)

	if *limit > 0 && *limit < d.Len() {
		d = d.Limit(*limit)
	}

	workers := parseIntList(*workersArg)
	if len(workers) == 0 {
		workers = []int{1, 2, 4, 8}
	}

	gm := webbench.DefaultGeometryModel()
	s := webbench.Describe(d, gm, workers)

	if *out != "" {
		data, _ := json.MarshalIndent(s, "", "  ")
		must(os.WriteFile(*out, data, 0644))
	} else {
		data, _ := json.MarshalIndent(s, "", "  ")
		fmt.Println(string(data))
	}

	printWebbenchSummary(os.Stderr, s, *dataset, *out)
}

func printWebbenchSummary(w *os.File, s webbench.Summary, src, out string) {
	fmt.Fprintf(w, "\n== fak webbench describe ==\n")
	fmt.Fprintf(w, "source        : %s\n", src)
	fmt.Fprintf(w, "instances     : %d\n", s.Instances)
	fmt.Fprintf(w, "difficulty    : %s\n", sortedCounts(s.DifficultyDist))
	fmt.Fprintf(w, "category      : %s\n", sortedCounts(s.CategoryDist))
	fmt.Fprintf(w, "geometry src  : %s\n", sortedCounts(s.GeometrySources))
	fmt.Fprintf(w, "turns         : min %d  median %d  max %d  (total %d navigation turns)\n",
		s.TurnsMin, s.TurnsMedian, s.TurnsMax, s.TotalTurns)
	fmt.Fprintf(w, "\nprefill-token work-elimination (deterministic floor, no model):\n")
	fmt.Fprintf(w, "  %-8s %16s %16s %16s   %8s %8s %8s\n", "workers", "A naive", "B per-agent", "C fak", "A/C", "B/C", "A/B")
	for _, p := range s.Prefill {
		fmt.Fprintf(w, "  %-8d %16s %16s %16s   %7.1fx %7.2fx %7.1fx\n",
			p.Workers,
			formatTokens(p.ANaive),
			formatTokens(p.BAgent),
			formatTokens(p.CFak),
			p.AOverC,
			p.BOverC,
			p.AOverB,
		)
	}
	fmt.Fprintf(w, "\n  A/C = net prefill work-elimination vs the naive re-prefill-every-turn harness\n")
	fmt.Fprintf(w, "  B/C = cross-worker prefix reuse (the value stack; bites at workers>1)\n")
	fmt.Fprintf(w, "  A/B = the turn-tax (re-prefill vs KV persistence), worker-independent\n")
	if out != "" {
		fmt.Fprintf(w, "\nSummary JSON written: %s\n", out)
	}
}

func cmdWebbenchEval(argv []string) {
	fs := flag.NewFlagSet("webbench eval", flag.ExitOnError)
	preds := fs.String("predictions", "", "path to predictions JSON (required)")
	benchmark := fs.String("benchmark", "browser-agent", "benchmark name (browser-agent, webvoyager, etc.)")
	runID := fs.String("run-id", "fak-webbench", "harness run id")
	maxWorkers := fs.Int("max-workers", 4, "harness parallelism")
	python := fs.String("python", "", "python interpreter (default: detected)")
	out := fs.String("out", "", "write the EvalResult JSON here (default: stdout)")
	_ = fs.Parse(argv)

	if *preds == "" {
		fmt.Fprintln(os.Stderr, "fak webbench eval: --predictions is required")
		os.Exit(2)
	}

	res, err := webbench.RunEval(webbench.EvalConfig{
		PredictionsPath: *preds,
		Benchmark:       *benchmark,
		RunID:           *runID,
		MaxWorkers:      *maxWorkers,
		Python:          *python,
	})
	must(err)

	if *out != "" {
		data, _ := json.MarshalIndent(res, "", "  ")
		must(os.WriteFile(*out, data, 0644))
	} else {
		data, _ := json.MarshalIndent(res, "", "  ")
		fmt.Println(string(data))
	}

	fmt.Fprintf(os.Stderr, "\n== fak webbench eval ==\n")
	if res.Available {
		fmt.Fprintf(os.Stderr, "PASSED %d / %d  (%.1f%% success rate)\n", res.Passed, res.Total, res.SuccessRatePct)
		if res.ReportPath != "" {
			fmt.Fprintf(os.Stderr, "report: %s\n", res.ReportPath)
		}
	} else {
		fmt.Fprintf(os.Stderr, "GATED on this box: %s\n", res.Reason)
		fmt.Fprintf(os.Stderr, "run on a box with the harness:\n  %s\n", res.Command)
	}
}

func cmdWebbenchCompare(argv []string) {
	fs := flag.NewFlagSet("webbench compare", flag.ExitOnError)
	dataset := fs.String("dataset", "", "path to web task dataset (required)")
	workersArg := fs.String("workers", "1,2,4,8", "worker sweep")
	limit := fs.Int("limit", 0, "cap to the first N instances (0 = all)")
	preds := fs.String("predictions", "", "predictions JSON to fold the success rate (optional)")
	benchResult := fs.String("bench-result", "", "external benchmark results for side-by-side (optional)")
	out := fs.String("out", "", "write the Comparison JSON here (default: stdout)")
	md := fs.String("md", "", "write the markdown report here (optional)")
	_ = fs.Parse(argv)
	_ = preds // TODO: use predictions to fold success rate

	if *dataset == "" {
		fmt.Fprintln(os.Stderr, "fak webbench compare: --dataset is required")
		os.Exit(2)
	}

	d, err := webbench.LoadDataset(*dataset)
	must(err)

	if *limit > 0 && *limit < d.Len() {
		d = d.Limit(*limit)
	}

	workers := parseIntList(*workersArg)
	if len(workers) == 0 {
		workers = []int{1, 2, 4, 8}
	}

	in := webbench.CompareInputs{
		Dataset:     d,
		Geometry:    webbench.DefaultGeometryModel(),
		Workers:     workers,
		BenchResult: *benchResult,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
	}

	c := webbench.BuildComparison(in)

	if *out != "" {
		data, _ := json.MarshalIndent(c, "", "  ")
		must(os.WriteFile(*out, data, 0644))
	} else {
		data, _ := json.MarshalIndent(c, "", "  ")
		fmt.Println(string(data))
	}

	if *md != "" {
		must(os.WriteFile(*md, []byte(webbench.RenderMarkdown(c)), 0644))
	}

	fmt.Fprintf(os.Stderr, "\n== fak webbench compare ==\n")
	fmt.Fprintf(os.Stderr, "source        : %s\n", *dataset)
	fmt.Fprintf(os.Stderr, "instances     : %d\n", c.Summary.Instances)
	for _, f := range c.Families {
		fmt.Fprintf(os.Stderr, "  %-30s %-11s %s\n", f.Name, "["+f.Kind+"]", f.Provenance)
	}
	if *md != "" {
		fmt.Fprintf(os.Stderr, "markdown: %s\n", *md)
	}
}

func formatTokens(n int64) string {
	if n < 1_000_000 {
		return fmt.Sprintf("%d", n)
	}
	m := float64(n) / 1_000_000
	if m < 1000 {
		return fmt.Sprintf("%.1f M", m)
	}
	g := m / 1000
	return fmt.Sprintf("%.2f G", g)
}

func sortedCounts(m map[string]int) string {
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
