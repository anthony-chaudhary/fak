package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/benchruns"
)

func cmdBenchRuns(argv []string) { os.Exit(runBenchRuns(os.Stdout, os.Stderr, argv)) }

func runBenchRuns(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		benchRunsUsage(stderr)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "list":
		return benchRunsList(stdout, stderr, rest)
	case "show":
		return benchRunsShow(stdout, stderr, rest)
	case "compare":
		return benchRunsCompare(stdout, stderr, rest)
	case "best":
		return benchRunsBest(stdout, stderr, rest)
	case "table":
		return benchRunsTable(stdout, stderr, rest)
	case "summary":
		return benchRunsSummary(stdout, stderr, rest)
	case "-h", "--help", "help":
		benchRunsUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak bench-runs: unknown subcommand %q\n", sub)
		benchRunsUsage(stderr)
		return 2
	}
}

func benchRunsList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	machine := fs.String("machine", "", "filter by machine ID")
	model := fs.String("model", "", "filter by model substring")
	precision := fs.String("precision", "", "filter by precision")
	since := fs.String("since", "", "filter by start date/string")
	until := fs.String("until", "", "filter by end date/string")
	format := fs.String("format", "table", "output format: table|json")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	cat, code := benchRunsCatalog(stderr, *workspace)
	if code != 0 {
		return code
	}
	runs := benchruns.FilterRuns(cat.Runs, benchruns.Filter{Machine: *machine, Model: *model, Precision: *precision, Since: *since, Until: *until})
	if *format == "json" {
		_ = writeIndentedJSON(stdout, runs)
		return 0
	}
	if *format != "table" {
		fmt.Fprintf(stderr, "fak bench-runs list: unknown --format %q\n", *format)
		return 2
	}
	fmt.Fprint(stdout, benchruns.RenderList(runs))
	return 0
}

func benchRunsShow(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs show", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: fak bench-runs show <run-id>")
		return 2
	}
	root := benchRunsRoot(*workspace)
	cat, code := benchRunsCatalog(stderr, root)
	if code != 0 {
		return code
	}
	d, err := benchruns.LoadRun(root, cat, fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs show: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, benchruns.RenderShow(d))
	return 0
}

func benchRunsCompare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 2 {
		fmt.Fprintln(stderr, "usage: fak bench-runs compare <run-id-1> <run-id-2>")
		return 2
	}
	root := benchRunsRoot(*workspace)
	cat, code := benchRunsCatalog(stderr, root)
	if code != 0 {
		return code
	}
	a, err := benchruns.LoadRun(root, cat, fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs compare: %v\n", err)
		return 1
	}
	b, err := benchruns.LoadRun(root, cat, fs.Arg(1))
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs compare: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, benchruns.RenderCompare(a, b))
	return 0
}

func benchRunsBest(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs best", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	model := fs.String("model", "", "filter by model")
	metric := fs.String("metric", "peak_tok_per_sec", "metric: peak_tok_per_sec|speedup")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	cat, code := benchRunsCatalog(stderr, *workspace)
	if code != 0 {
		return code
	}
	best, err := benchruns.Best(cat.Runs, *model, *metric)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs best: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, benchruns.RenderBest(best, *metric))
	return 0
}

func benchRunsTable(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs table", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	model := fs.String("model", "", "filter by model substring")
	precision := fs.String("precision", "", "filter by precision")
	format := fs.String("format", "markdown", "output format: markdown|json")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	cat, code := benchRunsCatalog(stderr, *workspace)
	if code != 0 {
		return code
	}
	runs := benchruns.FilterRuns(cat.Runs, benchruns.Filter{Model: *model, Precision: *precision})
	if *format == "json" {
		_ = writeIndentedJSON(stdout, runs)
		return 0
	}
	if *format != "markdown" {
		fmt.Fprintf(stderr, "fak bench-runs table: unknown --format %q\n", *format)
		return 2
	}
	if len(runs) == 0 {
		fmt.Fprintln(stderr, "No runs found")
		return 0
	}
	fmt.Fprint(stdout, benchruns.RenderMarkdownTable(runs))
	return 0
}

func benchRunsSummary(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-runs summary", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	groupBy := fs.String("group-by", "machine", "group by: machine|model")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *groupBy != "machine" && *groupBy != "model" {
		fmt.Fprintf(stderr, "fak bench-runs summary: unknown --group-by %q\n", *groupBy)
		return 2
	}
	cat, code := benchRunsCatalog(stderr, *workspace)
	if code != 0 {
		return code
	}
	fmt.Fprint(stdout, benchruns.RenderSummary(cat.Runs, *groupBy))
	return 0
}

func benchRunsCatalog(stderr io.Writer, workspace string) (benchruns.Catalog, int) {
	root := benchRunsRoot(workspace)
	cat, err := benchruns.LoadCatalog(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-runs: %v\n", err)
		fmt.Fprintln(stderr, "fak bench-runs: run `fak bench-catalog build` or refresh experiments/benchmark/catalog.json first")
		return benchruns.Catalog{}, 1
	}
	return cat, 0
}

func benchRunsRoot(workspace string) string {
	if workspace != "" {
		return workspace
	}
	return repoRoot()
}

func benchRunsUsage(w io.Writer) {
	fmt.Fprint(w, `fak bench-runs - query and compare recorded benchmark runs

usage:
  fak bench-runs list [--machine M] [--model M] [--precision P] [--format table|json]
  fak bench-runs show <run-id>
  fak bench-runs compare <run-id-1> <run-id-2>
  fak bench-runs best [--model M] [--metric peak_tok_per_sec|speedup]
  fak bench-runs table [--model M] [--precision P] [--format markdown|json]
  fak bench-runs summary [--group-by machine|model]
`)
}
