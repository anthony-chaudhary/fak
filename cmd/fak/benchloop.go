package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchloop"
	"github.com/anthony-chaudhary/fak/internal/nightrun"
)

func cmdBenchLoop(argv []string) { os.Exit(runBenchLoop(os.Stdout, os.Stderr, argv)) }

func runBenchLoop(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return benchLoopStatus(stdout, stderr, nil)
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "status":
		return benchLoopStatus(stdout, stderr, rest)
	case "next", "enter":
		return benchLoopNext(stdout, stderr, rest)
	case "walk":
		return benchLoopWalk(stdout, stderr, rest)
	case "run", "collect":
		return nightrunRun(stdout, stderr, rest)
	case "-h", "--help", "help":
		benchLoopUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak bench-loop: unknown subcommand %q\n", sub)
		benchLoopUsage(stderr)
		return 2
	}
}

type benchLoopFlags struct {
	workspace string
	overlay   string
	ledger    string
	now       string
	asJSON    bool
}

func parseBenchLoopFlags(name string, stderr io.Writer, argv []string) (*benchLoopFlags, int, bool) {
	fs := flag.NewFlagSet("fak bench-loop "+name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	f := &benchLoopFlags{}
	fs.StringVar(&f.workspace, "workspace", "", "workspace root (default: repo root)")
	fs.StringVar(&f.overlay, "overlay", "", "nightrun overlay file (default: <root>/"+nightrun.DefaultOverlayRel+" if present)")
	fs.StringVar(&f.ledger, "ledger", "", "nightrun ledger path (default: <root>/"+nightrun.DefaultLedgerRel+")")
	fs.StringVar(&f.now, "now", "", "evaluate as-of RFC3339, compact 20060102T150405Z, or YYYY-MM-DD")
	fs.BoolVar(&f.asJSON, "json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return nil, 2, false
	}
	return f, 0, true
}

func (f *benchLoopFlags) root() string {
	if f.workspace == "" {
		return repoRoot()
	}
	if abs, err := filepath.Abs(f.workspace); err == nil {
		return abs
	}
	return f.workspace
}

func (f *benchLoopFlags) nowOrZero() (time.Time, error) {
	if f.now == "" {
		return time.Time{}, nil
	}
	if t, err := time.Parse(time.RFC3339, f.now); err == nil {
		return t.UTC(), nil
	}
	if t, err := time.Parse("20060102T150405Z", f.now); err == nil {
		return t.UTC(), nil
	}
	if t, err := time.Parse("2006-01-02", f.now); err == nil {
		return t.UTC(), nil
	}
	return time.Time{}, fmt.Errorf("--now %q is not RFC3339, compact (20060102T150405Z), or YYYY-MM-DD", f.now)
}

func benchLoopReport(stderr io.Writer, argv []string, name string) (benchloop.Report, *benchLoopFlags, int, bool) {
	f, rc, ok := parseBenchLoopFlags(name, stderr, argv)
	if !ok {
		return benchloop.Report{}, nil, rc, false
	}
	now, err := f.nowOrZero()
	if err != nil {
		fmt.Fprintf(stderr, "fak bench-loop: %v\n", err)
		return benchloop.Report{}, nil, 2, false
	}
	rep := benchloop.Load(benchloop.Options{
		Root:        f.root(),
		OverlayPath: f.overlay,
		LedgerPath:  f.ledger,
		Now:         now,
	})
	return rep, f, 0, true
}

func benchLoopStatus(stdout, stderr io.Writer, argv []string) int {
	rep, f, rc, ok := benchLoopReport(stderr, argv, "status")
	if !ok {
		return rc
	}
	if f.asJSON {
		_ = writeIndentedJSONNoEscape(stdout, rep)
		return 0
	}
	fmt.Fprint(stdout, benchloop.RenderStatus(rep))
	return 0
}

func benchLoopNext(stdout, stderr io.Writer, argv []string) int {
	rep, f, rc, ok := benchLoopReport(stderr, argv, "next")
	if !ok {
		return rc
	}
	if f.asJSON {
		_ = writeIndentedJSONNoEscape(stdout, rep.NextAction)
		return 0
	}
	fmt.Fprint(stdout, benchloop.RenderNext(rep.NextAction))
	return 0
}

func benchLoopWalk(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bench-loop walk", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	walk := benchloop.Walk()
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, walk)
		return 0
	}
	fmt.Fprint(stdout, benchloop.RenderWalk(walk))
	return 0
}

func benchLoopUsage(w io.Writer) {
	fmt.Fprint(w, `fak bench-loop - benchmark super-loop manager

usage:
  fak bench-loop status [--json] [--workspace DIR] [--now STAMP]
  fak bench-loop next   [--json] [--workspace DIR] [--now STAMP]
  fak bench-loop walk   [--json]
  fak bench-loop run    [nightrun run flags...]

status folds the benchmark registry, recorded run catalog, nightrun ledger,
local capability-aware next selection, and benchmark-authority gap. run delegates
to fak nightrun run, which is dry-run unless --apply is passed.
`)
}
