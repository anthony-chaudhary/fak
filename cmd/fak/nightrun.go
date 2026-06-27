package main

// `fak nightrun` — the "run it all night" center of excellence: the one door an
// operator OR an agent uses to answer "what is the most important datum I can
// collect on THIS box right now?" and then collect it on a loop. It is the
// front end over internal/nightrun (the local-capability-aware next() selector
// + the durable collection ledger), unifying the benchmark grid and the curated
// open-witness backlog.
//
//	fak nightrun next   [--json]                 the single most important feasible datum
//	fak nightrun plan   [--json]                 the whole ranked queue for this box (+ why blocked)
//	fak nightrun run    [--apply] [--loop] [--max N] [--json]   collect (dry-run unless --apply)
//	fak nightrun ledger [--json]                 the durable collection history
//	fak nightrun caps   [--json]                 just the probed box capabilities
//
// next/plan/caps are pure reads. run is DRY-RUN by default (prints what it would
// execute, writes nothing); --apply executes real commands, captures each to an
// artifact, and appends an OBSERVED row to docs/nightrun/collected.jsonl.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/nightrun"
)

func cmdNightrun(argv []string) { os.Exit(runNightrun(os.Stdout, os.Stderr, argv)) }

func runNightrun(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		nightrunUsage(stderr)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "next":
		return nightrunNext(stdout, stderr, rest)
	case "plan", "list", "ls":
		return nightrunPlan(stdout, stderr, rest)
	case "run":
		return nightrunRun(stdout, stderr, rest)
	case "ledger", "history":
		return nightrunLedger(stdout, stderr, rest)
	case "caps", "capabilities":
		return nightrunCaps(stdout, stderr, rest)
	case "-h", "--help", "help":
		nightrunUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak nightrun: unknown subcommand %q\n", sub)
		nightrunUsage(stderr)
		return 2
	}
}

func nightrunUsage(w io.Writer) {
	fmt.Fprint(w, `fak nightrun — run it all night: collect the next() most important data on THIS box

usage:
  fak nightrun next   [--json]                              the single most important feasible datum
  fak nightrun plan   [--json]                              the whole ranked queue (+ why a task is blocked)
  fak nightrun run    [--apply] [--loop] [--max N] [--json] collect (dry-run unless --apply)
  fak nightrun ledger [--json]                              the durable collection history
  fak nightrun caps   [--json]                              the probed box capabilities

Start here:
  fak nightrun next      what should I collect right now, and the exact command to do it
`)
}

// nightrunFlags is the shared flag surface; each subcommand parses the subset it
// honours. Centralising it keeps --workspace/--overlay/--ledger/--now consistent.
type nightrunFlags struct {
	workspace string
	overlay   string
	ledger    string
	now       string
	asJSON    bool
	apply     bool
	loop      bool
	max       int
}

func parseNightrunFlags(name string, fs *flag.FlagSet, argv []string) (*nightrunFlags, error) {
	f := &nightrunFlags{}
	fs.StringVar(&f.workspace, "workspace", "", "workspace root (default: repo root)")
	fs.StringVar(&f.overlay, "overlay", "", "operator overlay file (default: <root>/"+nightrun.DefaultOverlayRel+" if present)")
	fs.StringVar(&f.ledger, "ledger", "", "collection ledger path (default: <root>/"+nightrun.DefaultLedgerRel+")")
	fs.StringVar(&f.now, "now", "", "evaluate as-of this RFC3339/compact stamp (default: now UTC; pin for deterministic output)")
	fs.BoolVar(&f.asJSON, "json", false, "emit machine-readable JSON")
	if name == "run" {
		fs.BoolVar(&f.apply, "apply", false, "execute real commands (default: dry-run — print what would run, write nothing)")
		fs.BoolVar(&f.loop, "loop", false, "keep collecting until the feasible queue is exhausted or --max is hit")
		fs.IntVar(&f.max, "max", 0, "cap the number of tasks collected this session (0 = unbounded within --loop)")
	}
	if err := fs.Parse(argv); err != nil {
		return nil, err
	}
	return f, nil
}

func (f *nightrunFlags) root() string {
	if f.workspace == "" {
		return repoRoot()
	}
	if abs, err := filepath.Abs(f.workspace); err == nil {
		return abs
	}
	return f.workspace
}

func (f *nightrunFlags) overlayPath(root string) string {
	if f.overlay != "" {
		return f.overlay
	}
	def := filepath.Join(root, filepath.FromSlash(nightrun.DefaultOverlayRel))
	if _, err := os.Stat(def); err == nil {
		return def
	}
	return "" // absent default overlay is fine — built-ins only
}

func (f *nightrunFlags) ledgerPath(root string) string {
	if f.ledger != "" {
		return f.ledger
	}
	return filepath.Join(root, filepath.FromSlash(nightrun.DefaultLedgerRel))
}

// nowOrWall resolves --now to a time, defaulting to the wall clock. A pinned
// stamp makes next/plan deterministic for tests and reproducible reports.
func (f *nightrunFlags) nowOrWall() (time.Time, error) {
	if f.now == "" {
		return time.Now().UTC(), nil
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

// load builds the capabilities, backlog, ledger, and now — the shared prelude of
// every subcommand.
func (f *nightrunFlags) load(stderr io.Writer) (root string, caps nightrun.Capabilities, tasks []nightrun.Task, ledger []nightrun.CollectRow, now time.Time, ok bool) {
	root = f.root()
	now, err := f.nowOrWall()
	if err != nil {
		fmt.Fprintf(stderr, "fak nightrun: %v\n", err)
		return root, caps, nil, nil, now, false
	}
	caps = nightrun.ProbeLocal(root)
	tasks, err = nightrun.Backlog(f.overlayPath(root))
	if err != nil {
		fmt.Fprintf(stderr, "fak nightrun: %v\n", err)
		return root, caps, nil, nil, now, false
	}
	ledger = nightrun.ReadLedgerFile(f.ledgerPath(root))
	return root, caps, tasks, ledger, now, true
}

func nightrunNext(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("nightrun next", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseNightrunFlags("next", fs, argv)
	if err != nil {
		return 2
	}
	_, caps, tasks, ledger, now, ok := f.load(stderr)
	if !ok {
		return 1
	}
	ranked := nightrun.Rank(tasks, caps, ledger, now)
	next, has := nightrun.Next(ranked)
	if f.asJSON {
		rep := nightrun.NextReport{
			Schema:       nightrun.NextSchema,
			GeneratedAt:  now.Format(time.RFC3339),
			Capabilities: caps,
			HasNext:      has,
		}
		if has {
			n := next
			rep.Next = &n
		} else {
			rep.Note = "no feasible task on this box right now"
		}
		emitNightrunJSON(stdout, rep)
		return 0
	}
	nightrun.RenderNext(stdout, caps, next, has)
	return 0
}

func nightrunPlan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("nightrun plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseNightrunFlags("plan", fs, argv)
	if err != nil {
		return 2
	}
	_, caps, tasks, ledger, now, ok := f.load(stderr)
	if !ok {
		return 1
	}
	ranked := nightrun.Rank(tasks, caps, ledger, now)
	feasible := 0
	for _, s := range ranked {
		if s.Feasible {
			feasible++
		}
	}
	if f.asJSON {
		emitNightrunJSON(stdout, nightrun.PlanReport{
			Schema:       nightrun.PlanSchema,
			GeneratedAt:  now.Format(time.RFC3339),
			Capabilities: caps,
			Feasible:     feasible,
			Total:        len(ranked),
			Ranked:       ranked,
		})
		return 0
	}
	nightrun.RenderPlan(stdout, caps, ranked)
	return 0
}

func nightrunRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("nightrun run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseNightrunFlags("run", fs, argv)
	if err != nil {
		return 2
	}
	root, caps, tasks, _, now, ok := f.load(stderr)
	if !ok {
		return 1
	}
	summary, err := nightrun.RunLoop(context.Background(), nightrun.RunOptions{
		Root:       root,
		Caps:       caps,
		Tasks:      tasks,
		Now:        now,
		Apply:      f.apply,
		Loop:       f.loop,
		Max:        f.max,
		LedgerPath: f.ledgerPath(root),
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak nightrun run: %v\n", err)
		return 1
	}
	if f.asJSON {
		emitNightrunJSON(stdout, summary)
		return 0
	}
	renderRunSummary(stdout, caps, summary)
	return 0
}

func nightrunLedger(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("nightrun ledger", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseNightrunFlags("ledger", fs, argv)
	if err != nil {
		return 2
	}
	root := f.root()
	rows := nightrun.ReadLedgerFile(f.ledgerPath(root))
	if f.asJSON {
		emitNightrunJSON(stdout, rows)
		return 0
	}
	nightrun.RenderLedger(stdout, rows)
	return 0
}

func nightrunCaps(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("nightrun caps", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseNightrunFlags("caps", fs, argv)
	if err != nil {
		return 2
	}
	caps := nightrun.ProbeLocal(f.root())
	if f.asJSON {
		emitNightrunJSON(stdout, caps)
		return 0
	}
	nightrun.RenderCapabilities(stdout, caps)
	return 0
}

// renderRunSummary prints the honest record of a run/dry-run session: each
// attempt, its outcome, and why the loop stopped.
func renderRunSummary(w io.Writer, caps nightrun.Capabilities, s nightrun.RunSummary) {
	nightrun.RenderCapabilities(w, caps)
	mode := "DRY-RUN (nothing executed; pass --apply to collect)"
	if s.Applied {
		mode = "APPLY (commands executed; ledger updated)"
	}
	fmt.Fprintf(w, "%s\n\n", mode)
	if len(s.Runs) == 0 {
		fmt.Fprintf(w, "no task attempted — %s\n", s.StopReason)
		return
	}
	for i, r := range s.Runs {
		fmt.Fprintf(w, "%d. [%s] %s\n   %s\n", i+1, r.Outcome, r.Task.ID, r.Task.Run)
		if r.Number != "" {
			fmt.Fprintf(w, "   number (observed): %s\n", r.Number)
		}
		if r.Artifact != "" {
			fmt.Fprintf(w, "   artifact: %s\n", r.Artifact)
		}
		if r.Err != "" {
			fmt.Fprintf(w, "   error: %s\n", r.Err)
		}
	}
	fmt.Fprintf(w, "\nstopped: %s\n", s.StopReason)
}

func emitNightrunJSON(w io.Writer, v any) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}
