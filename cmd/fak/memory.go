package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/memq"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// cmdMemory handles `fak memory <subcommand>`: the agent-facing memory-operation
// algebra. Subcommands:
//
//	drivers            list the registered memory strategies (the "canned queries")
//	explain            render a driver's (or an authored) query as a plan, WITHOUT running it
//	run                execute a driver's (or an authored) query against a backend
//
// The backend is the in-memory demo corpus by default, or a persisted recall core
// image with --dir. Effects are PROPOSED unless --apply is passed (fail-closed).
func cmdMemory(args []string) {
	if len(args) == 0 {
		memoryUsage()
		os.Exit(2)
	}
	switch args[0] {
	case "drivers":
		cmdMemoryDrivers(args[1:])
	case "explain":
		cmdMemoryExplain(args[1:])
	case "run":
		cmdMemoryRun(args[1:])
	case "-h", "--help", "help":
		memoryUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak memory: unknown subcommand %q\n", args[0])
		memoryUsage()
		os.Exit(2)
	}
}

func memoryUsage() {
	fmt.Fprint(os.Stderr, `usage: fak memory <subcommand>

  fak memory drivers [--json]
      list the registered memory strategies — each is a composable query in the
      memq algebra, not a hardcoded function (build SQL, not a specific query).

  fak memory explain --driver NAME [--intent STR] [--k N] [--budget BYTES]
                     [--query-file PLAN.json] [--dir IMAGE] [--json]
      render the plan WITHOUT executing it: every step, which steps are effects,
      and which mutate durable state (and so are proposal-only without --apply).

  fak memory run --driver NAME [--intent STR] [--k N] [--budget BYTES]
                 [--query-file PLAN.json] [--dir IMAGE] [--apply] [--out report.json] [--json]
      execute the query against a backend (the in-memory demo corpus, or a recall
      core image with --dir). Mutations are PROPOSED unless --apply is given.

An agent authors its OWN strategy by emitting a Query JSON (--query-file) instead of
picking a built-in driver — the same executor runs both.
`)
}

// memoryQuery resolves the query to run/explain: a --query-file JSON Query, else a
// built-in driver compiled with the given params. Exactly one source is required.
func memoryQuery(driver, queryFile, intent string, k int, budget int64) (memq.Query, string) {
	if queryFile != "" {
		b, err := os.ReadFile(pathutil.ExpandTilde(queryFile))
		must(err)
		var q memq.Query
		must(json.Unmarshal(b, &q))
		label := "authored:" + queryFile
		if q.Intent == "" {
			q.Intent = intent
		}
		return q, label
	}
	if driver == "" {
		fmt.Fprintln(os.Stderr, "fak memory: one of --driver or --query-file is required")
		os.Exit(2)
	}
	d, ok := memq.Get(driver)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak memory: unknown driver %q (try `fak memory drivers`)\n", driver)
		os.Exit(2)
	}
	return d.Build(memq.Params{Intent: intent, K: k, Budget: budget}), "driver:" + driver
}

// memoryBackend selects the backend: a recall core image at --dir, else the in-memory
// demo corpus. It returns the backend and a human label.
func memoryBackend(dir string) (memq.Backend, string) {
	if dir == "" {
		return memq.NewDemoStore(), "in-memory demo corpus"
	}
	dir = pathutil.ExpandTilde(dir)
	s, err := recall.Load(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak memory: could not load a recall image at %q (%v); using the in-memory demo corpus\n", dir, err)
		return memq.NewDemoStore(), "in-memory demo corpus"
	}
	return memq.NewRecallBackend(s, dir), "recall core image " + dir
}

func cmdMemoryDrivers(argv []string) {
	fs := flag.NewFlagSet("memory drivers", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit the driver list as JSON")
	intent := fs.String("intent", "the task at hand", "intent used to compile each driver's example plan")
	_ = fs.Parse(argv)

	drivers := memq.Drivers()
	if *asJSON {
		type row struct {
			Name string    `json:"name"`
			Doc  string    `json:"doc"`
			Plan memq.Plan `json:"plan"`
		}
		out := make([]row, 0, len(drivers))
		for _, d := range drivers {
			out = append(out, row{d.Name, d.Doc, memq.Explain(d.Build(memq.Params{Intent: *intent}))})
		}
		fmt.Println(string(jsonIndent(out)))
		return
	}
	fmt.Printf("== fak memory drivers (the composable strategies; each is a query, not a function) ==\n")
	for _, d := range drivers {
		fmt.Printf("\n%-10s %s\n", d.Name, d.Doc)
		p := memq.Explain(d.Build(memq.Params{Intent: *intent}))
		for _, s := range p.Steps {
			tag := "    "
			if s.Mutation {
				tag = "MUT "
			} else if s.Effect {
				tag = "eff "
			}
			fmt.Printf("    %s%-12s %s\n", tag, s.Kind, s.Detail)
		}
	}
	fmt.Printf("\nauthor your own: write a Query JSON and pass it to `fak memory run --query-file PLAN.json`.\n")
}

func cmdMemoryExplain(argv []string) {
	fs := flag.NewFlagSet("memory explain", flag.ExitOnError)
	driver := fs.String("driver", "", "built-in driver name (see `fak memory drivers`)")
	queryFile := fs.String("query-file", "", "path to an authored Query JSON (overrides --driver)")
	intent := fs.String("intent", "the task at hand", "the agent's intent (relevance ranking)")
	k := fs.Int("k", 0, "limit (driver-specific; 0 = driver default)")
	budget := fs.Int64("budget", 0, "byte budget (0 = unbounded)")
	dir := fs.String("dir", "", "recall core image directory (unused by explain; accepted for symmetry)")
	asJSON := fs.Bool("json", false, "emit the plan as JSON")
	_ = fs.Parse(argv)
	*dir = pathutil.ExpandTilde(*dir)
	_ = dir

	q, label := memoryQuery(*driver, *queryFile, *intent, *k, *budget)
	p := memq.Explain(q)
	if *asJSON {
		fmt.Println(string(jsonIndent(p)))
		return
	}
	fmt.Printf("== fak memory explain (%s) — plan only, nothing executed ==\n", label)
	fmt.Print(p.Text())
}

func cmdMemoryRun(argv []string) {
	fs := flag.NewFlagSet("memory run", flag.ExitOnError)
	driver := fs.String("driver", "", "built-in driver name (see `fak memory drivers`)")
	queryFile := fs.String("query-file", "", "path to an authored Query JSON (overrides --driver)")
	intent := fs.String("intent", "the task at hand", "the agent's intent (relevance ranking)")
	k := fs.Int("k", 0, "limit (driver-specific; 0 = driver default)")
	budget := fs.Int64("budget", 0, "byte budget (0 = unbounded)")
	dir := fs.String("dir", "", "recall core image directory (default: the in-memory demo corpus)")
	apply := fs.Bool("apply", false, "APPLY durable mutations (default: propose only — fail-closed)")
	out := fs.String("out", "memory-report.json", "report output path")
	asJSON := fs.Bool("json", false, "emit the full result as JSON to stdout")
	_ = fs.Parse(argv)
	*dir = pathutil.ExpandTilde(*dir)

	q, label := memoryQuery(*driver, *queryFile, *intent, *k, *budget)
	backend, blabel := memoryBackend(*dir)
	caps := memq.Caps{}
	if *apply {
		caps = memq.AllowAll()
	}
	res, err := memq.Run(ctx(), backend, q, caps)
	must(err)

	must(os.WriteFile(*out, jsonIndent(res), 0o644))
	if *asJSON {
		fmt.Println(string(jsonIndent(res)))
		return
	}

	fmt.Printf("== fak memory run (%s over %s) ==\n", label, blabel)
	if *apply {
		fmt.Println("mode: APPLY (durable mutations enabled)")
	} else {
		fmt.Println("mode: dry-run (mutations proposed, not applied; pass --apply to enact)")
	}
	for _, s := range res.Steps {
		note := ""
		if s.Note != "" {
			note = "  — " + s.Note
		}
		fmt.Printf("  [%d] %-12s %d -> %d%s\n", s.Index, s.Kind, s.In, s.Out, note)
	}
	if len(res.Rendered) > 0 {
		fmt.Printf("rendered into context (%d item(s), ~%d tokens):\n", res.Stats.Rendered, res.Stats.EstimatedTokens)
		for _, it := range res.Rendered {
			fmt.Printf("    + %-16s %s\n", it.ID, it.Descriptor)
		}
	}
	for _, e := range res.Effects {
		state := "PROPOSED"
		if e.Applied {
			state = "APPLIED"
		}
		fmt.Printf("effect %-12s [%s] %s\n", e.Kind, state, e.Note)
		if e.Derived != nil {
			fmt.Printf("    derived %s (%s, %d bytes): %s\n", e.Derived.ID, e.Derived.Durability, e.Derived.Bytes, e.Derived.Descriptor)
		}
	}
	for _, rf := range res.Refused {
		fmt.Printf("refused %-16s %s\n", rf.ID, rf.Reason)
	}
	fmt.Printf("stats: scanned=%d selected=%d rendered=%d refused=%d effects(proposed=%d applied=%d)\n",
		res.Stats.CellsScanned, res.Stats.CellsSelected, res.Stats.Rendered, res.Stats.Refused,
		res.Stats.EffectsProposed, res.Stats.EffectsApplied)
	fmt.Printf("report written: %s\n", *out)
}
