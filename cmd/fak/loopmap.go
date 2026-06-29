package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/loopmap"
)

// loopmap.go — the impure shell over internal/loopmap: the `fak loop-map` verb, the
// in-loop affordance that answers "what tool do I reach for RIGHT NOW?" (#1153, a child
// of the 10x dev-experience epic #1148). Where `fak loop-index-scorecard` MEASURES the
// agentic-coding loop, this FINDS the right verb at each stage so the downstream levers
// get used by default instead of by luck.
//
//	fak loop-map                      the whole stage -> verb map, grouped by loop stage
//	fak loop-map --stage verify       just the rows for one stage
//	fak loop-map --ask "<situation>"  match a free-text "what now?" to the best verb
//	fak loop-map --nudge              the "did you verify?" boundary reminder
//	fak loop-map --json               the machine-readable map (for tooling)

func cmdLoopMap(argv []string) {
	fs := flag.NewFlagSet("fak loop-map", flag.ContinueOnError)
	stage := fs.String("stage", "", "show only the rows for one loop stage (orient|plan|act|verify|ship|learn)")
	ask := fs.String("ask", "", "match a free-text \"what tool do I reach for now?\" situation to the best verb")
	nudge := fs.Bool("nudge", false, "print the \"did you verify?\" verify/ship-boundary reminder")
	asJSON := fs.Bool("json", false, "emit the machine-readable map JSON")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	switch {
	case *nudge:
		fmt.Println(loopmap.VerifyNudge())
	case *ask != "":
		e, ok := loopmap.Ask(*ask)
		if !ok {
			fmt.Printf("no stage matched %q — run `fak loop-map` to see the whole loop-stage -> tool map.\n", *ask)
			os.Exit(1)
		}
		if *asJSON {
			_ = writeIndentedJSONNoEscape(os.Stdout, e)
			return
		}
		fmt.Printf("[%s] %s\n  -> %s (%s)\n  why: %s\n", e.Stage, e.Situation, e.Verb, e.Kind, e.Why)
	case *asJSON:
		rows := loopmap.Map()
		if *stage != "" {
			rows = loopmap.ForStage(*stage)
		}
		_ = writeIndentedJSONNoEscape(os.Stdout, rows)
	case *stage != "":
		rows := loopmap.ForStage(*stage)
		if len(rows) == 0 {
			fmt.Printf("unknown loop stage %q — stages are: %s\n", *stage, strings.Join(loopmap.Stages(), ", "))
			os.Exit(2)
		}
		printLoopMapStage(os.Stdout, *stage, rows)
	default:
		fmt.Fprintln(os.Stdout, "loop-stage -> tool map — the right verb, one ask away at every stage (#1153)")
		fmt.Fprintln(os.Stdout, "  loop: orient -> plan -> act -> verify -> ship -> learn")
		fmt.Fprintln(os.Stdout)
		for _, st := range loopmap.Stages() {
			printLoopMapStage(os.Stdout, st, loopmap.ForStage(st))
		}
		fmt.Fprintf(os.Stdout, "nudge: %s\n", loopmap.VerifyNudge())
	}
}

// printLoopMapStage renders one stage's rows as a compact, copy-pasteable block.
func printLoopMapStage(w io.Writer, stage string, rows []loopmap.Entry) {
	for _, e := range rows {
		fmt.Fprintf(w, "[%s] when: %s\n", stage, e.Situation)
		fmt.Fprintf(w, "        reach for: %s (%s)\n", e.Verb, e.Kind)
		fmt.Fprintf(w, "        why: %s\n", e.Why)
	}
	fmt.Fprintln(w)
}
