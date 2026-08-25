package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/focusscore"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func cmdFocusScore(argv []string) { os.Exit(runFocusScore(os.Stdout, os.Stderr, argv)) }

// runFocusScore scores whether the fleet is CONVERGING on its live goal or fanning out
// too broad — bounded work-in-progress and every open objective's witnessed progress
// curve rising — re-derived from the trajectory-control ledger fak writes. Exit codes:
// 0 converging (no debt), 1 carries focus debt, 2 usage/IO error.
func runFocusScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak focus-score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	ledger := fs.String("ledger", "", "trajctl ledger path (default: <root>/"+trajctl.DefaultLedgerRel+")")
	wipCap := fs.Int("wip-cap", 0, "active-objective WIP cap (default: focusscore.DefaultWIPCap)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak focus-score: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	// An empty --ledger lets focusscore.Options derive the default from Root, the same
	// <root>/docs/nightrun/trajctl.jsonl `fak trajctl` writes to.
	payload := focusscore.Build(focusscore.Options{
		Root:       root,
		LedgerPath: *ledger,
		WIPCap:     *wipCap,
	})
	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak focus-score", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, focusscore.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	if code := emitJSONOrRender(stdout, stderr, "fak focus-score", *asJSON, payload, func(w io.Writer) {
		if *asMarkdown {
			fmt.Fprint(w, focusscore.Markdown(payload))
		} else {
			fmt.Fprintln(w, focusscore.Render(payload))
		}
	}); code != 0 {
		return code
	}
	if payload.OK {
		return 0
	}
	return 1
}
