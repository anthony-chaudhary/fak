package main

// support.go is the C11 read-out of the support-maturity epic (#1243, #1254): the
// `fak support` verb that folds the scorecard grid + the rung->regime router into ONE
// per-cell line, so a single question -- "where is X and what should I do about it?" --
// has a single answer. Each row is the whole instrument for one (model family x backend)
// cell: rung . regime . target . next-action . the witness behind the rung.
//
// It composes the shipped lowerings, all total + deterministic:
//
//	covmatrix.Grid()                     -- the scorecard's model x backend grid (the cell + support level)
//	supportmaturity.FromSupport          -- C1: the support level -> M0..M7 ladder rung
//	supportmaturity.WitnessFor           -- C2: the non-author witness that PROVES that rung
//	supportmaturity.NextActionFor        -- C7+C9: the rung -> dev-regime -> routed loop + directive
//	supportmaturityscore.DeclaredTargets -- the per-cell declared target rung (#1247)
//
// The fold reads only the committed grid and the ladder/router; the verb's output is
// golden-tested and its --json round-trips. The "target" column is the declared per-cell
// target rung from the scorecard's anti-Goodhart table (#1247).

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/supportmaturityscore"
)

type supportRow = supportmaturityscore.Row

func supportReadOut() []supportRow { return supportmaturityscore.ReadOut() }

func filterSupportRows(rows []supportRow, family, backend string) []supportRow {
	return supportmaturityscore.FilterReadOut(rows, family, backend)
}

func renderSupportReadOut(rows []supportRow) string {
	return supportmaturityscore.RenderReadOut(rows)
}

func cmdSupport(argv []string) {
	os.Exit(runSupport(os.Stdout, os.Stderr, argv))
}

// runSupport is the `fak support` entry point: it folds the per-cell read-out and renders
// it as an ASCII table (default) or round-trippable JSON (--json), optionally narrowed to
// the cell(s) named by --family / --backend (the "where is X" question).
func runSupport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak support", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the per-cell read-out as JSON (round-trips)")
	family := fs.String("family", "", "filter to model families whose name contains this (case-insensitive)")
	backend := fs.String("backend", "", "filter to this backend (case-insensitive exact match)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak support: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	rows := filterSupportRows(supportReadOut(), *family, *backend)

	if *asJSON {
		if err := writeIndentedJSON(stdout, rows); err != nil {
			fmt.Fprintf(stderr, "fak support: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, renderSupportReadOut(rows))
	return 0
}
