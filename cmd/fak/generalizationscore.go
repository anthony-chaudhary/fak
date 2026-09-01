package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/generalizationdebt"
)

func cmdGeneralizationScorecard(argv []string) {
	os.Exit(runGeneralizationScorecard(os.Stdout, os.Stderr, argv))
}

func runGeneralizationScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score generalization", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit deterministic JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak score generalization: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	report, err := generalizationdebt.Scan(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak score generalization: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak score generalization: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "generalization debt: %d finding(s), %d point(s) (%d accidental, %d accepted temporary)\n",
		report.Totals.Findings, report.Totals.DebtPoints, report.Totals.AccidentalUnaccepted, report.Totals.AcceptedTemporary)
	for _, f := range report.Findings {
		fmt.Fprintf(stdout, "%s:%d %s %s [%s; interest=%s/%s; points=%d]\n", f.Path, f.Line, f.Kind, f.Subject, f.Disposition, f.Interest.Band, f.Interest.Rate, f.DebtPoints)
	}
	return 0
}
