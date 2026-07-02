package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/sotacoverage"
)

func cmdSOTACoverageScorecard(argv []string) {
	os.Exit(runSOTACoverageScorecard(os.Stdout, os.Stderr, argv))
}

func runSOTACoverageScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sota-coverage-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	today := fs.String("today", "", "evaluate freshness against this YYYY-MM-DD date")
	check := fs.Bool("check", false, "exit nonzero when there is any HARD sota-debt")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	root := *workspace
	if root == "" {
		root = resolveRoot("")
		if root == "" {
			root = "."
		}
	}
	payload := sotacoverage.Collect(root, *today)
	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "sota-coverage-scorecard: encode json: %v\n", err)
			return 2
		}
	} else {
		fmt.Fprintln(stdout, sotacoverage.Render(payload))
	}
	if *check && !payload.OK {
		return 1
	}
	return 0
}
