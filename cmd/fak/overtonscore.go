package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/overtonscore"
)

func cmdOvertonScore(argv []string) {
	os.Exit(runOvertonScore(os.Stdout, os.Stderr, argv))
}

func runOvertonScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score overton", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit JSON report")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak score overton: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	report := overtonscore.Build(root)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak score overton: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "Overton Baseline Evaluation: %s (score %.1f, points %d, pressure %.3f, slack %.3f)\n",
		report.Grade, report.Score, report.OvertonPoints, report.Pressure, report.Slack)
	fmt.Fprintf(stdout, "Dispositions: orthodox_clean=%d, accepted_temporary=%d, accidental_unaccepted=%d\n",
		report.Dispositions[string(overtonscore.DispositionOrthodoxClean)],
		report.Dispositions[string(overtonscore.DispositionAcceptedTemporary)],
		report.Dispositions[string(overtonscore.DispositionAccidentalUnaccepted)])
	for _, ev := range report.Evaluations {
		fmt.Fprintf(stdout, "  [%-9s] %-20s = %8.3f (%s, points %d)\n",
			ev.Subsystem, ev.Metric, ev.Observed, ev.Disposition, ev.Points)
	}
	return 0
}
