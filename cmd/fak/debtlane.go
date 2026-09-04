package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/debtlane"
)

func cmdDebtLanes(argv []string) {
	os.Exit(runDebtLanes(os.Stdout, os.Stderr, argv))
}

func cmdDebtLanesScorecard(argv []string) {
	os.Exit(runDebtLanes(os.Stdout, os.Stderr, argv))
}

func runMaturityDebtLanes(stdout, stderr io.Writer, argv []string) int {
	return runDebtLanes(stdout, stderr, argv)
}

func runDebtLanes(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak debt-lanes", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	check := fs.Bool("check", false, "gate mode: exit non-zero if active maturity debt exists")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	laneFilter := fs.String("lane", "", "filter to a specific lane name")
	criticality := fs.String("criticality", "", "filter by criticality (core, enabling, stewardship, peripheral)")
	minGap := fs.Float64("min-gap", 0, "filter lanes with maturity gap >= min-gap")
	topN := fs.Int("top", 10, "number of top debt hotspots to display")

	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak debt-lanes: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	report, err := debtlane.Scan(debtlane.Options{
		WorkspaceRoot:     root,
		LaneFilter:        *laneFilter,
		CriticalityFilter: *criticality,
		MinGap:            *minGap,
		TopN:              *topN,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak debt-lanes: %v\n", err)
		return 2
	}

	if *comparePath != "" {
		baseBytes, err := os.ReadFile(*comparePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak debt-lanes: read compare baseline: %v\n", err)
			return 2
		}
		var base debtlane.Report
		if err := json.Unmarshal(baseBytes, &base); err != nil {
			fmt.Fprintf(stderr, "fak debt-lanes: decode compare baseline JSON: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, debtlane.Compare(report, base))
		if *check && !report.OK {
			return 1
		}
		return 0
	}

	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak debt-lanes: encode json: %v\n", err)
			return 1
		}
	case *asMarkdown:
		fmt.Fprint(stdout, debtlane.Markdown(report))
	default:
		fmt.Fprint(stdout, debtlane.Render(report))
	}

	if *check && !report.OK {
		return 1
	}
	return 0
}
