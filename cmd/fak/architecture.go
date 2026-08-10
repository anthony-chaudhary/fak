package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/archreport"
)

func cmdArchitecture(argv []string) { os.Exit(runArchitecture(os.Stdout, os.Stderr, argv)) }

func runArchitecture(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("architecture", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("workspace", "", "workspace root (defaults to current directory)")
	baseline := fs.String("baseline-workspace", "", "compare a baseline workspace to --workspace/current")
	leaf := fs.String("leaf", "", "report one internal leaf")
	jsonOut := fs.Bool("json", false, "emit fak-architecture/1 JSON")
	usage := fs.Bool("usage", false, "fold architecture invocations by ISO week")
	failOn := fs.String("fail-on", "", "comparison gate: introduced-violations")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak architecture: pass no positional arguments")
		return 2
	}
	if *failOn != "" && *failOn != "introduced-violations" {
		fmt.Fprintf(stderr, "fak architecture: invalid --fail-on %q (want introduced-violations)\n", *failOn)
		return 2
	}
	if *failOn != "" && *baseline == "" {
		fmt.Fprintln(stderr, "fak architecture: --fail-on requires --baseline-workspace")
		return 2
	}
	if *usage && *baseline != "" {
		fmt.Fprintln(stderr, "fak architecture: --usage cannot be combined with --baseline-workspace")
		return 2
	}
	if *leaf != "" && *baseline != "" {
		fmt.Fprintln(stderr, "fak architecture: --leaf cannot be combined with --baseline-workspace")
		return 2
	}
	if *usage {
		if *leaf != "" {
			fmt.Fprintln(stderr, "fak architecture: --usage cannot be combined with --leaf")
			return 2
		}
		return runArchitectureUsage(stdout, stderr, *jsonOut)
	}
	if *root == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak architecture: %v\n", err)
			return 1
		}
		*root = cwd
	}
	if *baseline != "" {
		before, err := archreport.Analyze(*baseline, "")
		if err != nil {
			fmt.Fprintf(stderr, "fak architecture: baseline: %v\n", err)
			return 1
		}
		after, err := archreport.Analyze(*root, "")
		if err != nil {
			fmt.Fprintf(stderr, "fak architecture: workspace: %v\n", err)
			return 1
		}
		return writeArchitectureDiff(stdout, stderr, archreport.Diff(before, after), *jsonOut, *failOn)
	}
	usagePath, usagePathErr := archreport.UsagePath()
	mode, format := "full", "text"
	if *leaf != "" {
		mode = "scoped"
	}
	if *jsonOut {
		format = "json"
	}
	report, err := archreport.Analyze(*root, *leaf)
	if err != nil {
		recordArchitectureUsage(stderr, usagePath, usagePathErr, archreport.Usage{At: time.Now().UTC().Format(time.RFC3339), Mode: mode, Format: format, Outcome: "error"})
		fmt.Fprintf(stderr, "fak architecture: %v\n", err)
		return 1
	}
	recordArchitectureUsage(stderr, usagePath, usagePathErr, archreport.Usage{At: time.Now().UTC().Format(time.RFC3339), Mode: mode, Format: format, Outcome: "ok", Diagnostics: len(report.Diagnostics), Violations: report.Violations})
	if *jsonOut {
		raw, err := report.JSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak architecture: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
		return 0
	}
	fmt.Fprintf(stdout, "architecture: %d leaves, %d upward violation(s)\n", sumArchitectureLeaves(report), report.Violations)
	for _, t := range report.Tiers {
		fmt.Fprintf(stdout, "  tier %d %-22s %d\n", t.Level, t.Name, t.Leaves)
	}
	for _, diagnostic := range report.Diagnostics {
		fmt.Fprintf(stdout, "  diagnostic %-24s leaf=%s: %s; recovery: %s\n", diagnostic.Kind, diagnostic.Leaf, diagnostic.Message, diagnostic.Recovery)
	}
	if *leaf == "" && len(report.Hotspots) > 0 {
		fmt.Fprintln(stdout, "  hotspots (direct fan-in):")
		for _, hotspot := range report.Hotspots {
			fmt.Fprintf(stdout, "    %-22s %d\n", hotspot.Name, hotspot.FanIn)
		}
	}
	for _, l := range report.Leaves {
		if *leaf != "" || len(l.Violations) > 0 {
			fmt.Fprintf(stdout, "  %-24s declared=%s(%d) floor=%s(%d) deps=%v dependents=%v", l.Name, l.DeclaredTierName, l.DeclaredTier, l.ImportFloorName, l.ImportFloor, l.Dependencies, l.Dependents)
			if len(l.Violations) > 0 {
				fmt.Fprintf(stdout, " violations=%v", l.Violations)
			}
			fmt.Fprintln(stdout)
		}
	}
	return 0
}
func sumArchitectureLeaves(r archreport.Report) int {
	n := 0
	for _, t := range r.Tiers {
		n += t.Leaves
	}
	return n
}

func recordArchitectureUsage(stderr io.Writer, path string, pathErr error, row archreport.Usage) {
	if pathErr != nil {
		fmt.Fprintf(stderr, "fak architecture: usage ledger warning: %v\n", pathErr)
		return
	}
	if err := archreport.AppendUsage(path, row); err != nil {
		fmt.Fprintf(stderr, "fak architecture: usage ledger warning: %v\n", err)
	}
}

func runArchitectureUsage(stdout, stderr io.Writer, jsonOut bool) int {
	path, err := archreport.UsagePath()
	if err != nil {
		fmt.Fprintf(stderr, "fak architecture: %v\n", err)
		return 1
	}
	weeks, err := archreport.FoldUsage(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak architecture: %v\n", err)
		return 1
	}
	if jsonOut {
		raw, err := json.MarshalIndent(struct {
			Schema string                 `json:"schema"`
			Weeks  []archreport.UsageWeek `json:"weeks"`
		}{Schema: "fak-architecture-usage-summary/1", Weeks: weeks}, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak architecture: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
		return 0
	}
	if len(weeks) == 0 {
		fmt.Fprintln(stdout, "architecture usage: no recorded invocations")
		return 0
	}
	for _, week := range weeks {
		fmt.Fprintf(stdout, "%s invocations=%d full=%d scoped=%d text=%d json=%d ok=%d error=%d\n", week.Week, week.Invocations, week.Full, week.Scoped, week.Text, week.JSON, week.OK, week.Error)
	}
	return 0
}

func writeArchitectureDiff(stdout, stderr io.Writer, diff archreport.ReportDiff, jsonOut bool, failOn string) int {
	if jsonOut {
		raw, err := diff.JSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak architecture: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
		if failOn == "introduced-violations" && diff.Verdict == "regression" {
			return 3
		}
		return 0
	}
	fmt.Fprintf(stdout, "architecture diff: %d change(s), verdict=%s\n", diff.Changes(), diff.Verdict)
	for _, leaf := range diff.AddedLeaves {
		fmt.Fprintf(stdout, "  + leaf %s\n", leaf)
	}
	for _, leaf := range diff.RemovedLeaves {
		fmt.Fprintf(stdout, "  - leaf %s\n", leaf)
	}
	for _, change := range diff.TierChanges {
		fmt.Fprintf(stdout, "  ~ tier %s %s(%d) -> %s(%d)\n", change.Leaf, change.BeforeName, change.Before, change.AfterName, change.After)
	}
	for _, edge := range diff.AddedEdges {
		fmt.Fprintf(stdout, "  + edge %s -> %s\n", edge.From, edge.To)
	}
	for _, edge := range diff.RemovedEdges {
		fmt.Fprintf(stdout, "  - edge %s -> %s\n", edge.From, edge.To)
	}
	for _, edge := range diff.IntroducedViolations {
		fmt.Fprintf(stdout, "  ! introduced violation %s\n", edge)
	}
	for _, edge := range diff.ResolvedViolations {
		fmt.Fprintf(stdout, "  resolved violation %s\n", edge)
	}
	if failOn == "introduced-violations" && diff.Verdict == "regression" {
		fmt.Fprintln(stdout, "  remediation: remove/invert introduced upward edges or move the shared seam down; comparison is baseline -> workspace")
		return 3
	}
	return 0
}
