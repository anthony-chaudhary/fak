package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
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
	failOn := fs.String("fail-on", "", "comparison gate: introduced-violations|introduced-diagnostics|increased-tier-gap|increased-violation-distance|increased-blast-radius|introduced-blast-impacts|increased-blast-path-length")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak architecture: pass no positional arguments")
		return 2
	}
	if *failOn != "" && *failOn != "introduced-violations" && *failOn != "introduced-diagnostics" && *failOn != "increased-tier-gap" && *failOn != "increased-violation-distance" && *failOn != "increased-blast-radius" && *failOn != "introduced-blast-impacts" && *failOn != "increased-blast-path-length" {
		fmt.Fprintf(stderr, "fak architecture: invalid --fail-on %q (want introduced-violations, introduced-diagnostics, increased-tier-gap, increased-violation-distance, increased-blast-radius, introduced-blast-impacts, or increased-blast-path-length)\n", *failOn)
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
	fmt.Fprintf(stdout, "architecture: %d leaves, %d upward violation(s), max tier distance %d\n", sumArchitectureLeaves(report), report.Violations, report.MaxViolationDistance)
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
	if *leaf == "" && len(report.SinkCandidates) > 0 {
		fmt.Fprintln(stdout, "  sink candidates (declared tier above import floor):")
		for _, candidate := range report.SinkCandidates {
			fmt.Fprintf(stdout, "    %-22s declared=%s(%d) floor=%s(%d) gap=%d\n", candidate.Name, candidate.DeclaredTierName, candidate.DeclaredTier, candidate.ImportFloorName, candidate.ImportFloor, candidate.TierGap)
		}
	}
	for _, l := range report.Leaves {
		if *leaf != "" || len(l.ViolationEdges) > 0 {
			fmt.Fprintf(stdout, "  %-24s declared=%s(%d) floor=%s(%d) gap=%d deps=%v dependents=%v blast-radius=%d", l.Name, l.DeclaredTierName, l.DeclaredTier, l.ImportFloorName, l.ImportFloor, l.TierGap, l.Dependencies, l.Dependents, l.BlastRadius)
			if len(l.ViolationEdges) > 0 {
				fmt.Fprint(stdout, " violations=[")
				for i, edge := range l.ViolationEdges {
					if i > 0 {
						fmt.Fprint(stdout, ", ")
					}
					fmt.Fprintf(stdout, "%s(%s) -> %s(%s), distance=%d", edge.From, edge.FromTierName, edge.To, edge.ToTierName, edge.TierDistance)
				}
				fmt.Fprint(stdout, "]")
			}
			fmt.Fprintln(stdout)
			if *leaf != "" && len(l.BlastPaths) > 0 {
				fmt.Fprintln(stdout, "    blast paths:")
				for _, path := range l.BlastPaths {
					fmt.Fprintf(stdout, "      %s: %s\n", path.Dependent, strings.Join(path.Path, " -> "))
				}
			}
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
		if architectureFailOnMatched(diff, failOn) {
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
	for _, change := range diff.TierGapChanges {
		fmt.Fprintf(stdout, "  ~ tier-gap %s floor %d -> %d, gap %d -> %d (%+d)\n", change.Leaf, change.BeforeFloor, change.AfterFloor, change.BeforeGap, change.AfterGap, change.Delta)
	}
	for _, change := range diff.FanInChanges {
		fmt.Fprintf(stdout, "  ~ fan-in %s %d -> %d (%+d)\n", change.Leaf, change.Before, change.After, change.Delta)
	}
	for _, change := range diff.BlastRadiusChanges {
		fmt.Fprintf(stdout, "  ~ blast-radius %s %d -> %d (%+d)\n", change.Leaf, change.Before, change.After, change.Delta)
	}
	for _, impact := range diff.IntroducedBlastImpacts {
		fmt.Fprintf(stdout, "  ! introduced blast-impact %s -> %s path=%s\n", impact.Source, impact.Dependent, strings.Join(impact.Path, " -> "))
	}
	for _, impact := range diff.ResolvedBlastImpacts {
		fmt.Fprintf(stdout, "  resolved blast-impact %s -> %s path=%s\n", impact.Source, impact.Dependent, strings.Join(impact.Path, " -> "))
	}
	for _, change := range diff.BlastPathChanges {
		fmt.Fprintf(stdout, "  ~ blast-path %s -> %s hops %d -> %d (%+d) before=%s after=%s\n", change.Source, change.Dependent, change.BeforeHops, change.AfterHops, change.Delta, strings.Join(change.BeforePath, " -> "), strings.Join(change.AfterPath, " -> "))
	}
	for _, edge := range diff.IntroducedViolationEdges {
		fmt.Fprintf(stdout, "  ! introduced violation %s(%s) -> %s(%s), distance=%d\n", edge.From, edge.FromTierName, edge.To, edge.ToTierName, edge.TierDistance)
	}
	for _, edge := range diff.ResolvedViolationEdges {
		fmt.Fprintf(stdout, "  resolved violation %s(%s) -> %s(%s), distance=%d\n", edge.From, edge.FromTierName, edge.To, edge.ToTierName, edge.TierDistance)
	}
	for _, change := range diff.ViolationDistanceChanges {
		fmt.Fprintf(stdout, "  ~ violation-distance %s -> %s %d -> %d (%+d)\n", change.From, change.To, change.BeforeDistance, change.AfterDistance, change.Delta)
	}
	for _, diagnostic := range diff.IntroducedDiagnostics {
		fmt.Fprintf(stdout, "  ! introduced diagnostic %s leaf=%s: %s; recovery: %s\n", diagnostic.Kind, diagnostic.Leaf, diagnostic.Message, diagnostic.Recovery)
	}
	for _, diagnostic := range diff.ResolvedDiagnostics {
		fmt.Fprintf(stdout, "  resolved diagnostic %s leaf=%s\n", diagnostic.Kind, diagnostic.Leaf)
	}
	if architectureFailOnMatched(diff, failOn) {
		if failOn == "introduced-violations" {
			fmt.Fprintln(stdout, "  remediation: remove/invert introduced upward edges or move the shared seam down; comparison is baseline -> workspace")
		} else if failOn == "introduced-diagnostics" {
			fmt.Fprintln(stdout, "  remediation: apply each introduced diagnostic recovery action; comparison is baseline -> workspace")
		} else if failOn == "increased-tier-gap" {
			fmt.Fprintln(stdout, "  remediation: restore the prior import floor or lower the over-declared tier; comparison is baseline -> workspace")
		} else if failOn == "increased-blast-radius" {
			fmt.Fprintln(stdout, "  remediation: remove/invert the new dependency path or move the shared seam down; comparison is baseline -> workspace")
		} else if failOn == "introduced-blast-impacts" {
			fmt.Fprintln(stdout, "  remediation: remove/invert each introduced path or move the shared seam down; comparison is baseline -> workspace")
		} else if failOn == "increased-blast-path-length" {
			fmt.Fprintln(stdout, "  remediation: restore the shorter dependency path or remove/invert the added intermediary edges; comparison is baseline -> workspace")
		} else {
			fmt.Fprintln(stdout, "  remediation: restore the prior endpoint tiers or remove/invert the upward edge; comparison is baseline -> workspace")
		}
		return 3
	}
	return 0
}

func architectureFailOnMatched(diff archreport.ReportDiff, failOn string) bool {
	switch failOn {
	case "introduced-violations":
		return len(diff.IntroducedViolationEdges) > 0
	case "introduced-diagnostics":
		return len(diff.IntroducedDiagnostics) > 0
	case "increased-tier-gap":
		for _, change := range diff.TierGapChanges {
			if change.Delta > 0 {
				return true
			}
		}
		return false
	case "increased-violation-distance":
		for _, change := range diff.ViolationDistanceChanges {
			if change.Delta > 0 {
				return true
			}
		}
		return false
	case "increased-blast-radius":
		for _, change := range diff.BlastRadiusChanges {
			if change.Delta > 0 {
				return true
			}
		}
		return false
	case "introduced-blast-impacts":
		return len(diff.IntroducedBlastImpacts) > 0
	case "increased-blast-path-length":
		for _, change := range diff.BlastPathChanges {
			if change.Delta > 0 {
				return true
			}
		}
		return false
	default:
		return false
	}
}
