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

func formatArchitectureSources(sources []archreport.SourceImport) string {
	parts := make([]string, 0, len(sources))
	for _, source := range sources {
		parts = append(parts, fmt.Sprintf("%s:%d:%d", source.Path, source.Line, source.Column))
	}
	return strings.Join(parts, ",")
}

func cmdArchitecture(argv []string) { os.Exit(runArchitecture(os.Stdout, os.Stderr, argv)) }

func runArchitecture(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("architecture", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("workspace", "", "workspace root (defaults to current directory)")
	baseline := fs.String("baseline-workspace", "", "compare a baseline workspace to --workspace/current")
	leaf := fs.String("leaf", "", "report one internal leaf")
	jsonOut := fs.Bool("json", false, "emit fak-architecture/1 JSON")
	usage := fs.Bool("usage", false, "fold architecture invocations by ISO week")
	failOn := fs.String("fail-on", "", "comparison gate: introduced-violations|introduced-diagnostics|increased-tier-gap|increased-violation-distance|introduced-or-increased-rootward-layer-skips|increased-fan-out|increased-dependency-reach|increased-dependency-depth|increased-blast-radius|introduced-blast-impacts|increased-blast-path-length|introduced-lateral-edges|introduced-lateral-couplings|introduced-or-increased-lateral-bridges|introduced-or-increased-lateral-articulation-points|resolved-lateral-resilient-pairs|decreased-lateral-edge-connectivity|decreased-lateral-vertex-connectivity|decreased-lateral-vertex-pair-cuts")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak architecture: pass no positional arguments")
		return 2
	}
	if *failOn != "" && *failOn != "introduced-violations" && *failOn != "introduced-diagnostics" && *failOn != "increased-tier-gap" && *failOn != "increased-violation-distance" && *failOn != "introduced-or-increased-rootward-layer-skips" && *failOn != "increased-fan-out" && *failOn != "increased-dependency-reach" && *failOn != "increased-dependency-depth" && *failOn != "increased-blast-radius" && *failOn != "introduced-blast-impacts" && *failOn != "increased-blast-path-length" && *failOn != "introduced-lateral-edges" && *failOn != "introduced-lateral-couplings" && *failOn != "introduced-or-increased-lateral-bridges" && *failOn != "introduced-or-increased-lateral-articulation-points" && *failOn != "resolved-lateral-resilient-pairs" && *failOn != "decreased-lateral-edge-connectivity" && *failOn != "decreased-lateral-vertex-connectivity" && *failOn != "decreased-lateral-vertex-pair-cuts" {
		fmt.Fprintf(stderr, "fak architecture: invalid --fail-on %q (want introduced-violations, introduced-diagnostics, increased-tier-gap, increased-violation-distance, introduced-or-increased-rootward-layer-skips, increased-fan-out, increased-dependency-reach, increased-dependency-depth, increased-blast-radius, introduced-blast-impacts, increased-blast-path-length, introduced-lateral-edges, introduced-lateral-couplings, introduced-or-increased-lateral-bridges, introduced-or-increased-lateral-articulation-points, resolved-lateral-resilient-pairs, decreased-lateral-edge-connectivity, decreased-lateral-vertex-connectivity, or decreased-lateral-vertex-pair-cuts)\n", *failOn)
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
	renderArchitectureReport(stdout, report, *leaf)
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
	for _, edge := range diff.IntroducedTypedEdges {
		fmt.Fprintf(stdout, "  + typed-edge %s(%s) -> %s(%s) delta=%+d direction=%s\n", edge.From, edge.FromTierName, edge.To, edge.ToTierName, edge.TierDelta, edge.Direction)
	}
	for _, edge := range diff.ResolvedTypedEdges {
		fmt.Fprintf(stdout, "  - typed-edge %s(%s) -> %s(%s) delta=%+d direction=%s\n", edge.From, edge.FromTierName, edge.To, edge.ToTierName, edge.TierDelta, edge.Direction)
	}
	for _, coupling := range diff.IntroducedLateralCouplings {
		fmt.Fprintf(stdout, "  ! introduced lateral-coupling tier=%s(%d) %s <-> %s\n", coupling.TierName, coupling.Tier, coupling.Left, coupling.Right)
	}
	for _, coupling := range diff.ResolvedLateralCouplings {
		fmt.Fprintf(stdout, "  resolved lateral-coupling tier=%s(%d) %s <-> %s\n", coupling.TierName, coupling.Tier, coupling.Left, coupling.Right)
	}
	for _, bridge := range diff.IntroducedLateralBridges {
		fmt.Fprintf(stdout, "  ! introduced lateral-bridge tier=%s(%d) %s--%s coupling-pairs=%d\n", bridge.TierName, bridge.Tier, bridge.Left, bridge.Right, bridge.CouplingPairs)
	}
	for _, bridge := range diff.ResolvedLateralBridges {
		fmt.Fprintf(stdout, "  resolved lateral-bridge tier=%s(%d) %s--%s coupling-pairs=%d\n", bridge.TierName, bridge.Tier, bridge.Left, bridge.Right, bridge.CouplingPairs)
	}
	for _, change := range diff.LateralBridgeChanges {
		fmt.Fprintf(stdout, "  ~ lateral-bridge %s--%s coupling-pairs %d -> %d (%+d)\n", change.Left, change.Right, change.BeforeCouplingPairs, change.AfterCouplingPairs, change.Delta)
	}
	for _, point := range diff.IntroducedLateralArticulationPoints {
		fmt.Fprintf(stdout, "  ! introduced lateral-articulation-point %s tier=%s(%d) fragments=%d coupling-pairs=%d\n", point.Name, point.TierName, point.Tier, point.FragmentCount, point.CouplingPairs)
	}
	for _, point := range diff.ResolvedLateralArticulationPoints {
		fmt.Fprintf(stdout, "  resolved lateral-articulation-point %s tier=%s(%d) fragments=%d coupling-pairs=%d\n", point.Name, point.TierName, point.Tier, point.FragmentCount, point.CouplingPairs)
	}
	for _, change := range diff.LateralArticulationPointChanges {
		fmt.Fprintf(stdout, "  ~ lateral-articulation-point %s fragments %d -> %d coupling-pairs %d -> %d (%+d)\n", change.Name, change.BeforeFragmentCount, change.AfterFragmentCount, change.BeforeCouplingPairs, change.AfterCouplingPairs, change.Delta)
	}
	for _, pair := range diff.IntroducedLateralResilientPairs {
		fmt.Fprintf(stdout, "  + lateral-resilient-pair tier=%s(%d) %s <=> %s\n", pair.TierName, pair.Tier, pair.Left, pair.Right)
	}
	for _, pair := range diff.ResolvedLateralResilientPairs {
		fmt.Fprintf(stdout, "  ! resolved lateral-resilient-pair tier=%s(%d) %s <=> %s\n", pair.TierName, pair.Tier, pair.Left, pair.Right)
	}
	for _, change := range diff.LateralEdgeConnectivityChanges {
		fmt.Fprintf(stdout, "  ~ lateral-edge-connectivity tier=%s(%d) %s <=> %s cut %d -> %d (%+d) witnesses %s -> %s partitions %v|%v -> %v|%v\n", change.TierName, change.Tier, change.Left, change.Right, change.BeforeCut, change.AfterCut, change.Delta, architectureCutEdges(change.BeforeCutEdges), architectureCutEdges(change.AfterCutEdges), change.BeforeSourceSide, change.BeforeSinkSide, change.AfterSourceSide, change.AfterSinkSide)
	}
	for _, change := range diff.LateralVertexConnectivityChanges {
		fmt.Fprintf(stdout, "  ~ lateral-vertex-connectivity tier=%s(%d) members=%v cut %d -> %d (%+d) separators %v -> %v\n", change.TierName, change.Tier, change.Members, change.BeforeCut, change.AfterCut, change.Delta, change.BeforeSeparator, change.AfterSeparator)
	}
	for _, change := range diff.LateralVertexPairCutChanges {
		fmt.Fprintf(stdout, "  ~ lateral-vertex-pair-cut tier=%s(%d) members=%v pair=%s--%s cut %d -> %d (%+d) separators %v -> %v\n", change.TierName, change.Tier, change.Members, change.Left, change.Right, change.BeforeCut, change.AfterCut, change.Delta, change.BeforeSeparator, change.AfterSeparator)
	}
	for _, change := range diff.TierGapChanges {
		fmt.Fprintf(stdout, "  ~ tier-gap %s floor %d -> %d, gap %d -> %d (%+d)\n", change.Leaf, change.BeforeFloor, change.AfterFloor, change.BeforeGap, change.AfterGap, change.Delta)
	}
	for _, change := range diff.FanInChanges {
		fmt.Fprintf(stdout, "  ~ fan-in %s %d -> %d (%+d)\n", change.Leaf, change.Before, change.After, change.Delta)
	}
	for _, skip := range diff.IntroducedRootwardLayerSkips {
		fmt.Fprintf(stdout, "  + rootward-layer-skip %s(%s:%d) -> %s(%s:%d) distance=%d skipped-tiers=%d\n", skip.From, skip.FromTierName, skip.FromTier, skip.To, skip.ToTierName, skip.ToTier, skip.TierDistance, skip.SkippedTiers)
	}
	for _, skip := range diff.ResolvedRootwardLayerSkips {
		fmt.Fprintf(stdout, "  - rootward-layer-skip %s -> %s distance=%d skipped-tiers=%d\n", skip.From, skip.To, skip.TierDistance, skip.SkippedTiers)
	}
	for _, change := range diff.RootwardLayerSkipChanges {
		fmt.Fprintf(stdout, "  ~ rootward-layer-skip %s(%s:%d) -> %s(%s:%d) distance %d -> %d skipped-tiers %d -> %d (%+d)\n", change.From, change.FromTierName, change.FromTier, change.To, change.ToTierName, change.ToTier, change.BeforeTierDistance, change.AfterTierDistance, change.BeforeSkippedTiers, change.AfterSkippedTiers, change.Delta)
	}
	for _, change := range diff.FanOutChanges {
		fmt.Fprintf(stdout, "  ~ fan-out %s %d -> %d (%+d)\n", change.Leaf, change.Before, change.After, change.Delta)
	}
	for _, change := range diff.DependencyReachChanges {
		fmt.Fprintf(stdout, "  ~ dependency-reach %s %d -> %d (%+d)\n", change.Leaf, change.Before, change.After, change.Delta)
	}
	for _, change := range diff.DependencyDepthChanges {
		fmt.Fprintf(stdout, "  ~ dependency-depth %s %d -> %d (%+d)\n", change.Leaf, change.Before, change.After, change.Delta)
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
		} else if failOn == "introduced-or-increased-rootward-layer-skips" {
			fmt.Fprintln(stdout, "  remediation: route through an intermediate tier or move the shared abstraction downward; comparison is baseline -> workspace")
		} else if failOn == "increased-fan-out" {
			fmt.Fprintln(stdout, "  remediation: consolidate direct dependencies or move the shared seam downward; comparison is baseline -> workspace")
		} else if failOn == "increased-dependency-reach" {
			fmt.Fprintln(stdout, "  remediation: remove the new transitive dependency path or move the shared seam downward to reduce footprint; comparison is baseline -> workspace")
		} else if failOn == "increased-dependency-depth" {
			fmt.Fprintln(stdout, "  remediation: shorten the dependency chain by removing an intermediary or moving the shared seam downward; comparison is baseline -> workspace")
		} else if failOn == "increased-blast-radius" {
			fmt.Fprintln(stdout, "  remediation: remove/invert the new dependency path or move the shared seam down; comparison is baseline -> workspace")
		} else if failOn == "introduced-blast-impacts" {
			fmt.Fprintln(stdout, "  remediation: remove/invert each introduced path or move the shared seam down; comparison is baseline -> workspace")
		} else if failOn == "increased-blast-path-length" {
			fmt.Fprintln(stdout, "  remediation: restore the shorter dependency path or remove/invert the added intermediary edges; comparison is baseline -> workspace")
		} else if failOn == "introduced-lateral-edges" {
			fmt.Fprintln(stdout, "  remediation: move the shared seam to a lower tier or extract a rootward dependency; comparison is baseline -> workspace")
		} else if failOn == "introduced-lateral-couplings" {
			fmt.Fprintln(stdout, "  remediation: remove the lateral bridge that joined the pair or extract their shared seam downward; comparison is baseline -> workspace")
		} else if failOn == "introduced-or-increased-lateral-bridges" {
			fmt.Fprintln(stdout, "  remediation: remove the articulation bridge or extract the shared seam downward to reduce induced coupling; comparison is baseline -> workspace")
		} else if failOn == "introduced-or-increased-lateral-articulation-points" {
			fmt.Fprintln(stdout, "  remediation: remove the package convergence seam or extract its shared responsibility downward; comparison is baseline -> workspace")
		} else if failOn == "resolved-lateral-resilient-pairs" {
			fmt.Fprintln(stdout, "  remediation: restore a lateral cycle or add a redundant same-tier path between the affected packages; comparison is baseline -> workspace")
		} else if failOn == "decreased-lateral-edge-connectivity" {
			fmt.Fprintln(stdout, "  remediation: restore or add an edge-disjoint same-tier path between each degraded pair; comparison is baseline -> workspace")
		} else if failOn == "decreased-lateral-vertex-connectivity" {
			fmt.Fprintln(stdout, "  remediation: diversify same-tier package paths so no critical separator remains a package-failure bottleneck; comparison is baseline -> workspace")
		} else if failOn == "decreased-lateral-vertex-pair-cuts" {
			fmt.Fprintln(stdout, "  remediation: add an internally vertex-disjoint same-tier package path around each degraded pair separator; comparison is baseline -> workspace")
		} else {
			fmt.Fprintln(stdout, "  remediation: restore the prior endpoint tiers or remove/invert the upward edge; comparison is baseline -> workspace")
		}
		return 3
	}
	return 0
}

func architectureCutEdges(edges []archreport.LateralCutEdge) []string {
	out := make([]string, 0, len(edges))
	for _, edge := range edges {
		out = append(out, edge.Left+"--"+edge.Right)
	}
	return out
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
	case "introduced-or-increased-rootward-layer-skips":
		if len(diff.IntroducedRootwardLayerSkips) > 0 {
			return true
		}
		for _, change := range diff.RootwardLayerSkipChanges {
			if change.Delta > 0 {
				return true
			}
		}
		return false
	case "increased-fan-out":
		for _, change := range diff.FanOutChanges {
			if change.Delta > 0 {
				return true
			}
		}
		return false
	case "increased-dependency-reach":
		for _, change := range diff.DependencyReachChanges {
			if change.Delta > 0 {
				return true
			}
		}
		return false
	case "increased-dependency-depth":
		for _, change := range diff.DependencyDepthChanges {
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
	case "introduced-lateral-edges":
		for _, edge := range diff.IntroducedTypedEdges {
			if edge.Direction == "lateral" {
				return true
			}
		}
		return false
	case "introduced-lateral-couplings":
		return len(diff.IntroducedLateralCouplings) > 0
	case "introduced-or-increased-lateral-bridges":
		if len(diff.IntroducedLateralBridges) > 0 {
			return true
		}
		for _, change := range diff.LateralBridgeChanges {
			if change.Delta > 0 {
				return true
			}
		}
		return false
	case "introduced-or-increased-lateral-articulation-points":
		if len(diff.IntroducedLateralArticulationPoints) > 0 {
			return true
		}
		for _, change := range diff.LateralArticulationPointChanges {
			if change.Delta > 0 {
				return true
			}
		}
		return false
	case "resolved-lateral-resilient-pairs":
		return len(diff.ResolvedLateralResilientPairs) > 0
	case "decreased-lateral-edge-connectivity":
		for _, change := range diff.LateralEdgeConnectivityChanges {
			if change.Delta < 0 {
				return true
			}
		}
		return false
	case "decreased-lateral-vertex-connectivity":
		for _, change := range diff.LateralVertexConnectivityChanges {
			if change.Delta < 0 {
				return true
			}
		}
		return false
	case "decreased-lateral-vertex-pair-cuts":
		for _, change := range diff.LateralVertexPairCutChanges {
			if change.Delta < 0 {
				return true
			}
		}
		return false
	default:
		return false
	}
}
