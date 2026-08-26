package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/archreport"
)

func renderArchitectureReport(stdout io.Writer, report archreport.Report, leaf string) {
	fmt.Fprintf(stdout, "architecture: %d leaves, %d upward violation(s), max tier distance %d\n", sumArchitectureLeaves(report), report.Violations, report.MaxViolationDistance)
	rootward, lateral, upward := 0, 0, 0
	for _, edge := range report.Edges {
		switch edge.Direction {
		case "rootward":
			rootward++
		case "lateral":
			lateral++
		case "upward":
			upward++
		}
	}
	fmt.Fprintf(stdout, "  typed edges: rootward=%d lateral=%d upward=%d\n", rootward, lateral, upward)
	for _, t := range report.Tiers {
		fmt.Fprintf(stdout, "  tier %d %-22s %d\n", t.Level, t.Name, t.Leaves)
	}
	for _, diagnostic := range report.Diagnostics {
		fmt.Fprintf(stdout, "  diagnostic %-24s leaf=%s: %s; recovery: %s\n", diagnostic.Kind, diagnostic.Leaf, diagnostic.Message, diagnostic.Recovery)
	}
	if leaf == "" && len(report.DependencyCycles) > 0 {
		fmt.Fprintln(stdout, "  dependency cycles (directed strongly connected components):")
		for _, cycle := range report.DependencyCycles {
			fmt.Fprintf(stdout, "    members=%v edges=%d\n", cycle.Members, len(cycle.Edges))
		}
	}
	if leaf == "" && len(report.Hotspots) > 0 {
		fmt.Fprintln(stdout, "  hotspots (direct fan-in):")
		for _, hotspot := range report.Hotspots {
			fmt.Fprintf(stdout, "    %-22s %d\n", hotspot.Name, hotspot.FanIn)
		}
	}
	if leaf == "" && len(report.FanOutHotspots) > 0 {
		fmt.Fprintln(stdout, "  fan-out hotspots (direct dependency burden):")
		for _, hotspot := range report.FanOutHotspots {
			fmt.Fprintf(stdout, "    %-22s %d\n", hotspot.Name, hotspot.FanOut)
		}
	}
	if leaf == "" && len(report.DependencyHotspots) > 0 {
		fmt.Fprintln(stdout, "  dependency hotspots (transitive forward burden):")
		for _, hotspot := range report.DependencyHotspots {
			fmt.Fprintf(stdout, "    %-22s reach=%d depth=%d fan-out=%d\n", hotspot.Name, hotspot.DependencyReach, hotspot.DependencyDepth, hotspot.FanOut)
		}
	}
	if leaf == "" && len(report.BlastHotspots) > 0 {
		fmt.Fprintln(stdout, "  blast hotspots (transitive impact):")
		for _, hotspot := range report.BlastHotspots {
			fmt.Fprintf(stdout, "    %-22s radius=%d max-hops=%d\n", hotspot.Name, hotspot.BlastRadius, hotspot.MaxHops)
		}
	}
	if len(report.RootwardLayerSkips) > 0 {
		fmt.Fprintln(stdout, "  rootward layer skips (legal abstraction bypasses):")
		for _, skip := range report.RootwardLayerSkips {
			fmt.Fprintf(stdout, "    %s(%s:%d) -> %s(%s:%d) distance=%d skipped-tiers=%d\n", skip.From, skip.FromTierName, skip.FromTier, skip.To, skip.ToTierName, skip.ToTier, skip.TierDistance, skip.SkippedTiers)
		}
	}
	if len(report.LateralComponents) > 0 {
		fmt.Fprintln(stdout, "  lateral components (same-tier coupling):")
		for _, component := range report.LateralComponents {
			fmt.Fprintf(stdout, "    %-22s members=%d edges=%d %v\n", component.TierName, component.MemberCount, component.EdgeCount, component.Members)
		}
	}
	if len(report.LateralBridges) > 0 {
		fmt.Fprintln(stdout, "  lateral bridges (articulation edges):")
		for _, bridge := range report.LateralBridges {
			fmt.Fprintf(stdout, "    %s -> %s tier=%s sides=%d/%d coupling-pairs=%d\n", bridge.From, bridge.To, bridge.TierName, len(bridge.LeftSide), len(bridge.RightSide), bridge.CouplingPairs)
		}
	}
	if len(report.LateralArticulationPoints) > 0 {
		fmt.Fprintln(stdout, "  lateral articulation points (package seams):")
		for _, point := range report.LateralArticulationPoints {
			sizes := make([]int, len(point.Fragments))
			for i, fragment := range point.Fragments {
				sizes[i] = len(fragment)
			}
			fmt.Fprintf(stdout, "    %s tier=%s fragments=%v coupling-pairs=%d\n", point.Name, point.TierName, sizes, point.CouplingPairs)
		}
	}
	if len(report.LateralBiconnectedBlocks) > 0 {
		fmt.Fprintln(stdout, "  lateral biconnected blocks (single-package resilient):")
		for _, block := range report.LateralBiconnectedBlocks {
			fmt.Fprintf(stdout, "    tier=%s members=%v edges=%d edge-connectivity=%d vertex-connectivity=%d separator=%v critical-pairs=%d\n", block.TierName, block.Members, block.EdgeCount, block.MinEdgeCut, block.MinVertexCut, block.CriticalSeparator, len(block.CriticalPairs))
			for _, pair := range block.VertexPairCuts {
				if pair.Cut == block.MinVertexCut {
					fmt.Fprintf(stdout, "      vertex-pair %s--%s cut=%d separator=%v\n", pair.Left, pair.Right, pair.Cut, pair.Separator)
				}
			}
			for _, pair := range block.CriticalPairs {
				edges := make([]string, 0, len(pair.CutEdges))
				for _, edge := range pair.CutEdges {
					edges = append(edges, edge.Left+"--"+edge.Right)
				}
				fmt.Fprintf(stdout, "      %s--%s cut=%d witness=%v partition=%v|%v\n", pair.Left, pair.Right, pair.Cut, edges, pair.SourceSide, pair.SinkSide)
			}
		}
	}
	if leaf == "" && len(report.SinkCandidates) > 0 {
		fmt.Fprintln(stdout, "  sink candidates (declared tier above import floor):")
		for _, candidate := range report.SinkCandidates {
			fmt.Fprintf(stdout, "    %-22s declared=%s(%d) floor=%s(%d) gap=%d\n", candidate.Name, candidate.DeclaredTierName, candidate.DeclaredTier, candidate.ImportFloorName, candidate.ImportFloor, candidate.TierGap)
		}
	}
	for _, l := range report.Leaves {
		if leaf != "" || len(l.ViolationEdges) > 0 {
			fmt.Fprintf(stdout, "  %-24s declared=%s(%d) floor=%s(%d) gap=%d deps=%v dependents=%v dependency-reach=%d dependency-depth=%d blast-radius=%d", l.Name, l.DeclaredTierName, l.DeclaredTier, l.ImportFloorName, l.ImportFloor, l.TierGap, l.Dependencies, l.Dependents, l.DependencyReach, l.DependencyDepth, l.BlastRadius)
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
			if leaf != "" && len(l.DependencyPaths) > 0 {
				fmt.Fprintln(stdout, "    dependency paths:")
				for _, path := range l.DependencyPaths {
					fmt.Fprintf(stdout, "      %s: %s\n", path.Dependency, strings.Join(path.Path, " -> "))
				}
			}
			if leaf != "" && len(l.DependencyCycle) > 0 {
				fmt.Fprintf(stdout, "    dependency cycle members=%v\n", l.DependencyCycle)
			}
			if leaf != "" && len(l.DependencyDominators) > 0 {
				fmt.Fprintln(stdout, "    mandatory dependency seams:")
				for _, seam := range l.DependencyDominators {
					fmt.Fprintf(stdout, "      %s via=%v path=%s\n", seam.Dependency, seam.Dominators, strings.Join(seam.Path, " -> "))
				}
			}
			if leaf != "" && len(l.RedundantDependencies) > 0 {
				fmt.Fprintln(stdout, "    redundant dependency edges:")
				for _, redundant := range l.RedundantDependencies {
					fmt.Fprintf(stdout, "      %s alternate=%s", redundant.Dependency, strings.Join(redundant.AlternatePath, " -> "))
					if len(redundant.Sources) > 0 {
						fmt.Fprintf(stdout, " source=%s", formatArchitectureSources(redundant.Sources))
					}
					fmt.Fprintln(stdout)
				}
			}
			if leaf != "" && len(l.BlastPaths) > 0 {
				fmt.Fprintln(stdout, "    blast paths:")
				for _, path := range l.BlastPaths {
					fmt.Fprintf(stdout, "      %s: %s\n", path.Dependent, strings.Join(path.Path, " -> "))
				}
			}
		}
	}
}
