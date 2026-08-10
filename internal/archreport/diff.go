package archreport

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"sort"
)

const DiffSchema = "fak-architecture-diff/1"

type TierChange struct {
	Leaf       string `json:"leaf"`
	Before     int    `json:"before"`
	BeforeName string `json:"before_name"`
	After      int    `json:"after"`
	AfterName  string `json:"after_name"`
}

type EdgeChange struct {
	From string `json:"from"`
	To   string `json:"to"`
}

type TierGapChange struct {
	Leaf         string `json:"leaf"`
	DeclaredTier int    `json:"declared_tier"`
	BeforeFloor  int    `json:"before_floor"`
	AfterFloor   int    `json:"after_floor"`
	BeforeGap    int    `json:"before_gap"`
	AfterGap     int    `json:"after_gap"`
	Delta        int    `json:"delta"`
}

type ViolationDistanceChange struct {
	From           string `json:"from"`
	To             string `json:"to"`
	BeforeDistance int    `json:"before_distance"`
	AfterDistance  int    `json:"after_distance"`
	Delta          int    `json:"delta"`
}

type FanInChange struct {
	Leaf   string `json:"leaf"`
	Before int    `json:"before"`
	After  int    `json:"after"`
	Delta  int    `json:"delta"`
}

type BlastRadiusChange struct {
	Leaf   string `json:"leaf"`
	Before int    `json:"before"`
	After  int    `json:"after"`
	Delta  int    `json:"delta"`
}

type BlastImpact struct {
	Source    string   `json:"source"`
	Dependent string   `json:"dependent"`
	Path      []string `json:"path"`
}

type LateralArticulationPointChange struct {
	Tier                int        `json:"tier"`
	TierName            string     `json:"tier_name"`
	Name                string     `json:"name"`
	BeforeFragments     [][]string `json:"before_fragments"`
	AfterFragments      [][]string `json:"after_fragments"`
	BeforeFragmentCount int        `json:"before_fragment_count"`
	AfterFragmentCount  int        `json:"after_fragment_count"`
	BeforeCouplingPairs int        `json:"before_coupling_pairs"`
	AfterCouplingPairs  int        `json:"after_coupling_pairs"`
	Delta               int        `json:"delta"`
}

type LateralBridgeChange struct {
	Tier                int      `json:"tier"`
	TierName            string   `json:"tier_name"`
	Left                string   `json:"left"`
	Right               string   `json:"right"`
	BeforeCouplingPairs int      `json:"before_coupling_pairs"`
	AfterCouplingPairs  int      `json:"after_coupling_pairs"`
	Delta               int      `json:"delta"`
	BeforeLeftSide      []string `json:"before_left_side"`
	BeforeRightSide     []string `json:"before_right_side"`
	AfterLeftSide       []string `json:"after_left_side"`
	AfterRightSide      []string `json:"after_right_side"`
}

type LateralEdgeConnectivityChange struct {
	Tier      int    `json:"tier"`
	TierName  string `json:"tier_name"`
	Left      string `json:"left"`
	Right     string `json:"right"`
	BeforeCut int    `json:"before_cut"`
	AfterCut  int    `json:"after_cut"`
	Delta     int    `json:"delta"`
}

type LateralResilientPair struct {
	Tier     int    `json:"tier"`
	TierName string `json:"tier_name"`
	Left     string `json:"left"`
	Right    string `json:"right"`
}

type LateralCoupling struct {
	Tier     int    `json:"tier"`
	TierName string `json:"tier_name"`
	Left     string `json:"left"`
	Right    string `json:"right"`
}

type BlastPathChange struct {
	Source     string   `json:"source"`
	Dependent  string   `json:"dependent"`
	BeforePath []string `json:"before_path"`
	AfterPath  []string `json:"after_path"`
	BeforeHops int      `json:"before_hops"`
	AfterHops  int      `json:"after_hops"`
	Delta      int      `json:"delta"`
}

type ReportDiff struct {
	Schema                              string                           `json:"schema"`
	Verdict                             string                           `json:"verdict"`
	AddedLeaves                         []string                         `json:"added_leaves,omitempty"`
	RemovedLeaves                       []string                         `json:"removed_leaves,omitempty"`
	TierChanges                         []TierChange                     `json:"tier_changes,omitempty"`
	AddedEdges                          []EdgeChange                     `json:"added_edges,omitempty"`
	RemovedEdges                        []EdgeChange                     `json:"removed_edges,omitempty"`
	IntroducedTypedEdges                []ArchitectureEdge               `json:"introduced_typed_edges,omitempty"`
	ResolvedTypedEdges                  []ArchitectureEdge               `json:"resolved_typed_edges,omitempty"`
	IntroducedLateralCouplings          []LateralCoupling                `json:"introduced_lateral_couplings,omitempty"`
	ResolvedLateralCouplings            []LateralCoupling                `json:"resolved_lateral_couplings,omitempty"`
	IntroducedLateralBridges            []LateralBridge                  `json:"introduced_lateral_bridges,omitempty"`
	ResolvedLateralBridges              []LateralBridge                  `json:"resolved_lateral_bridges,omitempty"`
	LateralBridgeChanges                []LateralBridgeChange            `json:"lateral_bridge_changes,omitempty"`
	IntroducedLateralArticulationPoints []LateralArticulationPoint       `json:"introduced_lateral_articulation_points,omitempty"`
	ResolvedLateralArticulationPoints   []LateralArticulationPoint       `json:"resolved_lateral_articulation_points,omitempty"`
	LateralArticulationPointChanges     []LateralArticulationPointChange `json:"lateral_articulation_point_changes,omitempty"`
	IntroducedLateralResilientPairs     []LateralResilientPair           `json:"introduced_lateral_resilient_pairs,omitempty"`
	ResolvedLateralResilientPairs       []LateralResilientPair           `json:"resolved_lateral_resilient_pairs,omitempty"`
	LateralEdgeConnectivityChanges      []LateralEdgeConnectivityChange  `json:"lateral_edge_connectivity_changes,omitempty"`
	FanInChanges                        []FanInChange                    `json:"fan_in_changes,omitempty"`
	BlastRadiusChanges                  []BlastRadiusChange              `json:"blast_radius_changes,omitempty"`
	IntroducedBlastImpacts              []BlastImpact                    `json:"introduced_blast_impacts,omitempty"`
	ResolvedBlastImpacts                []BlastImpact                    `json:"resolved_blast_impacts,omitempty"`
	BlastPathChanges                    []BlastPathChange                `json:"blast_path_changes,omitempty"`
	TierGapChanges                      []TierGapChange                  `json:"tier_gap_changes,omitempty"`
	IntroducedViolationEdges            []ViolationEdge                  `json:"introduced_violation_edges,omitempty"`
	ResolvedViolationEdges              []ViolationEdge                  `json:"resolved_violation_edges,omitempty"`
	ViolationDistanceChanges            []ViolationDistanceChange        `json:"violation_distance_changes,omitempty"`
	IntroducedViolations                []string                         `json:"introduced_violations,omitempty"` // Compatibility projection.
	ResolvedViolations                  []string                         `json:"resolved_violations,omitempty"`   // Compatibility projection.
	IntroducedDiagnostics               []Diagnostic                     `json:"introduced_diagnostics,omitempty"`
	ResolvedDiagnostics                 []Diagnostic                     `json:"resolved_diagnostics,omitempty"`
}

func Diff(before, after Report) ReportDiff {
	out := ReportDiff{Schema: DiffSchema, Verdict: "clean"}
	beforeLeaves, afterLeaves := leafIndex(before), leafIndex(after)
	for name, a := range afterLeaves {
		b, ok := beforeLeaves[name]
		if !ok {
			out.AddedLeaves = append(out.AddedLeaves, name)
			continue
		}
		if b.DeclaredTier != a.DeclaredTier {
			out.TierChanges = append(out.TierChanges, TierChange{Leaf: name, Before: b.DeclaredTier, BeforeName: b.DeclaredTierName, After: a.DeclaredTier, AfterName: a.DeclaredTierName})
		}
		if b.TierGap != a.TierGap {
			out.TierGapChanges = append(out.TierGapChanges, TierGapChange{Leaf: name, DeclaredTier: a.DeclaredTier, BeforeFloor: b.ImportFloor, AfterFloor: a.ImportFloor, BeforeGap: b.TierGap, AfterGap: a.TierGap, Delta: a.TierGap - b.TierGap})
		}
	}
	for name := range beforeLeaves {
		if _, ok := afterLeaves[name]; !ok {
			out.RemovedLeaves = append(out.RemovedLeaves, name)
		}
	}
	beforeEdges, afterEdges := edgeSet(before), edgeSet(after)
	for key, edge := range afterEdges {
		if _, ok := beforeEdges[key]; !ok {
			out.AddedEdges = append(out.AddedEdges, edge)
		}
	}
	for key, edge := range beforeEdges {
		if _, ok := afterEdges[key]; !ok {
			out.RemovedEdges = append(out.RemovedEdges, edge)
		}
	}
	beforeTypedEdges, afterTypedEdges := typedEdgeSet(before), typedEdgeSet(after)
	for key, edge := range afterTypedEdges {
		if _, ok := beforeTypedEdges[key]; !ok {
			out.IntroducedTypedEdges = append(out.IntroducedTypedEdges, edge)
		}
	}
	for key, edge := range beforeTypedEdges {
		if _, ok := afterTypedEdges[key]; !ok {
			out.ResolvedTypedEdges = append(out.ResolvedTypedEdges, edge)
		}
	}
	beforeCouplings, afterCouplings := lateralCouplingSet(before), lateralCouplingSet(after)
	for key, coupling := range afterCouplings {
		if _, ok := beforeCouplings[key]; !ok {
			out.IntroducedLateralCouplings = append(out.IntroducedLateralCouplings, coupling)
		}
	}
	for key, coupling := range beforeCouplings {
		if _, ok := afterCouplings[key]; !ok {
			out.ResolvedLateralCouplings = append(out.ResolvedLateralCouplings, coupling)
		}
	}
	beforeBridges, afterBridges := lateralBridgeSet(before), lateralBridgeSet(after)
	for key, bridge := range afterBridges {
		if beforeBridge, ok := beforeBridges[key]; !ok {
			out.IntroducedLateralBridges = append(out.IntroducedLateralBridges, bridge)
		} else if beforeBridge.CouplingPairs != bridge.CouplingPairs {
			out.LateralBridgeChanges = append(out.LateralBridgeChanges, LateralBridgeChange{Tier: bridge.Tier, TierName: bridge.TierName, Left: bridge.Left, Right: bridge.Right, BeforeCouplingPairs: beforeBridge.CouplingPairs, AfterCouplingPairs: bridge.CouplingPairs, Delta: bridge.CouplingPairs - beforeBridge.CouplingPairs, BeforeLeftSide: beforeBridge.LeftSide, BeforeRightSide: beforeBridge.RightSide, AfterLeftSide: bridge.LeftSide, AfterRightSide: bridge.RightSide})
		}
	}
	for key, bridge := range beforeBridges {
		if _, ok := afterBridges[key]; !ok {
			out.ResolvedLateralBridges = append(out.ResolvedLateralBridges, bridge)
		}
	}
	beforePoints, afterPoints := lateralArticulationPointSet(before), lateralArticulationPointSet(after)
	for key, point := range afterPoints {
		if beforePoint, ok := beforePoints[key]; !ok {
			out.IntroducedLateralArticulationPoints = append(out.IntroducedLateralArticulationPoints, point)
		} else if beforePoint.CouplingPairs != point.CouplingPairs || beforePoint.FragmentCount != point.FragmentCount || !reflect.DeepEqual(beforePoint.Fragments, point.Fragments) {
			out.LateralArticulationPointChanges = append(out.LateralArticulationPointChanges, LateralArticulationPointChange{Tier: point.Tier, TierName: point.TierName, Name: point.Name, BeforeFragments: beforePoint.Fragments, AfterFragments: point.Fragments, BeforeFragmentCount: beforePoint.FragmentCount, AfterFragmentCount: point.FragmentCount, BeforeCouplingPairs: beforePoint.CouplingPairs, AfterCouplingPairs: point.CouplingPairs, Delta: point.CouplingPairs - beforePoint.CouplingPairs})
		}
	}
	for key, point := range beforePoints {
		if _, ok := afterPoints[key]; !ok {
			out.ResolvedLateralArticulationPoints = append(out.ResolvedLateralArticulationPoints, point)
		}
	}
	beforeResilience, afterResilience := lateralResilientPairSet(before), lateralResilientPairSet(after)
	for key, pair := range afterResilience {
		if _, ok := beforeResilience[key]; !ok {
			out.IntroducedLateralResilientPairs = append(out.IntroducedLateralResilientPairs, pair)
		}
	}
	for key, pair := range beforeResilience {
		if _, ok := afterResilience[key]; !ok {
			out.ResolvedLateralResilientPairs = append(out.ResolvedLateralResilientPairs, pair)
		}
	}
	beforeCuts, afterCuts := lateralPairCutSet(before), lateralPairCutSet(after)
	for key, afterCut := range afterCuts {
		if beforeCut, ok := beforeCuts[key]; ok && beforeCut.Cut != afterCut.Cut {
			out.LateralEdgeConnectivityChanges = append(out.LateralEdgeConnectivityChanges, LateralEdgeConnectivityChange{Tier: afterCut.Tier, TierName: afterCut.TierName, Left: afterCut.Left, Right: afterCut.Right, BeforeCut: beforeCut.Cut, AfterCut: afterCut.Cut, Delta: afterCut.Cut - beforeCut.Cut})
		}
	}
	for name, a := range afterLeaves {
		beforeFanIn := 0
		if b, ok := beforeLeaves[name]; ok {
			beforeFanIn = len(b.Dependents)
		}
		afterFanIn := len(a.Dependents)
		if beforeFanIn != afterFanIn {
			out.FanInChanges = append(out.FanInChanges, FanInChange{Leaf: name, Before: beforeFanIn, After: afterFanIn, Delta: afterFanIn - beforeFanIn})
		}
		if b, ok := beforeLeaves[name]; ok {
			if b.BlastRadius != a.BlastRadius {
				out.BlastRadiusChanges = append(out.BlastRadiusChanges, BlastRadiusChange{Leaf: name, Before: b.BlastRadius, After: a.BlastRadius, Delta: a.BlastRadius - b.BlastRadius})
			}
			beforeImpacts, afterImpacts := blastImpactSet(name, b), blastImpactSet(name, a)
			for key, impact := range afterImpacts {
				if _, ok := beforeImpacts[key]; !ok {
					out.IntroducedBlastImpacts = append(out.IntroducedBlastImpacts, impact)
				}
			}
			for key, impact := range beforeImpacts {
				if _, ok := afterImpacts[key]; !ok {
					out.ResolvedBlastImpacts = append(out.ResolvedBlastImpacts, impact)
				}
			}
			for key, afterImpact := range afterImpacts {
				if beforeImpact, ok := beforeImpacts[key]; ok && !slices.Equal(beforeImpact.Path, afterImpact.Path) {
					beforeHops, afterHops := len(beforeImpact.Path)-1, len(afterImpact.Path)-1
					out.BlastPathChanges = append(out.BlastPathChanges, BlastPathChange{Source: name, Dependent: afterImpact.Dependent, BeforePath: beforeImpact.Path, AfterPath: afterImpact.Path, BeforeHops: beforeHops, AfterHops: afterHops, Delta: afterHops - beforeHops})
				}
			}
		}
	}
	for name, b := range beforeLeaves {
		if _, ok := afterLeaves[name]; !ok && len(b.Dependents) > 0 {
			out.FanInChanges = append(out.FanInChanges, FanInChange{Leaf: name, Before: len(b.Dependents), After: 0, Delta: -len(b.Dependents)})
		}
	}
	beforeDiagnostics, afterDiagnostics := diagnosticSet(before), diagnosticSet(after)
	for key, diagnostic := range afterDiagnostics {
		if _, ok := beforeDiagnostics[key]; !ok {
			out.IntroducedDiagnostics = append(out.IntroducedDiagnostics, diagnostic)
		}
	}
	for key, diagnostic := range beforeDiagnostics {
		if _, ok := afterDiagnostics[key]; !ok {
			out.ResolvedDiagnostics = append(out.ResolvedDiagnostics, diagnostic)
		}
	}
	beforeViolations, afterViolations := violationEdgeSet(before), violationEdgeSet(after)
	for key, edge := range afterViolations {
		if _, ok := beforeViolations[key]; !ok {
			out.IntroducedViolationEdges = append(out.IntroducedViolationEdges, edge)
		}
	}
	for key, edge := range beforeViolations {
		if _, ok := afterViolations[key]; !ok {
			out.ResolvedViolationEdges = append(out.ResolvedViolationEdges, edge)
		}
	}
	for key, afterEdge := range afterViolations {
		if beforeEdge, ok := beforeViolations[key]; ok && beforeEdge.TierDistance != afterEdge.TierDistance {
			out.ViolationDistanceChanges = append(out.ViolationDistanceChanges, ViolationDistanceChange{From: afterEdge.From, To: afterEdge.To, BeforeDistance: beforeEdge.TierDistance, AfterDistance: afterEdge.TierDistance, Delta: afterEdge.TierDistance - beforeEdge.TierDistance})
		}
	}
	sortViolationEdges(out.IntroducedViolationEdges)
	sortViolationEdges(out.ResolvedViolationEdges)
	sortViolationDistanceChanges(out.ViolationDistanceChanges)
	out.IntroducedViolations = violationStrings(out.IntroducedViolationEdges)
	out.ResolvedViolations = violationStrings(out.ResolvedViolationEdges)
	sort.Strings(out.AddedLeaves)
	sort.Strings(out.RemovedLeaves)
	sort.Slice(out.TierChanges, func(i, j int) bool { return out.TierChanges[i].Leaf < out.TierChanges[j].Leaf })
	sortEdges(out.AddedEdges)
	sortEdges(out.RemovedEdges)
	sortArchitectureEdges(out.IntroducedTypedEdges)
	sortArchitectureEdges(out.ResolvedTypedEdges)
	sortLateralCouplings(out.IntroducedLateralCouplings)
	sortLateralCouplings(out.ResolvedLateralCouplings)
	sortLateralBridges(out.IntroducedLateralBridges)
	sortLateralBridges(out.ResolvedLateralBridges)
	sortLateralBridgeChanges(out.LateralBridgeChanges)
	sortLateralArticulationPoints(out.IntroducedLateralArticulationPoints)
	sortLateralArticulationPoints(out.ResolvedLateralArticulationPoints)
	sortLateralArticulationPointChanges(out.LateralArticulationPointChanges)
	sortLateralResilientPairs(out.IntroducedLateralResilientPairs)
	sortLateralResilientPairs(out.ResolvedLateralResilientPairs)
	sortLateralEdgeConnectivityChanges(out.LateralEdgeConnectivityChanges)
	sortFanInChanges(out.FanInChanges)
	sortBlastRadiusChanges(out.BlastRadiusChanges)
	sortBlastImpacts(out.IntroducedBlastImpacts)
	sortBlastImpacts(out.ResolvedBlastImpacts)
	sortBlastPathChanges(out.BlastPathChanges)
	sortTierGapChanges(out.TierGapChanges)
	sortDiagnostics(out.IntroducedDiagnostics)
	sortDiagnostics(out.ResolvedDiagnostics)
	if len(out.IntroducedViolationEdges) > 0 || len(out.IntroducedDiagnostics) > 0 || hasIncreasedTierGap(out.TierGapChanges) || hasIncreasedViolationDistance(out.ViolationDistanceChanges) || hasIncreasedBlastRadius(out.BlastRadiusChanges) || len(out.IntroducedBlastImpacts) > 0 || hasIncreasedBlastPathLength(out.BlastPathChanges) || len(out.IntroducedLateralCouplings) > 0 || len(out.IntroducedLateralBridges) > 0 || hasIncreasedLateralBridgeImpact(out.LateralBridgeChanges) || len(out.IntroducedLateralArticulationPoints) > 0 || hasIncreasedLateralArticulationPointImpact(out.LateralArticulationPointChanges) || len(out.ResolvedLateralResilientPairs) > 0 || hasDecreasedLateralEdgeConnectivity(out.LateralEdgeConnectivityChanges) {
		out.Verdict = "regression"
	}
	return out
}

func (d ReportDiff) Changes() int {
	return len(d.AddedLeaves) + len(d.RemovedLeaves) + len(d.TierChanges) + len(d.AddedEdges) + len(d.RemovedEdges) + len(d.IntroducedViolationEdges) + len(d.ResolvedViolationEdges) + len(d.IntroducedDiagnostics) + len(d.ResolvedDiagnostics)
}
func (d ReportDiff) JSON() ([]byte, error) { return json.MarshalIndent(d, "", "  ") }
func leafIndex(r Report) map[string]Leaf {
	out := map[string]Leaf{}
	for _, l := range r.Leaves {
		out[l.Name] = l
	}
	return out
}
func blastImpactSet(source string, leaf Leaf) map[string]BlastImpact {
	out := make(map[string]BlastImpact, len(leaf.BlastPaths))
	for _, path := range leaf.BlastPaths {
		impact := BlastImpact{Source: source, Dependent: path.Dependent, Path: append([]string(nil), path.Path...)}
		out[source+"\x00"+path.Dependent] = impact
	}
	return out
}

type lateralPairCut struct {
	Tier                  int
	TierName, Left, Right string
	Cut                   int
}

func lateralPairCutSet(r Report) map[string]lateralPairCut {
	out := map[string]lateralPairCut{}
	for _, block := range r.LateralBiconnectedBlocks {
		for _, pair := range block.PairCuts {
			key := fmt.Sprintf("%d\x00%s\x00%s", block.Tier, pair.Left, pair.Right)
			if old, ok := out[key]; !ok || pair.Cut > old.Cut {
				out[key] = lateralPairCut{block.Tier, block.TierName, pair.Left, pair.Right, pair.Cut}
			}
		}
	}
	return out
}
func sortLateralEdgeConnectivityChanges(c []LateralEdgeConnectivityChange) {
	sort.Slice(c, func(i, j int) bool {
		if (c[i].Delta < 0) != (c[j].Delta < 0) {
			return c[i].Delta < 0
		}
		im, jm := c[i].Delta, c[j].Delta
		if im < 0 {
			im = -im
		}
		if jm < 0 {
			jm = -jm
		}
		if im != jm {
			return im > jm
		}
		if c[i].Tier != c[j].Tier {
			return c[i].Tier < c[j].Tier
		}
		if c[i].Left != c[j].Left {
			return c[i].Left < c[j].Left
		}
		return c[i].Right < c[j].Right
	})
}
func hasDecreasedLateralEdgeConnectivity(c []LateralEdgeConnectivityChange) bool {
	for _, x := range c {
		if x.Delta < 0 {
			return true
		}
	}
	return false
}

func lateralResilientPairSet(r Report) map[string]LateralResilientPair {
	out := map[string]LateralResilientPair{}
	for _, block := range r.LateralBiconnectedBlocks {
		for i := 0; i < len(block.Members); i++ {
			for j := i + 1; j < len(block.Members); j++ {
				left, right := block.Members[i], block.Members[j]
				if right < left {
					left, right = right, left
				}
				pair := LateralResilientPair{Tier: block.Tier, TierName: block.TierName, Left: left, Right: right}
				out[fmt.Sprintf("%d\x00%s\x00%s", block.Tier, left, right)] = pair
			}
		}
	}
	return out
}
func sortLateralResilientPairs(pairs []LateralResilientPair) {
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Tier != pairs[j].Tier {
			return pairs[i].Tier < pairs[j].Tier
		}
		if pairs[i].Left != pairs[j].Left {
			return pairs[i].Left < pairs[j].Left
		}
		return pairs[i].Right < pairs[j].Right
	})
}

func lateralArticulationPointSet(r Report) map[string]LateralArticulationPoint {
	out := make(map[string]LateralArticulationPoint, len(r.LateralArticulationPoints))
	for _, point := range r.LateralArticulationPoints {
		out[fmt.Sprintf("%d\x00%s", point.Tier, point.Name)] = point
	}
	return out
}
func sortLateralArticulationPoints(points []LateralArticulationPoint) {
	sort.Slice(points, func(i, j int) bool {
		if points[i].CouplingPairs != points[j].CouplingPairs {
			return points[i].CouplingPairs > points[j].CouplingPairs
		}
		if points[i].FragmentCount != points[j].FragmentCount {
			return points[i].FragmentCount > points[j].FragmentCount
		}
		if points[i].Tier != points[j].Tier {
			return points[i].Tier < points[j].Tier
		}
		return points[i].Name < points[j].Name
	})
}
func sortLateralArticulationPointChanges(changes []LateralArticulationPointChange) {
	sort.Slice(changes, func(i, j int) bool {
		if (changes[i].Delta > 0) != (changes[j].Delta > 0) {
			return changes[i].Delta > 0
		}
		im, jm := changes[i].Delta, changes[j].Delta
		if im < 0 {
			im = -im
		}
		if jm < 0 {
			jm = -jm
		}
		if im != jm {
			return im > jm
		}
		if changes[i].Tier != changes[j].Tier {
			return changes[i].Tier < changes[j].Tier
		}
		return changes[i].Name < changes[j].Name
	})
}
func hasIncreasedLateralArticulationPointImpact(changes []LateralArticulationPointChange) bool {
	for _, change := range changes {
		if change.Delta > 0 {
			return true
		}
	}
	return false
}

func lateralBridgeSet(r Report) map[string]LateralBridge {
	out := make(map[string]LateralBridge, len(r.LateralBridges))
	for _, bridge := range r.LateralBridges {
		out[fmt.Sprintf("%d\x00%s\x00%s", bridge.Tier, bridge.Left, bridge.Right)] = bridge
	}
	return out
}

func sortLateralBridges(bridges []LateralBridge) {
	sort.Slice(bridges, func(i, j int) bool {
		if bridges[i].CouplingPairs != bridges[j].CouplingPairs {
			return bridges[i].CouplingPairs > bridges[j].CouplingPairs
		}
		if bridges[i].Tier != bridges[j].Tier {
			return bridges[i].Tier < bridges[j].Tier
		}
		if bridges[i].Left != bridges[j].Left {
			return bridges[i].Left < bridges[j].Left
		}
		return bridges[i].Right < bridges[j].Right
	})
}

func sortLateralBridgeChanges(changes []LateralBridgeChange) {
	sort.Slice(changes, func(i, j int) bool {
		if (changes[i].Delta > 0) != (changes[j].Delta > 0) {
			return changes[i].Delta > 0
		}
		iMagnitude, jMagnitude := changes[i].Delta, changes[j].Delta
		if iMagnitude < 0 {
			iMagnitude = -iMagnitude
		}
		if jMagnitude < 0 {
			jMagnitude = -jMagnitude
		}
		if iMagnitude != jMagnitude {
			return iMagnitude > jMagnitude
		}
		if changes[i].Tier != changes[j].Tier {
			return changes[i].Tier < changes[j].Tier
		}
		if changes[i].Left != changes[j].Left {
			return changes[i].Left < changes[j].Left
		}
		return changes[i].Right < changes[j].Right
	})
}

func hasIncreasedLateralBridgeImpact(changes []LateralBridgeChange) bool {
	for _, change := range changes {
		if change.Delta > 0 {
			return true
		}
	}
	return false
}

func lateralCouplingSet(r Report) map[string]LateralCoupling {
	out := map[string]LateralCoupling{}
	for _, component := range r.LateralComponents {
		for i := 0; i < len(component.Members); i++ {
			for j := i + 1; j < len(component.Members); j++ {
				left, right := component.Members[i], component.Members[j]
				if right < left {
					left, right = right, left
				}
				coupling := LateralCoupling{Tier: component.Tier, TierName: component.TierName, Left: left, Right: right}
				out[fmt.Sprintf("%d\x00%s\x00%s", component.Tier, left, right)] = coupling
			}
		}
	}
	return out
}

func sortLateralCouplings(couplings []LateralCoupling) {
	sort.Slice(couplings, func(i, j int) bool {
		if couplings[i].Tier != couplings[j].Tier {
			return couplings[i].Tier < couplings[j].Tier
		}
		if couplings[i].Left != couplings[j].Left {
			return couplings[i].Left < couplings[j].Left
		}
		return couplings[i].Right < couplings[j].Right
	})
}

func typedEdgeSet(r Report) map[string]ArchitectureEdge {
	out := make(map[string]ArchitectureEdge, len(r.Edges))
	for _, edge := range r.Edges {
		out[edge.From+"\x00"+edge.To] = edge
	}
	return out
}

func sortArchitectureEdges(edges []ArchitectureEdge) {
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].From != edges[j].From {
			return edges[i].From < edges[j].From
		}
		return edges[i].To < edges[j].To
	})
}

func edgeSet(r Report) map[string]EdgeChange {
	out := map[string]EdgeChange{}
	for _, l := range r.Leaves {
		for _, to := range l.Dependencies {
			e := EdgeChange{From: l.Name, To: to}
			out[l.Name+"\x00"+to] = e
		}
	}
	return out
}
func violationEdgeSet(r Report) map[string]ViolationEdge {
	out := map[string]ViolationEdge{}
	for _, leaf := range r.Leaves {
		for _, edge := range leaf.ViolationEdges {
			out[edge.From+"\x00"+edge.To] = edge
		}
	}
	return out
}

func sortEdges(edges []EdgeChange) {
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].From != edges[j].From {
			return edges[i].From < edges[j].From
		}
		return edges[i].To < edges[j].To
	})
}

func diagnosticSet(r Report) map[string]Diagnostic {
	out := map[string]Diagnostic{}
	for _, diagnostic := range r.Diagnostics {
		out[diagnostic.Kind+"\x00"+diagnostic.Leaf] = diagnostic
	}
	return out
}

func sortDiagnostics(diagnostics []Diagnostic) {
	sort.Slice(diagnostics, func(i, j int) bool {
		if diagnostics[i].Kind != diagnostics[j].Kind {
			return diagnostics[i].Kind < diagnostics[j].Kind
		}
		return diagnostics[i].Leaf < diagnostics[j].Leaf
	})
}

func sortBlastPathChanges(changes []BlastPathChange) {
	sort.Slice(changes, func(i, j int) bool {
		if (changes[i].Delta > 0) != (changes[j].Delta > 0) {
			return changes[i].Delta > 0
		}
		iMagnitude, jMagnitude := changes[i].Delta, changes[j].Delta
		if iMagnitude < 0 {
			iMagnitude = -iMagnitude
		}
		if jMagnitude < 0 {
			jMagnitude = -jMagnitude
		}
		if iMagnitude != jMagnitude {
			return iMagnitude > jMagnitude
		}
		if changes[i].Source != changes[j].Source {
			return changes[i].Source < changes[j].Source
		}
		return changes[i].Dependent < changes[j].Dependent
	})
}

func sortBlastImpacts(impacts []BlastImpact) {
	sort.Slice(impacts, func(i, j int) bool {
		if impacts[i].Source != impacts[j].Source {
			return impacts[i].Source < impacts[j].Source
		}
		return impacts[i].Dependent < impacts[j].Dependent
	})
}

func sortBlastRadiusChanges(changes []BlastRadiusChange) {
	sort.Slice(changes, func(i, j int) bool {
		if (changes[i].Delta > 0) != (changes[j].Delta > 0) {
			return changes[i].Delta > 0
		}
		iMagnitude, jMagnitude := changes[i].Delta, changes[j].Delta
		if iMagnitude < 0 {
			iMagnitude = -iMagnitude
		}
		if jMagnitude < 0 {
			jMagnitude = -jMagnitude
		}
		if iMagnitude != jMagnitude {
			return iMagnitude > jMagnitude
		}
		return changes[i].Leaf < changes[j].Leaf
	})
}

func sortFanInChanges(changes []FanInChange) {
	sort.Slice(changes, func(i, j int) bool {
		iGrowth, jGrowth := changes[i].Delta > 0, changes[j].Delta > 0
		if iGrowth != jGrowth {
			return iGrowth
		}
		iMagnitude, jMagnitude := changes[i].Delta, changes[j].Delta
		if iMagnitude < 0 {
			iMagnitude = -iMagnitude
		}
		if jMagnitude < 0 {
			jMagnitude = -jMagnitude
		}
		if iMagnitude != jMagnitude {
			return iMagnitude > jMagnitude
		}
		return changes[i].Leaf < changes[j].Leaf
	})
}

func sortTierGapChanges(changes []TierGapChange) {
	sort.Slice(changes, func(i, j int) bool {
		ig, jg := changes[i].Delta > 0, changes[j].Delta > 0
		if ig != jg {
			return ig
		}
		ai, aj := changes[i].Delta, changes[j].Delta
		if ai < 0 {
			ai = -ai
		}
		if aj < 0 {
			aj = -aj
		}
		if ai != aj {
			return ai > aj
		}
		return changes[i].Leaf < changes[j].Leaf
	})
}

func hasIncreasedTierGap(changes []TierGapChange) bool {
	for _, change := range changes {
		if change.Delta > 0 {
			return true
		}
	}
	return false
}

func hasIncreasedBlastPathLength(changes []BlastPathChange) bool {
	for _, change := range changes {
		if change.Delta > 0 {
			return true
		}
	}
	return false
}

func hasIncreasedBlastRadius(changes []BlastRadiusChange) bool {
	for _, change := range changes {
		if change.Delta > 0 {
			return true
		}
	}
	return false
}

func hasIncreasedViolationDistance(changes []ViolationDistanceChange) bool {
	for _, change := range changes {
		if change.Delta > 0 {
			return true
		}
	}
	return false
}

func sortViolationDistanceChanges(changes []ViolationDistanceChange) {
	sort.Slice(changes, func(i, j int) bool {
		iGrowth, jGrowth := changes[i].Delta > 0, changes[j].Delta > 0
		if iGrowth != jGrowth {
			return iGrowth
		}
		iMagnitude, jMagnitude := changes[i].Delta, changes[j].Delta
		if iMagnitude < 0 {
			iMagnitude = -iMagnitude
		}
		if jMagnitude < 0 {
			jMagnitude = -jMagnitude
		}
		if iMagnitude != jMagnitude {
			return iMagnitude > jMagnitude
		}
		if changes[i].From != changes[j].From {
			return changes[i].From < changes[j].From
		}
		return changes[i].To < changes[j].To
	})
}
