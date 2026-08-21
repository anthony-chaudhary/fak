package archreport

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
)

const modulePrefix = "github.com/anthony-chaudhary/fak/internal/"

const DiagnosticStaleTierDeclaration = "stale-tier-declaration"

type Tier struct {
	Level  int    `json:"level"`
	Name   string `json:"name"`
	Leaves int    `json:"leaves"`
}
type Leaf struct {
	Name                   string                `json:"name"`
	DeclaredTier           int                   `json:"declared_tier"`
	DeclaredTierName       string                `json:"declared_tier_name"`
	ImportFloor            int                   `json:"import_floor"`
	ImportFloorName        string                `json:"import_floor_name"`
	TierGap                int                   `json:"tier_gap"`
	Dependencies           []string              `json:"dependencies"`
	TransitiveDependencies []string              `json:"transitive_dependencies"`
	DependencyReach        int                   `json:"dependency_reach"`
	DependencyDepth        int                   `json:"dependency_depth"`
	DependencyPaths        []DependencyPath      `json:"dependency_paths"`
	DependencyDominators   []DependencyDominator `json:"dependency_dominators"`
	RedundantDependencies  []RedundantDependency `json:"redundant_dependencies"`
	DependencyCycle        []string              `json:"dependency_cycle"`
	Dependents             []string              `json:"dependents,omitempty"`
	TransitiveDependents   []string              `json:"transitive_dependents"`
	BlastRadius            int                   `json:"blast_radius"`
	BlastPaths             []BlastPath           `json:"blast_paths"`
	ViolationEdges         []ViolationEdge       `json:"violation_edges,omitempty"`
	Violations             []string              `json:"violations,omitempty"` // Compatibility projection; use ViolationEdges.
}

type DependencyDominator struct {
	Dependency string   `json:"dependency"`
	Dominators []string `json:"dominators"`
	Path       []string `json:"path"`
}

type DependencyPath struct {
	Dependency string   `json:"dependency"`
	Path       []string `json:"path"`
}

type RedundantDependency struct {
	Dependency    string         `json:"dependency"`
	AlternatePath []string       `json:"alternate_path"`
	Sources       []SourceImport `json:"sources,omitempty"`
}

// SourceImport identifies the exact source-owned import that creates a package edge.
type SourceImport struct {
	Path   string `json:"path"`
	Line   int    `json:"line"`
	Column int    `json:"column"`
}

type BlastPath struct {
	Dependent string   `json:"dependent"`
	Path      []string `json:"path"`
}

type Hotspot struct {
	Name  string `json:"name"`
	FanIn int    `json:"fan_in"`
}

type FanOutHotspot struct {
	Name   string `json:"name"`
	FanOut int    `json:"fan_out"`
}

type DependencyHotspot struct {
	Name            string `json:"name"`
	FanOut          int    `json:"fan_out"`
	DependencyReach int    `json:"dependency_reach"`
	DependencyDepth int    `json:"dependency_depth"`
}

type BlastHotspot struct {
	Name        string `json:"name"`
	BlastRadius int    `json:"blast_radius"`
	MaxHops     int    `json:"max_hops"`
}

type LateralCutEdge struct {
	Left  string `json:"left"`
	Right string `json:"right"`
}

type LateralCriticalPair struct {
	Left       string           `json:"left"`
	Right      string           `json:"right"`
	Cut        int              `json:"cut"`
	CutEdges   []LateralCutEdge `json:"cut_edges"`
	SourceSide []string         `json:"source_side"`
	SinkSide   []string         `json:"sink_side"`
}

type LateralVertexPairCut struct {
	Left      string   `json:"left"`
	Right     string   `json:"right"`
	Cut       int      `json:"cut"`
	Separator []string `json:"separator"`
}

type LateralBiconnectedBlock struct {
	Tier              int                    `json:"tier"`
	TierName          string                 `json:"tier_name"`
	Members           []string               `json:"members"`
	MemberCount       int                    `json:"member_count"`
	EdgeCount         int                    `json:"edge_count"`
	MinEdgeCut        int                    `json:"min_edge_cut"`
	MinVertexCut      int                    `json:"min_vertex_cut"`
	CriticalSeparator []string               `json:"critical_separator"`
	VertexPairCuts    []LateralVertexPairCut `json:"vertex_pair_cuts"`
	CriticalPairs     []LateralCriticalPair  `json:"critical_pairs"`
	PairCuts          []LateralCriticalPair  `json:"pair_cuts"`
}

type LateralArticulationPoint struct {
	Tier          int        `json:"tier"`
	TierName      string     `json:"tier_name"`
	Name          string     `json:"name"`
	Fragments     [][]string `json:"fragments"`
	FragmentCount int        `json:"fragment_count"`
	CouplingPairs int        `json:"coupling_pairs"`
}

type LateralBridge struct {
	Tier          int      `json:"tier"`
	TierName      string   `json:"tier_name"`
	From          string   `json:"from"`
	To            string   `json:"to"`
	Left          string   `json:"left"`
	Right         string   `json:"right"`
	LeftSide      []string `json:"left_side"`
	RightSide     []string `json:"right_side"`
	CouplingPairs int      `json:"coupling_pairs"`
}

type LateralComponent struct {
	Tier        int      `json:"tier"`
	TierName    string   `json:"tier_name"`
	Members     []string `json:"members"`
	MemberCount int      `json:"member_count"`
	EdgeCount   int      `json:"edge_count"`
}

type DependencyCycle struct {
	Members []string              `json:"members"`
	Edges   []DependencyCycleEdge `json:"edges"`
}

type DependencyCycleEdge struct {
	From string `json:"from"`
	To   string `json:"to"`
}

type ArchitectureEdge struct {
	From         string         `json:"from"`
	FromTier     int            `json:"from_tier"`
	FromTierName string         `json:"from_tier_name"`
	To           string         `json:"to"`
	ToTier       int            `json:"to_tier"`
	ToTierName   string         `json:"to_tier_name"`
	TierDelta    int            `json:"tier_delta"`
	Direction    string         `json:"direction"`
	Sources      []SourceImport `json:"sources,omitempty"`
}

type RootwardLayerSkip struct {
	From         string `json:"from"`
	FromTier     int    `json:"from_tier"`
	FromTierName string `json:"from_tier_name"`
	To           string `json:"to"`
	ToTier       int    `json:"to_tier"`
	ToTierName   string `json:"to_tier_name"`
	TierDistance int    `json:"tier_distance"`
	SkippedTiers int    `json:"skipped_tiers"`
}

type ViolationEdge struct {
	From         string `json:"from"`
	FromTier     int    `json:"from_tier"`
	FromTierName string `json:"from_tier_name"`
	To           string `json:"to"`
	ToTier       int    `json:"to_tier"`
	ToTierName   string `json:"to_tier_name"`
	TierDistance int    `json:"tier_distance"`
}

func (e ViolationEdge) String() string { return e.From + " -> " + e.To }

type SinkCandidate struct {
	Name             string `json:"name"`
	DeclaredTier     int    `json:"declared_tier"`
	DeclaredTierName string `json:"declared_tier_name"`
	ImportFloor      int    `json:"import_floor"`
	ImportFloorName  string `json:"import_floor_name"`
	TierGap          int    `json:"tier_gap"`
}

type Diagnostic struct {
	Kind     string `json:"kind"`
	Leaf     string `json:"leaf"`
	Message  string `json:"message"`
	Recovery string `json:"recovery"`
}

type Report struct {
	Schema                    string                     `json:"schema"`
	Tiers                     []Tier                     `json:"tiers"`
	Leaves                    []Leaf                     `json:"leaves"`
	Hotspots                  []Hotspot                  `json:"hotspots,omitempty"`
	FanOutHotspots            []FanOutHotspot            `json:"fan_out_hotspots,omitempty"`
	DependencyHotspots        []DependencyHotspot        `json:"dependency_hotspots,omitempty"`
	BlastHotspots             []BlastHotspot             `json:"blast_hotspots,omitempty"`
	Edges                     []ArchitectureEdge         `json:"edges,omitempty"`
	DependencyCycles          []DependencyCycle          `json:"dependency_cycles,omitempty"`
	RootwardLayerSkips        []RootwardLayerSkip        `json:"rootward_layer_skips,omitempty"`
	LateralComponents         []LateralComponent         `json:"lateral_components,omitempty"`
	LateralBridges            []LateralBridge            `json:"lateral_bridges,omitempty"`
	LateralArticulationPoints []LateralArticulationPoint `json:"lateral_articulation_points,omitempty"`
	LateralBiconnectedBlocks  []LateralBiconnectedBlock  `json:"lateral_biconnected_blocks,omitempty"`
	Diagnostics               []Diagnostic               `json:"diagnostics,omitempty"`
	SinkCandidates            []SinkCandidate            `json:"sink_candidates,omitempty"`
	Violations                int                        `json:"violations"`
	MaxViolationDistance      int                        `json:"max_violation_distance"`
}

func Analyze(root, onlyLeaf string) (Report, error) {
	tiers, names, err := parseContract(filepath.Join(root, "internal", "architest", "architest_test.go"))
	if err != nil {
		return Report{}, err
	}
	if onlyLeaf != "" {
		if _, ok := tiers[onlyLeaf]; !ok {
			return Report{}, fmt.Errorf("leaf %q has no tier declaration; choose a declared leaf from internal/architest/architest_test.go or add its tier there", onlyLeaf)
		}
	}
	counts := map[int]int{}
	for _, level := range tiers {
		counts[level]++
	}
	report := Report{Schema: "fak-architecture/1"}
	levels := make([]int, 0, len(counts))
	for level := range counts {
		levels = append(levels, level)
	}
	sort.Ints(levels)
	for _, level := range levels {
		report.Tiers = append(report.Tiers, Tier{Level: level, Name: tierName(names, level), Leaves: counts[level]})
	}
	leaves := make([]string, 0, len(tiers))
	for leaf := range tiers {
		leaves = append(leaves, leaf)
	}
	sort.Strings(leaves)
	allLeaves := make([]Leaf, 0, len(leaves))
	byName := make(map[string]int, len(leaves))
	importSources := make(map[string]map[string][]SourceImport, len(leaves))
	for _, name := range leaves {
		dir := filepath.Join(root, "internal", name)
		if _, err := os.Stat(dir); err != nil {
			if os.IsNotExist(err) {
				report.Diagnostics = append(report.Diagnostics, Diagnostic{
					Kind:     DiagnosticStaleTierDeclaration,
					Leaf:     name,
					Message:  fmt.Sprintf("declared package directory %s does not exist", dir),
					Recovery: "create the package or remove its stale tier declaration",
				})
				continue
			}
			return Report{}, fmt.Errorf("inspect declared leaf %q: stat declared package directory %s: %w; restore access and retry", name, dir, err)
		}
		deps, sources, err := internalImports(root, dir)
		if err != nil {
			return Report{}, fmt.Errorf("inspect declared leaf %q: %w", name, err)
		}
		importSources[name] = sources
		declared, floor := tiers[name], 1
		if name == "abi" {
			floor = 0
		}
		var violationEdges []ViolationEdge
		for _, dep := range deps {
			if level, ok := tiers[dep]; ok {
				if level > floor {
					floor = level
				}
				if level > declared {
					violationEdges = append(violationEdges, ViolationEdge{From: name, FromTier: declared, FromTierName: tierName(names, declared), To: dep, ToTier: level, ToTierName: tierName(names, level), TierDistance: level - declared})
				}
			}
		}
		sortViolationEdges(violationEdges)
		violations := violationStrings(violationEdges)
		byName[name] = len(allLeaves)
		allLeaves = append(allLeaves, Leaf{Name: name, DeclaredTier: declared, DeclaredTierName: tierName(names, declared), ImportFloor: floor, ImportFloorName: tierName(names, floor), TierGap: declared - floor, Dependencies: deps, ViolationEdges: violationEdges, Violations: violations})
	}
	for _, leaf := range allLeaves {
		for _, dep := range leaf.Dependencies {
			if _, ok := byName[dep]; !ok {
				continue
			}
			toTier := tiers[dep]
			delta := toTier - leaf.DeclaredTier
			direction := "lateral"
			if delta < 0 {
				direction = "rootward"
			} else if delta > 0 {
				direction = "upward"
			}
			report.Edges = append(report.Edges, ArchitectureEdge{From: leaf.Name, FromTier: leaf.DeclaredTier, FromTierName: leaf.DeclaredTierName, To: dep, ToTier: toTier, ToTierName: tierName(names, toTier), TierDelta: delta, Direction: direction, Sources: importSources[leaf.Name][dep]})
			if direction == "rootward" && -delta > 1 {
				report.RootwardLayerSkips = append(report.RootwardLayerSkips, RootwardLayerSkip{From: leaf.Name, FromTier: leaf.DeclaredTier, FromTierName: leaf.DeclaredTierName, To: dep, ToTier: toTier, ToTierName: tierName(names, toTier), TierDistance: -delta, SkippedTiers: -delta - 1})
			}
		}
	}
	sort.Slice(report.Edges, func(i, j int) bool {
		if report.Edges[i].From != report.Edges[j].From {
			return report.Edges[i].From < report.Edges[j].From
		}
		return report.Edges[i].To < report.Edges[j].To
	})
	sort.Slice(report.RootwardLayerSkips, func(i, j int) bool {
		if report.RootwardLayerSkips[i].SkippedTiers != report.RootwardLayerSkips[j].SkippedTiers {
			return report.RootwardLayerSkips[i].SkippedTiers > report.RootwardLayerSkips[j].SkippedTiers
		}
		if report.RootwardLayerSkips[i].TierDistance != report.RootwardLayerSkips[j].TierDistance {
			return report.RootwardLayerSkips[i].TierDistance > report.RootwardLayerSkips[j].TierDistance
		}
		if report.RootwardLayerSkips[i].From != report.RootwardLayerSkips[j].From {
			return report.RootwardLayerSkips[i].From < report.RootwardLayerSkips[j].From
		}
		return report.RootwardLayerSkips[i].To < report.RootwardLayerSkips[j].To
	})
	report.LateralComponents = lateralComponents(report.Edges, tiers, names)
	report.LateralBridges = lateralBridges(report.Edges, report.LateralComponents)
	report.LateralArticulationPoints = lateralArticulationPoints(report.Edges, report.LateralComponents)
	report.LateralBiconnectedBlocks = lateralBiconnectedBlocks(report.Edges, report.LateralComponents)
	for i := range allLeaves {
		liveViolations := allLeaves[i].ViolationEdges[:0]
		for _, edge := range allLeaves[i].ViolationEdges {
			if _, ok := byName[edge.To]; ok {
				liveViolations = append(liveViolations, edge)
			}
		}
		allLeaves[i].ViolationEdges = liveViolations
		allLeaves[i].Violations = violationStrings(liveViolations)
	}
	for _, importer := range allLeaves {
		for _, dependency := range importer.Dependencies {
			if i, ok := byName[dependency]; ok {
				allLeaves[i].Dependents = append(allLeaves[i].Dependents, importer.Name)
			}
		}
	}
	report.DependencyCycles = dependencyCycles(allLeaves, byName)
	for _, cycle := range report.DependencyCycles {
		for _, member := range cycle.Members {
			allLeaves[byName[member]].DependencyCycle = append([]string(nil), cycle.Members...)
		}
	}
	for i := range allLeaves {
		sort.Strings(allLeaves[i].Dependents)
		allLeaves[i].DependencyPaths = dependencyPaths(allLeaves[i].Name, allLeaves, byName)
		allLeaves[i].DependencyDominators = dependencyDominators(allLeaves[i].Name, allLeaves, byName, allLeaves[i].DependencyPaths)
		allLeaves[i].RedundantDependencies = redundantDependencies(allLeaves[i].Name, allLeaves, byName)
		for j := range allLeaves[i].RedundantDependencies {
			dep := allLeaves[i].RedundantDependencies[j].Dependency
			allLeaves[i].RedundantDependencies[j].Sources = importSources[allLeaves[i].Name][dep]
		}
		allLeaves[i].TransitiveDependencies = make([]string, len(allLeaves[i].DependencyPaths))
		for j, path := range allLeaves[i].DependencyPaths {
			allLeaves[i].TransitiveDependencies[j] = path.Dependency
			if depth := len(path.Path) - 1; depth > allLeaves[i].DependencyDepth {
				allLeaves[i].DependencyDepth = depth
			}
		}
		allLeaves[i].DependencyReach = len(allLeaves[i].DependencyPaths)
		allLeaves[i].BlastPaths = blastPaths(allLeaves[i].Name, allLeaves, byName)
		allLeaves[i].TransitiveDependents = make([]string, len(allLeaves[i].BlastPaths))
		for j, path := range allLeaves[i].BlastPaths {
			allLeaves[i].TransitiveDependents[j] = path.Dependent
		}
		allLeaves[i].BlastRadius = len(allLeaves[i].BlastPaths)
		if len(allLeaves[i].Dependencies) > 0 {
			report.FanOutHotspots = append(report.FanOutHotspots, FanOutHotspot{Name: allLeaves[i].Name, FanOut: len(allLeaves[i].Dependencies)})
		}
		if allLeaves[i].DependencyReach > 0 {
			report.DependencyHotspots = append(report.DependencyHotspots, DependencyHotspot{Name: allLeaves[i].Name, FanOut: len(allLeaves[i].Dependencies), DependencyReach: allLeaves[i].DependencyReach, DependencyDepth: allLeaves[i].DependencyDepth})
		}
		if allLeaves[i].BlastRadius > 0 {
			maxHops := 0
			for _, path := range allLeaves[i].BlastPaths {
				if hops := len(path.Path) - 1; hops > maxHops {
					maxHops = hops
				}
			}
			report.BlastHotspots = append(report.BlastHotspots, BlastHotspot{Name: allLeaves[i].Name, BlastRadius: allLeaves[i].BlastRadius, MaxHops: maxHops})
		}
		if len(allLeaves[i].Dependents) > 0 {
			report.Hotspots = append(report.Hotspots, Hotspot{Name: allLeaves[i].Name, FanIn: len(allLeaves[i].Dependents)})
		}
	}
	for _, leaf := range allLeaves {
		if leaf.DeclaredTier > 1 && leaf.TierGap >= 2 {
			report.SinkCandidates = append(report.SinkCandidates, SinkCandidate{Name: leaf.Name, DeclaredTier: leaf.DeclaredTier, DeclaredTierName: leaf.DeclaredTierName, ImportFloor: leaf.ImportFloor, ImportFloorName: leaf.ImportFloorName, TierGap: leaf.TierGap})
		}
	}
	sort.Slice(report.SinkCandidates, func(i, j int) bool {
		if report.SinkCandidates[i].TierGap != report.SinkCandidates[j].TierGap {
			return report.SinkCandidates[i].TierGap > report.SinkCandidates[j].TierGap
		}
		return report.SinkCandidates[i].Name < report.SinkCandidates[j].Name
	})
	sort.Slice(report.Hotspots, func(i, j int) bool {
		if report.Hotspots[i].FanIn != report.Hotspots[j].FanIn {
			return report.Hotspots[i].FanIn > report.Hotspots[j].FanIn
		}
		return report.Hotspots[i].Name < report.Hotspots[j].Name
	})
	sort.Slice(report.FanOutHotspots, func(i, j int) bool {
		if report.FanOutHotspots[i].FanOut != report.FanOutHotspots[j].FanOut {
			return report.FanOutHotspots[i].FanOut > report.FanOutHotspots[j].FanOut
		}
		return report.FanOutHotspots[i].Name < report.FanOutHotspots[j].Name
	})
	sort.Slice(report.DependencyHotspots, func(i, j int) bool {
		if report.DependencyHotspots[i].DependencyReach != report.DependencyHotspots[j].DependencyReach {
			return report.DependencyHotspots[i].DependencyReach > report.DependencyHotspots[j].DependencyReach
		}
		if report.DependencyHotspots[i].DependencyDepth != report.DependencyHotspots[j].DependencyDepth {
			return report.DependencyHotspots[i].DependencyDepth > report.DependencyHotspots[j].DependencyDepth
		}
		if report.DependencyHotspots[i].FanOut != report.DependencyHotspots[j].FanOut {
			return report.DependencyHotspots[i].FanOut > report.DependencyHotspots[j].FanOut
		}
		return report.DependencyHotspots[i].Name < report.DependencyHotspots[j].Name
	})
	sort.Slice(report.BlastHotspots, func(i, j int) bool {
		if report.BlastHotspots[i].BlastRadius != report.BlastHotspots[j].BlastRadius {
			return report.BlastHotspots[i].BlastRadius > report.BlastHotspots[j].BlastRadius
		}
		if report.BlastHotspots[i].MaxHops != report.BlastHotspots[j].MaxHops {
			return report.BlastHotspots[i].MaxHops > report.BlastHotspots[j].MaxHops
		}
		return report.BlastHotspots[i].Name < report.BlastHotspots[j].Name
	})
	if onlyLeaf == "" {
		report.Leaves = allLeaves
	} else {
		if i, ok := byName[onlyLeaf]; ok {
			report.Leaves = []Leaf{allLeaves[i]}
		}
		report.Hotspots = nil
		report.FanOutHotspots = nil
		report.DependencyHotspots = nil
		report.BlastHotspots = nil
		cycles := report.DependencyCycles[:0]
		for _, cycle := range report.DependencyCycles {
			if slices.Contains(cycle.Members, onlyLeaf) {
				cycles = append(cycles, cycle)
			}
		}
		report.DependencyCycles = cycles
		edges := report.Edges[:0]
		for _, edge := range report.Edges {
			if edge.From == onlyLeaf {
				edges = append(edges, edge)
			}
		}
		report.Edges = edges
		skips := report.RootwardLayerSkips[:0]
		for _, skip := range report.RootwardLayerSkips {
			if skip.From == onlyLeaf {
				skips = append(skips, skip)
			}
		}
		report.RootwardLayerSkips = skips
		blocks := report.LateralBiconnectedBlocks[:0]
		for _, block := range report.LateralBiconnectedBlocks {
			if slices.Contains(block.Members, onlyLeaf) {
				blocks = append(blocks, block)
			}
		}
		report.LateralBiconnectedBlocks = blocks
		points := report.LateralArticulationPoints[:0]
		for _, point := range report.LateralArticulationPoints {
			keep := point.Name == onlyLeaf
			for _, fragment := range point.Fragments {
				keep = keep || slices.Contains(fragment, onlyLeaf)
			}
			if keep {
				points = append(points, point)
			}
		}
		report.LateralArticulationPoints = points
		bridges := report.LateralBridges[:0]
		for _, bridge := range report.LateralBridges {
			if slices.Contains(bridge.LeftSide, onlyLeaf) || slices.Contains(bridge.RightSide, onlyLeaf) {
				bridges = append(bridges, bridge)
			}
		}
		report.LateralBridges = bridges
		components := report.LateralComponents[:0]
		for _, component := range report.LateralComponents {
			if slices.Contains(component.Members, onlyLeaf) {
				components = append(components, component)
			}
		}
		report.LateralComponents = components
		report.SinkCandidates = nil
		diagnostics := report.Diagnostics[:0]
		for _, diagnostic := range report.Diagnostics {
			if diagnostic.Leaf == onlyLeaf {
				diagnostics = append(diagnostics, diagnostic)
			}
		}
		report.Diagnostics = diagnostics
	}
	for _, leaf := range report.Leaves {
		report.Violations += len(leaf.ViolationEdges)
		for _, edge := range leaf.ViolationEdges {
			if edge.TierDistance > report.MaxViolationDistance {
				report.MaxViolationDistance = edge.TierDistance
			}
		}
	}
	return report, nil
}

func blockVertexConnectivity(members []string, edges map[string][2]string) (int, []string, []LateralVertexPairCut) {
	adjacency := map[string]map[string]struct{}{}
	for _, member := range members {
		adjacency[member] = map[string]struct{}{}
	}
	for _, edge := range edges {
		left, right := edge[0], edge[1]
		if _, ok := adjacency[left]; !ok {
			continue
		}
		if _, ok := adjacency[right]; !ok || left == right {
			continue
		}
		adjacency[left][right] = struct{}{}
		adjacency[right][left] = struct{}{}
	}
	minCut := len(members) - 1
	separator := []string(nil)
	for _, member := range members {
		neighbors := make([]string, 0, len(adjacency[member]))
		for neighbor := range adjacency[member] {
			neighbors = append(neighbors, neighbor)
		}
		sort.Strings(neighbors)
		if len(neighbors) < minCut || len(neighbors) == minCut && lexicalStringsLess(neighbors, separator) {
			minCut, separator = len(neighbors), neighbors
		}
	}
	var pairCuts []LateralVertexPairCut
	for i, source := range members {
		for _, sink := range members[i+1:] {
			if _, adjacent := adjacency[source][sink]; adjacent {
				continue
			}
			cut, candidate := unitVertexMaxFlow(source, sink, members, adjacency)
			pairCuts = append(pairCuts, LateralVertexPairCut{Left: source, Right: sink, Cut: cut, Separator: candidate})
			if cut < minCut || cut == minCut && lexicalStringsLess(candidate, separator) {
				minCut, separator = cut, candidate
			}
		}
	}
	return minCut, separator, pairCuts
}

func lexicalStringsLess(left, right []string) bool {
	if right == nil {
		return true
	}
	return strings.Join(left, "\x00") < strings.Join(right, "\x00")
}

func unitVertexMaxFlow(source, sink string, members []string, adjacency map[string]map[string]struct{}) (int, []string) {
	const inSuffix, outSuffix = "\x00in", "\x00out"
	inNode := func(member string) string { return member + inSuffix }
	outNode := func(member string) string { return member + outSuffix }
	capacity := map[string]map[string]int{}
	addCapacity := func(from, to string, amount int) {
		if capacity[from] == nil {
			capacity[from] = map[string]int{}
		}
		if capacity[to] == nil {
			capacity[to] = map[string]int{}
		}
		capacity[from][to] += amount
	}
	infinity := len(members) + 1
	for _, member := range members {
		amount := 1
		if member == source || member == sink {
			amount = infinity
		}
		addCapacity(inNode(member), outNode(member), amount)
	}
	for _, left := range members {
		neighbors := make([]string, 0, len(adjacency[left]))
		for right := range adjacency[left] {
			neighbors = append(neighbors, right)
		}
		sort.Strings(neighbors)
		for _, right := range neighbors {
			addCapacity(outNode(left), inNode(right), infinity)
		}
	}
	start, target := outNode(source), inNode(sink)
	flow := 0
	for {
		parent := map[string]string{start: ""}
		queue := []string{start}
		for len(queue) > 0 {
			current := queue[0]
			queue = queue[1:]
			neighbors := make([]string, 0, len(capacity[current]))
			for next, residual := range capacity[current] {
				if residual > 0 {
					neighbors = append(neighbors, next)
				}
			}
			sort.Strings(neighbors)
			for _, next := range neighbors {
				if _, seen := parent[next]; seen {
					continue
				}
				parent[next] = current
				queue = append(queue, next)
			}
		}
		if _, found := parent[target]; !found {
			break
		}
		bottleneck := infinity
		for node := target; node != start; node = parent[node] {
			if residual := capacity[parent[node]][node]; residual < bottleneck {
				bottleneck = residual
			}
		}
		for node := target; node != start; node = parent[node] {
			previous := parent[node]
			capacity[previous][node] -= bottleneck
			capacity[node][previous] += bottleneck
		}
		flow += bottleneck
	}
	reachable := map[string]struct{}{start: {}}
	queue := []string{start}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		neighbors := make([]string, 0, len(capacity[current]))
		for next, residual := range capacity[current] {
			if residual > 0 {
				neighbors = append(neighbors, next)
			}
		}
		sort.Strings(neighbors)
		for _, next := range neighbors {
			if _, seen := reachable[next]; seen {
				continue
			}
			reachable[next] = struct{}{}
			queue = append(queue, next)
		}
	}
	separator := make([]string, 0, flow)
	for _, member := range members {
		if member == source || member == sink {
			continue
		}
		_, inReachable := reachable[inNode(member)]
		_, outReachable := reachable[outNode(member)]
		if inReachable && !outReachable {
			separator = append(separator, member)
		}
	}
	return flow, separator
}

func blockEdgeConnectivity(members []string, edges map[string][2]string) (int, []LateralCriticalPair, []LateralCriticalPair) {
	memberSet := map[string]struct{}{}
	for _, member := range members {
		memberSet[member] = struct{}{}
	}
	minCut := -1
	var pairs []LateralCriticalPair
	var allPairs []LateralCriticalPair
	for i := 0; i < len(members); i++ {
		for j := i + 1; j < len(members); j++ {
			cut, cutEdges, sourceSide, sinkSide := unitEdgeMaxFlow(members[i], members[j], edges, memberSet)
			pair := LateralCriticalPair{Left: members[i], Right: members[j], Cut: cut, CutEdges: cutEdges, SourceSide: sourceSide, SinkSide: sinkSide}
			allPairs = append(allPairs, pair)
			if minCut < 0 || cut < minCut {
				minCut = cut
				pairs = nil
			}
			if cut == minCut {
				pairs = append(pairs, pair)
			}
		}
	}
	return minCut, pairs, allPairs
}
func unitEdgeMaxFlow(source, sink string, edges map[string][2]string, members map[string]struct{}) (int, []LateralCutEdge, []string, []string) {
	unique := map[string][2]string{}
	for _, edge := range edges {
		left, right := edge[0], edge[1]
		if right < left {
			left, right = right, left
		}
		if _, ok := members[left]; !ok {
			continue
		}
		if _, ok := members[right]; !ok {
			continue
		}
		unique[left+"\x00"+right] = [2]string{left, right}
	}
	capacity := map[string]map[string]int{}
	for _, edge := range unique {
		u, v := edge[0], edge[1]
		if capacity[u] == nil {
			capacity[u] = map[string]int{}
		}
		if capacity[v] == nil {
			capacity[v] = map[string]int{}
		}
		capacity[u][v] = 1
		capacity[v][u] = 1
	}
	flow := 0
	for {
		parent := map[string]string{source: ""}
		queue := []string{source}
		for len(queue) > 0 {
			u := queue[0]
			queue = queue[1:]
			neighbors := make([]string, 0, len(capacity[u]))
			for v, c := range capacity[u] {
				if c > 0 {
					neighbors = append(neighbors, v)
				}
			}
			sort.Strings(neighbors)
			for _, v := range neighbors {
				if _, ok := parent[v]; ok {
					continue
				}
				parent[v] = u
				queue = append(queue, v)
				if v == sink {
					break
				}
			}
			if _, ok := parent[sink]; ok {
				break
			}
		}
		if _, ok := parent[sink]; !ok {
			break
		}
		for v := sink; v != source; {
			u := parent[v]
			capacity[u][v]--
			capacity[v][u]++
			v = u
		}
		flow++
	}
	reachable := map[string]struct{}{source: {}}
	queue := []string{source}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		neighbors := make([]string, 0, len(capacity[u]))
		for v, residual := range capacity[u] {
			if residual > 0 {
				neighbors = append(neighbors, v)
			}
		}
		sort.Strings(neighbors)
		for _, v := range neighbors {
			if _, seen := reachable[v]; seen {
				continue
			}
			reachable[v] = struct{}{}
			queue = append(queue, v)
		}
	}
	keys := make([]string, 0, len(unique))
	for key := range unique {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	cutEdges := make([]LateralCutEdge, 0, flow)
	for _, key := range keys {
		edge := unique[key]
		_, leftReachable := reachable[edge[0]]
		_, rightReachable := reachable[edge[1]]
		if leftReachable != rightReachable {
			left, right := edge[0], edge[1]
			if right < left {
				left, right = right, left
			}
			cutEdges = append(cutEdges, LateralCutEdge{Left: left, Right: right})
		}
	}
	sort.Slice(cutEdges, func(i, j int) bool {
		if cutEdges[i].Left != cutEdges[j].Left {
			return cutEdges[i].Left < cutEdges[j].Left
		}
		return cutEdges[i].Right < cutEdges[j].Right
	})
	sourceSide, sinkSide := make([]string, 0, len(reachable)), make([]string, 0, len(members)-len(reachable))
	for member := range members {
		if _, ok := reachable[member]; ok {
			sourceSide = append(sourceSide, member)
		} else {
			sinkSide = append(sinkSide, member)
		}
	}
	sort.Strings(sourceSide)
	sort.Strings(sinkSide)
	return flow, cutEdges, sourceSide, sinkSide
}

func lateralBiconnectedBlocks(edges []ArchitectureEdge, components []LateralComponent) []LateralBiconnectedBlock {
	adjacency := map[string][]string{}
	undirected := map[string][2]string{}
	for _, edge := range edges {
		if edge.Direction != "lateral" {
			continue
		}
		left, right := edge.From, edge.To
		if right < left {
			left, right = right, left
		}
		key := left + "\x00" + right
		if _, ok := undirected[key]; ok {
			continue
		}
		undirected[key] = [2]string{left, right}
		adjacency[left] = append(adjacency[left], right)
		adjacency[right] = append(adjacency[right], left)
	}
	for name := range adjacency {
		sort.Strings(adjacency[name])
	}
	disc, low, parent := map[string]int{}, map[string]int{}, map[string]string{}
	time := 0
	var stack [][2]string
	var memberSets []map[string]struct{}
	var visit func(string)
	visit = func(u string) {
		time++
		disc[u], low[u] = time, time
		for _, v := range adjacency[u] {
			left, right := u, v
			if right < left {
				left, right = right, left
			}
			e := [2]string{left, right}
			if disc[v] == 0 {
				parent[v] = u
				stack = append(stack, e)
				visit(v)
				if low[v] < low[u] {
					low[u] = low[v]
				}
				if low[v] >= disc[u] {
					members := map[string]struct{}{}
					for len(stack) > 0 {
						last := stack[len(stack)-1]
						stack = stack[:len(stack)-1]
						members[last[0]] = struct{}{}
						members[last[1]] = struct{}{}
						if last == e {
							break
						}
					}
					if len(members) >= 3 {
						memberSets = append(memberSets, members)
					}
				}
			} else if parent[u] != v && disc[v] < disc[u] {
				stack = append(stack, e)
				if disc[v] < low[u] {
					low[u] = disc[v]
				}
			}
		}
	}
	starts := make([]string, 0, len(adjacency))
	for name := range adjacency {
		starts = append(starts, name)
	}
	sort.Strings(starts)
	for _, start := range starts {
		if disc[start] == 0 {
			visit(start)
		}
	}
	tierByMember := map[string]struct {
		level int
		name  string
	}{}
	for _, component := range components {
		for _, member := range component.Members {
			tierByMember[member] = struct {
				level int
				name  string
			}{component.Tier, component.TierName}
		}
	}
	var out []LateralBiconnectedBlock
	for _, set := range memberSets {
		members := make([]string, 0, len(set))
		for member := range set {
			members = append(members, member)
		}
		sort.Strings(members)
		edgeCount := 0
		for _, e := range undirected {
			_, a := set[e[0]]
			_, b := set[e[1]]
			if a && b {
				edgeCount++
			}
		}
		tier := tierByMember[members[0]]
		minCut, criticalPairs, pairCuts := blockEdgeConnectivity(members, undirected)
		minVertexCut, separator, vertexPairCuts := blockVertexConnectivity(members, undirected)
		out = append(out, LateralBiconnectedBlock{
			Tier: tier.level, TierName: tier.name, Members: members, MemberCount: len(members), EdgeCount: edgeCount,
			MinEdgeCut: minCut, MinVertexCut: minVertexCut, CriticalSeparator: separator, VertexPairCuts: vertexPairCuts,
			CriticalPairs: criticalPairs, PairCuts: pairCuts,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].MemberCount != out[j].MemberCount {
			return out[i].MemberCount > out[j].MemberCount
		}
		if out[i].EdgeCount != out[j].EdgeCount {
			return out[i].EdgeCount > out[j].EdgeCount
		}
		if out[i].MinEdgeCut != out[j].MinEdgeCut {
			return out[i].MinEdgeCut > out[j].MinEdgeCut
		}
		if out[i].Tier != out[j].Tier {
			return out[i].Tier < out[j].Tier
		}
		return strings.Join(out[i].Members, "\x00") < strings.Join(out[j].Members, "\x00")
	})
	return out
}

func lateralArticulationPoints(edges []ArchitectureEdge, components []LateralComponent) []LateralArticulationPoint {
	var out []LateralArticulationPoint
	for _, component := range components {
		members := map[string]struct{}{}
		for _, member := range component.Members {
			members[member] = struct{}{}
		}
		for _, removed := range component.Members {
			remaining := make(map[string]struct{}, len(members)-1)
			for member := range members {
				if member != removed {
					remaining[member] = struct{}{}
				}
			}
			var fragments [][]string
			seen := map[string]struct{}{}
			for _, start := range component.Members {
				if start == removed {
					continue
				}
				if _, ok := seen[start]; ok {
					continue
				}
				fragment := reachableWithoutVertex(start, removed, edges, remaining)
				for _, member := range fragment {
					seen[member] = struct{}{}
				}
				fragments = append(fragments, fragment)
			}
			if len(fragments) <= 1 {
				continue
			}
			sort.Slice(fragments, func(i, j int) bool { return strings.Join(fragments[i], "\x00") < strings.Join(fragments[j], "\x00") })
			pairs := 0
			for i := 0; i < len(fragments); i++ {
				for j := i + 1; j < len(fragments); j++ {
					pairs += len(fragments[i]) * len(fragments[j])
				}
			}
			out = append(out, LateralArticulationPoint{Tier: component.Tier, TierName: component.TierName, Name: removed, Fragments: fragments, FragmentCount: len(fragments), CouplingPairs: pairs})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CouplingPairs != out[j].CouplingPairs {
			return out[i].CouplingPairs > out[j].CouplingPairs
		}
		if out[i].FragmentCount != out[j].FragmentCount {
			return out[i].FragmentCount > out[j].FragmentCount
		}
		if out[i].Tier != out[j].Tier {
			return out[i].Tier < out[j].Tier
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func reachableWithoutVertex(start, removed string, edges []ArchitectureEdge, members map[string]struct{}) []string {
	seen := map[string]struct{}{start: {}}
	pending := []string{start}
	for len(pending) > 0 {
		current := pending[0]
		pending = pending[1:]
		var next []string
		for _, edge := range edges {
			if edge.Direction != "lateral" || edge.From == removed || edge.To == removed {
				continue
			}
			neighbor := ""
			if edge.From == current {
				neighbor = edge.To
			} else if edge.To == current {
				neighbor = edge.From
			}
			if neighbor == "" {
				continue
			}
			if _, ok := members[neighbor]; !ok {
				continue
			}
			if _, ok := seen[neighbor]; !ok {
				seen[neighbor] = struct{}{}
				next = append(next, neighbor)
			}
		}
		sort.Strings(next)
		pending = append(pending, next...)
	}
	out := make([]string, 0, len(seen))
	for member := range seen {
		out = append(out, member)
	}
	sort.Strings(out)
	return out
}

func lateralBridges(edges []ArchitectureEdge, components []LateralComponent) []LateralBridge {
	var out []LateralBridge
	for _, component := range components {
		members := map[string]struct{}{}
		for _, member := range component.Members {
			members[member] = struct{}{}
		}
		for _, edge := range edges {
			if edge.Direction != "lateral" {
				continue
			}
			if _, ok := members[edge.From]; !ok {
				continue
			}
			if _, ok := members[edge.To]; !ok {
				continue
			}
			left, right := edge.From, edge.To
			if right < left {
				left, right = right, left
			}
			leftSide := reachableWithoutEdge(left, left, right, edges, members)
			if slices.Contains(leftSide, right) {
				continue
			}
			leftSet := map[string]struct{}{}
			for _, member := range leftSide {
				leftSet[member] = struct{}{}
			}
			var rightSide []string
			for _, member := range component.Members {
				if _, ok := leftSet[member]; !ok {
					rightSide = append(rightSide, member)
				}
			}
			out = append(out, LateralBridge{Tier: component.Tier, TierName: component.TierName, From: edge.From, To: edge.To, Left: left, Right: right, LeftSide: leftSide, RightSide: rightSide, CouplingPairs: len(leftSide) * len(rightSide)})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CouplingPairs != out[j].CouplingPairs {
			return out[i].CouplingPairs > out[j].CouplingPairs
		}
		if out[i].Tier != out[j].Tier {
			return out[i].Tier < out[j].Tier
		}
		if out[i].Left != out[j].Left {
			return out[i].Left < out[j].Left
		}
		return out[i].Right < out[j].Right
	})
	return out
}

func reachableWithoutEdge(start, skipLeft, skipRight string, edges []ArchitectureEdge, members map[string]struct{}) []string {
	seen := map[string]struct{}{start: {}}
	pending := []string{start}
	for len(pending) > 0 {
		current := pending[0]
		pending = pending[1:]
		var next []string
		for _, edge := range edges {
			if edge.Direction != "lateral" {
				continue
			}
			left, right := edge.From, edge.To
			if right < left {
				left, right = right, left
			}
			if left == skipLeft && right == skipRight {
				continue
			}
			neighbor := ""
			if edge.From == current {
				neighbor = edge.To
			} else if edge.To == current {
				neighbor = edge.From
			}
			if neighbor == "" {
				continue
			}
			if _, ok := members[neighbor]; !ok {
				continue
			}
			if _, ok := seen[neighbor]; !ok {
				seen[neighbor] = struct{}{}
				next = append(next, neighbor)
			}
		}
		sort.Strings(next)
		pending = append(pending, next...)
	}
	out := make([]string, 0, len(seen))
	for member := range seen {
		out = append(out, member)
	}
	sort.Strings(out)
	return out
}

func lateralComponents(edges []ArchitectureEdge, tiers map[string]int, names []string) []LateralComponent {
	adjacency := map[string]map[string]struct{}{}
	for _, edge := range edges {
		if edge.Direction != "lateral" {
			continue
		}
		if adjacency[edge.From] == nil {
			adjacency[edge.From] = map[string]struct{}{}
		}
		if adjacency[edge.To] == nil {
			adjacency[edge.To] = map[string]struct{}{}
		}
		adjacency[edge.From][edge.To] = struct{}{}
		adjacency[edge.To][edge.From] = struct{}{}
	}
	seen := map[string]struct{}{}
	var out []LateralComponent
	starts := make([]string, 0, len(adjacency))
	for name := range adjacency {
		starts = append(starts, name)
	}
	sort.Strings(starts)
	for _, start := range starts {
		if _, ok := seen[start]; ok {
			continue
		}
		pending, members := []string{start}, []string{}
		seen[start] = struct{}{}
		for len(pending) > 0 {
			current := pending[0]
			pending = pending[1:]
			members = append(members, current)
			next := make([]string, 0, len(adjacency[current]))
			for neighbor := range adjacency[current] {
				next = append(next, neighbor)
			}
			sort.Strings(next)
			for _, neighbor := range next {
				if _, ok := seen[neighbor]; !ok {
					seen[neighbor] = struct{}{}
					pending = append(pending, neighbor)
				}
			}
		}
		sort.Strings(members)
		memberSet := map[string]struct{}{}
		for _, member := range members {
			memberSet[member] = struct{}{}
		}
		edgeCount := 0
		for _, edge := range edges {
			if edge.Direction == "lateral" {
				_, from := memberSet[edge.From]
				_, to := memberSet[edge.To]
				if from && to {
					edgeCount++
				}
			}
		}
		tier := tiers[start]
		out = append(out, LateralComponent{Tier: tier, TierName: tierName(names, tier), Members: members, MemberCount: len(members), EdgeCount: edgeCount})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].MemberCount != out[j].MemberCount {
			return out[i].MemberCount > out[j].MemberCount
		}
		if out[i].EdgeCount != out[j].EdgeCount {
			return out[i].EdgeCount > out[j].EdgeCount
		}
		if out[i].Tier != out[j].Tier {
			return out[i].Tier < out[j].Tier
		}
		return strings.Join(out[i].Members, "\x00") < strings.Join(out[j].Members, "\x00")
	})
	return out
}

func dependencyCycles(leaves []Leaf, byName map[string]int) []DependencyCycle {
	index := 0
	indices, lowlink := map[string]int{}, map[string]int{}
	onStack := map[string]bool{}
	var stack []string
	var components [][]string
	var visit func(string)
	visit = func(node string) {
		indices[node], lowlink[node] = index, index
		index++
		stack = append(stack, node)
		onStack[node] = true
		dependencies := append([]string(nil), leaves[byName[node]].Dependencies...)
		sort.Strings(dependencies)
		for _, dependency := range dependencies {
			if _, ok := byName[dependency]; !ok {
				continue
			}
			if _, seen := indices[dependency]; !seen {
				visit(dependency)
				lowlink[node] = min(lowlink[node], lowlink[dependency])
			} else if onStack[dependency] {
				lowlink[node] = min(lowlink[node], indices[dependency])
			}
		}
		if lowlink[node] != indices[node] {
			return
		}
		var component []string
		for {
			last := len(stack) - 1
			member := stack[last]
			stack = stack[:last]
			onStack[member] = false
			component = append(component, member)
			if member == node {
				break
			}
		}
		sort.Strings(component)
		components = append(components, component)
	}
	names := make([]string, 0, len(byName))
	for name := range byName {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if _, seen := indices[name]; !seen {
			visit(name)
		}
	}
	var out []DependencyCycle
	for _, members := range components {
		memberSet := map[string]struct{}{}
		for _, member := range members {
			memberSet[member] = struct{}{}
		}
		var edges []DependencyCycleEdge
		for _, from := range members {
			for _, to := range leaves[byName[from]].Dependencies {
				if _, ok := memberSet[to]; ok {
					edges = append(edges, DependencyCycleEdge{From: from, To: to})
				}
			}
		}
		sort.Slice(edges, func(i, j int) bool {
			if edges[i].From != edges[j].From {
				return edges[i].From < edges[j].From
			}
			return edges[i].To < edges[j].To
		})
		if len(members) == 1 && len(edges) == 0 {
			continue
		}
		out = append(out, DependencyCycle{Members: members, Edges: edges})
	}
	sort.Slice(out, func(i, j int) bool { return lexicalStringsLess(out[i].Members, out[j].Members) })
	return out
}

func redundantDependencies(source string, leaves []Leaf, byName map[string]int) []RedundantDependency {
	dependencies := append([]string(nil), leaves[byName[source]].Dependencies...)
	sort.Strings(dependencies)
	var out []RedundantDependency
	for _, dependency := range dependencies {
		if path := alternateDependencyPath(source, dependency, leaves, byName); len(path) > 0 {
			out = append(out, RedundantDependency{Dependency: dependency, AlternatePath: path})
		}
	}
	return out
}

func alternateDependencyPath(source, destination string, leaves []Leaf, byName map[string]int) []string {
	paths := map[string][]string{source: {source}}
	pending := []string{source}
	for len(pending) > 0 {
		current := pending[0]
		pending = pending[1:]
		dependencies := append([]string(nil), leaves[byName[current]].Dependencies...)
		sort.Strings(dependencies)
		for _, dependency := range dependencies {
			if current == source && dependency == destination {
				continue
			}
			if _, seen := paths[dependency]; seen {
				continue
			}
			paths[dependency] = append(append([]string(nil), paths[current]...), dependency)
			if dependency == destination {
				return paths[dependency]
			}
			pending = append(pending, dependency)
		}
	}
	return nil
}

func dependencyDominators(source string, leaves []Leaf, byName map[string]int, paths []DependencyPath) []DependencyDominator {
	reachable := map[string]struct{}{source: {}}
	for _, path := range paths {
		reachable[path.Dependency] = struct{}{}
	}
	predecessors := map[string][]string{}
	for node := range reachable {
		for _, dependency := range leaves[byName[node]].Dependencies {
			if _, ok := reachable[dependency]; ok {
				predecessors[dependency] = append(predecessors[dependency], node)
			}
		}
	}
	for node := range predecessors {
		sort.Strings(predecessors[node])
	}
	all := map[string]struct{}{}
	for node := range reachable {
		all[node] = struct{}{}
	}
	dominators := map[string]map[string]struct{}{source: {source: {}}}
	nodes := make([]string, 0, len(reachable))
	for node := range reachable {
		if node != source {
			nodes = append(nodes, node)
		}
	}
	sort.Strings(nodes)
	for _, node := range nodes {
		dominators[node] = cloneStringSet(all)
	}
	changed := true
	for changed {
		changed = false
		for _, node := range nodes {
			preds := predecessors[node]
			var next map[string]struct{}
			for _, pred := range preds {
				if next == nil {
					next = cloneStringSet(dominators[pred])
				} else {
					for candidate := range next {
						if _, ok := dominators[pred][candidate]; !ok {
							delete(next, candidate)
						}
					}
				}
			}
			if next == nil {
				next = map[string]struct{}{}
			}
			next[node] = struct{}{}
			if !stringSetsEqual(next, dominators[node]) {
				dominators[node] = next
				changed = true
			}
		}
	}
	pathByDependency := map[string][]string{}
	for _, path := range paths {
		pathByDependency[path.Dependency] = path.Path
	}
	var out []DependencyDominator
	for _, dependency := range nodes {
		var strict []string
		for candidate := range dominators[dependency] {
			if candidate != source && candidate != dependency {
				strict = append(strict, candidate)
			}
		}
		if len(strict) == 0 {
			continue
		}
		sort.Slice(strict, func(i, j int) bool { return dominatorOrder(strict[i], strict[j], dominators) })
		out = append(out, DependencyDominator{Dependency: dependency, Dominators: strict, Path: append([]string(nil), pathByDependency[dependency]...)})
	}
	return out
}
func dependencyPaths(name string, leaves []Leaf, byName map[string]int) []DependencyPath {
	paths := map[string][]string{name: {name}}
	pending := []string{name}
	for len(pending) > 0 {
		current := pending[0]
		pending = pending[1:]
		dependencies := append([]string(nil), leaves[byName[current]].Dependencies...)
		sort.Strings(dependencies)
		for _, dependency := range dependencies {
			candidate := append(append([]string(nil), paths[current]...), dependency)
			existing, seen := paths[dependency]
			if seen && (len(existing) < len(candidate) || len(existing) == len(candidate) && strings.Join(existing, "\x00") <= strings.Join(candidate, "\x00")) {
				continue
			}
			paths[dependency] = candidate
			pending = append(pending, dependency)
		}
	}
	delete(paths, name)
	dependencies := make([]string, 0, len(paths))
	for dependency := range paths {
		dependencies = append(dependencies, dependency)
	}
	sort.Strings(dependencies)
	out := make([]DependencyPath, 0, len(dependencies))
	for _, dependency := range dependencies {
		out = append(out, DependencyPath{Dependency: dependency, Path: paths[dependency]})
	}
	return out
}

func blastPaths(name string, leaves []Leaf, byName map[string]int) []BlastPath {
	paths := map[string][]string{name: {name}}
	pending := []string{name}
	for len(pending) > 0 {
		current := pending[0]
		pending = pending[1:]
		dependents := append([]string(nil), leaves[byName[current]].Dependents...)
		sort.Strings(dependents)
		for _, dependent := range dependents {
			if _, ok := paths[dependent]; ok {
				continue
			}
			paths[dependent] = append(append([]string(nil), paths[current]...), dependent)
			pending = append(pending, dependent)
		}
	}
	delete(paths, name)
	dependents := make([]string, 0, len(paths))
	for dependent := range paths {
		dependents = append(dependents, dependent)
	}
	sort.Strings(dependents)
	out := make([]BlastPath, 0, len(dependents))
	for _, dependent := range dependents {
		out = append(out, BlastPath{Dependent: dependent, Path: paths[dependent]})
	}
	return out
}

func (r Report) JSON() ([]byte, error) { return json.MarshalIndent(r, "", "  ") }
