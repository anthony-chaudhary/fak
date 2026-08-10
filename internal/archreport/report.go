package archreport

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
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
	Name                 string          `json:"name"`
	DeclaredTier         int             `json:"declared_tier"`
	DeclaredTierName     string          `json:"declared_tier_name"`
	ImportFloor          int             `json:"import_floor"`
	ImportFloorName      string          `json:"import_floor_name"`
	TierGap              int             `json:"tier_gap"`
	Dependencies         []string        `json:"dependencies"`
	Dependents           []string        `json:"dependents,omitempty"`
	TransitiveDependents []string        `json:"transitive_dependents"`
	BlastRadius          int             `json:"blast_radius"`
	BlastPaths           []BlastPath     `json:"blast_paths"`
	ViolationEdges       []ViolationEdge `json:"violation_edges,omitempty"`
	Violations           []string        `json:"violations,omitempty"` // Compatibility projection; use ViolationEdges.
}

type BlastPath struct {
	Dependent string   `json:"dependent"`
	Path      []string `json:"path"`
}

type Hotspot struct {
	Name  string `json:"name"`
	FanIn int    `json:"fan_in"`
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

type LateralBiconnectedBlock struct {
	Tier          int                   `json:"tier"`
	TierName      string                `json:"tier_name"`
	Members       []string              `json:"members"`
	MemberCount   int                   `json:"member_count"`
	EdgeCount     int                   `json:"edge_count"`
	MinEdgeCut    int                   `json:"min_edge_cut"`
	CriticalPairs []LateralCriticalPair `json:"critical_pairs"`
	PairCuts      []LateralCriticalPair `json:"pair_cuts"`
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

type ArchitectureEdge struct {
	From         string `json:"from"`
	FromTier     int    `json:"from_tier"`
	FromTierName string `json:"from_tier_name"`
	To           string `json:"to"`
	ToTier       int    `json:"to_tier"`
	ToTierName   string `json:"to_tier_name"`
	TierDelta    int    `json:"tier_delta"`
	Direction    string `json:"direction"`
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
	BlastHotspots             []BlastHotspot             `json:"blast_hotspots,omitempty"`
	Edges                     []ArchitectureEdge         `json:"edges,omitempty"`
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
		deps, err := internalImports(dir)
		if err != nil {
			return Report{}, fmt.Errorf("inspect declared leaf %q: %w", name, err)
		}
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
			report.Edges = append(report.Edges, ArchitectureEdge{From: leaf.Name, FromTier: leaf.DeclaredTier, FromTierName: leaf.DeclaredTierName, To: dep, ToTier: toTier, ToTierName: tierName(names, toTier), TierDelta: delta, Direction: direction})
		}
	}
	sort.Slice(report.Edges, func(i, j int) bool {
		if report.Edges[i].From != report.Edges[j].From {
			return report.Edges[i].From < report.Edges[j].From
		}
		return report.Edges[i].To < report.Edges[j].To
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
	for i := range allLeaves {
		sort.Strings(allLeaves[i].Dependents)
		allLeaves[i].BlastPaths = blastPaths(allLeaves[i].Name, allLeaves, byName)
		allLeaves[i].TransitiveDependents = make([]string, len(allLeaves[i].BlastPaths))
		for j, path := range allLeaves[i].BlastPaths {
			allLeaves[i].TransitiveDependents[j] = path.Dependent
		}
		allLeaves[i].BlastRadius = len(allLeaves[i].BlastPaths)
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
		report.BlastHotspots = nil
		edges := report.Edges[:0]
		for _, edge := range report.Edges {
			if edge.From == onlyLeaf {
				edges = append(edges, edge)
			}
		}
		report.Edges = edges
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
		out = append(out, LateralBiconnectedBlock{Tier: tier.level, TierName: tier.name, Members: members, MemberCount: len(members), EdgeCount: edgeCount, MinEdgeCut: minCut, CriticalPairs: criticalPairs, PairCuts: pairCuts})
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

func parseContract(path string) (map[string]int, []string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return nil, nil, fmt.Errorf("parse architecture contract %s: %w; repair the Go syntax before reporting", path, err)
	}
	tiers := map[string]int{}
	var names []string
	ast.Inspect(f, func(n ast.Node) bool {
		vs, ok := n.(*ast.ValueSpec)
		if !ok || len(vs.Names) != 1 || len(vs.Values) != 1 {
			return true
		}
		lit, ok := vs.Values[0].(*ast.CompositeLit)
		if !ok {
			return true
		}
		switch vs.Names[0].Name {
		case "tier":
			for _, e := range lit.Elts {
				kv, ok := e.(*ast.KeyValueExpr)
				if !ok {
					continue
				}
				k, ok1 := kv.Key.(*ast.BasicLit)
				v, ok2 := kv.Value.(*ast.BasicLit)
				if !ok1 || !ok2 {
					continue
				}
				name, e1 := strconv.Unquote(k.Value)
				level, e2 := strconv.Atoi(v.Value)
				if e1 == nil && e2 == nil {
					tiers[name] = level
				}
			}
		case "tierName":
			for _, e := range lit.Elts {
				b, ok := e.(*ast.BasicLit)
				if ok {
					if name, e := strconv.Unquote(b.Value); e == nil {
						names = append(names, name)
					}
				}
			}
		}
		return true
	})
	if len(tiers) == 0 || len(names) == 0 {
		return nil, nil, fmt.Errorf("architecture contract missing tier or tierName in %s; restore both declarations", path)
	}
	return tiers, names, nil
}
func internalImports(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("declared package directory %s does not exist; create the package or remove its stale tier declaration", dir)
		}
		return nil, fmt.Errorf("read declared package directory %s: %w; restore read access and retry", dir, err)
	}
	set := map[string]bool{}
	fset := token.NewFileSet()
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, e.Name()), nil, parser.ImportsOnly)
		if err != nil {
			return nil, fmt.Errorf("parse imports in %s: %w; repair the Go syntax before reporting", filepath.Join(dir, e.Name()), err)
		}
		for _, imp := range f.Imports {
			p, err := strconv.Unquote(imp.Path.Value)
			if err == nil && strings.HasPrefix(p, modulePrefix) {
				leaf := strings.SplitN(strings.TrimPrefix(p, modulePrefix), "/", 2)[0]
				set[leaf] = true
			}
		}
	}
	out := make([]string, 0, len(set))
	for d := range set {
		out = append(out, d)
	}
	sort.Strings(out)
	return out, nil
}
func tierName(names []string, level int) string {
	if level >= 0 && level < len(names) {
		return names[level]
	}
	return "unknown"
}

func sortViolationEdges(edges []ViolationEdge) {
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].TierDistance != edges[j].TierDistance {
			return edges[i].TierDistance > edges[j].TierDistance
		}
		if edges[i].From != edges[j].From {
			return edges[i].From < edges[j].From
		}
		return edges[i].To < edges[j].To
	})
}

func violationStrings(edges []ViolationEdge) []string {
	if len(edges) == 0 {
		return nil
	}
	out := make([]string, len(edges))
	for i, edge := range edges {
		out[i] = edge.String()
	}
	return out
}
