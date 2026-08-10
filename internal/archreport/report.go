package archreport

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
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
	Name             string          `json:"name"`
	DeclaredTier     int             `json:"declared_tier"`
	DeclaredTierName string          `json:"declared_tier_name"`
	ImportFloor      int             `json:"import_floor"`
	ImportFloorName  string          `json:"import_floor_name"`
	TierGap          int             `json:"tier_gap"`
	Dependencies     []string        `json:"dependencies"`
	Dependents       []string        `json:"dependents,omitempty"`
	ViolationEdges   []ViolationEdge `json:"violation_edges,omitempty"`
	Violations       []string        `json:"violations,omitempty"` // Compatibility projection; use ViolationEdges.
}

type Hotspot struct {
	Name  string `json:"name"`
	FanIn int    `json:"fan_in"`
}

type ViolationEdge struct {
	From         string `json:"from"`
	FromTier     int    `json:"from_tier"`
	FromTierName string `json:"from_tier_name"`
	To           string `json:"to"`
	ToTier       int    `json:"to_tier"`
	ToTierName   string `json:"to_tier_name"`
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
	Schema         string          `json:"schema"`
	Tiers          []Tier          `json:"tiers"`
	Leaves         []Leaf          `json:"leaves"`
	Hotspots       []Hotspot       `json:"hotspots,omitempty"`
	Diagnostics    []Diagnostic    `json:"diagnostics,omitempty"`
	SinkCandidates []SinkCandidate `json:"sink_candidates,omitempty"`
	Violations     int             `json:"violations"`
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
					violationEdges = append(violationEdges, ViolationEdge{From: name, FromTier: declared, FromTierName: tierName(names, declared), To: dep, ToTier: level, ToTierName: tierName(names, level)})
				}
			}
		}
		sortViolationEdges(violationEdges)
		violations := violationStrings(violationEdges)
		byName[name] = len(allLeaves)
		allLeaves = append(allLeaves, Leaf{Name: name, DeclaredTier: declared, DeclaredTierName: tierName(names, declared), ImportFloor: floor, ImportFloorName: tierName(names, floor), TierGap: declared - floor, Dependencies: deps, ViolationEdges: violationEdges, Violations: violations})
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
	if onlyLeaf == "" {
		report.Leaves = allLeaves
	} else {
		if i, ok := byName[onlyLeaf]; ok {
			report.Leaves = []Leaf{allLeaves[i]}
		}
		report.Hotspots = nil
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
	}
	return report, nil
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
