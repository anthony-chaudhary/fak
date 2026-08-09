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

type Tier struct {
	Level  int    `json:"level"`
	Name   string `json:"name"`
	Leaves int    `json:"leaves"`
}
type Leaf struct {
	Name             string   `json:"name"`
	DeclaredTier     int      `json:"declared_tier"`
	DeclaredTierName string   `json:"declared_tier_name"`
	ImportFloor      int      `json:"import_floor"`
	ImportFloorName  string   `json:"import_floor_name"`
	Dependencies     []string `json:"dependencies"`
	Violations       []string `json:"violations,omitempty"`
}
type Report struct {
	Schema     string `json:"schema"`
	Tiers      []Tier `json:"tiers"`
	Leaves     []Leaf `json:"leaves"`
	Violations int    `json:"violations"`
}

func Analyze(root, onlyLeaf string) (Report, error) {
	tiers, names, err := parseContract(filepath.Join(root, "internal", "architest", "architest_test.go"))
	if err != nil {
		return Report{}, err
	}
	if onlyLeaf != "" {
		if _, ok := tiers[onlyLeaf]; !ok {
			return Report{}, fmt.Errorf("leaf %q has no tier declaration", onlyLeaf)
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
		if onlyLeaf == "" || leaf == onlyLeaf {
			leaves = append(leaves, leaf)
		}
	}
	sort.Strings(leaves)
	for _, name := range leaves {
		deps, err := internalImports(filepath.Join(root, "internal", name))
		if err != nil {
			return Report{}, err
		}
		declared, floor := tiers[name], 1
		if name == "abi" {
			floor = 0
		}
		var violations []string
		for _, dep := range deps {
			if level, ok := tiers[dep]; ok {
				if level > floor {
					floor = level
				}
				if level > declared {
					violations = append(violations, name+" -> "+dep)
				}
			}
		}
		sort.Strings(violations)
		report.Violations += len(violations)
		report.Leaves = append(report.Leaves, Leaf{Name: name, DeclaredTier: declared, DeclaredTierName: tierName(names, declared), ImportFloor: floor, ImportFloorName: tierName(names, floor), Dependencies: deps, Violations: violations})
	}
	return report, nil
}

func (r Report) JSON() ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

func parseContract(path string) (map[string]int, []string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return nil, nil, err
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
		return nil, nil, fmt.Errorf("architecture contract missing tier or tierName in %s", path)
	}
	return tiers, names, nil
}
func internalImports(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	set := map[string]bool{}
	fset := token.NewFileSet()
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, e.Name()), nil, parser.ImportsOnly)
		if err != nil {
			return nil, err
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
