package archreport

import (
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

func cloneStringSet(source map[string]struct{}) map[string]struct{} {
	out := make(map[string]struct{}, len(source))
	for value := range source {
		out[value] = struct{}{}
	}
	return out
}
func stringSetsEqual(left, right map[string]struct{}) bool {
	if len(left) != len(right) {
		return false
	}
	for value := range left {
		if _, ok := right[value]; !ok {
			return false
		}
	}
	return true
}
func dominatorOrder(left, right string, dominators map[string]map[string]struct{}) bool {
	_, leftDominatesRight := dominators[right][left]
	_, rightDominatesLeft := dominators[left][right]
	if leftDominatesRight != rightDominatesLeft {
		return leftDominatesRight
	}
	return left < right
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
