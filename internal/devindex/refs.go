package devindex

import (
	"fmt"
	"sort"
	"strings"
)

// SymbolID names a Go symbol by import path plus symbol name. The package path is
// the full import path known to the import graph, not only the local package name.
type SymbolID struct {
	Package string `json:"package"`
	Symbol  string `json:"symbol"`
}

// String renders the CLI/query spelling: <pkg>.<Symbol>.
func (s SymbolID) String() string {
	s = cleanSymbolID(s)
	if s.Package == "" || s.Symbol == "" {
		return ""
	}
	return s.Package + "." + s.Symbol
}

// ParseSymbolID parses <pkg>.<Symbol>. It splits at the last dot so module hosts
// such as github.com stay part of the package path.
func ParseSymbolID(query string) (SymbolID, error) {
	q := strings.TrimSpace(query)
	dot := strings.LastIndex(q, ".")
	lastSlash := strings.LastIndex(q, "/")
	if dot <= lastSlash || dot < 0 || dot == len(q)-1 {
		return SymbolID{}, fmt.Errorf("symbol target %q must be <pkg>.<Symbol>", query)
	}
	id := cleanSymbolID(SymbolID{Package: q[:dot], Symbol: q[dot+1:]})
	if id.Package == "" || id.Symbol == "" || strings.ContainsAny(id.Symbol, `/\`) {
		return SymbolID{}, fmt.Errorf("symbol target %q must be <pkg>.<Symbol>", query)
	}
	return id, nil
}

// PackageImports is one node from the already-built import graph. Imports,
// TestImports, and XTestImports are all dependency edges; callers should pass the
// same folded go-list shape that fak affected uses.
type PackageImports struct {
	ImportPath   string   `json:"import_path"`
	Imports      []string `json:"imports,omitempty"`
	TestImports  []string `json:"test_imports,omitempty"`
	XTestImports []string `json:"xtest_imports,omitempty"`
}

// SymbolReference says FromPackage directly references Target. Test marks a
// reference seen only in test files; the blast-radius closure treats it as a real
// dependency seed because test-only callers still need re-verification.
type SymbolReference struct {
	FromPackage string   `json:"from_package"`
	Target      SymbolID `json:"target"`
	Test        bool     `json:"test,omitempty"`
}

// ReferenceIndex is the pure, deterministic index behind the blast-radius query.
// It owns no IO: the CLI/gateway layer gathers go-list packages and symbol refs,
// then asks this index "what depends on this symbol?" before editing.
type ReferenceIndex struct {
	packages   []PackageImports
	references []SymbolReference
}

// NewReferenceIndex returns an immutable-by-convention copy of the supplied graph
// and references. The query methods never mutate caller-owned slices.
func NewReferenceIndex(packages []PackageImports, refs []SymbolReference) ReferenceIndex {
	idx := ReferenceIndex{
		packages:   make([]PackageImports, len(packages)),
		references: make([]SymbolReference, len(refs)),
	}
	for i, p := range packages {
		idx.packages[i] = PackageImports{
			ImportPath:   cleanPackagePath(p.ImportPath),
			Imports:      cloneStrings(p.Imports),
			TestImports:  cloneStrings(p.TestImports),
			XTestImports: cloneStrings(p.XTestImports),
		}
	}
	for i, r := range refs {
		idx.references[i] = SymbolReference{
			FromPackage: cleanPackagePath(r.FromPackage),
			Target:      cleanSymbolID(r.Target),
			Test:        r.Test,
		}
	}
	return idx
}

// BlastRadiusPackage is one dependent package ranked by shortest import distance
// to a direct symbol reference. Distance 1 means the package directly references
// the target symbol; larger distances are transitive importers of a direct user.
type BlastRadiusPackage struct {
	ImportPath string `json:"import_path"`
	Distance   int    `json:"distance"`
	Direct     bool   `json:"direct,omitempty"`
}

// BlastRadiusResult is the complete answer for one symbol query. Packages excludes
// the defining package itself and contains direct plus transitive dependents.
type BlastRadiusResult struct {
	Target   SymbolID             `json:"target"`
	Packages []BlastRadiusPackage `json:"packages"`
}

// BlastRadius is a convenience wrapper for querying a one-shot package/ref set.
func BlastRadius(packages []PackageImports, refs []SymbolReference, target SymbolID) BlastRadiusResult {
	return NewReferenceIndex(packages, refs).BlastRadius(target)
}

// BlastRadius returns direct and transitive packages that depend on target, ranked
// by shortest distance and then import path. Test imports participate in the graph,
// matching fak affected's import-edge semantics without running a build.
func (idx ReferenceIndex) BlastRadius(target SymbolID) BlastRadiusResult {
	target = cleanSymbolID(target)
	direct := idx.directReferencePackages(target)
	distances := make(map[string]int, len(direct))
	queue := make([]string, 0, len(direct))
	for _, pkg := range direct {
		distances[pkg] = 1
		queue = append(queue, pkg)
	}

	importedBy := idx.reverseImportGraph()
	for head := 0; head < len(queue); head++ {
		cur := queue[head]
		for _, importer := range importedBy[cur] {
			if importer == target.Package {
				continue
			}
			if _, seen := distances[importer]; seen {
				continue
			}
			distances[importer] = distances[cur] + 1
			queue = append(queue, importer)
		}
	}

	rows := make([]BlastRadiusPackage, 0, len(distances))
	directSet := stringSet(direct)
	for pkg, distance := range distances {
		rows = append(rows, BlastRadiusPackage{
			ImportPath: pkg,
			Distance:   distance,
			Direct:     directSet[pkg],
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Distance != rows[j].Distance {
			return rows[i].Distance < rows[j].Distance
		}
		return rows[i].ImportPath < rows[j].ImportPath
	})
	return BlastRadiusResult{Target: target, Packages: rows}
}

func (idx ReferenceIndex) directReferencePackages(target SymbolID) []string {
	seen := map[string]bool{}
	var out []string
	for _, ref := range idx.references {
		from := cleanPackagePath(ref.FromPackage)
		if from == "" || from == target.Package {
			continue
		}
		if cleanSymbolID(ref.Target) != target || seen[from] {
			continue
		}
		seen[from] = true
		out = append(out, from)
	}
	sort.Strings(out)
	return out
}

func (idx ReferenceIndex) reverseImportGraph() map[string][]string {
	importedBy := map[string][]string{}
	for _, pkg := range idx.packages {
		from := cleanPackagePath(pkg.ImportPath)
		if from == "" {
			continue
		}
		for _, group := range [][]string{pkg.Imports, pkg.TestImports, pkg.XTestImports} {
			for _, dep := range group {
				to := cleanPackagePath(dep)
				if to == "" || to == from {
					continue
				}
				importedBy[to] = append(importedBy[to], from)
			}
		}
	}
	for dep := range importedBy {
		importedBy[dep] = dedupeSortedStrings(importedBy[dep])
	}
	return importedBy
}

func cleanSymbolID(s SymbolID) SymbolID {
	return SymbolID{Package: cleanPackagePath(s.Package), Symbol: strings.TrimSpace(s.Symbol)}
}

func cleanPackagePath(pkg string) string {
	return strings.Trim(normPath(strings.TrimSpace(pkg)), "/")
}

func cloneStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, len(in))
	copy(out, in)
	return out
}

func stringSet(in []string) map[string]bool {
	out := make(map[string]bool, len(in))
	for _, s := range in {
		out[s] = true
	}
	return out
}

func dedupeSortedStrings(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}
