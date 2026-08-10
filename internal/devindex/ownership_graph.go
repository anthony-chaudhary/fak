package devindex

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const moduleInternalPrefix = "github.com/anthony-chaudhary/fak/internal/"

// PackageOwnership declares an implementation package that belongs only to the
// development artifact. Runtime must have no direct or transitive path to it.
type PackageOwnership struct {
	Path      string       `json:"path"`
	Owner     CommandOwner `json:"owner"`
	Rationale string       `json:"rationale"`
}

// DevOnlyPackages starts the package boundary with leaves whose contracts are
// intrinsically repository-development control-plane work. The list expands as
// command families move; every entry is enforced immediately by GraphLeaks.
var DevOnlyPackages = []PackageOwnership{
	{Path: moduleInternalPrefix + "amdgpu", Owner: OwnerDev, Rationale: "probes development-host AMD GPU diagnostics and counters"},
	{Path: moduleInternalPrefix + "commitsubject", Owner: OwnerDev, Rationale: "audits repository commit-subject grammar coverage"},
	{Path: moduleInternalPrefix + "devcmd", Owner: OwnerDev, Rationale: "hosts repository-development command implementations for fak-dev"},
	{Path: moduleInternalPrefix + "devindex", Owner: OwnerDev, Rationale: "indexes and audits the fak repository command surface"},
	{Path: moduleInternalPrefix + "readmevisualaudit", Owner: OwnerDev, Rationale: "audits repository README visual and asset health"},
	{Path: moduleInternalPrefix + "planaudit", Owner: OwnerDev, Rationale: "audits repository plan documents for drift"},
	{Path: moduleInternalPrefix + "issuesync", Owner: OwnerDev, Rationale: "synchronizes fak repository GitHub issues"},
	{Path: moduleInternalPrefix + "wiki", Owner: OwnerDev, Rationale: "audits repository documentation structure, citations, freshness, and coverage"},
	{Path: moduleInternalPrefix + "sweep", Owner: OwnerDev, Rationale: "groups and commits shared-checkout development work"},
	{Path: moduleInternalPrefix + "worktreeworker", Owner: OwnerDev, Rationale: "manages isolated repository worker worktrees"},
}

// ImportNode is the stable subset of `go list -deps -json` used by the boundary
// witness. Tests can construct synthetic graphs without invoking the toolchain.
type ImportNode struct {
	ImportPath string   `json:"ImportPath"`
	Imports    []string `json:"Imports"`
}

// GraphLeak is one shortest witnessed path from a runtime root to a dev-only
// package. Path includes both the root and forbidden package.
type GraphLeak struct {
	Root      string   `json:"root"`
	Forbidden string   `json:"forbidden"`
	Path      []string `json:"path"`
}

// GraphReport is deterministic machine-readable evidence for an artifact's
// dependency closure.
type GraphReport struct {
	Root          string      `json:"root"`
	PackageCount  int         `json:"package_count"`
	InternalCount int         `json:"internal_count"`
	Leaks         []GraphLeak `json:"leaks"`
}

// LoadImportGraph asks the Go toolchain for the complete dependency graph of a
// package pattern. The caller controls Dir so this can run in a clean archive.
func LoadImportGraph(dir, pattern string) ([]ImportNode, error) {
	cmd := graphCommand("go", "list", "-deps", "-json", pattern)
	cmd.Dir = dir
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	dec := json.NewDecoder(stdout)
	var nodes []ImportNode
	for dec.More() {
		var node ImportNode
		if err := dec.Decode(&node); err != nil {
			_ = cmd.Process.Kill()
			return nil, fmt.Errorf("decode go list graph: %w", err)
		}
		nodes = append(nodes, node)
	}
	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("go list -deps %s: %w", pattern, err)
	}
	return nodes, nil
}

// BuildGraphReport finds shortest runtime-to-dev paths. It deliberately tests
// reachability, not package-name substrings, so transitive leaks are visible.
func BuildGraphReport(root string, nodes []ImportNode, packages []PackageOwnership) GraphReport {
	adj := make(map[string][]string, len(nodes))
	forbidden := make(map[string]bool, len(packages))
	internal := 0
	for _, node := range nodes {
		imports := append([]string(nil), node.Imports...)
		sort.Strings(imports)
		adj[node.ImportPath] = imports
		if strings.HasPrefix(node.ImportPath, moduleInternalPrefix) {
			internal++
		}
	}
	for _, pkg := range packages {
		if pkg.Owner == OwnerDev {
			forbidden[pkg.Path] = true
		}
	}
	report := GraphReport{Root: root, PackageCount: len(nodes), InternalCount: internal}
	queue := []string{root}
	paths := map[string][]string{root: {root}}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		if forbidden[cur] {
			report.Leaks = append(report.Leaks, GraphLeak{Root: root, Forbidden: cur, Path: paths[cur]})
			continue
		}
		for _, next := range adj[cur] {
			if _, ok := paths[next]; ok {
				continue
			}
			paths[next] = append(append([]string(nil), paths[cur]...), next)
			queue = append(queue, next)
		}
	}
	sort.Slice(report.Leaks, func(i, j int) bool { return report.Leaks[i].Forbidden < report.Leaks[j].Forbidden })
	return report
}
