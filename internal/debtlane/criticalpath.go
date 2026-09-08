package debtlane

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Canonical production execution spine roots.
const (
	RootFakUp    = "cmd/fak up"
	RootFakServe = "cmd/fak serve"
)

// DefaultProductionRoots lists the canonical roots for all-in-one and serving execution spines.
var DefaultProductionRoots = []string{
	RootFakUp,
	RootFakServe,
}

// CriticalPathInfo records deterministic root-to-lane reachability and provenance.
type CriticalPathInfo struct {
	OnCriticalPath bool     `json:"on_critical_path"`
	Root           string   `json:"root,omitempty"`  // Canonical root entry point (e.g. "cmd/fak up" or "cmd/fak serve")
	Roots          []string `json:"roots,omitempty"` // All reachable production roots
	Distance       int      `json:"distance"`        // Shortest hop distance from root (0 for root itself)
	Path           []string `json:"path,omitempty"`  // Shortest hop sequence from root to this node
}

// CriticalPathMap maps a unit of work or lane name to its CriticalPathInfo.
type CriticalPathMap map[string]CriticalPathInfo

// MapCriticalPaths computes deterministic root-to-lane reachability from the specified roots over an import graph.
// If roots is empty, it checks for known production roots present in graph, or defaults to DefaultProductionRoots.
func MapCriticalPaths(roots []string, graph map[string]map[string]struct{}) CriticalPathMap {
	result := make(CriticalPathMap)
	if graph == nil {
		return result
	}

	effectiveRoots := roots
	if len(effectiveRoots) == 0 {
		for _, r := range DefaultProductionRoots {
			if _, ok := graph[r]; ok {
				effectiveRoots = append(effectiveRoots, r)
			}
		}
		if len(effectiveRoots) == 0 {
			for _, alias := range []string{"up", "serve", "cmd/fak"} {
				if _, ok := graph[alias]; ok {
					effectiveRoots = append(effectiveRoots, alias)
				}
			}
		}
		if len(effectiveRoots) == 0 {
			effectiveRoots = DefaultProductionRoots
		}
	}

	type queueItem struct {
		node string
		root string
		dist int
		path []string
	}

	var queue []queueItem
	visited := make(map[string]int) // node -> shortest distance

	// Initialize BFS queue with declared roots
	for _, root := range effectiveRoots {
		if edges, ok := graph[root]; ok {
			p := []string{root}
			result[root] = CriticalPathInfo{
				OnCriticalPath: true,
				Root:           root,
				Roots:          []string{root},
				Distance:       0,
				Path:           p,
			}
			visited[root] = 0

			sortedDeps := make([]string, 0, len(edges))
			for dep := range edges {
				sortedDeps = append(sortedDeps, dep)
			}
			sort.Strings(sortedDeps)

			for _, dep := range sortedDeps {
				depPath := []string{root, dep}
				queue = append(queue, queueItem{
					node: dep,
					root: root,
					dist: 1,
					path: depPath,
				})
			}
		} else {
			for k, edges := range graph {
				if strings.EqualFold(k, root) {
					p := []string{root}
					result[root] = CriticalPathInfo{
						OnCriticalPath: true,
						Root:           root,
						Roots:          []string{root},
						Distance:       0,
						Path:           p,
					}
					visited[root] = 0

					sortedDeps := make([]string, 0, len(edges))
					for dep := range edges {
						sortedDeps = append(sortedDeps, dep)
					}
					sort.Strings(sortedDeps)

					for _, dep := range sortedDeps {
						depPath := []string{root, dep}
						queue = append(queue, queueItem{
							node: dep,
							root: root,
							dist: 1,
							path: depPath,
						})
					}
				}
			}
		}
	}

	// BFS traversal
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		prevDist, seen := visited[curr.node]
		if seen {
			if prevDist <= curr.dist {
				if existing, ok := result[curr.node]; ok {
					found := false
					for _, r := range existing.Roots {
						if r == curr.root {
							found = true
							break
						}
					}
					if !found {
						existing.Roots = append(existing.Roots, curr.root)
						result[curr.node] = existing
					}
				}
				continue
			}
		}

		visited[curr.node] = curr.dist

		existing, ok := result[curr.node]
		rootsList := []string{curr.root}
		if ok {
			found := false
			for _, r := range existing.Roots {
				if r == curr.root {
					found = true
					break
				}
			}
			if !found {
				rootsList = append(existing.Roots, curr.root)
			} else {
				rootsList = existing.Roots
			}
		}

		result[curr.node] = CriticalPathInfo{
			OnCriticalPath: true,
			Root:           curr.root,
			Roots:          rootsList,
			Distance:       curr.dist,
			Path:           curr.path,
		}

		if deps, ok := graph[curr.node]; ok {
			sortedDeps := make([]string, 0, len(deps))
			for dep := range deps {
				sortedDeps = append(sortedDeps, dep)
			}
			sort.Strings(sortedDeps)

			for _, dep := range sortedDeps {
				if d, ok := visited[dep]; ok && d <= curr.dist+1 {
					continue
				}
				nextPath := make([]string, len(curr.path)+1)
				copy(nextPath, curr.path)
				nextPath[len(curr.path)] = dep

				queue = append(queue, queueItem{
					node: dep,
					root: curr.root,
					dist: curr.dist + 1,
					path: nextPath,
				})
			}
		}
	}

	return result
}

// MapProductionCriticalPaths extracts root dependencies from cmd/fak up and cmd/fak serve and maps reachability.
func MapProductionCriticalPaths(workspaceRoot string, graph map[string]map[string]struct{}) CriticalPathMap {
	if graph == nil && workspaceRoot != "" {
		graph, _ = BuildInternalImportGraph(workspaceRoot)
	}

	unifiedGraph := make(map[string]map[string]struct{})
	for k, v := range graph {
		edges := make(map[string]struct{}, len(v))
		for edge := range v {
			edges[edge] = struct{}{}
		}
		unifiedGraph[k] = edges
	}

	upDeps := make(map[string]struct{})
	serveDeps := make(map[string]struct{})

	cmdDir := filepath.Join(workspaceRoot, "cmd")
	if info, err := os.Stat(cmdDir); err == nil && info.IsDir() {
		_ = filepath.WalkDir(cmdDir, func(path string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil || d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			baseName := strings.ToLower(d.Name())
			isUp := strings.HasPrefix(baseName, "up")
			isServe := strings.HasPrefix(baseName, "serve") || baseName == "server.go" || baseName == "doctor_serve.go"

			if !isUp && !isServe {
				return nil
			}

			content, readErr := os.ReadFile(path)
			if readErr != nil {
				return nil
			}

			matches := importRe.FindAllStringSubmatch(string(content), -1)
			for _, m := range matches {
				if len(m) == 2 {
					if isUp {
						upDeps[m[1]] = struct{}{}
					}
					if isServe {
						serveDeps[m[1]] = struct{}{}
					}
				}
			}
			return nil
		})
	}

	// Always wire allinone if present in internal/allinone
	allinonePath := filepath.Join(workspaceRoot, "internal", "allinone")
	if info, err := os.Stat(allinonePath); err == nil && info.IsDir() {
		upDeps["allinone"] = struct{}{}
	}

	if len(upDeps) > 0 {
		unifiedGraph[RootFakUp] = upDeps
	}
	if len(serveDeps) > 0 {
		unifiedGraph[RootFakServe] = serveDeps
	}

	return MapCriticalPaths(DefaultProductionRoots, unifiedGraph)
}

// LookupCriticalPath resolves critical-path reachability and provenance for a lane.
func LookupCriticalPath(lane DebtLane, cpMap CriticalPathMap) (CriticalPathInfo, bool) {
	if len(cpMap) == 0 {
		return CriticalPathInfo{Distance: -1}, false
	}

	if info, ok := cpMap[lane.Lane]; ok {
		return info, true
	}
	if info, ok := cpMap[lane.UnitOfWork]; ok {
		return info, true
	}
	normUnit := filepathToSlash(lane.UnitOfWork)
	if info, ok := cpMap[normUnit]; ok {
		return info, true
	}
	base := filepath.Base(lane.UnitOfWork)
	if info, ok := cpMap[base]; ok {
		return info, true
	}

	lowerLane := strings.ToLower(lane.Lane)
	lowerBase := strings.ToLower(base)
	for k, info := range cpMap {
		lowerK := strings.ToLower(k)
		if lowerK == lowerLane || lowerK == lowerBase || strings.HasSuffix(lowerK, "/"+lowerLane) || strings.HasSuffix(lowerK, "/"+lowerBase) {
			return info, true
		}
	}

	return CriticalPathInfo{Distance: -1}, false
}

// RankCandidatesUnderPerfFocus sorts candidate debt lanes prioritizing production critical path debt.
func RankCandidatesUnderPerfFocus(candidates []DebtLane, cpMap CriticalPathMap) []DebtLane {
	sorted := make([]DebtLane, len(candidates))
	copy(sorted, candidates)

	for i := range sorted {
		if sorted[i].CriticalPath == nil && len(cpMap) > 0 {
			if info, ok := LookupCriticalPath(sorted[i], cpMap); ok {
				sorted[i].CriticalPath = &info
			}
		}
	}

	isPerfCriticalDebt := func(l DebtLane) bool {
		if l.Criticality != CriticalityCore && l.Criticality != CriticalityEnabling {
			return false
		}
		return !l.Evidence.Benchmarked || !l.Evidence.Dogfooded || l.Evidence.ModularityDeficit || l.Evidence.HasModelHardcoding
	}

	sort.SliceStable(sorted, func(i, j int) bool {
		iCrit := sorted[i].CriticalPath != nil && sorted[i].CriticalPath.OnCriticalPath
		jCrit := sorted[j].CriticalPath != nil && sorted[j].CriticalPath.OnCriticalPath
		if iCrit != jCrit {
			return iCrit
		}

		iPerf := isPerfCriticalDebt(sorted[i])
		jPerf := isPerfCriticalDebt(sorted[j])
		if iPerf != jPerf {
			return iPerf
		}

		if sorted[i].TotalDebt != sorted[j].TotalDebt {
			return sorted[i].TotalDebt > sorted[j].TotalDebt
		}
		if sorted[i].CarryingCost != sorted[j].CarryingCost {
			return sorted[i].CarryingCost > sorted[j].CarryingCost
		}
		if sorted[i].MaturityGap != sorted[j].MaturityGap {
			return sorted[i].MaturityGap > sorted[j].MaturityGap
		}
		return sorted[i].Lane < sorted[j].Lane
	})

	return sorted
}
