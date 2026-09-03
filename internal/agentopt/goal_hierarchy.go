package agentopt

import (
	"fmt"
	"path"
	"sort"
	"strings"
)

// Family 10: Goal hierarchy decomposition and world models.
//
// Goal hierarchy validation ensures that subagent task graphs form valid
// directed acyclic graphs (DAGs), that parallel subtasks execute within
// disjoint write scopes, and that turn budgets remain strictly bounded.

// GoalNode models a discrete task unit in a goal hierarchy graph.
type GoalNode struct {
	ID          string   `json:"id"`
	Description string   `json:"description,omitempty"`
	DependsOn   []string `json:"depends_on,omitempty"`
	TargetPaths []string `json:"target_paths,omitempty"`
	MaxTurns    int      `json:"max_turns,omitempty"`
}

// SubtaskSpec represents the execution specification for a decomposed subtask.
type SubtaskSpec struct {
	ID          string   `json:"id"`
	Description string   `json:"description,omitempty"`
	TargetPaths []string `json:"target_paths,omitempty"`
	MaxTurns    int      `json:"max_turns"`
}

// SubtaskSpec converts the goal node into its corresponding subtask specification.
func (g GoalNode) SubtaskSpec() SubtaskSpec {
	return SubtaskSpec{
		ID:          g.ID,
		Description: g.Description,
		TargetPaths: append([]string(nil), g.TargetPaths...),
		MaxTurns:    g.MaxTurns,
	}
}

// ScopeOverlap records target path conflicts between concurrent subtasks.
type ScopeOverlap struct {
	TaskA string   `json:"task_a"`
	TaskB string   `json:"task_b"`
	Paths []string `json:"paths"`
}

// TurnViolation records a subtask whose execution turn budget is invalid or exceeds bounds.
type TurnViolation struct {
	TaskID   string `json:"task_id"`
	MaxTurns int    `json:"max_turns"`
	Reason   string `json:"reason"`
}

// HierarchyReport records the structural and semantic findings of DAG validation.
type HierarchyReport struct {
	Valid                bool            `json:"valid"`
	CyclesDetected       bool            `json:"cycles_detected"`
	CyclePaths           [][]string      `json:"cycle_paths,omitempty"`
	ScopeOverlaps        []ScopeOverlap  `json:"scope_overlaps,omitempty"`
	TurnBudgetViolations []TurnViolation `json:"turn_violations,omitempty"`
	TurnBudgetExceeded   bool            `json:"turn_budget_exceeded"`
	AtomicViolations     []string        `json:"atomic_violations,omitempty"`
	Violations           []string        `json:"violations,omitempty"`
	ExecutionOrder       []string        `json:"execution_order,omitempty"`
	ParallelStages       [][]string      `json:"parallel_stages,omitempty"`
	Subtasks             []SubtaskSpec   `json:"subtasks,omitempty"`
	TotalTurnBudget      int             `json:"total_turn_budget"`
	CriticalPathTurns    int             `json:"critical_path_turns"`
}

// ValidationReport is an alias for HierarchyReport.
type ValidationReport = HierarchyReport

// GoalHierarchyValidator validates goal DAGs for cyclic dependencies,
// atomicity, write scope disjointness, and turn budget bounds.
type GoalHierarchyValidator struct {
	MaxTurnsPerSubtask int  // optional ceiling per subtask; 0 means default (50)
	MaxTotalTurns      int  // optional ceiling across all subtasks; 0 means unconstrained
	RequireTargetPaths bool // whether each subtask must declare at least one target path
}

// NewGoalHierarchyValidator constructs a validator with standard defaults.
func NewGoalHierarchyValidator() *GoalHierarchyValidator {
	return &GoalHierarchyValidator{
		MaxTurnsPerSubtask: 50,
		MaxTotalTurns:      200,
		RequireTargetPaths: false,
	}
}

// normalizeTargetPath canonicalizes a target path or directory prefix for comparison.
func normalizeTargetPath(raw string) string {
	cleaned := strings.TrimSpace(raw)
	cleaned = strings.TrimPrefix(cleaned, "./")
	cleaned = strings.TrimSuffix(cleaned, "/**")
	cleaned = strings.TrimSuffix(cleaned, "/*")
	cleaned = strings.TrimSuffix(cleaned, "/")
	if cleaned == "" {
		return ""
	}
	return path.Clean(cleaned)
}

// targetPathsOverlap checks whether two target paths collide by exact match or directory prefix.
func targetPathsOverlap(pathA, pathB string) bool {
	normA := normalizeTargetPath(pathA)
	normB := normalizeTargetPath(pathB)
	if normA == "" || normB == "" {
		return false
	}
	if normA == normB {
		return true
	}
	if strings.HasPrefix(normB, normA+"/") || strings.HasPrefix(normA, normB+"/") {
		return true
	}
	return false
}

// ValidateGoalDAG evaluates a set of goal nodes against DAG ordering,
// atomicity, write scope disjointness, and turn limits.
func (v *GoalHierarchyValidator) ValidateGoalDAG(nodes []GoalNode) ValidationReport {
	report := HierarchyReport{
		Valid:                true,
		CyclePaths:           make([][]string, 0),
		ScopeOverlaps:        make([]ScopeOverlap, 0),
		TurnBudgetViolations: make([]TurnViolation, 0),
		AtomicViolations:     make([]string, 0),
		Violations:           make([]string, 0),
		ExecutionOrder:       make([]string, 0),
		ParallelStages:       make([][]string, 0),
		Subtasks:             make([]SubtaskSpec, 0),
	}

	if len(nodes) == 0 {
		return report
	}

	subtaskLimit := 50
	if v != nil && v.MaxTurnsPerSubtask > 0 {
		subtaskLimit = v.MaxTurnsPerSubtask
	}

	nodeMap := make(map[string]GoalNode, len(nodes))
	seenIDs := make(map[string]bool, len(nodes))
	var allIDs []string

	totalTurns := 0

	// Step 1: Structural atomicity and local property validation
	for _, node := range nodes {
		id := strings.TrimSpace(node.ID)
		if id == "" {
			msg := "atomic violation: subtask ID cannot be empty"
			report.AtomicViolations = append(report.AtomicViolations, msg)
			report.Violations = append(report.Violations, msg)
			continue
		}

		if seenIDs[id] {
			msg := fmt.Sprintf("atomic violation: duplicate subtask ID %q", id)
			report.AtomicViolations = append(report.AtomicViolations, msg)
			report.Violations = append(report.Violations, msg)
			continue
		}
		seenIDs[id] = true
		allIDs = append(allIDs, id)
		nodeMap[id] = node

		report.Subtasks = append(report.Subtasks, node.SubtaskSpec())
		totalTurns += node.MaxTurns

		if strings.TrimSpace(node.Description) == "" {
			msg := fmt.Sprintf("atomic violation: subtask %q description cannot be empty", id)
			report.AtomicViolations = append(report.AtomicViolations, msg)
			report.Violations = append(report.Violations, msg)
		}

		if node.MaxTurns <= 0 {
			msg := fmt.Sprintf("atomic violation: subtask %q execution turn budget must be positive, got %d", id, node.MaxTurns)
			report.AtomicViolations = append(report.AtomicViolations, msg)
			report.Violations = append(report.Violations, msg)
			report.TurnBudgetViolations = append(report.TurnBudgetViolations, TurnViolation{
				TaskID:   id,
				MaxTurns: node.MaxTurns,
				Reason:   "execution turn budget must be positive",
			})
		} else if subtaskLimit > 0 && node.MaxTurns > subtaskLimit {
			msg := fmt.Sprintf("turn budget violation: subtask %q turns (%d) exceeds limit (%d)", id, node.MaxTurns, subtaskLimit)
			report.Violations = append(report.Violations, msg)
			report.TurnBudgetViolations = append(report.TurnBudgetViolations, TurnViolation{
				TaskID:   id,
				MaxTurns: node.MaxTurns,
				Reason:   fmt.Sprintf("exceeds subtask limit of %d turns", subtaskLimit),
			})
		}

		if v != nil && v.RequireTargetPaths && len(node.TargetPaths) == 0 {
			msg := fmt.Sprintf("atomic violation: subtask %q must declare at least one target path", id)
			report.AtomicViolations = append(report.AtomicViolations, msg)
			report.Violations = append(report.Violations, msg)
		}

		for _, p := range node.TargetPaths {
			if strings.TrimSpace(p) == "" {
				msg := fmt.Sprintf("atomic violation: subtask %q contains empty target path entry", id)
				report.AtomicViolations = append(report.AtomicViolations, msg)
				report.Violations = append(report.Violations, msg)
			}
		}

		for _, dep := range node.DependsOn {
			if dep == id {
				msg := fmt.Sprintf("atomic violation: subtask %q cannot depend on itself", id)
				report.AtomicViolations = append(report.AtomicViolations, msg)
				report.Violations = append(report.Violations, msg)
			}
		}
	}

	report.TotalTurnBudget = totalTurns
	if v != nil && v.MaxTotalTurns > 0 && totalTurns > v.MaxTotalTurns {
		msg := fmt.Sprintf("turn budget violation: total turns (%d) exceeds overall budget (%d)", totalTurns, v.MaxTotalTurns)
		report.Violations = append(report.Violations, msg)
		report.TurnBudgetExceeded = true
	}

	// Step 2: Dependency references check
	for _, id := range allIDs {
		node := nodeMap[id]
		for _, dep := range node.DependsOn {
			if !seenIDs[dep] {
				msg := fmt.Sprintf("dependency violation: subtask %q depends on unknown subtask %q", id, dep)
				report.Violations = append(report.Violations, msg)
			}
		}
	}

	sort.Strings(allIDs)

	// Step 3: Cycle detection via DFS
	color := make(map[string]int, len(allIDs)) // 0: unvisited, 1: visiting, 2: visited
	var dfsPath []string
	recordedCycles := make(map[string]bool)

	var checkCycleDFS func(curr string)
	checkCycleDFS = func(curr string) {
		color[curr] = 1
		dfsPath = append(dfsPath, curr)

		node := nodeMap[curr]
		// Sort dependencies for deterministic traversal
		deps := append([]string(nil), node.DependsOn...)
		sort.Strings(deps)

		for _, dep := range deps {
			if !seenIDs[dep] {
				continue
			}
			if color[dep] == 1 {
				// Found a cycle
				startIdx := -1
				for i, item := range dfsPath {
					if item == dep {
						startIdx = i
						break
					}
				}
				if startIdx >= 0 {
					rawCycle := append([]string(nil), dfsPath[startIdx:]...)
					rawCycle = append(rawCycle, dep)

					cycleKey := strings.Join(rawCycle, " -> ")
					if !recordedCycles[cycleKey] {
						recordedCycles[cycleKey] = true
						report.CyclePaths = append(report.CyclePaths, rawCycle)
						report.CyclesDetected = true
						msg := fmt.Sprintf("cyclic dependency detected: %s", cycleKey)
						report.Violations = append(report.Violations, msg)
					}
				}
			} else if color[dep] == 0 {
				checkCycleDFS(dep)
			}
		}

		dfsPath = dfsPath[:len(dfsPath)-1]
		color[curr] = 2
	}

	for _, id := range allIDs {
		if color[id] == 0 {
			checkCycleDFS(id)
		}
	}

	// Step 4: Transitive dependency reachability (who depends on whom)
	// reaches[u][v] is true if task u directly or transitively depends on task v.
	reaches := make(map[string]map[string]bool, len(allIDs))
	for _, id := range allIDs {
		reaches[id] = make(map[string]bool)
		queue := append([]string(nil), nodeMap[id].DependsOn...)
		visitedDep := make(map[string]bool)

		for len(queue) > 0 {
			d := queue[0]
			queue = queue[1:]
			if visitedDep[d] {
				continue
			}
			visitedDep[d] = true
			reaches[id][d] = true
			if parent, ok := nodeMap[d]; ok {
				queue = append(queue, parent.DependsOn...)
			}
		}
	}

	// Step 5: Scope disjointness check across parallel tasks
	// Two tasks u and v are parallel if neither depends on the other.
	for i := 0; i < len(allIDs); i++ {
		for j := i + 1; j < len(allIDs); j++ {
			u := allIDs[i]
			w := allIDs[j]

			// If u depends on w, or w depends on u, they run sequentially (not parallel)
			if reaches[u][w] || reaches[w][u] {
				continue
			}

			// Independent / parallel tasks: verify target write paths are disjoint
			var conflicts []string
			for _, pA := range nodeMap[u].TargetPaths {
				for _, pB := range nodeMap[w].TargetPaths {
					if targetPathsOverlap(pA, pB) {
						conflicts = append(conflicts, fmt.Sprintf("%s overlaps with %s", pA, pB))
					}
				}
			}

			if len(conflicts) > 0 {
				overlap := ScopeOverlap{
					TaskA: u,
					TaskB: w,
					Paths: conflicts,
				}
				report.ScopeOverlaps = append(report.ScopeOverlaps, overlap)
				msg := fmt.Sprintf("write scope overlap between parallel subtasks %q and %q: %s", u, w, strings.Join(conflicts, ", "))
				report.Violations = append(report.Violations, msg)
			}
		}
	}

	// Step 6: Topological sort and parallel execution stages (Kahn's algorithm)
	if !report.CyclesDetected {
		// inDegree represents how many dependencies task u is waiting on
		inDegree := make(map[string]int, len(allIDs))
		dependents := make(map[string][]string, len(allIDs)) // who depends on task u

		for _, id := range allIDs {
			validDeps := 0
			for _, dep := range nodeMap[id].DependsOn {
				if seenIDs[dep] {
					validDeps++
					dependents[dep] = append(dependents[dep], id)
				}
			}
			inDegree[id] = validDeps
		}

		var currentStage []string
		for _, id := range allIDs {
			if inDegree[id] == 0 {
				currentStage = append(currentStage, id)
			}
		}
		sort.Strings(currentStage)

		// Calculate critical path turns
		pathTurns := make(map[string]int, len(allIDs))

		for len(currentStage) > 0 {
			report.ParallelStages = append(report.ParallelStages, currentStage)
			report.ExecutionOrder = append(report.ExecutionOrder, currentStage...)

			var nextStage []string
			for _, completedID := range currentStage {
				curMax := pathTurns[completedID] + nodeMap[completedID].MaxTurns
				if curMax > report.CriticalPathTurns {
					report.CriticalPathTurns = curMax
				}

				for _, depID := range dependents[completedID] {
					if curMax > pathTurns[depID] {
						pathTurns[depID] = curMax
					}
					inDegree[depID]--
					if inDegree[depID] == 0 {
						nextStage = append(nextStage, depID)
					}
				}
			}

			sort.Strings(nextStage)
			currentStage = nextStage
		}

		if len(report.ExecutionOrder) < len(allIDs) {
			report.CyclesDetected = true
			msg := "unresolved dependency cycle detected in graph execution order"
			report.Violations = append(report.Violations, msg)
		}
	}

	report.Valid = len(report.Violations) == 0
	return report
}
