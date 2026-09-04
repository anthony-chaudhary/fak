// Package ciyaml provides dependency resolution and cycle detection for CI workflows.
//
// Invariant: dependency graphs are validated as directed acyclic graphs before execution.
package ciyaml

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

var (
	// ErrCycleDetected indicates that the workflow contains a circular dependency among jobs.
	ErrCycleDetected = errors.New("ciyaml: dependency cycle detected")
	// ErrMissingDependency indicates that a job depends on an undefined job key.
	ErrMissingDependency = errors.New("ciyaml: missing dependency key")
)

// CycleError describes the exact cycle path discovered in the dependency graph.
type CycleError struct {
	Cycle []string
}

// Error returns a human-readable description of the circular dependency cycle.
func (e *CycleError) Error() string {
	return fmt.Sprintf("%v: %s", ErrCycleDetected, strings.Join(e.Cycle, " -> "))
}

// Unwrap returns ErrCycleDetected for errors.Is inspection.
func (e *CycleError) Unwrap() error {
	return ErrCycleDetected
}

// MissingDependencyError describes a job that requires an undefined dependency.
type MissingDependencyError struct {
	Job        string
	Dependency string
}

// Error returns a human-readable description of the missing dependency key.
func (e *MissingDependencyError) Error() string {
	return fmt.Sprintf("%v: job %q requires undefined job %q", ErrMissingDependency, e.Job, e.Dependency)
}

// Unwrap returns ErrMissingDependency for errors.Is inspection.
func (e *MissingDependencyError) Unwrap() error {
	return ErrMissingDependency
}

// Dependencies returns the deduplicated list of job IDs this job depends on,
// combining both Needs and DependsOn declarations in deterministic order.
func (j Job) Dependencies() []string {
	seen := make(map[string]bool)
	var deps []string
	for _, d := range j.Needs {
		d = strings.TrimSpace(d)
		if d != "" && !seen[d] {
			seen[d] = true
			deps = append(deps, d)
		}
	}
	for _, d := range j.DependsOn {
		d = strings.TrimSpace(d)
		if d != "" && !seen[d] {
			seen[d] = true
			deps = append(deps, d)
		}
	}
	sort.Strings(deps)
	return deps
}

// ResolveDAG returns a deterministic topological order of job IDs (dependencies executed first).
// If any job depends on an undefined job, it returns a MissingDependencyError.
// If a dependency cycle is detected, it returns a CycleError.
//
// Invariant: dependency graphs are validated as directed acyclic graphs before execution.
func (w *Workflow) ResolveDAG() ([]string, error) {
	return ResolveDAG(w)
}

// ValidateDAG validates that all job dependencies exist and form an acyclic graph.
// It returns an error if any dependency is missing or if a cycle is detected.
func (w *Workflow) ValidateDAG() error {
	return ValidateDAG(w)
}

// ResolveDAG returns a deterministic topological order of job IDs (dependencies executed first).
// If any job depends on an undefined job, it returns a MissingDependencyError.
// If a dependency cycle is detected, it returns a CycleError.
//
// Invariant: dependency graphs are validated as directed acyclic graphs before execution.
func ResolveDAG(w *Workflow) ([]string, error) {
	if w == nil || len(w.Jobs) == 0 {
		return nil, nil
	}

	// 1. Verify missing dependency keys
	// Sort job IDs for deterministic error reporting
	jobIDs := make([]string, 0, len(w.Jobs))
	for id := range w.Jobs {
		jobIDs = append(jobIDs, id)
	}
	sort.Strings(jobIDs)

	for _, jobID := range jobIDs {
		job := w.Jobs[jobID]
		for _, dep := range job.Dependencies() {
			if _, exists := w.Jobs[dep]; !exists {
				return nil, &MissingDependencyError{
					Job:        jobID,
					Dependency: dep,
				}
			}
		}
	}

	// 2. Check for self-dependencies (direct 1-node cycle)
	for _, jobID := range jobIDs {
		job := w.Jobs[jobID]
		for _, dep := range job.Dependencies() {
			if dep == jobID {
				return nil, &CycleError{
					Cycle: []string{jobID, jobID},
				}
			}
		}
	}

	// 3. Build graph
	// inDegree: jobID -> number of dependencies that must finish before jobID can run
	// dependents: dep -> list of jobs waiting on dep
	inDegree := make(map[string]int)
	dependents := make(map[string][]string)

	for _, id := range jobIDs {
		inDegree[id] = 0
	}

	for _, jobID := range jobIDs {
		job := w.Jobs[jobID]
		deps := job.Dependencies()
		inDegree[jobID] = len(deps)
		for _, dep := range deps {
			dependents[dep] = append(dependents[dep], jobID)
		}
	}

	// 4. Initial queue: jobs with inDegree == 0 (no dependencies)
	var queue []string
	for _, id := range jobIDs {
		if inDegree[id] == 0 {
			queue = append(queue, id)
		}
	}
	sort.Strings(queue)

	var order []string
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		order = append(order, curr)

		var newlyReady []string
		for _, depJob := range dependents[curr] {
			inDegree[depJob]--
			if inDegree[depJob] == 0 {
				newlyReady = append(newlyReady, depJob)
			}
		}
		if len(newlyReady) > 0 {
			sort.Strings(newlyReady)
			queue = append(queue, newlyReady...)
			sort.Strings(queue)
		}
	}

	// 5. If topological sort did not cover all jobs, cycle exists!
	if len(order) < len(w.Jobs) {
		cycle := findCycle(w.Jobs, inDegree)
		return nil, &CycleError{Cycle: cycle}
	}

	return order, nil
}

// ValidateDAG validates that all job dependencies exist and form an acyclic graph.
func ValidateDAG(w *Workflow) error {
	_, err := ResolveDAG(w)
	return err
}

// findCycle finds a representative cycle path using depth-first search on remaining jobs.
func findCycle(jobs map[string]Job, inDegree map[string]int) []string {
	var candidates []string
	for id, deg := range inDegree {
		if deg > 0 {
			candidates = append(candidates, id)
		}
	}
	sort.Strings(candidates)

	// 0: unvisited, 1: visiting (in current path), 2: visited (no cycle found from here)
	state := make(map[string]int)
	var path []string

	var dfs func(u string) []string
	dfs = func(u string) []string {
		state[u] = 1
		path = append(path, u)

		// Check outgoing edges: job u depends on deps
		// For cycle representation: u -> dep -> ... -> u
		deps := jobs[u].Dependencies()
		for _, v := range deps {
			if inDegree[v] == 0 {
				continue
			}
			if state[v] == 1 {
				// Cycle detected: extract from v to current u, then append v
				for i, p := range path {
					if p == v {
						cycle := make([]string, len(path)-i+1)
						copy(cycle, path[i:])
						cycle[len(cycle)-1] = v
						return cycle
					}
				}
				return []string{v, u, v}
			}
			if state[v] == 0 {
				if c := dfs(v); len(c) > 0 {
					return c
				}
			}
		}

		path = path[:len(path)-1]
		state[u] = 2
		return nil
	}

	for _, start := range candidates {
		if state[start] == 0 {
			if c := dfs(start); len(c) > 0 {
				return c
			}
		}
	}

	// Fallback if DFS didn't isolate a cycle
	return candidates
}
