package workflow

// workflow.go — the workflow DSL, the compiler that turns a declared spec into a
// validated DAG, and the built-in pattern constructors (map-reduce, fan-out).
//
// A workflow is authored as a JSON document (which, being a strict subset of YAML
// 1.2, is also a valid YAML document — so "define workflows in YAML/JSON" is honored
// with zero external dependencies; the repo's no-deps invariant rules out a YAML
// package). It declares either an explicit list of tasks with dependencies, or one of
// the built-in patterns as sugar that the compiler expands into the same task graph.

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// TaskSpec is one declared unit of work in a workflow document. Op names the operation
// the Runner dispatches on; Needs lists the IDs of upstream tasks whose outputs this
// task depends on (the DAG edges); Payload is a static input handed to the Runner;
// Retries is the number of EXTRA attempts on failure (0 = one attempt, no retry).
type TaskSpec struct {
	ID      string   `json:"id"`
	Op      string   `json:"op,omitempty"`
	Needs   []string `json:"needs,omitempty"`
	Payload string   `json:"payload,omitempty"`
	Retries int      `json:"retries,omitempty"`
}

// MapReduceSpec declares the map-reduce pattern: one Map op fanned over Items (one map
// task per item, all independent), whose outputs all feed a single Reduce task.
type MapReduceSpec struct {
	Map     string   `json:"map"`     // op run once per item
	Reduce  string   `json:"reduce"`  // op run once over all map outputs
	Items   []string `json:"items"`   // the inputs to map over
	Retries int      `json:"retries"` // retries applied to every generated task
}

// FanOutSpec declares the fan-out pattern: one Source task whose output feeds N
// independent Branch tasks in parallel, optionally joined by a single Join task that
// depends on every branch. An empty Join omits the join (a pure fan-out).
type FanOutSpec struct {
	Source   string   `json:"source"`   // op run first; every branch depends on it
	Branches []string `json:"branches"` // ops run in parallel, each depending on source
	Join     string   `json:"join"`     // optional op depending on every branch
	Retries  int      `json:"retries"`
}

// Spec is a whole workflow document. Exactly one shape is populated: an explicit Tasks
// list, or one of the built-in patterns. Compile rejects an ambiguous or empty spec.
type Spec struct {
	Name      string         `json:"name"`
	Tasks     []TaskSpec     `json:"tasks,omitempty"`
	MapReduce *MapReduceSpec `json:"map_reduce,omitempty"`
	FanOut    *FanOutSpec    `json:"fan_out,omitempty"`
}

// Node is one compiled, validated unit of work in the DAG.
type Node struct {
	ID      string
	Op      string
	Needs   []string
	Payload string
	Retries int
}

// Graph is a compiled workflow: a validated DAG (unique IDs, every dependency exists,
// no cycles). Nodes are stored in a deterministic topological order so execution and
// reporting are reproducible.
type Graph struct {
	Name  string
	Nodes []Node
}

// ParseSpec decodes a workflow document (JSON; also valid YAML since JSON ⊂ YAML).
// Unknown fields are rejected so a typo'd key is a loud error, not a silent no-op.
func ParseSpec(doc []byte) (Spec, error) {
	var s Spec
	dec := json.NewDecoder(strings.NewReader(string(doc)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return Spec{}, fmt.Errorf("workflow: parse spec: %w", err)
	}
	return s, nil
}

// Compile validates a spec and expands any built-in pattern into the canonical DAG.
// It is the single gate every workflow passes through, whether authored as explicit
// tasks or as a pattern, so the executor only ever sees a sound graph.
func Compile(s Spec) (*Graph, error) {
	tasks, err := s.tasks()
	if err != nil {
		return nil, err
	}
	if len(tasks) == 0 {
		return nil, fmt.Errorf("workflow: empty — declare tasks, map_reduce, or fan_out")
	}
	ordered, err := validateDAG(tasks)
	if err != nil {
		return nil, err
	}
	g := &Graph{Name: s.Name, Nodes: make([]Node, 0, len(ordered))}
	for _, t := range ordered {
		g.Nodes = append(g.Nodes, Node{
			ID: t.ID, Op: t.Op, Needs: t.Needs, Payload: t.Payload, Retries: t.Retries,
		})
	}
	return g, nil
}

// CompileJSON is the convenience door: parse + compile in one call.
func CompileJSON(doc []byte) (*Graph, error) {
	s, err := ParseSpec(doc)
	if err != nil {
		return nil, err
	}
	return Compile(s)
}

// tasks returns the explicit task list, expanding a built-in pattern if one is set.
// Exactly one of {Tasks, MapReduce, FanOut} may be populated.
func (s Spec) tasks() ([]TaskSpec, error) {
	set := 0
	if len(s.Tasks) > 0 {
		set++
	}
	if s.MapReduce != nil {
		set++
	}
	if s.FanOut != nil {
		set++
	}
	if set > 1 {
		return nil, fmt.Errorf("workflow: declare exactly one of tasks, map_reduce, or fan_out")
	}
	switch {
	case s.MapReduce != nil:
		return s.MapReduce.expand(), nil
	case s.FanOut != nil:
		return s.FanOut.expand(), nil
	default:
		return s.Tasks, nil
	}
}

// expand turns a map-reduce spec into tasks: map:<i> for each item, then a single
// reduce task that needs every map task.
func (m *MapReduceSpec) expand() []TaskSpec {
	tasks := make([]TaskSpec, 0, len(m.Items)+1)
	needs := make([]string, 0, len(m.Items))
	for i, item := range m.Items {
		id := fmt.Sprintf("map:%d", i)
		tasks = append(tasks, TaskSpec{ID: id, Op: m.Map, Payload: item, Retries: m.Retries})
		needs = append(needs, id)
	}
	tasks = append(tasks, TaskSpec{ID: "reduce", Op: m.Reduce, Needs: needs, Retries: m.Retries})
	return tasks
}

// expand turns a fan-out spec into tasks: a source, one branch per op depending on the
// source, and (if Join is set) a join depending on every branch.
func (f *FanOutSpec) expand() []TaskSpec {
	tasks := make([]TaskSpec, 0, len(f.Branches)+2)
	tasks = append(tasks, TaskSpec{ID: "source", Op: f.Source, Retries: f.Retries})
	branchIDs := make([]string, 0, len(f.Branches))
	for i, op := range f.Branches {
		id := fmt.Sprintf("branch:%d", i)
		tasks = append(tasks, TaskSpec{ID: id, Op: op, Needs: []string{"source"}, Retries: f.Retries})
		branchIDs = append(branchIDs, id)
	}
	if f.Join != "" {
		tasks = append(tasks, TaskSpec{ID: "join", Op: f.Join, Needs: branchIDs, Retries: f.Retries})
	}
	return tasks
}

// MapReduce builds a compiled map-reduce graph directly (the programmatic twin of the
// map_reduce DSL block). Useful when items are computed rather than authored.
func MapReduce(name, mapOp, reduceOp string, items []string) (*Graph, error) {
	return Compile(Spec{Name: name, MapReduce: &MapReduceSpec{Map: mapOp, Reduce: reduceOp, Items: items}})
}

// FanOut builds a compiled fan-out graph directly. An empty joinOp yields a pure
// fan-out with no join.
func FanOut(name, sourceOp string, branchOps []string, joinOp string) (*Graph, error) {
	return Compile(Spec{Name: name, FanOut: &FanOutSpec{Source: sourceOp, Branches: branchOps, Join: joinOp}})
}

// validateDAG checks IDs are non-empty and unique, every dependency names an existing
// task, no task depends on itself, and the dependency graph is acyclic. It returns the
// tasks in a deterministic topological order (ties broken by ID) — that order is both
// the executor's wave order and the report order, so a run is reproducible.
func validateDAG(tasks []TaskSpec) ([]TaskSpec, error) {
	byID := make(map[string]TaskSpec, len(tasks))
	for _, t := range tasks {
		if t.ID == "" {
			return nil, fmt.Errorf("workflow: a task has an empty id")
		}
		if _, dup := byID[t.ID]; dup {
			return nil, fmt.Errorf("workflow: duplicate task id %q", t.ID)
		}
		byID[t.ID] = t
	}
	// Kahn's algorithm with a sorted ready frontier for a deterministic order.
	indeg := make(map[string]int, len(tasks))
	dependents := make(map[string][]string, len(tasks))
	for _, t := range tasks {
		indeg[t.ID] = 0
	}
	for _, t := range tasks {
		seen := map[string]bool{}
		for _, d := range t.Needs {
			if d == t.ID {
				return nil, fmt.Errorf("workflow: task %q depends on itself", t.ID)
			}
			if _, ok := byID[d]; !ok {
				return nil, fmt.Errorf("workflow: task %q needs unknown task %q", t.ID, d)
			}
			if seen[d] {
				continue // a duplicate edge is harmless; count it once
			}
			seen[d] = true
			indeg[t.ID]++
			dependents[d] = append(dependents[d], t.ID)
		}
	}
	var ready []string
	for id, n := range indeg {
		if n == 0 {
			ready = append(ready, id)
		}
	}
	sort.Strings(ready)
	var order []TaskSpec
	for len(ready) > 0 {
		id := ready[0]
		ready = ready[1:]
		order = append(order, byID[id])
		var unlocked []string
		for _, dep := range dependents[id] {
			indeg[dep]--
			if indeg[dep] == 0 {
				unlocked = append(unlocked, dep)
			}
		}
		if len(unlocked) > 0 {
			ready = append(ready, unlocked...)
			sort.Strings(ready)
		}
	}
	if len(order) != len(tasks) {
		return nil, fmt.Errorf("workflow: dependency cycle among %s", cycleMembers(indeg))
	}
	return order, nil
}

// cycleMembers names the tasks still carrying unmet dependencies after a topo sort —
// i.e. the members of the cycle (or downstream of it) — for an actionable error.
func cycleMembers(indeg map[string]int) string {
	var stuck []string
	for id, n := range indeg {
		if n > 0 {
			stuck = append(stuck, id)
		}
	}
	sort.Strings(stuck)
	return strings.Join(stuck, ", ")
}
