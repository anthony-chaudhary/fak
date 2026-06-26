package workflow

// execute.go — the DAG execution engine and its fault-tolerance core.
//
// Execute walks a compiled Graph wave by wave: every task whose dependencies have all
// SUCCEEDED is ready, and a wave's ready tasks run concurrently (bounded by
// Options.Concurrency). Each task's Runner receives the outputs of the tasks it needs,
// so data flows along the DAG edges. Fault tolerance is three rules:
//
//   - Retries: a task is attempted 1+Retries times; the first success wins.
//   - Skip propagation: if a task ultimately fails (or is skipped), every task that
//     transitively depends on it is marked Skipped — never run with a missing input.
//   - Fail-fast (default) vs continue-on-error: on the first failure, fail-fast skips
//     all not-yet-run tasks and stops; continue-on-error skips only the failed task's
//     descendants and keeps running independent branches.
//
// The engine is deterministic given a deterministic Runner: ready tasks are processed
// in sorted-ID order and results are folded in that order, so Result.Order and every
// status are reproducible regardless of goroutine scheduling.

import (
	"context"
	"runtime"
	"sort"
	"sync"
)

// Status is the terminal disposition of a task in a run.
type Status string

const (
	StatusSucceeded Status = "succeeded"
	StatusFailed    Status = "failed"
	StatusSkipped   Status = "skipped" // not run: an upstream task failed/was skipped, or fail-fast aborted the run
)

// RunInput is what a Runner receives for one task: the node plus the outputs of every
// task it depends on, keyed by upstream task ID.
type RunInput struct {
	Node Node
	Deps map[string]string
}

// Runner performs the actual unit of work for a task. The orchestration layer owns
// scheduling, dependency resolution, retries, and failure propagation; the Runner owns
// only "given this op and these inputs, produce an output or an error." A model call, a
// tool call, or a pure function all fit behind this one seam.
type Runner interface {
	Run(ctx context.Context, in RunInput) (output string, err error)
}

// RunnerFunc adapts a plain function to a Runner.
type RunnerFunc func(ctx context.Context, in RunInput) (string, error)

// Run implements Runner.
func (f RunnerFunc) Run(ctx context.Context, in RunInput) (string, error) {
	return f(ctx, in)
}

// NodeResult is the outcome of one task.
type NodeResult struct {
	ID       string
	Status   Status
	Output   string
	Attempts int
	Err      string // non-empty only when Status == StatusFailed
}

// Result is the outcome of a whole run: one NodeResult per task, the deterministic
// order they settled in, and whether any task failed.
type Result struct {
	Workflow string
	Nodes    map[string]NodeResult
	Order    []string
	Failed   bool
}

// Options tune a run. The zero value is valid and fail-fast (the safe default): a
// failing task aborts the run rather than letting downstream tasks run on partial data.
type Options struct {
	Concurrency     int  // max tasks in flight per wave (default: GOMAXPROCS)
	ContinueOnError bool // false (default) = fail-fast; true = skip only descendants of a failure
}

// Execute runs a compiled graph with the given Runner.
func Execute(ctx context.Context, g *Graph, r Runner, opt Options) Result {
	if opt.Concurrency <= 0 {
		opt.Concurrency = runtime.GOMAXPROCS(0)
	}
	byID := make(map[string]Node, len(g.Nodes))
	dependents := make(map[string][]string, len(g.Nodes))
	for _, n := range g.Nodes {
		byID[n.ID] = n
	}
	for _, n := range g.Nodes {
		for _, d := range n.Needs {
			dependents[d] = append(dependents[d], n.ID)
		}
	}

	res := Result{Workflow: g.Name, Nodes: make(map[string]NodeResult, len(g.Nodes))}
	done := make(map[string]bool, len(g.Nodes))

	// settle records a terminal result exactly once, in deterministic fold order.
	settle := func(nr NodeResult) {
		if done[nr.ID] {
			return
		}
		done[nr.ID] = true
		res.Nodes[nr.ID] = nr
		res.Order = append(res.Order, nr.ID)
		if nr.Status == StatusFailed {
			res.Failed = true
		}
	}
	// skip marks a task Skipped and propagates to every transitive dependent.
	var skip func(id, reason string)
	skip = func(id, reason string) {
		if done[id] {
			return
		}
		settle(NodeResult{ID: id, Status: StatusSkipped, Err: reason})
		for _, dep := range dependents[id] {
			skip(dep, "upstream "+id+" did not succeed")
		}
	}

	for {
		// Ready = not done, with every dependency done AND succeeded. A node whose
		// dependency is done-but-not-succeeded is handled by skip propagation, so it
		// will already be done before we get here.
		var ready []string
		for _, n := range g.Nodes {
			if done[n.ID] {
				continue
			}
			ok := true
			for _, d := range n.Needs {
				if !done[d] || res.Nodes[d].Status != StatusSucceeded {
					ok = false
					break
				}
			}
			if ok {
				ready = append(ready, n.ID)
			}
		}
		if len(ready) == 0 {
			break
		}
		sort.Strings(ready)

		// Run the wave concurrently, bounded by Concurrency; collect into a map and
		// fold in sorted order so the result is scheduling-independent.
		wave := make(map[string]NodeResult, len(ready))
		var mu sync.Mutex
		var wg sync.WaitGroup
		sem := make(chan struct{}, opt.Concurrency)
		for _, id := range ready {
			id := id
			node := byID[id]
			deps := make(map[string]string, len(node.Needs))
			for _, d := range node.Needs {
				deps[d] = res.Nodes[d].Output
			}
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-sem }()
				nr := runOne(ctx, r, node, deps)
				mu.Lock()
				wave[id] = nr
				mu.Unlock()
			}()
		}
		wg.Wait()

		failed := false
		for _, id := range ready { // sorted → deterministic fold
			nr := wave[id]
			settle(nr)
			if nr.Status == StatusFailed {
				failed = true
				if opt.ContinueOnError {
					for _, dep := range dependents[id] {
						skip(dep, "upstream "+id+" failed")
					}
				}
			}
		}
		if failed && !opt.ContinueOnError {
			// Fail-fast: skip everything not yet settled and stop.
			for _, n := range g.Nodes {
				if !done[n.ID] {
					settle(NodeResult{ID: n.ID, Status: StatusSkipped, Err: "run aborted by an earlier failure (fail-fast)"})
				}
			}
			break
		}
	}
	return res
}

// runOne runs a single task with its retry budget, honoring context cancellation. The
// first successful attempt wins; otherwise the last error is recorded.
func runOne(ctx context.Context, r Runner, node Node, deps map[string]string) NodeResult {
	attempts := 1 + node.Retries
	if attempts < 1 {
		attempts = 1
	}
	var lastErr error
	for i := 0; i < attempts; i++ {
		if err := ctx.Err(); err != nil {
			return NodeResult{ID: node.ID, Status: StatusFailed, Attempts: i, Err: err.Error()}
		}
		out, err := r.Run(ctx, RunInput{Node: node, Deps: deps})
		if err == nil {
			return NodeResult{ID: node.ID, Status: StatusSucceeded, Output: out, Attempts: i + 1}
		}
		lastErr = err
	}
	return NodeResult{ID: node.ID, Status: StatusFailed, Attempts: attempts, Err: lastErr.Error()}
}
