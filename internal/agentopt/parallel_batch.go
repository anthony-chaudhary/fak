package agentopt

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// BatchCall represents one tool call submitted as part of a multi-tool turn.
type BatchCall struct {
	ID         string         `json:"id"`
	Name       string         `json:"name"`
	Args       map[string]any `json:"args"`
	ReadPaths  []string       `json:"read_paths,omitempty"`
	WritePaths []string       `json:"write_paths,omitempty"`
	ReadOnly   bool           `json:"read_only"`
}

// BatchResult records the output, timing, and execution stage of a completed call.
type BatchResult struct {
	CallID   string        `json:"call_id"`
	Output   string        `json:"output"`
	Error    error         `json:"error,omitempty"`
	Duration time.Duration `json:"duration"`
	Stage    int           `json:"stage"`
}

// BatchExecutor schedules and runs multi-tool batches with dependency graph resolution.
type BatchExecutor struct {
	MaxConcurrency int
}

// NewBatchExecutor constructs a BatchExecutor with bounded concurrency.
func NewBatchExecutor(maxConcurrency int) *BatchExecutor {
	if maxConcurrency <= 0 {
		maxConcurrency = 8
	}
	return &BatchExecutor{MaxConcurrency: maxConcurrency}
}

// normalizePath cleans a path for conflict matching.
func normalizePath(p string) string {
	return filepath.Clean(filepath.ToSlash(strings.TrimSpace(p)))
}

// pathsConflict reports whether any path in setA matches or overlaps with a path in setB.
func pathsConflict(setA, setB []string) bool {
	for _, a := range setA {
		na := normalizePath(a)
		for _, b := range setB {
			nb := normalizePath(b)
			if na == nb || strings.HasPrefix(na, nb+"/") || strings.HasPrefix(nb, na+"/") {
				return true
			}
		}
	}
	return false
}

// callsHaveConflict reports whether callB depends on callA (where A precedes B).
func callsHaveConflict(a, b BatchCall) bool {
	// A pure read before another pure read never conflicts.
	if a.ReadOnly && b.ReadOnly {
		return false
	}
	// If a writes and b reads (RAW)
	if len(a.WritePaths) > 0 && len(b.ReadPaths) > 0 && pathsConflict(a.WritePaths, b.ReadPaths) {
		return true
	}
	// If a reads and b writes (WAR)
	if len(a.ReadPaths) > 0 && len(b.WritePaths) > 0 && pathsConflict(a.ReadPaths, b.WritePaths) {
		return true
	}
	// If both write to overlapping paths (WAW)
	if len(a.WritePaths) > 0 && len(b.WritePaths) > 0 && pathsConflict(a.WritePaths, b.WritePaths) {
		return true
	}
	// Mutating call with unknown write footprint conservatively depends on all prior operations.
	if !a.ReadOnly && len(a.WritePaths) == 0 {
		return true
	}
	if !b.ReadOnly && len(b.WritePaths) == 0 {
		return true
	}
	return false
}

// PartitionStages resolves the dependency graph and groups calls into sequential tiers.
// Calls within the same stage have zero data dependencies and can execute concurrently.
func (b *BatchExecutor) PartitionStages(calls []BatchCall) [][]BatchCall {
	if len(calls) == 0 {
		return nil
	}

	n := len(calls)
	// Track which stage index each call is assigned to.
	stageOf := make([]int, n)

	for i := 0; i < n; i++ {
		maxDepStage := -1
		for j := 0; j < i; j++ {
			if callsHaveConflict(calls[j], calls[i]) {
				if stageOf[j] > maxDepStage {
					maxDepStage = stageOf[j]
				}
			}
		}
		stageOf[i] = maxDepStage + 1
	}

	numStages := 0
	for _, s := range stageOf {
		if s+1 > numStages {
			numStages = s + 1
		}
	}

	stages := make([][]BatchCall, numStages)
	for i, c := range calls {
		s := stageOf[i]
		stages[s] = append(stages[s], c)
	}

	return stages
}

// Execute runs all calls in the batch respecting stage dependency boundaries.
// Within each stage, calls run in parallel up to MaxConcurrency.
func (b *BatchExecutor) Execute(
	ctx context.Context,
	calls []BatchCall,
	execFn func(ctx context.Context, call BatchCall) (string, error),
) ([]BatchResult, error) {
	stages := b.PartitionStages(calls)
	var allResults []BatchResult
	var mu sync.Mutex

	for stageIdx, stageCalls := range stages {
		if err := ctx.Err(); err != nil {
			return allResults, err
		}

		results := make([]BatchResult, len(stageCalls))
		sem := make(chan struct{}, b.MaxConcurrency)
		var wg sync.WaitGroup

		for i, call := range stageCalls {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int, c BatchCall) {
				defer wg.Done()
				defer func() { <-sem }()

				start := time.Now()
				out, err := execFn(ctx, c)
				elapsed := time.Since(start)

				res := BatchResult{
					CallID:   c.ID,
					Output:   out,
					Error:    err,
					Duration: elapsed,
					Stage:    stageIdx,
				}
				mu.Lock()
				results[idx] = res
				mu.Unlock()
			}(i, call)
		}

		wg.Wait()
		allResults = append(allResults, results...)
	}

	return allResults, nil
}
