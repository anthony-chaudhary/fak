package reflexagent

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

// ReflexTask represents a bounded, leaf-level task assigned to a reflex micro-agent.
type ReflexTask struct {
	ID           string   `json:"id"`
	Description  string   `json:"description"`
	LaneName     string   `json:"lane_name"`
	LaneKind     string   `json:"lane_kind"`
	TreePatterns []string `json:"tree_patterns"`
	ExecuteFn    func(ctx context.Context) (any, error)
}

// TaskResult captures the execution outcome and metrics for a reflex task.
type TaskResult struct {
	TaskID    string        `json:"task_id"`
	LaneName  string        `json:"lane_name"`
	Success   bool          `json:"success"`
	Output    any           `json:"output,omitempty"`
	Error     string        `json:"error,omitempty"`
	SpawnTime time.Duration `json:"spawn_time"`
	RunTime   time.Duration `json:"run_time"`
}

// ReflexMicroAgentProfile represents the fast-spawn execution profile.
// It bypasses heavy session initialization and multi-agent coordination,
// executing atomic tasks concurrently under tree-disjoint lane leases.
type ReflexMicroAgentProfile struct {
	mu           sync.RWMutex
	arbiter      *agentopt.ConcurrencyClassArbiter
	spawnCounter int64
}

// NewReflexMicroAgentProfile creates a reflex micro-agent profile backed by a concurrency arbiter.
func NewReflexMicroAgentProfile(arbiter *agentopt.ConcurrencyClassArbiter) *ReflexMicroAgentProfile {
	if arbiter == nil {
		arbiter = agentopt.NewConcurrencyClassArbiter(map[string]int{
			"leaf":    16,
			"cluster": 4,
			"global":  1,
		})
	}
	return &ReflexMicroAgentProfile{
		arbiter: arbiter,
	}
}

// SpawnAndExecute launches a micro-agent worker for a task, acquiring a tree-disjoint lease,
// running the leaf task with sub-millisecond setup overhead, and releasing the lease.
func (p *ReflexMicroAgentProfile) SpawnAndExecute(ctx context.Context, task ReflexTask) (*TaskResult, error) {
	spawnStart := time.Now()
	workerID := fmt.Sprintf("reflex-worker-%d", atomic.AddInt64(&p.spawnCounter, 1))

	laneKind := task.LaneKind
	if laneKind == "" {
		laneKind = "leaf"
	}
	laneName := task.LaneName
	if laneName == "" {
		laneName = fmt.Sprintf("lane-%s", task.ID)
	}

	// 1. Acquire tree-disjoint lane lease
	arbRes := p.arbiter.AcquireLease(agentopt.LaneLeaseRequest{
		LaneKind:     laneKind,
		LaneName:     laneName,
		TreePatterns: task.TreePatterns,
		WorkerID:     workerID,
	})

	if !arbRes.Granted {
		return &TaskResult{
			TaskID:    task.ID,
			LaneName:  laneName,
			Success:   false,
			Error:     fmt.Sprintf("lease acquisition refused: %s", arbRes.Reason),
			SpawnTime: time.Since(spawnStart),
		}, fmt.Errorf("lease acquisition refused: %s", arbRes.Reason)
	}

	defer p.arbiter.ReleaseLease(workerID, laneName)

	spawnDuration := time.Since(spawnStart)

	// 2. Execute leaf task
	runStart := time.Now()
	var out any
	var runErr error
	if task.ExecuteFn != nil {
		out, runErr = task.ExecuteFn(ctx)
	}

	runDuration := time.Since(runStart)

	res := &TaskResult{
		TaskID:    task.ID,
		LaneName:  laneName,
		Success:   runErr == nil,
		Output:    out,
		SpawnTime: spawnDuration,
		RunTime:   runDuration,
	}
	if runErr != nil {
		res.Error = runErr.Error()
	}

	return res, runErr
}

// RunParallel runs a slice of reflex tasks in parallel, respecting lane tree-disjointness.
func (p *ReflexMicroAgentProfile) RunParallel(ctx context.Context, tasks []ReflexTask) ([]*TaskResult, error) {
	results := make([]*TaskResult, len(tasks))
	var wg sync.WaitGroup
	errCh := make(chan error, len(tasks))

	for i, task := range tasks {
		wg.Add(1)
		go func(idx int, t ReflexTask) {
			defer wg.Done()
			res, err := p.SpawnAndExecute(ctx, t)
			results[idx] = res
			if err != nil {
				errCh <- err
			}
		}(i, task)
	}

	wg.Wait()
	close(errCh)

	var firstErr error
	if len(errCh) > 0 {
		firstErr = <-errCh
	}

	return results, firstErr
}
