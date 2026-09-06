package reflexagent

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

// ReflexTask specifies a bounded unit of work executed by an ephemeral worker under a lane lease.
type ReflexTask struct {
	ID           string   `json:"id"`
	Description  string   `json:"description"`
	LaneName     string   `json:"lane_name"`
	LaneKind     string   `json:"lane_kind"`
	TreePatterns []string `json:"tree_patterns"`
	ExecuteFn    func(ctx context.Context) (any, error)
}

// TaskResult captures execution outcome, timing measurements, and output payload for a task.
type TaskResult struct {
	TaskID    string        `json:"task_id"`
	LaneName  string        `json:"lane_name"`
	Success   bool          `json:"success"`
	Output    any           `json:"output,omitempty"`
	Error     string        `json:"error,omitempty"`
	SpawnTime time.Duration `json:"spawn_time"`
	RunTime   time.Duration `json:"run_time"`
}

// ReflexMicroAgentProfile coordinates worker lifecycle and manages lane lease concurrency.
type ReflexMicroAgentProfile struct {
	mu           sync.RWMutex
	arbiter      *agentopt.ConcurrencyClassArbiter
	spawnCounter int64
}

// NewReflexMicroAgentProfile constructs a profile using the supplied concurrency arbiter.
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

// SpawnAndExecute obtains a lane lease, executes the task callback, and releases the lease.
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

// RunParallel dispatches tasks concurrently across goroutines while preserving disjoint lease safety.
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
