package main

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// barrierPlanner is an agent.Planner that witnesses concurrent host execution:
// every Complete() registers as an in-flight turn and blocks on a release channel
// until the test lets the whole concurrent cohort through. Its recorded peak is the
// direct evidence that ONE shared host stepped agents in parallel -- not a relabeled
// serial loop -- which is the whole property this file exists to prove.
type barrierPlanner struct {
	active *atomic.Int64
	peak   *atomic.Int64
	gate   chan struct{}
}

func (p *barrierPlanner) Model() string { return "shared-host-barrier" }

func (p *barrierPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	n := p.active.Add(1)
	for old := p.peak.Load(); n > old && !p.peak.CompareAndSwap(old, n); old = p.peak.Load() {
	}
	defer p.active.Add(-1)
	select {
	case <-p.gate:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "shared-host step"}}, nil
}

// newSharedHostRow builds one synthetic deferred enrollment row the way the wave
// execute loop does: a fresh per-row agent instance over the shared host, its own
// routed enrollment plan, and an identity finish closure.
func newSharedHostRow(rank, issue int, root string, opts dispatchTickOptions) *dispatchWaveHostShareRow {
	payload := map[string]any{"backend": opts.Backend}
	return &dispatchWaveHostShareRow{
		rank:    rank,
		target:  issue,
		runsDir: root,
		opts:    opts,
		plan:    dispatchtick.PlanHostEnrollment("docs", issue, "", []string{"docs/**"}),
		agent:   &hostEnrollAgent{issue: issue, root: root, lane: "docs", tree: []string{"docs/**"}, maxTurns: 1},
		lease:   map[string]any{"acquired": false, "id": ""},
		payload: payload,
		finish:  func(m map[string]any) map[string]any { return m },
	}
}

// TestSharedHostEnrollsWholeWaveIntoOneHost is the mass-concurrency acceptance
// witness: a live micro wave of N rows admits every row into ONE dispatchWaveHostShare
// (one host construction), drains ONCE, and maps each reaped result back to its row
// with host_result.done=true. The detached exec spawner is wired to fail the test if
// the shared path ever reaches it.
func TestSharedHostEnrollsWholeWaveIntoOneHost(t *testing.T) {
	const n = 16
	root := t.TempDir()

	oldSpawner := dispatchIssueWorkerSpawner
	spawned := false
	dispatchIssueWorkerSpawner = func(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		spawned = true
		return dispatchSpawnResult{}, nil
	}
	t.Cleanup(func() { dispatchIssueWorkerSpawner = oldSpawner })

	rows := make([]dispatchWaveExecutionPlan, n)
	for i := range rows {
		rows[i] = dispatchWaveExecutionPlan{Rank: i, Backend: "micro", Target: dispatchLaunchTarget{Lane: "docs", Issue: 100 + i, Tree: []string{"docs/**"}}}
	}
	base := dispatchTickOptions{Workspace: root, Backend: "micro", Live: true, MaxWorkers: 8, Account: &dispatchtick.Account{Tag: "acct"}}
	share, err := newDispatchWaveHostShare(rows, base)
	if err != nil {
		t.Fatalf("newDispatchWaveHostShare: %v", err)
	}

	for i := range rows {
		share.enroll(newSharedHostRow(i, 100+i, root, base))
	}
	if got := share.admittedRows(); got != n {
		t.Fatalf("admitted=%d, want %d", got, n)
	}
	if spawned {
		t.Fatal("shared-host path called the DETACHED exec spawner")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	share.drainAndFinish(ctx)

	done := 0
	for _, r := range share.rows {
		if dispatchMapBool(r.payload, "ok") && r.payload["action"] == "enrolled" {
			res := mapAt(r.payload, "host_result")
			if res["done"] == true {
				done++
			}
		}
	}
	if done != n {
		t.Fatalf("done rows=%d, want %d (receipt carries one host construction)", done, n)
	}

	rec := map[string]any{}
	dispatchWaveRecordSharedHost(rec, share, microagent.BudgetForWave(8, n), len(share.rows))
	sh := mapAt(rec, "shared_host")
	if dispatchMapInt(sh, "host_constructions") != 1 {
		t.Fatalf("host_constructions=%v, want 1", sh["host_constructions"])
	}
	if dispatchMapInt(sh, "enrolled") != n || dispatchMapInt(sh, "admitted") != n {
		t.Fatalf("shared_host enrolled/admitted = %v/%v, want %d/%d", sh["enrolled"], sh["admitted"], n, n)
	}
	share.Close()
}

// TestSharedHostPeakConcurrencyExceedsOne is the fan-out witness: N counting agents
// admitted into ONE shared host with an 8-worker budget must be stepped in parallel,
// so observed peak concurrency rises above 1 (and never exceeds the declared budget).
// This is the property a per-row Config{Workers:1} host could never produce.
func TestSharedHostPeakConcurrencyExceedsOne(t *testing.T) {
	const n = 16
	active, peak := &atomic.Int64{}, &atomic.Int64{}
	gate := make(chan struct{})

	base := dispatchTickOptions{}
	oldWorker := dispatchHostEnrollWorker
	dispatchHostEnrollWorker = func(opts dispatchTickOptions, account dispatchtick.Account) agent.Planner {
		return &barrierPlanner{active: active, peak: peak, gate: gate}
	}
	t.Cleanup(func() { dispatchHostEnrollWorker = oldWorker })

	rows := make([]dispatchWaveExecutionPlan, n)
	for i := range rows {
		rows[i] = dispatchWaveExecutionPlan{Rank: i, Backend: "micro"}
	}
	share, err := newDispatchWaveHostShare(rows, base)
	if err != nil {
		t.Fatalf("newDispatchWaveHostShare: %v", err)
	}

	for i := 0; i < n; i++ {
		share.enroll(&dispatchWaveHostShareRow{
			rank:   i,
			target: 200 + i,
			opts:   base,
			plan:   dispatchtick.PlanHostEnrollment("docs", 200+i, "", []string{"docs/**"}),
			agent:  &hostEnrollAgent{issue: 200 + i, maxTurns: 1},
		})
	}

	// Release the barrier once the cohort has had time to fill the worker slots.
	go func() {
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			if peak.Load() > 1 {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
		close(gate)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	results, err := share.host.DrainAll(ctx)
	if err != nil {
		t.Fatalf("DrainAll: %v", err)
	}
	if len(results) != n {
		t.Fatalf("reaped %d results, want %d", len(results), n)
	}
	t.Logf("shared-host peak concurrency witness: %d (budget %d)", peak.Load(), microagent.BudgetForWave(base.MaxWorkers, n))
	if peak.Load() <= 1 {
		t.Fatalf("peak concurrency=%d, want >1 (the shared host must step agents in parallel)", peak.Load())
	}
	wantBudget := microagent.BudgetForWave(base.MaxWorkers, n)
	if peak.Load() > int64(wantBudget) {
		t.Fatalf("peak concurrency=%d exceeded the declared budget %d", peak.Load(), wantBudget)
	}
	share.Close()
}

// TestSharedHostZeroAdmittedSkipsDrain pins the empty-batch edge: when every row is
// refused (or none is enrolled), the share admits zero and drainAndFinish must not
// drain or panic -- the per-row refusals are the whole story.
func TestSharedHostZeroAdmittedSkipsDrain(t *testing.T) {
	rows := []dispatchWaveExecutionPlan{{Rank: 0, Backend: "micro"}}
	share, err := newDispatchWaveHostShare(rows, dispatchTickOptions{})
	if err != nil {
		t.Fatalf("newDispatchWaveHostShare: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	share.drainAndFinish(ctx)
	if got := share.admittedRows(); got != 0 {
		t.Fatalf("admitted=%d, want 0", got)
	}
	share.Close()
}
