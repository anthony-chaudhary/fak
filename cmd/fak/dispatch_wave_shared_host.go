package main

import (
	"context"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// dispatchWaveHostShare is the ONE microagent host a live --backend micro wave
// shares across every row it enrolls (#2030 / native-harness massive-concurrency).
//
// Before this seam the wave executed its rows strictly sequentially and each row's
// dispatchTickHostEnroll built its OWN Config{Workers:1, Queue:1} host, ran exactly
// one prototype agent, and tore it down (cmd/fak/dispatch_tick_hostenroll.go). A
// 1000-issue wave therefore paid ~1000 host constructions at global peak
// concurrency 1, and the verified microagent capacity (one host, N agents) was
// unreachable from the real dispatch path. CONCURRENCY IS A PROPERTY OF THE HOST,
// NOT OF THE TICK: this share constructs ONE host sized from the wave's declared
// worker budget, enrolls every admitted row into it, drains ONCE, and maps each
// reaped result back to the row that produced it.
//
// Tree-safety and per-row accounting are preserved BY CONSTRUCTION: routing, the
// shared duplicate/collision/lane gates, and the lane-lease fence all still run
// per-row in dispatchTickHostEnroll over the SAME fence tree the detached path
// would hold; only the host lifecycle is shared. A held peer lease still refuses
// that row (and only that row). Each enrolled agent keeps its own hostEnrollAgent
// instance (one spawn + one done of per-agent audit) and its own lease, released
// per-agent on a clean retire exactly as the standalone path does (#4324).
type dispatchWaveHostShare struct {
	host *microagent.WaveHost
	sink *hostEnrollSink

	rows  []*dispatchWaveHostShareRow
	byRow map[int]*dispatchWaveHostShareRow
}

// dispatchWaveHostShareRow is one wave row's deferred enrollment: everything the
// standalone dispatchTickHostEnroll would have carried into a private host, held
// here until the shared drain so the row's result can be mapped back after ONE
// DrainAll.
type dispatchWaveHostShareRow struct {
	rank    int
	target  int
	runsDir string
	opts    dispatchTickOptions
	plan    dispatchtick.HostEnrollment
	agent   *hostEnrollAgent
	lease   map[string]any
	payload map[string]any
	finish  func(map[string]any) map[string]any

	// admitted is true once the shared host accepted this row's agent. refusal
	// carries the typed host refusal verbatim for a row the bounded queue refused.
	admitted bool
	refusal  error
	pending  bool
}

func (s *dispatchWaveHostShare) row(rank int) *dispatchWaveHostShareRow {
	if s == nil {
		return nil
	}
	return s.byRow[rank]
}

// newDispatchWaveHostShare builds the ONE host a live micro wave shares. It sizes
// the resident worker budget from the wave's declared --max-workers ceiling and the
// row count (microagent.BudgetForWave: the declared cap, never the backlog) and the
// pending queue from the row count so a full wave is admitted without refusing its
// own tail. The gateway is the row-0 planner: the host holds ONE gateway and hands
// the same handle to every Step, which is exactly the shared-gateway inversion the
// in-process path exists to buy.
func newDispatchWaveHostShare(rows []dispatchWaveExecutionPlan, base dispatchTickOptions) (*dispatchWaveHostShare, error) {
	gw := agent.Planner(dispatchHostEnrollWorker(base, base.accountOrZero()))
	sink := &hostEnrollSink{}
	workers := microagent.BudgetForWave(base.MaxWorkers, len(rows))
	host, err := microagent.NewWaveHost(gw, microagent.Config{Audit: sink}, microagent.WaveConfig{Workers: workers, Queue: len(rows)})
	if err != nil {
		return nil, err
	}
	return &dispatchWaveHostShare{
		host:  host,
		sink:  sink,
		rows:  make([]*dispatchWaveHostShareRow, 0, len(rows)),
		byRow: make(map[int]*dispatchWaveHostShareRow, len(rows)),
	}, nil
}

// enroll records and admits one row's agent into the shared host. It mirrors the
// standalone Spawn: a row whose agent is refused is reported per-row and the rest
// of the batch still runs. No drain happens here -- the wave drains once, later.
func (s *dispatchWaveHostShare) enroll(r *dispatchWaveHostShareRow) {
	r.pending = true
	s.rows = append(s.rows, r)
	s.byRow[r.rank] = r
	res := s.host.Enroll([]microagent.Enrollment{{ID: r.plan.AgentID, Agent: r.agent}})
	if len(res) == 1 {
		r.admitted = res[0].Admitted
		r.refusal = res[0].Err
	}
}

// admittedRows counts the rows this share accepted, for the receipt's admitted.
func (s *dispatchWaveHostShare) admittedRows() int {
	n := 0
	for _, r := range s.rows {
		if r.admitted {
			n++
		}
	}
	return n
}

// drainAndFinish drains the shared host ONCE and maps every reaped result back to
// its row. Each admitted row then renders its final host_result/host_audit, releases
// its own lane lease on a clean retire (#4324), records its payload, and runs its
// own finish closure. A row whose agent was refused, or whose agent did not retire
// done, is rendered as a per-row ENROLL_FAILED -- the rest of the batch is untouched.
func (s *dispatchWaveHostShare) drainAndFinish(ctx context.Context) {
	if s == nil {
		return
	}
	byID := map[string]microagent.Result{}
	if s.admittedRows() > 0 {
		results, _ := s.host.DrainAll(ctx)
		for _, res := range results {
			byID[res.ID] = res
		}
	}
	for _, r := range s.rows {
		s.finishRow(r, byID)
	}
}

// finishRow renders one row's deferred outcome into its payload. It is the shared
// twin of the standalone tail of dispatchTickHostEnroll: same keys, same audit
// counters, same #4324 release-on-exit, same verdict vocabulary.
func (s *dispatchWaveHostShare) finishRow(r *dispatchWaveHostShareRow, byID map[string]microagent.Result) {
	payload := r.payload
	payload["host_audit"] = map[string]any{
		"spawns":          s.sink.count(microagent.EventSpawn),
		"dones":           s.sink.count(microagent.EventDone),
		"errors":          s.sink.count(microagent.EventError),
		"cancels":         s.sink.count(microagent.EventCancel),
		"distinct_agents": s.sink.agentCount(),
	}
	if !r.admitted {
		payload["host_result"] = map[string]any{
			"agent_id": r.plan.AgentID,
			"done":     false,
			"steps":    0,
		}
		dispatchHostEnrollFailed(r.runsDir, r.opts, payload, r.finish, fmt.Sprintf("host refused to enroll microagent %q for issue #%d: %v", r.plan.AgentID, r.target, r.refusal))
		return
	}
	res := byID[r.plan.AgentID]
	payload["host_result"] = map[string]any{
		"agent_id":   r.plan.AgentID,
		"steps":      res.Steps,
		"done":       res.Done,
		"turns":      r.agent.metrics.Turns,
		"tool_calls": r.agent.metrics.ToolCalls,
		"metrics":    r.agent.metrics,
	}
	if !res.Done {
		dispatchHostEnrollFailed(r.runsDir, r.opts, payload, r.finish, fmt.Sprintf("microagent %q for issue #%d did not retire done (done=%v drain_err=%v)", r.plan.AgentID, r.target, res.Done, res.Err))
		return
	}
	payload["lease_release"] = releaseInProcessLaneLease(r.opts.Workspace, r.lease)
	payload["launch_id"] = newHostEnrollmentLaunchID(r.opts.Backend, r.target)
	payload["ok"] = true
	payload["action"] = "enrolled"
	payload["verdict"] = "ENROLLED"
	payload["reason"] = fmt.Sprintf("enrolled issue #%d (lane %q) as microagent %q into the shared in-process %s host (%d step(s), %d turn(s), %d tool call(s), lease tree %v)", r.target, r.plan.Lane, r.plan.AgentID, r.opts.Backend, res.Steps, r.agent.metrics.Turns, r.agent.metrics.ToolCalls, r.plan.Tree)
	recordDispatchPayload(r.runsDir, r.opts.Backend, payload)
	r.finish(payload)
}

// Close releases the shared host. Idempotent.
func (s *dispatchWaveHostShare) Close() {
	if s != nil && s.host != nil {
		s.host.Close()
	}
}

// dispatchWaveRecordSharedHost stamps the wave-level shared-host receipt: ONE host
// construction, its resident worker budget, and the enrolled/admitted counts. It is
// additive -- a non-micro wave never records it, so those receipts stay byte-identical.
func dispatchWaveRecordSharedHost(rec map[string]any, share *dispatchWaveHostShare, workers, enrolled int) {
	if rec == nil || share == nil {
		return
	}
	rec["shared_host"] = map[string]any{
		"host_constructions": 1,
		"peak_workers":       workers,
		"enrolled":           enrolled,
		"admitted":           share.admittedRows(),
	}
}

// dispatchWaveDrainSharedHost is the live-micro wave's one drain seam. The wave
// calls it after the per-row execution loop has enrolled every row; it drains the
// shared host once, maps results back, and closes it.
func dispatchWaveDrainSharedHost(ctx context.Context, share *dispatchWaveHostShare) {
	if share == nil {
		return
	}
	share.drainAndFinish(ctx)
	share.Close()
}

// sharedHostDrainTimeout bounds how long a wave waits for the whole enrolled batch
// to retire. It scales the single-tick backstop by the wave's shape so a large
// batch that fans out across K workers is not cut off at the single-agent budget.
func sharedHostDrainTimeout(rows, workers int) time.Duration {
	if workers < 1 {
		workers = 1
	}
	waves := (rows + workers - 1) / workers
	if waves < 1 {
		waves = 1
	}
	return time.Duration(waves) * dispatchHostEnrollDrainTimeout
}

// accountOrZero returns the tick's account, or the zero account when none is set.
func (o dispatchTickOptions) accountOrZero() dispatchtick.Account {
	if o.Account != nil {
		return *o.Account
	}
	return dispatchtick.Account{}
}
