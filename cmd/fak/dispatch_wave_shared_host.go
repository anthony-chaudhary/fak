package main

import (
	"context"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// dispatchSharedHost is the ONE microagent host a live --backend micro driver
// shares across every agent it enrolls (#2030 / native-harness massive-concurrency).
//
// Before this seam each driver executed its units strictly sequentially and each
// unit's dispatchTickHostEnroll built its OWN Config{Workers:1, Queue:1} host, ran
// exactly one prototype agent, and tore it down (cmd/fak/dispatch_tick_hostenroll.go).
// A 1000-issue wave therefore paid ~1000 host constructions at global peak
// concurrency 1, and the verified microagent capacity (one host, N agents) was
// unreachable from the real dispatch path. CONCURRENCY IS A PROPERTY OF THE HOST,
// NOT OF THE TICK: this share constructs ONE host sized from the driver's declared
// worker budget, enrolls every admitted unit into it, drains ONCE, and maps each
// reaped result back to the unit that produced it.
//
// This type is the REUSABLE lifecycle seam (#13084): the one-shot `fak dispatch
// wave` and the continuous `fak dispatch sweep` / `auto` drivers all build one of
// these, enroll per unit/tick, and drain once at the end of the run. The wave
// enrolls a priced batch and drains when the batch is admitted; the sweep enrolls
// one agent per tick and drains after the whole loop returns. Neither drains
// more than once: microagent's Host latches `draining` on the first Drain and
// never resets it (internal/microagent/microagent.go), so "enroll per tick then
// drain per tick" is structurally impossible — one host, one drain, one run.
//
// Tree-safety and per-unit accounting are preserved BY CONSTRUCTION: routing, the
// shared duplicate/collision/lane gates, and the lane-lease fence all still run
// per-unit in dispatchTickHostEnroll over the SAME fence tree the detached path
// would hold; only the host lifecycle is shared. A held peer lease still refuses
// that unit (and only that unit). Each enrolled agent keeps its own hostEnrollAgent
// instance (one spawn + one done of per-agent audit) and its own lease, released
// per-agent on a clean retire exactly as the standalone path does (#4324).
type dispatchSharedHost struct {
	host *microagent.WaveHost
	sink *hostEnrollSink

	rows  []*dispatchWaveHostShareRow
	byRow map[int]*dispatchWaveHostShareRow

	// drains counts how many times this share drained. It is the ONE-drain
	// witness the receipt emits (always 1 on the live-micro path) and it
	// makes a second drain a visible regression rather than an assumption.
	drains int
	// drainErr is the run's drain outcome: nil on a clean retire, the typed
	// microagent.ErrWaveTimeout when a wedged agent outlived the drain budget.
	drainErr error
	// closeErr is non-nil when the BOUNDED close abandoned a wedged agent
	// instead of blocking on workers.Wait() (#13080). It is the hang-proofing
	// witness: a run with an uncooperative agent still returns.
	closeErr error
	// abandoned is set when closeErr is a deadline/cancel, i.e. the run gave up
	// waiting for an agent whose Step ignored ctx rather than stalling forever.
	abandoned bool
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

func (s *dispatchSharedHost) row(rank int) *dispatchWaveHostShareRow {
	if s == nil {
		return nil
	}
	return s.byRow[rank]
}

// newDispatchSharedHost builds the ONE host a live micro run shares. It sizes the
// resident worker budget from the run's declared --max-workers ceiling and the
// batch size (microagent.BudgetForWave: the declared cap, never the backlog) and
// the pending queue from the batch size so a full batch is admitted without
// refusing its own tail. The gateway is the base planner: the host holds ONE
// gateway and hands the same handle to every Step, which is exactly the
// shared-gateway inversion the in-process path exists to buy.
//
// This is the reusable lifecycle seam (#13084): `batchCap` is how many agents the
// run may enroll before its single drain (a wave's row count, or a sweep's
// --max-agents ceiling), while `workerBudget` is the declared concurrency cap both
// callers pass in their base options. Capacity is never widened to the batch: a
// 1000-agent sweep on a 6-worker budget still runs 6-wide and queues the rest.
func newDispatchSharedHost(base dispatchTickOptions, batchCap int) (*dispatchSharedHost, error) {
	gw := agent.Planner(dispatchHostEnrollWorker(base, base.accountOrZero()))
	sink := &hostEnrollSink{}
	workers := microagent.BudgetForWave(base.MaxWorkers, batchCap)
	host, err := microagent.NewWaveHost(gw, microagent.Config{Audit: sink}, microagent.WaveConfig{Workers: workers, Queue: batchCap})
	if err != nil {
		return nil, err
	}
	return &dispatchSharedHost{
		host:  host,
		sink:  sink,
		rows:  make([]*dispatchWaveHostShareRow, 0, batchCap),
		byRow: make(map[int]*dispatchWaveHostShareRow, batchCap),
	}, nil
}

// newDispatchWaveHostShare is the wave's thin binding of the shared lifecycle seam:
// a live micro wave shares ONE host sized to its row batch. Kept as a named shim so
// the wave executor reads as it always did while the lifecycle itself is common.
func newDispatchWaveHostShare(rows []dispatchWaveExecutionPlan, base dispatchTickOptions) (*dispatchSharedHost, error) {
	return newDispatchSharedHost(base, len(rows))
}

// enroll records and admits one row's agent into the shared host. It mirrors the
// standalone Spawn: a row whose agent is refused is reported per-row and the rest
// of the batch still runs. No drain happens here -- the wave drains once, later.
func (s *dispatchSharedHost) enroll(r *dispatchWaveHostShareRow) {
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
func (s *dispatchSharedHost) admittedRows() int {
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
func (s *dispatchSharedHost) drainAndFinish(ctx context.Context) {
	if s == nil {
		return
	}
	byID := map[string]microagent.Result{}
	if s.admittedRows() > 0 {
		s.drains++
		results, drainErr := s.host.DrainAll(ctx)
		s.drainErr = drainErr
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
func (s *dispatchSharedHost) finishRow(r *dispatchWaveHostShareRow, byID map[string]microagent.Result) {
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
func (s *dispatchSharedHost) Close() {
	if s != nil && s.host != nil {
		s.host.Close()
	}
}

// closeWithin is the wave's BOUNDED close seam (#13080): it releases the shared
// host within ctx, and when a wedged agent (one whose Step ignores ctx) outlives
// the budget it records the refusal on the share and RETURNS instead of blocking
// on the host's workers.Wait(). It is deliberately not Close(): the base Close
// keeps its unbounded semantics so every other caller stays byte-identical, while
// the wave — the one caller that can hold an untrusted batch — is hang-proof.
func (s *dispatchSharedHost) closeWithin(ctx context.Context) {
	if s == nil || s.host == nil {
		return
	}
	if err := s.host.CloseContext(ctx); err != nil {
		s.closeErr = err
		s.abandoned = true
	}
}

// closeBounded releases the shared host with the #13080 hang-proof margin,
// recording an abandoned wedge on the share instead of blocking. It is the seam a
// driver's deferred early-return backup calls, so the bounded-close budget lives in
// exactly ONE place and no caller names its own timeout context.
func (s *dispatchSharedHost) closeBounded() {
	if s == nil || s.host == nil {
		return
	}
	closeCtx, cancel := context.WithTimeout(context.Background(), dispatchWaveSharedHostCloseMargin)
	defer cancel()
	s.closeWithin(closeCtx)
}

// dispatchSharedHostReceipt stamps the run-level shared-host receipt: ONE host
// construction, its resident worker budget, and the enrolled/admitted counts. It is
// additive -- a non-micro run never records it, so those receipts stay byte-identical.
// Both the wave and the sweep call it, so `host_constructions == 1` means the same
// thing (one host for the whole run) on either driver.
func dispatchSharedHostReceipt(rec map[string]any, share *dispatchSharedHost, workers, enrolled int) {
	if rec == nil || share == nil {
		return
	}
	rec["shared_host"] = map[string]any{
		"host_constructions": 1,
		"peak_workers":       workers,
		"enrolled":           enrolled,
		"admitted":           share.admittedRows(),
		"drains":             share.drains,
	}
	if share.drainErr != nil {
		rec["shared_host"].(map[string]any)["drain_incomplete"] = share.drainErr.Error()
	}
	// #13080: a close that returned ctx.Err() abandoned a wedged (ctx-ignoring)
	// agent rather than blocking on workers.Wait(). Record it as an explicit
	// refusal so the run is never silently hung and the wedge is auditable.
	if share.abandoned {
		rec["shared_host"].(map[string]any)["abandoned"] = true
		rec["shared_host"].(map[string]any)["close_refusal"] = share.closeErr.Error()
	}
}

// dispatchWaveRecordSharedHost is the wave's thin alias of the shared receipt
// stamper, kept so the wave executor and its tests name the wave's own receipt.
func dispatchWaveRecordSharedHost(rec map[string]any, share *dispatchSharedHost, workers, enrolled int) {
	dispatchSharedHostReceipt(rec, share, workers, enrolled)
}

// dispatchSharedHostDrain is the run's one drain seam. The driver calls it after
// its enroll loop has admitted every agent; it drains the shared host once, maps
// results back, and closes it — BOUNDED, so a wedged agent cannot hang the run
// (#13080). The drain gets the declared budget ctx while the close gets a fresh,
// independent margin: an agent that ignored the drain ctx is exactly the one the
// close margin must survive, so reusing the (already expired) drain ctx would make
// every timeout a guaranteed abandonment.
func dispatchSharedHostDrain(ctx context.Context, share *dispatchSharedHost) {
	if share == nil {
		return
	}
	share.drainAndFinish(ctx)
	closeCtx, cancel := context.WithTimeout(context.Background(), dispatchWaveSharedHostCloseMargin)
	defer cancel()
	share.closeWithin(closeCtx)
}

// dispatchWaveDrainSharedHost is the wave's thin alias of the one-drain seam.
func dispatchWaveDrainSharedHost(ctx context.Context, share *dispatchSharedHost) {
	dispatchSharedHostDrain(ctx, share)
}

// dispatchWaveSharedHostCloseMargin is the bounded close budget the wave grants
// after the drain budget ends (#13080). It is deliberately small: every agent
// that honors ctx has already retired by now, so anything still resident is a
// wedge the wave must abandon, not wait out.
const dispatchWaveSharedHostCloseMargin = 5 * time.Second

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
