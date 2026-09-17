package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
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

	// De-mock (fak#13085/#13082): a live shared host no longer falls back to a canned
	// planner, so the test supplies its mock through the explicit test hook.
	oldHook := dispatchHostEnrollWorkerHook
	dispatchHostEnrollWorkerHook = func(dispatchTickOptions, dispatchtick.Account) agent.Planner {
		return hostEnrollPlanner{}
	}
	t.Cleanup(func() { dispatchHostEnrollWorkerHook = oldHook })

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

// ---------------------------------------------------------------------------
// End-to-end loop witnesses (#13080.2 / #13083).
//
// The helper-level tests above drive the seams directly. These two drive the REAL
// executeDispatchWavePlan — the actual loop wiring a live `fak dispatch wave`
// takes — so a regression in the shared-host arm of that loop is caught at the
// dispatch surface, not only at the helper boundary.
// ---------------------------------------------------------------------------

// sharedHostWavePlanRows builds the same execution-plan rows
// dispatchWaveExecutionPlans emits for a micro wave: one per issue, all on the
// docs lane, with the wave membership the tick's sidecars carry. Keeping the row
// shape identical is what makes the loop witness faithful — the test drives the
// production loop, not a re-implementation of it.
func sharedHostWavePlanRows(backend string, n int) []dispatchWaveExecutionPlan {
	rows := make([]dispatchWaveExecutionPlan, n)
	for i := range rows {
		rows[i] = dispatchWaveExecutionPlan{
			Rank:     i,
			WaveID:   "wave-e2e",
			WaveSize: n,
			Backend:  backend,
			WorkKind: "docs",
			Target:   dispatchWaveLaunchTarget(dispatchWaveCandidate{Lane: "docs", Issue: 300 + i, Tree: []string{"docs/**"}}),
			Account:  dispatchtick.AccountSidecar(dispatchtick.Account{Tag: "acct-e2e", Model: "micro-e2e"}),
		}
	}
	return rows
}

// sharedHostWaveRouterPayload is the routed backlog the wave's per-row tick reads.
func sharedHostWaveRouterPayload(issues []int) dispatchtick.RouterPayload {
	return dispatchtick.RouterPayload{
		Schema: dispatchtick.RouterSchema,
		OK:     true,
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"docs": {Tree: []string{"docs/**"}, Issues: issues, Count: len(issues)},
		},
	}
}

// TestDispatchWaveSharedHostEndToEndMicroBackend drives the REAL
// executeDispatchWavePlan with a micro backend and witnesses the shared-host
// contract at the loop surface: ONE host construction, exactly ONE drain, and a
// spawned count that matches the enrolled rows. It is the loop-level twin of
// TestSharedHostEnrollsWholeWaveIntoOneHost (#13080.2).
func TestDispatchWaveSharedHostEndToEndMicroBackend(t *testing.T) {
	const n = 4
	root := t.TempDir()
	rows := sharedHostWavePlanRows("micro", n)
	issues := make([]int, n)
	for i := range rows {
		issues[i] = rows[i].Target.Issue
	}

	var hostConstructions int32
	withDispatchJSONHelper(t, func(root string, args ...string) (map[string]any, error) {
		return map[string]any{"ok": true}, nil
	})
	// The router + issue fetch the per-row tick consults.
	oldRoute := dispatchRouteIssues
	dispatchRouteIssues = func(string, io.Writer) (dispatchtick.RouterPayload, error) {
		return sharedHostWaveRouterPayload(issues), nil
	}
	t.Cleanup(func() { dispatchRouteIssues = oldRoute })
	oldFetch := dispatchFetchIssue
	dispatchFetchIssue = func(root string, issue int) dispatchIssueInfo {
		return dispatchIssueInfo{Number: issue, Title: "e2e micro wave issue", Body: "resolve it", Labels: []string{"docs"}}
	}
	t.Cleanup(func() { dispatchFetchIssue = oldFetch })

	// The detached exec spawner must never be reached by the micro path.
	oldSpawner := dispatchIssueWorkerSpawner
	detached := int32(0)
	dispatchIssueWorkerSpawner = func(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		atomic.AddInt32(&detached, 1)
		return dispatchSpawnResult{}, nil
	}
	t.Cleanup(func() { dispatchIssueWorkerSpawner = oldSpawner })

	// ONE shared gateway is constructed per host build; count it to witness
	// host_constructions == 1 independently of the receipt.
	oldWorker := dispatchHostEnrollWorker
	dispatchHostEnrollWorker = func(opts dispatchTickOptions, account dispatchtick.Account) agent.Planner {
		atomic.AddInt32(&hostConstructions, 1)
		return &mockInProcessToolPlanner{}
	}
	t.Cleanup(func() { dispatchHostEnrollWorker = oldWorker })

	maxWorkers := 8
	live := true
	asJSON := true
	settle := 0.0
	excludeLane := ""
	codexLoopGate := dispatchCodexLoopGateDefaultThreshold()
	gateSince := 0.0
	gateLimit := dispatchCodexLoopGateDefaultLimitValue()
	rec := newDispatchWaveRecord(root, live, "micro", "docs", "", dispatchGoalProfileThroughput, n, n, nil)
	var out, errb bytes.Buffer
	code := executeDispatchWavePlan(&out, &errb, dispatchWaveExecutionRequest{
		root: root, plan: rows, maxWorkers: &maxWorkers, excludeLane: &excludeLane,
		live: &live, settleSeconds: &settle, codexLoopGate: &codexLoopGate,
		gateSinceHours: &gateSince, gateLimit: &gateLimit, asJSON: &asJSON, record: rec,
	})
	if code != 0 {
		t.Fatalf("executeDispatchWavePlan exit = %d, want 0 (stderr: %s)\n%s", code, errb.String(), out.String())
	}
	if got := atomic.LoadInt32(&detached); got != 0 {
		t.Fatalf("micro wave reached the DETACHED exec spawner %d time(s); it must enroll in-process", got)
	}

	sh := mapAt(rec, "shared_host")
	if dispatchMapInt(sh, "host_constructions") != 1 {
		t.Fatalf("shared_host.host_constructions = %v, want 1", sh["host_constructions"])
	}
	if dispatchMapInt(sh, "drains") != 1 {
		t.Fatalf("shared_host.drains = %v, want exactly 1 (one drain per wave)", sh["drains"])
	}
	if dispatchMapInt(sh, "enrolled") != n || dispatchMapInt(sh, "admitted") != n {
		t.Fatalf("shared_host enrolled/admitted = %v/%v, want %d/%d", sh["enrolled"], sh["admitted"], n, n)
	}
	if sh["abandoned"] == true {
		t.Fatalf("shared_host.abandoned = true on a cooperative batch (close_refusal=%v)", sh["close_refusal"])
	}
	if got := dispatchMapInt(rec, "spawned"); got != n {
		t.Fatalf("wave spawned = %d, want %d (one enrolled row each)", got, n)
	}
	// Exactly one host construction is the whole point of the shared-host arm.
	if got := atomic.LoadInt32(&hostConstructions); got != 1 {
		t.Fatalf("planner constructed %d time(s), want 1 shared gateway", got)
	}

	// Every row retired done with tool calls under its lease.
	ticks, _ := rec["ticks"].([]any)
	if len(ticks) != n {
		t.Fatalf("wave ticks = %d, want %d", len(ticks), n)
	}
	done := 0
	for _, tk := range ticks {
		m, _ := tk.(map[string]any)
		res := mapAt(m, "host_result")
		if res["done"] == true {
			done++
		}
	}
	if done != n {
		t.Fatalf("rows retired done = %d, want %d", done, n)
	}
}

// liveE2EEndpointEnv returns the first configured live endpoint env var, or ""
// when none is set. The integration witness SKIPS with this reason rather than
// silently passing when the operator has no live model backend (#13083).
func liveE2EEndpointEnv() (string, string) {
	for _, name := range []string{"FAK_E2E_LIVE_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "FAK_GATEWAY_URL"} {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return name, v
		}
	}
	return "", ""
}

// stuckStepPlanner is the adversarial gateway for the wave-seam close witness: its
// Complete IGNORES ctx and blocks until released, which is exactly the agent kind
// that times a drain out and then wedged the old unconditional Close.
type stuckStepPlanner struct {
	release chan struct{}
	started *atomic.Int64
}

func (p *stuckStepPlanner) Model() string { return "stuck-step" }

func (p *stuckStepPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	p.started.Add(1)
	<-p.release // deliberately not select-on-ctx
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "released"}}, nil
}

// stuckWaveAgent is a microagent whose Step ignores ctx and blocks, so the shared
// host cannot retire it and the drain times out.
type stuckWaveAgent struct{ release chan struct{} }

func (a *stuckWaveAgent) Step(context.Context, microagent.Gateway) (bool, error) {
	<-a.release
	return true, nil
}

// TestDispatchWaveSharedHostCloseIsBoundedAgainstACtxIgnoringAgent is the #13080
// acceptance witness at the WAVE seam: a share whose gateway ignores ctx drains to
// a timeout, yet dispatchWaveDrainSharedHost returns within the drain budget plus
// the bounded close margin AND stamps the refusal on the receipt
// (shared_host.drains == 1, abandoned == true, close_refusal present) instead of
// hanging on workers.Wait().
func TestDispatchWaveSharedHostCloseIsBoundedAgainstACtxIgnoringAgent(t *testing.T) {
	root := t.TempDir()
	release := make(chan struct{})
	defer close(release)
	started := &atomic.Int64{}

	// The host's ONE gateway is read from this seam at construction, so installing
	// the stuck planner BEFORE newDispatchWaveHostShare is what puts the wedge on
	// the real enrolled-agent Step path (hostEnrollAgent -> RunGovernedArm -> gw).
	oldWorker := dispatchHostEnrollWorker
	dispatchHostEnrollWorker = func(opts dispatchTickOptions, account dispatchtick.Account) agent.Planner {
		return &stuckStepPlanner{release: release, started: started}
	}
	t.Cleanup(func() { dispatchHostEnrollWorker = oldWorker })

	base := dispatchTickOptions{Workspace: root, Backend: "micro", Live: true, MaxWorkers: 2, Account: &dispatchtick.Account{Tag: "acct"}}
	rows := []dispatchWaveExecutionPlan{{Rank: 0, Backend: "micro", Target: dispatchLaunchTarget{Lane: "docs", Issue: 500, Tree: []string{"docs/**"}}}}
	share, err := newDispatchWaveHostShare(rows, base)
	if err != nil {
		t.Fatalf("newDispatchWaveHostShare: %v", err)
	}
	share.enroll(&dispatchWaveHostShareRow{
		rank:    0,
		target:  500,
		runsDir: root,
		opts:    base,
		plan:    dispatchtick.PlanHostEnrollment("docs", 500, "", []string{"docs/**"}),
		agent:   &hostEnrollAgent{issue: 500, root: root, lane: "docs", tree: []string{"docs/**"}, maxTurns: 1},
		lease:   map[string]any{"acquired": false, "id": ""},
		payload: map[string]any{"backend": "micro"},
		finish:  func(m map[string]any) map[string]any { return m },
	})
	if share.admittedRows() != 1 {
		t.Fatalf("admitted=%d, want 1", share.admittedRows())
	}

	// A tiny drain budget: the agent ignores ctx, so the drain times out at once and
	// the bounded close is what must return.
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	began := time.Now()
	done := make(chan struct{})
	go func() {
		dispatchWaveDrainSharedHost(ctx, share)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		t.Fatal("dispatchWaveDrainSharedHost hung on a ctx-ignoring agent")
	}
	if elapsed := time.Since(began); elapsed > 10*time.Second {
		t.Fatalf("drain seam took %v, want bounded by drain budget + close margin", elapsed)
	}
	if share.drains != 1 {
		t.Fatalf("drains=%d, want exactly 1", share.drains)
	}
	if !share.abandoned {
		t.Fatal("abandoned=false; the bounded close must record the wedge refusal")
	}
	if share.closeErr == nil {
		t.Fatal("closeErr=nil; the wedge refusal must be recorded")
	}

	rec := map[string]any{}
	dispatchWaveRecordSharedHost(rec, share, microagent.BudgetForWave(2, 1), len(share.rows))
	sh := mapAt(rec, "shared_host")
	if dispatchMapInt(sh, "drains") != 1 {
		t.Fatalf("receipt drains=%v, want 1", sh["drains"])
	}
	if sh["abandoned"] != true || dispatchMapString(sh, "close_refusal") == "" {
		t.Fatalf("receipt did not record the close refusal: %#v", sh)
	}
	if sh["drain_incomplete"] == nil {
		t.Fatalf("receipt missing drain_incomplete for a timed-out drain: %#v", sh)
	}
}

// TestDispatchWaveSharedHostLiveBackendEndToEnd is the #13083 integration witness:
// it drives the REAL executeDispatchWavePlan with a LIVE model backend for N >= 4
// agents and asserts the shared-host density contract end to end -- ONE host
// construction, exactly ONE drain, every agent retired done after real tool calls
// (tool_calls > 0), and zero post-wave lane collisions (every in-process lease was
// handed back, so the wave leaves no live lane behind).
//
// It requires a live endpoint. When none is configured the test SKIPS with an
// explicit, recorded reason -- it never silently passes. Gate vars (first match
// wins): FAK_E2E_LIVE_BASE_URL | OPENAI_BASE_URL | OPENAI_API_BASE | FAK_GATEWAY_URL.
func TestDispatchWaveSharedHostLiveBackendEndToEnd(t *testing.T) {
	envName, endpoint := liveE2EEndpointEnv()
	if envName == "" {
		t.Skipf("SKIP (witnessed no-live-endpoint): none of FAK_E2E_LIVE_BASE_URL, OPENAI_BASE_URL, OPENAI_API_BASE, FAK_GATEWAY_URL is set; the #13083 live wave witness requires a configured model backend")
	}
	t.Logf("live wave witness: using %s=%s", envName, endpoint)

	const n = 4
	root := t.TempDir()
	rows := sharedHostWavePlanRows("micro", n)
	issues := make([]int, n)
	for i := range rows {
		issues[i] = rows[i].Target.Issue
	}

	withDispatchJSONHelper(t, func(root string, args ...string) (map[string]any, error) {
		return map[string]any{"ok": true}, nil
	})
	oldRoute := dispatchRouteIssues
	dispatchRouteIssues = func(string, io.Writer) (dispatchtick.RouterPayload, error) {
		return sharedHostWaveRouterPayload(issues), nil
	}
	t.Cleanup(func() { dispatchRouteIssues = oldRoute })
	oldFetch := dispatchFetchIssue
	dispatchFetchIssue = func(root string, issue int) dispatchIssueInfo {
		return dispatchIssueInfo{Number: issue, Title: "live wave issue", Body: "resolve it with a real tool call", Labels: []string{"docs"}}
	}
	t.Cleanup(func() { dispatchFetchIssue = oldFetch })
	// The live path must NOT reach the detached exec spawner.
	oldSpawner := dispatchIssueWorkerSpawner
	detached := int32(0)
	dispatchIssueWorkerSpawner = func(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		atomic.AddInt32(&detached, 1)
		return dispatchSpawnResult{}, nil
	}
	t.Cleanup(func() { dispatchIssueWorkerSpawner = oldSpawner })

	var hostConstructions int32
	oldWorker := dispatchHostEnrollWorker
	dispatchHostEnrollWorker = func(opts dispatchTickOptions, account dispatchtick.Account) agent.Planner {
		atomic.AddInt32(&hostConstructions, 1)
		// The REAL live planner selection (no mock): a configured endpoint yields a
		// provider HTTP planner, so the agents take real model turns.
		return defaultDispatchHostEnrollWorker(opts, account)
	}
	t.Cleanup(func() { dispatchHostEnrollWorker = oldWorker })

	maxWorkers := 8
	live := true
	asJSON := true
	settle := 0.0
	excludeLane := ""
	codexLoopGate := dispatchCodexLoopGateDefaultThreshold()
	gateSince := 0.0
	gateLimit := dispatchCodexLoopGateDefaultLimitValue()
	rec := newDispatchWaveRecord(root, live, "micro", "docs", "", dispatchGoalProfileThroughput, n, n, nil)
	var out, errb bytes.Buffer
	code := executeDispatchWavePlan(&out, &errb, dispatchWaveExecutionRequest{
		root: root, plan: rows, maxWorkers: &maxWorkers, excludeLane: &excludeLane,
		live: &live, settleSeconds: &settle, codexLoopGate: &codexLoopGate,
		gateSinceHours: &gateSince, gateLimit: &gateLimit, asJSON: &asJSON, record: rec,
	})
	if code != 0 {
		t.Fatalf("executeDispatchWavePlan exit = %d, want 0 (stderr: %s)\n%s", code, errb.String(), out.String())
	}
	if got := atomic.LoadInt32(&detached); got != 0 {
		t.Fatalf("live micro wave reached the DETACHED exec spawner %d time(s)", got)
	}
	sh := mapAt(rec, "shared_host")
	if dispatchMapInt(sh, "host_constructions") != 1 {
		t.Fatalf("shared_host.host_constructions = %v, want 1", sh["host_constructions"])
	}
	if dispatchMapInt(sh, "drains") != 1 {
		t.Fatalf("shared_host.drains = %v, want exactly 1", sh["drains"])
	}
	if got := dispatchMapInt(rec, "spawned"); got != n {
		t.Fatalf("wave spawned = %d, want %d", got, n)
	}
	if got := atomic.LoadInt32(&hostConstructions); got != 1 {
		t.Fatalf("planner constructed %d time(s), want 1 shared gateway", got)
	}

	// Every row must have retired done THROUGH REAL TOOL CALLS and released its lease.
	ticks, _ := rec["ticks"].([]any)
	if len(ticks) != n {
		t.Fatalf("wave ticks = %d, want %d", len(ticks), n)
	}
	doneRows := 0
	for _, tk := range ticks {
		m, _ := tk.(map[string]any)
		res := mapAt(m, "host_result")
		if res["done"] != true {
			t.Fatalf("rank %v did not retire done: host_result=%#v reason=%v", m["wave_rank"], res, m["reason"])
		}
		doneRows++
		if dispatchMapInt(res, "tool_calls") <= 0 {
			t.Fatalf("rank %v tool_calls=%v, want > 0 (a real agent must call a tool)", m["wave_rank"], res["tool_calls"])
		}
		if rel := dispatchMapString(m, "lease_release"); rel != "released" && rel != "no_lease_id" {
			t.Fatalf("rank %v lease_release=%q, want released/no_lease_id (no stranded lane)", m["wave_rank"], rel)
		}
	}
	if doneRows != n {
		t.Fatalf("done rows = %d, want %d", doneRows, n)
	}

	// Zero post-wave lane collisions: a fresh scan of the runs dir must report no
	// live lane -- the wave's own in-process leases were all handed back.
	snap := scanRunsSnapshot(filepath.Join(root, dispatchtick.RunsDirName), time.Now())
	if live := snap.liveLanes(); len(live) != 0 {
		t.Fatalf("post-wave live lanes = %v, want none (zero lane collisions)", live)
	}
}
