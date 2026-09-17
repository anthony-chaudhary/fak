package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// dispatchTickHostEnroll is the cmd/fak glue for the #2030 host-enroll path: the
// counterpart of dispatchTickLiveSpawn for the micro backend. Instead of exec-spawning
// one detached guarded CLI process per lane, it enrolls the routed issue as a single
// microagent into a real in-process host (internal/microagent, M2). It runs AFTER the
// shared duplicate / collision / lane-busy / no-issue gates and BEFORE the CLI-only
// model/guard/command machinery (BuildWorkerCommand refuses micro), so the host path is
// a first-class, config-selected peer of the detached spawn — a default fleet never
// sets --backend micro, so nothing here runs on a default tick.
//
// Tree-safety is preserved by construction: the live branch acquires the SAME lane
// lease over the SAME fence tree the detached path would hold (acquireDispatchLaneLease
// + PlanHostEnrollment copy the routed tree verbatim), so a held peer lease refuses the
// enrollment exactly as it refuses a detached spawn (M11 lane-lease disjointness). The
// host runs behind ONE shared gateway and ONE audit sink, and every lifecycle event is
// recorded per-agent — the in-process contrast to one hash-chained JSONL per detached
// guard process.
//
// Generation frame (#2030 is gen/second-next — an architectural OPTION behind an
// explicit gate, never default exposure). Closing evidence:
//   - Promotion evidence: dispatch_tick_hostenroll_test.go witnesses a routed issue
//     enrolling into a real host (spawn+done audit, retired done) WITHOUT touching the
//     detached exec spawner, under a live lane lease over the routed tree. Promote the
//     micro path off opt-in once the #2033 density/$-per-task benchmark confirms the
//     per-agent process weight was the binding cost.
//   - Demotion / retirement: drop the micro path if #2033 shows per-agent cost is
//     dominated by provider seats/rate limits (the host buys no density), or if the
//     isolation floor (#2018) demands a per-agent OS process anyway.
//   - Real loop: hostEnrollAgent.Step resolves the issue through the OWNED agent loop —
//     it arms the focused code tool catalog (ArmFocusedCodeTools over a.root) and calls
//     agent.RunGovernedArm, so the enrolled microagent proposes and EXECUTES tool calls
//     (edits included) against the fence tree under ONE host-shared gateway. This is not
//     a prototype no-op.
//
// Model planner selection (live path): a live tick WILL NOT enroll on a canned planner.
// It constructs a REAL HTTP planner from the first available provider route and fails
// the enrollment with a typed no_planner reason when none can be built. The supported
// LOCAL route is an OpenAI-compatible endpoint: point OPENAI_BASE_URL (or the legacy
// OPENAI_API_BASE) at the local gateway (e.g. http://127.0.0.1:8080/v1) with
// OPENAI_API_KEY set to any non-empty token. ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY and
// FAK_GATEWAY_URL (with FAK_API_KEY) are the hosted/gateway alternatives. The offline
// canned hostEnrollPlanner is reachable ONLY via the explicit test hook
// (dispatchHostEnrollWorkerHook) or the non-live WOULD_ENROLL path — never a live run.
func dispatchTickHostEnroll(root, runsDir string, opts dispatchTickOptions, pick dispatchLanePick, leaseID string, account dispatchtick.Account, target int, payload map[string]any, finish func(map[string]any) map[string]any) map[string]any {
	plan := dispatchtick.PlanHostEnrollment(pick.Lane, target, leaseID, pick.Tree)
	payload["host_enrollment"] = map[string]any{
		"agent_id": plan.AgentID,
		"lane":     plan.Lane,
		"issue":    plan.Issue,
		"lease_id": plan.LeaseID,
		"tree":     append([]string(nil), plan.Tree...),
	}

	if !opts.Live {
		payload["ok"] = true
		payload["action"] = "would_enroll"
		payload["verdict"] = "WOULD_ENROLL"
		payload["reason"] = fmt.Sprintf("safe to enroll issue #%d (lane %q) as microagent %q into the in-process %s host under account %q; lease tree %v matches the detached path", target, pick.Lane, plan.AgentID, opts.Backend, account.Tag, plan.Tree)
		return finish(payload)
	}

	// Live: acquire the SAME lane lease the detached path would hold, so lane-lease
	// disjointness (M11) is enforced identically — a held peer lease refuses here
	// exactly as it refuses a detached spawn.
	lease := acquireDispatchLaneLease(root, leaseID, pick.Lane, pick.Tree, opts.WorkerTimeoutS+dispatchtick.LeaseTTLMarginS, opts.Goal)
	if applyDispatchLaneLease(payload, lease, fmt.Sprintf("lane %q lease is held by a live peer; not enrolling issue #%d into the host", pick.Lane, target)) {
		recordDispatchPayload(runsDir, opts.Backend, payload)
		return finish(payload)
	}

	// Live WAVE enrollment into ONE shared host (#2030 massive-concurrency). A wave
	// sets opts.SharedHost so every row's routing/lease/gate evaluation lands in the
	// SAME host instead of constructing a private Config{Workers:1,Queue:1} host per
	// tick -- the host lifecycle is shared, the per-row fence is not. Enrollment is
	// deferred: no drain happens here, and this row's payload is finalized once the
	// wave drains the batch (dispatch_wave_shared_host.go). A standalone tick leaves
	// SharedHost nil and takes the byte-identical private-host path below.
	if opts.SharedHost != nil {
		agentInst := &hostEnrollAgent{issue: target, root: root, lane: pick.Lane, tree: pick.Tree, maxTurns: opts.MaxTurns}
		opts.SharedHost.enroll(&dispatchWaveHostShareRow{
			rank:    opts.WaveSharedRank,
			target:  target,
			runsDir: runsDir,
			opts:    opts,
			plan:    plan,
			agent:   agentInst,
			lease:   lease,
			payload: payload,
			finish:  finish,
		})
		payload["host_pending"] = true
		return payload
	}
	// Enroll the routed issue as ONE microagent into a real in-process host over one
	// shared gateway and one audit sink — the M2 host lifecycle
	// (Spawn -> Step -> retire -> Reap), not an exec.Command spawn.
	sink := &hostEnrollSink{}
	planner := dispatchHostEnrollWorker(opts, account)
	if planner == nil {
		return dispatchHostEnrollNoPlanner(runsDir, opts, payload, finish, account)
	}
	host, err := microagent.NewHost(planner, microagent.Config{Workers: 1, Queue: 1, Audit: sink})
	if err != nil {
		return dispatchHostEnrollFailed(runsDir, opts, payload, finish, fmt.Sprintf("microagent host construct failed for issue #%d: %v", target, err))
	}
	defer host.Close()
	agentInst := &hostEnrollAgent{
		issue:    target,
		root:     root,
		lane:     pick.Lane,
		tree:     pick.Tree,
		maxTurns: opts.MaxTurns,
	}
	if err := host.Spawn(plan.AgentID, agentInst); err != nil {
		return dispatchHostEnrollFailed(runsDir, opts, payload, finish, fmt.Sprintf("host refused to enroll microagent %q for issue #%d: %v", plan.AgentID, target, err))
	}
	ctx, cancel := context.WithTimeout(context.Background(), dispatchHostEnrollDrainTimeout)
	defer cancel()
	drainErr := host.Drain(ctx)
	results := host.Reap()

	steps, done := 0, false
	for _, r := range results {
		if r.ID == plan.AgentID {
			steps, done = r.Steps, r.Done
		}
	}
	payload["host_audit"] = map[string]any{
		"spawns":          sink.count(microagent.EventSpawn),
		"dones":           sink.count(microagent.EventDone),
		"errors":          sink.count(microagent.EventError),
		"cancels":         sink.count(microagent.EventCancel),
		"distinct_agents": sink.agentCount(),
	}
	payload["host_result"] = map[string]any{
		"agent_id":   plan.AgentID,
		"steps":      steps,
		"done":       done,
		"turns":      agentInst.metrics.Turns,
		"tool_calls": agentInst.metrics.ToolCalls,
		"metrics":    agentInst.metrics,
	}

	if drainErr != nil || !done {
		return dispatchHostEnrollFailed(runsDir, opts, payload, finish, fmt.Sprintf("microagent %q for issue #%d did not retire done (done=%v drain_err=%v)", plan.AgentID, target, done, drainErr))
	}

	// #4324 release-on-exit, in-process twin. The microagent retired DONE after a clean
	// drain, so the work this lane lease fenced is finished inside this very process and
	// nothing more will be written under it — but unlike a detached worker there is no
	// witness sweep that will ever grade this slot, so without an explicit hand-back the
	// lease strands for its full TTL every single time and refuses peers against a holder
	// that has already returned. The fenced CAS delete is the same one the witness sweep
	// uses. Only THIS path releases: every dispatchHostEnrollFailed return above (host
	// construct fault, refused spawn, drain error or a microagent that did not retire
	// done) is an abnormal exit whose lane may be mid-step, and those correctly keep the
	// lease until TTL expiry. Fail-open — the outcome is surfaced, never propagated.
	payload["lease_release"] = releaseInProcessLaneLease(root, lease)
	payload["launch_id"] = newHostEnrollmentLaunchID(opts.Backend, target)
	payload["ok"] = true
	payload["action"] = "enrolled"
	payload["verdict"] = "ENROLLED"
	payload["reason"] = fmt.Sprintf("enrolled issue #%d (lane %q) as microagent %q into the in-process %s host under %q (%d step(s), %d turn(s), %d tool call(s), lease tree %v)", target, pick.Lane, plan.AgentID, opts.Backend, account.Tag, steps, agentInst.metrics.Turns, agentInst.metrics.ToolCalls, plan.Tree)
	recordDispatchPayload(runsDir, opts.Backend, payload)
	return finish(payload)
}

// newHostEnrollmentLaunchID creates a per-enrollment identity without inventing an OS
// process PID. It is generated once and copied into every alias and immutable receipt.
func newHostEnrollmentLaunchID(backend string, issue int) string {
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err == nil {
		return fmt.Sprintf("%s-%d-%s", backend, issue, hex.EncodeToString(nonce[:]))
	}
	// crypto/rand failure must not erase the launch witness. The timestamp fallback is
	// process-local but still distinguishes sequential in-process enrollments.
	return fmt.Sprintf("%s-%d-%d", backend, issue, time.Now().UnixNano())
}

// dispatchHostEnrollFailed marks a live host-enroll payload as a non-benign
// ENROLL_FAILED (exit code 1, like SPAWN_FAILED) and records it. Kept separate so every
// live failure edge records the run and returns through finish identically.
func dispatchHostEnrollFailed(runsDir string, opts dispatchTickOptions, payload map[string]any, finish func(map[string]any) map[string]any, reason string) map[string]any {
	payload["ok"] = false
	payload["action"] = "enroll_failed"
	payload["verdict"] = "ENROLL_FAILED"
	payload["reason"] = reason
	recordDispatchPayload(runsDir, opts.Backend, payload)
	return finish(payload)
}

// dispatchHostEnrollNoPlanner is the typed de-mock failure for a live enrollment with no
// real model planner: it stamps enroll_failure=no_planner alongside the ENROLL_FAILED
// verdict so an operator (and the tests) can distinguish "no model configured" from a
// host/spawn/drain fault, and can never mistake a no-op for a real ENROLLED run.
func dispatchHostEnrollNoPlanner(runsDir string, opts dispatchTickOptions, payload map[string]any, finish func(map[string]any) map[string]any, account dispatchtick.Account) map[string]any {
	model := firstString(opts.WorkerModel, account.Model)
	if model == "" {
		model = "claude-3-5-sonnet-20241022"
	}
	payload["enroll_failure"] = enrollFailureNoPlanner
	err := &errHostEnrollNoPlanner{model: model, tried: []string{"anthropic", "openai", "fak-gateway"}}
	return dispatchHostEnrollFailed(runsDir, opts, payload, finish, err.Error())
}

// dispatchHostEnrollDrainTimeout bounds how long the tick waits for the enrolled
// microagent to retire after its governed arm loop. It is a generous liveness
// backstop, not a steady-state budget.
const dispatchHostEnrollDrainTimeout = 30 * time.Second

// hostEnrollPlanner is the canned offline agent.Planner used ONLY by the non-live
// WOULD_ENROLL path and the explicit test hook (dispatchHostEnrollWorkerHook): it
// returns a deterministic completion with no network, credentials, or model call,
// so unit tests and dry runs never depend on an upstream. It is NEVER returned by a
// live tick, whose planner selection fails typed (enrollFailureNoPlanner) instead.
type hostEnrollPlanner struct{}

func (hostEnrollPlanner) Model() string { return "micro-offline-canned" }

func (hostEnrollPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "microagent offline canned step (non-live/test only)"}}, nil
}

// enrollFailureNoPlanner is the typed enroll_failure class stamped on a live micro
// enrollment that could not construct a real model planner. It is the de-mock
// contract: a live tick reports ENROLL_FAILED/no_planner rather than a fake ENROLLED.
const enrollFailureNoPlanner = "no_planner"

// errHostEnrollNoPlanner is the typed error returned when no real planner is available.
type errHostEnrollNoPlanner struct {
	model string
	tried []string
}

func (e *errHostEnrollNoPlanner) Error() string {
	return fmt.Sprintf("no live model planner for the micro host-enroll path: model %q has no constructible endpoint (tried %s); set OPENAI_BASE_URL (local OpenAI-compatible gateway, or legacy OPENAI_API_BASE) with OPENAI_API_KEY, or ANTHROPIC_BASE_URL with ANTHROPIC_API_KEY, or FAK_GATEWAY_URL with FAK_API_KEY; a non-live run uses the offline canned planner", e.model, strings.Join(e.tried, ", "))
}

// dispatchHostEnrollWorkerHook is an optional test hook to intercept host-enroll worker
// creation. It is the ONLY route by which a test reaches the canned hostEnrollPlanner on
// the live path; the default worker never returns it on live.
var dispatchHostEnrollWorkerHook func(opts dispatchTickOptions, account dispatchtick.Account) agent.Planner

// dispatchHostEnrollWorker returns the agent.Planner for in-process issue resolution.
// It returns a real provider HTTP planner when a live endpoint is configured, the
// canned hostEnrollPlanner when NOT live (dry-run/test), and nil when live with no
// constructible planner — never a mock. A nil result is a typed failure the caller
// must surface; see defaultDispatchHostEnrollWorker.
var dispatchHostEnrollWorker = defaultDispatchHostEnrollWorker

func defaultDispatchHostEnrollWorker(opts dispatchTickOptions, account dispatchtick.Account) agent.Planner {
	if dispatchHostEnrollWorkerHook != nil {
		if p := dispatchHostEnrollWorkerHook(opts, account); p != nil {
			return p
		}
	}
	if !opts.Live {
		return hostEnrollPlanner{}
	}

	model := firstString(opts.WorkerModel, account.Model)
	if model == "" {
		model = "claude-3-5-sonnet-20241022"
	}

	if key := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); key != "" || strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL")) != "" {
		baseURL := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL"))
		if p, err := agent.NewProviderHTTPPlanner("anthropic", baseURL, model, key); err == nil {
			return p
		}
	}

	if key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); key != "" || strings.TrimSpace(os.Getenv("OPENAI_BASE_URL")) != "" || strings.TrimSpace(os.Getenv("OPENAI_API_BASE")) != "" {
		baseURL := firstString(strings.TrimSpace(os.Getenv("OPENAI_BASE_URL")), strings.TrimSpace(os.Getenv("OPENAI_API_BASE")))
		if p, err := agent.NewProviderHTTPPlanner("openai", baseURL, model, key); err == nil {
			return p
		}
	}

	if gw := strings.TrimSpace(os.Getenv("FAK_GATEWAY_URL")); gw != "" {
		return agent.NewHTTPPlanner(gatewayBaseURL(gw), model, strings.TrimSpace(os.Getenv("FAK_API_KEY")))
	}

	return nil
}

// hostEnrollAgent executes in-process issue resolution through the owned agent loop.
type hostEnrollAgent struct {
	issue    int
	root     string
	lane     string
	tree     []string
	maxTurns int
	opts     []agent.RunOption
	metrics  agent.ArmMetrics
	traces   []agent.CallTrace
	err      error
}

func (a *hostEnrollAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	task := fmt.Sprintf("Resolve issue #%d in lane %s (fence tree: %v)", a.issue, a.lane, a.tree)
	maxTurns := a.maxTurns
	if maxTurns <= 0 {
		maxTurns = 10
	}
	runOpts := append([]agent.RunOption(nil), a.opts...)
	if a.root != "" {
		codeCat, armErr := agent.ArmFocusedCodeTools(a.root)
		if armErr == nil {
			defer agent.DisarmCodeTools()
			runOpts = append(runOpts, agent.WithToolCatalog(codeCat))
		}
	}
	m, traces, err := agent.RunGovernedArm(ctx, gw, task, maxTurns, runOpts...)
	a.metrics = m
	a.traces = traces
	a.err = err
	if err != nil {
		return false, err
	}
	return true, nil
}

// hostEnrollSink is the host's ONE audit sink: it counts per-agent lifecycle events so
// the tick can WITNESS that per-agent audit (M11) is preserved on the in-process path.
type hostEnrollSink struct {
	mu     sync.Mutex
	kinds  map[microagent.EventKind]int
	agents map[string]bool
}

func (s *hostEnrollSink) Record(ev microagent.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.kinds == nil {
		s.kinds = map[microagent.EventKind]int{}
		s.agents = map[string]bool{}
	}
	s.kinds[ev.Kind]++
	s.agents[ev.Agent] = true
}

func (s *hostEnrollSink) count(k microagent.EventKind) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.kinds[k]
}

func (s *hostEnrollSink) agentCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.agents)
}
