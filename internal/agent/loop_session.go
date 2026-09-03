package agent

// loop_session.go — the session-control seam for the agent turn loop. It threads a
// per-session DRIVE state (internal/session.Table) into RunArm as an OPTIONAL
// trailing option, so the loop reads its budget/pace/run-state each turn instead of
// running blindly to a fixed maxTurns. With no option passed, runConfig.table is nil
// and session.Table.Decide is a permissive no-op (nil receiver) — so every existing
// caller and the default loop are byte-for-byte unchanged. This is the live-loop
// half of docs/notes/SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md.

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/toolproc"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// RunOption configures an optional behavior of RunArm / Run. The zero set of options
// is the historical behavior; each option opts into one capability (today: a session
// drive-state table). It is the variadic-options idiom so adding a capability never
// breaks an existing positional call site.
type RunOption func(*runConfig)

// ModelRequestBoundary is the exact logical request handed to one Planner call.
// Messages is the single context-planned slice; Injected identifies directives
// spliced immediately before planning that survived into this boundary.
type ModelRequestBoundary struct {
	Model      string
	Turn       int
	Stream     bool
	MaxTokens  int
	Messages   []Message
	Tools      []ToolDef
	Injected   []Message
	InputClaim *InputClaimBinding
}

// ModelRequestObserver persists or audits one model boundary. Returning an error
// refuses the Planner call so the wire cannot advance without its receipt.
type ModelRequestObserver func(ModelRequestBoundary) error

// WithModelRequestObserver observes each exact model-boundary request. A nil
// observer is a literal no-op and preserves the historical loop.
func WithModelRequestObserver(observer ModelRequestObserver) RunOption {
	return func(c *runConfig) { c.modelRequestObserver = observer }
}

// InterruptedTurn is the exact assistant prefix delivered before a streamed
// model turn ended abnormally. Chunks preserve sink-delivery boundaries; Reason
// is the closed client-safe terminal classification for the completion error.
type InterruptedTurn struct {
	Turn   int
	Chunks []string
	Reason Termination
}

// InterruptedTurnObserver persists or audits an abnormal streamed turn. It is
// invoked even when Chunks is empty so the terminal reason can close the turn
// explicitly without inventing an assistant message.
type InterruptedTurnObserver func(InterruptedTurn) error

// WithInterruptedTurnObserver observes streamed completion failures after any
// admitted input claim has been released. A nil observer preserves the
// historical non-durable path.
func WithInterruptedTurnObserver(observer InterruptedTurnObserver) RunOption {
	return func(c *runConfig) { c.interruptedTurnObserver = observer }
}

// AdmittedInputClaim is the exact directive set removed from the live inbox for
// one turn. Claim runs before prompt assembly, so later arrivals cannot leak into
// the request being built.
type AdmittedInputClaim struct {
	Turn   int
	Inputs []Message
}

// InputClaimBinding is the durable identity returned by the claim store and
// carried unchanged into the exact model-request receipt.
type InputClaimBinding struct {
	ID     string
	SHA256 string
	Count  int
}

// InputClaimLifecycle persists a claim before assembly and releases it after a
// failed assembly or dispatch. Release must be idempotent.
type InputClaimLifecycle struct {
	Claim   func(AdmittedInputClaim) (InputClaimBinding, error)
	Release func(InputClaimBinding, string) error
}

// WithInputClaimLifecycle wires durable native-loop input ownership.
func WithInputClaimLifecycle(lifecycle InputClaimLifecycle) RunOption {
	return func(c *runConfig) { c.inputClaims = &lifecycle }
}

// PromptAssembler is the model-facing context assembly seam. The optional
// function makes blocking and failure behavior directly witnessable; nil keeps
// the existing SessionPlanner path.
type PromptAssembler func(context.Context, []Message) ([]Message, error)

// WithPromptAssembler overrides prompt assembly for an owned loop.
func WithPromptAssembler(assembler PromptAssembler) RunOption {
	return func(c *runConfig) { c.promptAssembler = assembler }
}

// WithFinalGate requires an independently checked post-condition before a model
// final answer may end the owned loop. A failed check returns the fact to the model
// in-band and the next iteration re-runs the normal session/budget gate first.
func WithFinalGate(check func() (satisfied bool, missingWitness string)) RunOption {
	return func(c *runConfig) { c.finalGate = check }
}

// WithResponseProfileSource records where the already-resolved response profile came
// from. Selection remains on syspromptmmu's existing environment seam; this option adds
// provenance to the run artifact without creating another profile mechanism.
func WithResponseProfileSource(source string) RunOption {
	return func(c *runConfig) { c.responseProfileSource = strings.TrimSpace(source) }
}

// WithGracefulDrain configures the loop to perform a graceful drain and final
// synthesis turn with tools disabled when the turn cap or budget stop condition is reached.
func WithGracefulDrain(enabled bool) RunOption {
	return func(c *runConfig) { c.gracefulDrain = enabled }
}

// runConfig is the resolved option set for one RunArm invocation. The zero value is
// the historical loop (nil table => permissive Decide => no per-turn gate; nil route
// => Engine left unset => kernel default for every tool call).
type runConfig struct {
	table     *session.Table
	gate      *SessionGate
	trace     string
	route     *modelroute.Manifest
	roster    *modelroute.Roster
	principal string
	// spawnPlace is the OPTIONAL per-spawn placement policy (#5420, WithSpawnPlacement
	// in spawn_place.go). When set, a tool call that creates delegated work gets its own
	// walk down the roster's zone ladder for its own declared work class instead of
	// inheriting the engine this turn was routed to. nil => no spawn is ever placed and
	// the loop is byte-for-byte the historical loop.
	spawnPlace            *SpawnPlacementPolicy
	spec                  *abi.Speculator
	contextPlanner        *SessionPlanner
	contextBaselineOutput int
	toolTerminalWake      *ToolTerminalWakeQueue
	finalGate             func() (bool, string)
	responseProfileSource string
	gracefulDrain         bool
	// observer is the typed loop-progress sink (#5148, WithProgressObserver in
	// loop_observe.go). nil => every emitProgress is a no-op and the loop is
	// byte-for-byte the historical loop.
	observer    ProgressObserver
	progressSeq uint64
	// midflight is the owned per-run mid-flight verb mailbox (#5158, WithMidflightVerbs
	// in loop_midflight.go). The loop consumes its queued interrupt / drop-pending-call /
	// set-budget verbs at each CLEAN turn boundary. nil => no mailbox wired and every
	// mid-flight consult is a no-op, so the loop is byte-for-byte the historical loop.
	midflight *MidflightVerbs
	// conversation / toolCatalog are the WIRE seam (#6657, loop_wire.go): a served
	// request's ordered transcript and its request-scoped tool declarations. Both empty
	// => the historical fixed seed (system prompt + task, ToolCatalog()).
	conversation []Message
	toolCatalog  []ToolDef
	// modelRequestObserver runs synchronously after directive splicing and the
	// one context-planning pass, immediately before the Planner call.
	modelRequestObserver    ModelRequestObserver
	interruptedTurnObserver InterruptedTurnObserver
	inputClaims             *InputClaimLifecycle
	promptAssembler         PromptAssembler
}

// ToolTerminalWakeKind is the typed reason a background-tool terminal
// transition re-enters its owning turn loop.
const ToolTerminalWakeKind = "WAKE_TOOL_TERMINAL"

// ToolTerminalWake carries the folded terminal verdict that caused a loop wake.
type ToolTerminalWake struct {
	Kind    string        `json:"kind"`
	TraceID string        `json:"trace_id"`
	Session string        `json:"session"`
	Verdict toolproc.Proc `json:"verdict"`
}

// ToolTerminalWakeRecord makes enqueue/defer/dispatch decisions inspectable.
type ToolTerminalWakeRecord struct {
	Wake   ToolTerminalWake `json:"wake"`
	Status string           `json:"status"`
}

// ToolTerminalWakeQueue is an owned, one-session wake mailbox and journal.
type ToolTerminalWakeQueue struct {
	trace   string
	signal  chan struct{}
	mu      sync.Mutex
	queued  []ToolTerminalWake
	records []ToolTerminalWakeRecord
	pending *ToolTerminalWake
}

// NewToolTerminalWakeQueue constructs the mailbox for one live session.
func NewToolTerminalWakeQueue(trace string) *ToolTerminalWakeQueue {
	return &ToolTerminalWakeQueue{trace: trace, signal: make(chan struct{}, 1)}
}

// Enqueue is suitable for toolprocgate.Supervisor.SetTerminalSink. Verdicts
// owned by another session are ignored rather than waking the wrong loop.
func (q *ToolTerminalWakeQueue) Enqueue(p toolproc.Proc) {
	if q == nil || p.Session != q.trace || (p.State != toolproc.StateDone && p.State != toolproc.StateKilled) {
		return
	}
	w := ToolTerminalWake{Kind: ToolTerminalWakeKind, TraceID: p.CallID, Session: p.Session, Verdict: p}
	q.mu.Lock()
	q.queued = append(q.queued, w)
	q.records = append(q.records, ToolTerminalWakeRecord{Wake: w, Status: "ENQUEUED"})
	q.mu.Unlock()
	select {
	case q.signal <- struct{}{}:
	default:
	}
}

// Journal returns a stable copy of the wake decision journal.
func (q *ToolTerminalWakeQueue) Journal() []ToolTerminalWakeRecord {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := make([]ToolTerminalWakeRecord, len(q.records))
	copy(out, q.records)
	return out
}

func (q *ToolTerminalWakeQueue) next() ToolTerminalWake {
	q.mu.Lock()
	defer q.mu.Unlock()
	w := q.queued[0]
	q.pending = &w
	return w
}

// release returns an unadmitted claim to the queue and re-arms its wake signal.
func (q *ToolTerminalWakeQueue) release() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.pending = nil
	if len(q.queued) > 0 {
		select {
		case q.signal <- struct{}{}:
		default:
		}
	}
}

func (q *ToolTerminalWakeQueue) mark(status string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.pending == nil {
		return
	}
	q.records = append(q.records, ToolTerminalWakeRecord{Wake: *q.pending, Status: status})
	if status == "DISPATCHED" {
		q.queued = q.queued[1:]
		q.pending = nil
		if len(q.queued) > 0 {
			select {
			case q.signal <- struct{}{}:
			default:
			}
		}
	}
}

// SessionGate is the FUNCTION-shaped per-turn session-control seam — the same gate
// WithSessionTable installs, but for a caller that holds injected hook functions
// rather than the concrete *session.Table. The gateway is the motivating caller: it
// stays decoupled from internal/session (it carries SessionDecideFunc/SessionDebitFunc,
// not a Table), so it cannot pass WithSessionTable; it wires those exact hooks here
// instead, and RunArm gates each turn boundary on the SAME live drive state the proxy
// path reads. Either field may be nil (a nil Decide proceeds with no cap; a nil Debit
// drops the usage report), so a partial gate is safe.
type SessionGate struct {
	// Decide gates one turn boundary: it returns the per-turn output cap (0 = no cap),
	// whether the loop should PROCEED, the inter-turn pace gap in ms, and the closed
	// stop reason when it should not proceed. It mirrors session.Table.Decide projected
	// onto primitives.
	Decide func(trace string) (maxTokens int, proceed bool, minGapMs int, reason string)
	// Debit reports a completed turn's usage back to the drive state (output + context
	// tokens), the function-shaped twin of session.Table.DebitUsage.
	Debit func(trace string, outputTokens, contextTokens int)
	// Wait parks a non-terminal hold until the session resumes. It is called only for
	// a PAUSED function-shaped gate; table-based harness callers keep their historical
	// single-shot "stop this arm" behavior.
	Wait func(trace string) (resumed bool, reason string)
	// Nudge returns the model-facing context advisory for this turn boundary ("" =
	// nothing to say) — the function-shaped twin of session.Table.ContextNudge
	// (#2197): when the session's cost ring shows the context window grew sharply
	// last turn, the loop splices the returned line into this turn's input so the
	// model corrects its own context use. Optional; nil drops the nudge, never the
	// turn.
	Nudge func(trace string) string
	// Checkpoint records the in-flight turn's write-ahead retry checkpoint (#1363, epic
	// #1193) — the function-shaped twin of session.Table.SetPendingTurn. RunArm binds the
	// planner's PendingTurnCheckpoint hook to it, so a retry inside HTTPPlanner.Complete
	// writes how far the turn had gotten (attempt/last-status/start) keyed on this run's
	// trace; the zero value (attempt=0,lastStatus=0,startedAt=0) CLEARS it on completion.
	// A restart reading a non-zero checkpoint re-enters that turn instead of a fresh
	// turn-0. Optional; nil drops the checkpoint, never the turn.
	Checkpoint func(trace string, attempt, lastStatus int, startedAtUnixNano int64)
	// ResumeCheckpoint returns the write-ahead turn checkpoint the session carries at loop
	// entry (#1363/#4124) — the READ twin of Checkpoint. A run keyed on a session that was
	// Restore'd with a non-zero PendingTurn returns (attempt, lastStatus, startedAtUnixNano)
	// here so runArm re-enters that turn instead of a fresh turn-0; the zero triple means
	// nothing was in flight. Optional; nil defers to the concrete table (or, with neither
	// wired, no resume) — the function-shaped twin of reading table.Get(trace).PendingTurn.
	ResumeCheckpoint func(trace string) (attempt, lastStatus int, startedAtUnixNano int64)
	// TerminateSignal returns the channel closed when the session enters Terminating
	// (#2758) — the function-shaped twin of session.Table.TerminateSignal. When wired,
	// runArm cancels the in-flight turn's context on the signal and dispatches no
	// further tool call (the forceful stop), instead of waiting for the boundary like
	// a drain. Optional; nil keeps terminate at boundary granularity (the Decide gate
	// still stops the arm with the closed TERMINATED reason at its next turn).
	TerminateSignal func(trace string) <-chan struct{}
}

// WithSessionTable wires a per-session drive-state table and the trace id this run is
// keyed under into RunArm. Each turn boundary the loop calls table.Decide(trace) to
// gate the turn on the session's live run-state + budget + pace, and Debit reports the
// turn's token usage back. A nil table is accepted (it degrades to the historical
// loop), so a caller can pass the option unconditionally.
func WithSessionTable(table *session.Table, trace string) RunOption {
	return func(c *runConfig) {
		c.table = table
		c.trace = trace
	}
}

// WithToolTerminalWake wires the owned background-tool terminal mailbox.
func WithToolTerminalWake(q *ToolTerminalWakeQueue) RunOption {
	return func(c *runConfig) { c.toolTerminalWake = q }
}

// WithSessionGate wires a FUNCTION-shaped session gate (and the trace id this run is
// keyed under) into RunArm — the decoupled twin of WithSessionTable for a caller that
// holds Decide/Debit hooks rather than the concrete *session.Table (the gateway native
// serve loop). Each turn boundary the loop calls gate.Decide(trace) to gate on the live
// run-state + budget + pace, and gate.Debit reports the turn's token usage back. A zero
// SessionGate is accepted (it degrades to the historical loop), so a caller may pass the
// option unconditionally. Wiring the trace also arms drainSteer for this run.
func WithSessionGate(g SessionGate, trace string) RunOption {
	return func(c *runConfig) {
		gate := g
		c.gate = &gate
		c.trace = trace
	}
}

// WithRouteManifest wires an OPTIONAL per-tool-call routing policy into the in-process
// agent loop. When set, the fak arm classifies each tool call into a
// modelroute.Subject{Aspect: AspectToolCall, Tool: ...}, routes it, and binds the
// chosen model for a single-model PICK to abi.ToolCall.Engine BEFORE k.Syscall — the
// same pre-submit ordering the gateway child uses, so the residency PDP adjudicates the
// real route (#598 / epic #595). A nil manifest is accepted and degrades to the
// historical loop (Engine left unset => the loop's kernel.New("localtools") default), so
// a caller may pass the option unconditionally.
func WithRouteManifest(m *modelroute.Manifest) RunOption {
	return func(c *runConfig) { c.route = m }
}

// WithRouteAccounts wires an OPTIONAL model-ACCOUNT roster (the bring-your-own-account
// switcher, #2528) into the in-process agent loop. When set alongside a routing
// manifest, each single-model PICK the manifest chooses is RESOLVED through the roster
// to the account-bound, residency-honest Target.EngineRoute() ("openai:acct/gpt-5.5",
// "local:box/llama3.2") BEFORE it is bound to abi.ToolCall.Engine — the same seam the
// served gateway uses (Server.resolveRoute), so the standalone loop and the gateway can
// no longer diverge on WHOSE account a routed id dispatches to. A nil roster is accepted
// and leaves the abstract routed id verbatim (byte-for-byte the pre-roster loop), so a
// caller may pass the option unconditionally; a manifest is still required for the roster
// to bind anything (the roster is only consulted for a resolved PICK). An id the roster
// cannot resolve (no binding, no default account) is a FAIL-LOUD error at the call site,
// never a silent fallback to the kernel default.
func WithRouteAccounts(r *modelroute.Roster) RunOption {
	return func(c *runConfig) { c.roster = r }
}

// WithRoutePrincipal wires the caller's tenant ISOLATION principal (the org/project a
// keyset key authenticated as, #5332) into the in-process agent loop, so the account
// roster's RESIDENCY arm adjudicates the same principal the served gateway does. It is
// the second half of WithRouteAccounts: the roster answers WHICH account a routed id
// binds to, and the principal answers WHETHER this caller may dispatch through that
// account at all. Without it a roster-wired loop would resolve an account-bound route
// that the gateway's resolveRoute REFUSES for the same call, which is the divergence
// WithRouteAccounts exists to close (#5644) — so a caller that wires a roster on a
// multi-tenant path must wire the principal too.
//
// An EMPTY principal is the unattributed caller (no keyset, or the single
// --require-key-env bearer) and is fail-CLOSED against a restricted account, exactly as
// Target.Admits specifies; an account naming NO principals is unrestricted and admits
// everyone, so a pre-#5332 roster resolves byte-for-byte as before. A caller may pass
// the option unconditionally.
func WithRoutePrincipal(principal string) RunOption {
	return func(c *runConfig) { c.principal = principal }
}

// WithSpeculator wires the SEAM-4 predicted-next-path engine (#809) into RunArm so the
// loop SPECULATES the next tool call ahead of the model: after a turn's tool calls run,
// the loop predicts the model's next call, runs it effect-free under a speculative epoch,
// and SUSPENDS it (holds the provisional result in a BufferSink) — then RESUMES when the
// model's authoritative next call is known, promoting on a match or squashing on a miss,
// all within the same turn index. This is the live, non-test caller of Speculator.Predict
// the suspend-and-resume turn primitive needs (#1318). A nil speculator (the default) is
// accepted and degrades to the historical loop — no prediction, no suspension — so a
// caller may pass the option unconditionally.
func WithSpeculator(s *abi.Speculator) RunOption {
	return func(c *runConfig) { c.spec = s }
}

// WithContextPlanner wires a persistent per-session context planner into RunArm. When
// the session gate lowers this turn's output cap, RunArm composes that Pace into the
// planner's resident-context Budget before rendering the prompt. A nil planner is a
// no-op, preserving the historical loop.
func WithContextPlanner(sp *SessionPlanner, baselineOutput int) RunOption {
	return func(c *runConfig) {
		c.contextPlanner = sp
		c.contextBaselineOutput = baselineOutput
	}
}

// routeToolEngine returns the engine route to bind to abi.ToolCall.Engine for one tool
// call under this run's optional routing manifest, or "" for the kernel default. It
// classifies the call into a Subject{Aspect: AspectToolCall, Tool: tool} and returns the
// matched Plan.Primary() for a single-model PICK; a nil manifest or an ENSEMBLE plan
// returns "" (the kernel default — an ensemble fan-out is a separate dispatch concern,
// #597, never collapsed to one member here). It mirrors the gateway's routeEngine
// exactly so the agent loop and the gateway can never diverge on what a call routes to.
func (c runConfig) routeToolEngine(tool string, callMeta ...map[string]string) string {
	if c.route == nil {
		return ""
	}
	meta := metaFor(tool)
	if len(callMeta) != 0 && callMeta[0] != nil {
		meta = callMeta[0]
	}
	d := c.route.Route(modelroute.Subject{
		Aspect: modelroute.AspectToolCall,
		Tool:   tool,
		Labels: nativeRouteLabels(meta, c.principal),
	})
	if d.Plan.IsEnsemble() {
		return ""
	}
	return d.Plan.Primary()
}

// nativeRouteLabels mirrors gateway.routeLabels for the signals the owned loop
// attests at call time. read_only comes from the same metadata that reaches the
// kernel, sensitivity accepts the gateway's canonical and compatibility spellings,
// and tenant is the authenticated principal wired into this run. Keeping this at
// the route boundary means the manifest and the eventual ToolCall classify the same
// call rather than reconstructing a weaker, tool-name-only subject.
func nativeRouteLabels(meta map[string]string, principal string) map[string]string {
	labels := map[string]string{"read_only": "false"}
	if meta["readOnlyHint"] == "true" {
		labels["read_only"] = "true"
	}
	sensitivity := meta["sensitivity"]
	if sensitivity == "" {
		sensitivity = meta["data_sensitivity"]
	}
	if sensitivity != "" {
		labels["sensitivity"] = sensitivity
	}
	if principal != "" {
		labels["tenant"] = principal
	}
	return labels
}

// resolveToolEngine returns the FINAL engine route to bind to abi.ToolCall.Engine for
// one tool call: the abstract routed id from routeToolEngine, then — when an account
// roster is wired (WithRouteAccounts, #2528) — resolved through it to the account-bound,
// residency-honest Target.EngineRoute(), and finally gated on whether this run's
// principal is admitted to that account (WithRoutePrincipal, #5332). It mirrors the
// gateway's resolveRoute exactly, so the in-process loop and the served gateway can never
// diverge on WHOSE account a routed id lands on — the PRECEDENCE is manifest first
// (which abstract id), roster second (which account), principal third (may this caller
// use it); a roster binds nothing without a manifest, because the roster is only
// consulted for an id the manifest already PICKed. An id the roster cannot resolve (no
// binding and no default account) is a FAIL-LOUD error carrying the recovery hint —
// never a silent fallback to the default engine, so a misconfigured roster cannot
// dispatch a routed call to the wrong account. A nil roster, or an empty/ensemble route
// (id ""), returns the abstract id verbatim: byte-for-byte the pre-roster loop and the
// kernel default respectively.
func (c runConfig) resolveToolEngine(tool string, callMeta ...map[string]string) (string, error) {
	id := c.routeToolEngine(tool, callMeta...)
	if c.roster == nil || id == "" {
		return id, nil
	}
	t, err := c.roster.Resolve(id)
	if err != nil {
		return "", fmt.Errorf("route accounts: %w (fix the roster binding for %q or set a default account; no silent fallback)", err, id)
	}
	if !t.Admits(c.principal) {
		// Name the principal and the account, never the credential — the same shape (and
		// the same fail-closed verdict) the gateway's resolveRoute reports, so an operator
		// reads one refusal whichever path served the turn. An empty principal is rendered
		// as <unattributed> so "wrong tenant" is distinguishable from "no tenant at all".
		who := c.principal
		if strings.TrimSpace(who) == "" {
			who = "<unattributed>"
		}
		return "", fmt.Errorf("route accounts: principal %s is not admitted to account %q (routed model %q): that account's principals allowlist scopes it to another tenant (#5332) — add this principal to the account, or route this run through an account it is provisioned for", who, t.Account, id)
	}
	return t.EngineRoute(), nil
}

// resolveRunConfig folds the options into a runConfig.
func resolveRunConfig(opts []RunOption) runConfig {
	var c runConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&c)
		}
	}
	return c
}

// gateTurn applies the per-turn session gate at a turn boundary. It returns the
// per-turn output-token cap to lower into the planner (0 = no cap / planner default),
// whether the loop should PROCEED with this turn, and the stop reason when it should
// not. A nil table proceeds with no cap (the historical loop). When pace asks for an
// inter-turn gap, gateTurn sleeps it here (respecting ctx cancellation) so a throttled
// session is paced without the loop body needing to know about timing.
//
// PAUSED is a non-terminal hold. A concrete table still returns proceed=false so the
// historical harness can stop a single-shot arm cleanly; a function-shaped gate with a
// Wait hook parks and re-Decides at the resumed boundary. DRAINING/STOPPED/budget-
// exhausted return proceed=false with a terminal reason.
func (c runConfig) gateTurn(ctx contextLike) (maxTokens int, proceed bool, reason string) {
	// Function-shaped gate (the gateway native loop): prefer it when wired. It carries
	// the same Decide semantics as the table, projected onto primitives.
	if c.gate != nil && c.gate.Decide != nil {
		for {
			mt, proceed, gap, reason := c.gate.Decide(c.trace)
			if proceed {
				if gap > 0 {
					select {
					case <-ctx.Done():
						return 0, false, reason
					case <-time.After(time.Duration(gap) * time.Millisecond):
					}
				}
				return mt, true, ""
			}
			if reason == session.ReasonPaused && c.toolTerminalWake != nil {
				c.toolTerminalWake.mark("DEFERRED")
			}
			if reason != session.ReasonPaused || c.gate.Wait == nil {
				return 0, false, reason
			}
			resumed, waitReason := c.gate.Wait(c.trace)
			if !resumed {
				if waitReason == "" {
					waitReason = reason
				}
				return 0, false, waitReason
			}
			// Re-Decide at the resumed boundary so turn/budget debits and pace are
			// taken from the live state after the operator released the hold.
		}
	}
	if c.table == nil {
		return 0, true, ""
	}
	v := c.table.Decide(c.trace)
	if !v.Proceed {
		return 0, false, v.Reason
	}
	if v.MinGapMs > 0 {
		select {
		case <-ctx.Done():
			return 0, false, v.Reason
		case <-time.After(time.Duration(v.MinGapMs) * time.Millisecond):
		}
	}
	return v.MaxTokens, true, ""
}

// applyPace composes the session gate's per-turn output cap into the optional
// persistent context planner. With no planner this is a no-op.
func (c runConfig) applyPace(maxTokens int) {
	if c.contextPlanner == nil {
		return
	}
	c.contextPlanner.ApplyPace(session.Pace{MaxTokensPerTurn: maxTokens}, c.contextBaselineOutput)
}

// hasCheckpointSink reports whether this run can record a write-ahead turn checkpoint
// (#1363): a function-shaped gate.Checkpoint OR a concrete drive table. RunArm consults
// it to decide whether to bind the planner's PendingTurnCheckpoint hook at all, so a run
// with no session wiring stays byte-for-byte the historical loop (no checkpoint written).
func (c runConfig) hasCheckpointSink() bool {
	return (c.gate != nil && c.gate.Checkpoint != nil) || c.table != nil
}

// checkpointPending records (or, with the zero value, clears) the in-flight turn's
// write-ahead retry checkpoint keyed on this run's trace (#1363). It is the writer half
// the planner's PendingTurnCheckpoint hook binds to: a retry inside Complete calls it
// with the 1-based attempt in progress + last upstream status + turn start, and turn
// completion calls it with the zero value to clear. Source preference mirrors gateTurn:
// a wired function-shaped gate owns the seam; otherwise the concrete table. A terminal
// session rejects the write inside SetPendingTurn, exactly like every other drive field.
func (c runConfig) checkpointPending(attempt, lastStatus int, startedAtUnixNano int64) {
	if c.gate != nil && c.gate.Checkpoint != nil {
		c.gate.Checkpoint(c.trace, attempt, lastStatus, startedAtUnixNano)
		return
	}
	if c.table != nil {
		c.table.SetPendingTurn(c.trace, session.PendingTurn{
			Attempt:           attempt,
			LastStatus:        lastStatus,
			StartedAtUnixNano: startedAtUnixNano,
		})
	}
}

// resumeCheckpoint reads the in-flight turn's write-ahead checkpoint this run should RE-ENTER at
// loop entry (#1363/#4124) — the read twin of checkpointPending. Source preference mirrors
// gateTurn: a wired function-shaped gate.ResumeCheckpoint owns the seam; otherwise the concrete
// table's Get(trace).PendingTurn. Returns the zero PendingTurn (IsZero) when no session is wired,
// the trace is empty, or nothing is checkpointed — so a run with no resume state stays byte-for-
// byte the historical loop and reports a fresh turn-0.
func (c runConfig) resumeCheckpoint() session.PendingTurn {
	if c.trace == "" {
		return session.PendingTurn{}
	}
	if c.gate != nil && c.gate.ResumeCheckpoint != nil {
		attempt, lastStatus, startedAt := c.gate.ResumeCheckpoint(c.trace)
		return session.PendingTurn{Attempt: attempt, LastStatus: lastStatus, StartedAtUnixNano: startedAt}
	}
	if c.table != nil {
		return c.table.Get(c.trace).PendingTurn
	}
	return session.PendingTurn{}
}

// bindPendingCheckpoint returns the planner RunArm should drive. When this run can record
// a checkpoint (hasCheckpointSink) and p is a DIRECT *HTTPPlanner — where the retry loop
// lives; a wrapped/dual planner's per-request trace binding is future work per #4122 — it
// returns a per-run SHALLOW COPY whose PendingTurnCheckpoint writes the checkpoint keyed on
// this run's trace. The copy is per-run, so concurrent arms sharing one planner never
// cross-write each other's trace, and Complete's hot path never reassigns planner fields so
// the copy is behavior-identical. Otherwise it returns p unchanged — byte-for-byte the
// historical loop (no checkpoint written).
func bindPendingCheckpoint(p Planner, cfg runConfig) Planner {
	hp, ok := p.(*HTTPPlanner)
	if !ok || !cfg.hasCheckpointSink() {
		return p
	}
	clone := *hp
	clone.PendingTurnCheckpoint = func(attempt, lastStatus int, startedAtUnixNano int64) {
		cfg.checkpointPending(attempt, lastStatus, startedAtUnixNano)
	}
	return &clone
}

// promptMessages returns the context-planned prompt for this turn when a persistent
// planner is wired. The authoritative message history stays lossless in RunArm; only
// the model-facing prompt is shortened.
func (c runConfig) promptMessages(ctx context.Context, messages []Message) ([]Message, error) {
	if c.promptAssembler != nil {
		planned, err := c.promptAssembler(ctx, append([]Message(nil), messages...))
		if err != nil {
			return nil, err
		}
		if len(planned) == 0 {
			return messages, nil
		}
		return planned, nil
	}
	if c.contextPlanner == nil {
		return messages, nil
	}
	planned := c.contextPlanner.RenderTurn(ctx, messages)
	if len(planned) == 0 {
		return messages, nil
	}
	return planned, nil
}

func (c runConfig) claimTurnInputs(turn int, inputs []Message) (InputClaimBinding, error) {
	if len(inputs) == 0 || c.inputClaims == nil {
		return InputClaimBinding{}, nil
	}
	if c.inputClaims.Claim == nil || c.inputClaims.Release == nil {
		return InputClaimBinding{}, fmt.Errorf("input claim lifecycle requires claim and release callbacks")
	}
	binding, err := c.inputClaims.Claim(AdmittedInputClaim{
		Turn: turn, Inputs: append([]Message(nil), inputs...),
	})
	if err != nil {
		return InputClaimBinding{}, err
	}
	if binding.ID == "" || binding.SHA256 == "" || binding.Count != len(inputs) {
		return InputClaimBinding{}, fmt.Errorf("input claim store returned an invalid binding")
	}
	return binding, nil
}

func (c runConfig) releaseInputClaim(binding InputClaimBinding, reason string) error {
	if binding.ID == "" {
		return nil
	}
	if c.inputClaims == nil || c.inputClaims.Release == nil {
		return fmt.Errorf("input claim %q has no release callback", binding.ID)
	}
	return c.inputClaims.Release(binding, reason)
}

func claimedInputsSurviveAssembly(claimed, planned []Message) bool {
	matched := 0
	for _, message := range planned {
		if matched < len(claimed) && reflect.DeepEqual(claimed[matched], message) {
			matched++
		}
	}
	return matched == len(claimed)
}

// terminateSignal returns the channel closed when this run's session enters
// Terminating (#2758), or nil when no terminate source is wired. Source preference
// mirrors gateTurn: a wired function-shaped gate owns the seam; otherwise the
// concrete table. A nil channel blocks forever in a select, so the historical loop
// (no table, no gate) is byte-for-byte unchanged.
func (c runConfig) terminateSignal() <-chan struct{} {
	if c.gate != nil && c.gate.TerminateSignal != nil {
		return c.gate.TerminateSignal(c.trace)
	}
	if c.table != nil {
		return c.table.TerminateSignal(c.trace)
	}
	return nil
}

// drainSteer non-blocking-receives any operator steer enqueued for this run on the
// a2achan Session-locale bus and returns its text to splice into the next turn's
// input. It is the CONSUMER half of the steer path (#850): the producer (the serve
// process's steerSession, cmd/fak/main.go) does a2achan.Send onto {Session, trace}
// when POST /session/{id}/steer fires; this drains it at the turn boundary so a
// RUNNING session actually picks the steer up, not just enqueues it.
//
// The channel is keyed by the run's trace id (the same id WithSessionTable wires),
// so a run with no trace (c.trace == "") has no mailbox and drains nothing. TryRecv
// is non-blocking: an empty mailbox returns ok=false (VerdictDefer) and we splice
// nothing — zero cost on the common no-steer path. The operator body is Shared +
// Tainted (a cross-principal widening, screened on ingress), so a VerdictQuarantine
// is DROPPED, never spliced: only an explicitly-allowed body becomes turn input.
// Multiple queued steers coalesce (drained in order) into one appended block.
func (c runConfig) drainSteer() string {
	if c.trace == "" {
		return ""
	}
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: c.trace}
	var out string
	for {
		msg, v, ok := a2achan.TryRecv(context.Background(), key, a2achan.CapA2ARecv)
		if !ok {
			break // empty mailbox (VerdictDefer) — nothing more to splice
		}
		if v.Kind != abi.VerdictAllow {
			sessionctl.RecordSteerNext(c.trace, "", sessionctl.ApplyResult{
				Refusal: fmt.Sprintf("a2achan receive verdict %v", v.Kind),
			})
			continue // quarantined/screened operator input — drop, keep draining
		}
		if len(msg.Body.Inline) == 0 {
			continue
		}
		payload := string(msg.Body.Inline)
		sessionctl.RecordSteerNext(c.trace, payload, sessionctl.ApplyResult{Applied: true})
		if out != "" {
			out += "\n"
		}
		out += payload
	}
	return out
}

// contextNudge is the turn-boundary advisory read (#2197), the consumer twin of
// drainSteer for the CONTEXT-SPIKE signal: when the session's cost ring shows the
// context window grew suddenly and materially last turn, it returns the rendered
// model-facing nudge to splice into THIS turn's input; otherwise "". It mirrors
// gateTurn's source preference — a wired function-shaped gate owns the boundary
// (its optional Nudge hook answers; a gate without one nudges nothing), else the
// table answers via session.Table.ContextNudge. With no trace, gate, or table the
// historical loop is byte-for-byte unchanged. Advisory only: the nudge informs the
// model's next turn; it never gates, debits, or transitions anything.
func (c runConfig) contextNudge() string {
	if c.trace == "" {
		return ""
	}
	if c.gate != nil {
		if c.gate.Nudge == nil {
			return ""
		}
		return c.gate.Nudge(c.trace)
	}
	return c.table.ContextNudge(c.trace)
}

// debitTurn reports a completed turn's usage to the session table so the output and
// context budget axes decrement. A nil table is a no-op.
func (c runConfig) debitTurn(usage Usage) {
	if c.gate != nil && c.gate.Debit != nil {
		c.gate.Debit(c.trace, usage.CompletionTokens, usage.ContextWindowTokens())
		return
	}
	if c.table != nil {
		c.table.DebitUsage(c.trace, session.Usage{
			OutputTokens:  usage.CompletionTokens,
			ContextTokens: usage.ContextWindowTokens(),
		})
	}
}

// debitToolCall spends ONE unit of the session's tool-call runaway budget (#5235, the
// #2887 floor) for a call about to be DISPATCHED, and returns the closed stop reason
// when the ceiling is crossed ("" = proceed). It is debitTurn's per-CALL twin: where the
// turn axis debits once per model round-trip and is only read at a turn boundary, this
// axis is debited per dispatched tool call, so a single turn that emits a long tool-call
// loop is cut MID-TURN at the ceiling instead of running the whole batch out. Exhaustion
// drives the session to Draining/Stopped inside session.Table.DebitToolCall, yielding the
// closed ReasonBudgetToolCalls witness the caller stamps on StoppedBySession.
//
// A nil table is a no-op returning "" (the historical loop is byte-for-byte unchanged),
// and an unconfigured axis is permissive by the 0=off convention, so wiring a session
// without a calls= envelope never cuts a run. The function-shaped SessionGate carries no
// tool-call axis yet, so a gateway-served run debits nothing here (#5235 out of scope).
//
// It cuts ONLY for its own axis. Table.DebitToolCall shares its run-state head with
// Decide, so a session already Draining/Stopped/Paused answers !Proceed carrying THAT
// reason rather than a budget one — and honoring it here would silently re-adjudicate
// run-state at a point the loop deliberately does not. That would break the #2758
// drain/terminate contract in the drain direction: a drain must let the in-flight turn
// dispatch every call the model already announced and take the stop at the NEXT
// BOUNDARY, while only a terminate cuts dispatch mid-turn (already handled by the
// stopTerminated safe point above the call site). So a non-budget reason is passed over
// as "proceed" and left to the gate that owns it.
func (c runConfig) debitToolCall() string {
	if c.table == nil {
		return ""
	}
	if v := c.table.DebitToolCall(c.trace); !v.Proceed && v.Reason == session.ReasonBudgetToolCalls {
		return v.Reason
	}
	return ""
}

// sampleOptsFor turns a per-turn output-token cap into the variadic SampleOpt slice
// for p.Complete. A non-positive cap returns NO options, so the planner call is
// byte-identical to the pre-seam p.Complete(ctx, messages, tools) — the historical
// path is untouched. A positive cap lowers WithMaxTokens, capping THIS turn's output
// (the pace throttle), which WithMaxTokens itself further guards (it no-ops on n<=0).
func sampleOptsFor(maxTokens int) []SampleOpt {
	if maxTokens <= 0 {
		return nil
	}
	return []SampleOpt{WithMaxTokens(maxTokens)}
}

// contextLike is the narrow slice of context.Context gateTurn needs (Done). It lets
// the helper be unit-tested with a fake without importing a real context, and keeps
// this file's surface honest about what it uses.
type contextLike interface {
	Done() <-chan struct{}
}
