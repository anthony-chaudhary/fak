package microagent

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// sessiongateway.go — the in-process session-adjudicating gateway (#2005, epic
// #2000 M5). It is the DIRECT-CALL twin of the served gateway's session control
// surface: the `fak serve`/`fak guard` assembly wraps the shared planner with the
// injected closures decideSession/debitSession (cmd/fak/main.go), which are thin
// wrappers over ONE process-local session.Table's Decide and DebitUsage. This
// gateway routes those SAME two calls around the SAME one shared Table — but for
// N in-process goroutine microagents making a plain Go method call, with NO HTTP
// hop and NO per-agent loopback listener (the whole point of #2005 scope 3).
//
// SchedulingGateway (slotsched.go) is the sibling wrap: it fronts the ONE shared
// gateway with the cooperative slot pool. SessionGateway is the session-drive
// layer and composes with it — NewSessionGateway(NewSchedulingGateway(base,
// sched), tbl) gives every hosted agent one seat pool AND one session control
// plane, still one shared gateway, still no socket.
//
// Concurrency (#2005 scope 2): the whole session drive is one *session.Table,
// whose Decide and DebitUsage each take the table's single lock exactly once, so N
// goroutine callers are serialized on the hot path with no per-agent state outside
// the Table. sessiongateway_test.go witnesses this race-clean under -race.
//
// Generation intent: gen/second-next architectural OPTION (#2005). Nothing in the
// default serve/guard/dispatch path constructs a SessionGateway; it is the seam a
// microagent Host opts into to drive the shared session control plane in-process.
// Closing evidence for the generation frame:
//
//   - Promotion evidence: the concurrent-load witness (sessiongateway_test.go)
//     drives N goroutine microagents through ONE SessionGateway over ONE
//     session.Table, race-clean, one Table entry per agent, the base gateway
//     dialed exactly once per proceeding turn and ZERO net.Listener opened — the
//     direct-call contrast to the #2002 smoke test's httptest hop. Promote once
//     `fak micro`'s live drive path (cmd/fak/micro.go) targets a real gateway.New
//     assembly behind this wrap instead of the Mock, and a density measurement
//     (#2033) confirms the removed per-agent process/socket weight was the binding
//     cost.
//   - Demotion / retirement criteria: retire this wrap if the served gateway grows
//     an in-process (non-HTTP) call entry point of its own that a microagent can
//     invoke directly — then the session drive rides that path and this adapter
//     buys nothing — or if #2033 shows per-agent cost is dominated by provider
//     seats/rate limits, not local socket/process weight.
//   - Invalidating assumption: the per-agent session key is carried on the context
//     via WithTrace (mirroring SchedulingGateway's WithPriority), so an UNTAGGED
//     Complete adjudicates the empty-trace default session — permissive and shared.
//     If the real hosted loop cannot thread its trace id onto the model-call
//     context (e.g. a backend that re-enters through an opaque client), this wrap
//     under-isolates the session drive and the Host must forward the trace itself
//     (tag the per-job context in Spawn) before it carries production agents.

// traceKey tags a context with the per-agent session trace id the SessionGateway
// adjudicates on. It mirrors slotsched.go's priorityKey: an unexported key type so
// only this package's WithTrace/TraceFromContext touch the value.
type traceKey struct{}

// WithTrace tags ctx with the session trace id for the model call made under it —
// the seam a hosted microagent uses to forward its id (which the Host also uses as
// the session.Table TraceID) to the shared session-adjudicating gateway WITHOUT
// the gateway importing the host's id scheme. It is the session twin of
// WithPriority. Higher layers pass the same id the Host spawned the agent under so
// the per-turn gate and the per-agent lifecycle entry are the same Table row.
func WithTrace(ctx context.Context, trace string) context.Context {
	return context.WithValue(ctx, traceKey{}, trace)
}

// TraceFromContext reads a WithTrace tag, defaulting to "" (the empty-trace default
// session: Running/unbounded — a permissive, shared gate). It is SessionGateway's
// default trace hook.
func TraceFromContext(ctx context.Context) string {
	if s, ok := ctx.Value(traceKey{}).(string); ok {
		return s
	}
	return ""
}

// ErrSessionGated is returned by SessionGateway.Complete when the shared session
// control plane refuses the turn at its boundary (Decide.Proceed=false) BEFORE the
// base gateway is dialed — a stopped, paused, drained, or budget-exhausted session
// must not spend a model call. The wrapped Reason (the session.Reason* closed
// vocabulary, e.g. BUDGET_TURNS_EXHAUSTED) rides the error so a caller sees which
// closed cause gated the turn, not free text.
var ErrSessionGated = errors.New("microagent: shared session control plane gated the turn before dial")

// SessionGateway wraps the ONE shared gateway with the shared session control
// plane: every Complete first calls session.Table.Decide (the per-turn boundary
// gate — run-state + turn/token/pace budget) and, only on Proceed, dials the base
// gateway, then records the returned usage with session.Table.DebitUsage so the
// next turn's Decide observes budget exhaustion. It satisfies agent.Planner (==
// Gateway), so it drops straight into NewHost in place of the raw gateway.
//
// This is the in-process realization of the served decideSession/debitSession
// closures (cmd/fak/main.go) over the SAME session.Table primitive, reached by a
// direct method call with no HTTP hop and no per-agent socket (#2005).
type SessionGateway struct {
	gw    Gateway
	tbl   *session.Table
	trace func(context.Context) string
}

// NewSessionGateway wraps gw so every model call is gated and debited against tbl,
// the ONE shared session control plane for the whole host. A nil tbl is a valid
// no-op-permissive drive (session.Table's own nil-receiver contract: Decide
// proceeds unbounded, DebitUsage is a no-op), so the wrap degrades to a plain
// pass-through rather than panicking — matching the "a loop with no table wired
// behaves byte-identically" discipline the session package documents. The trace
// hook defaults to TraceFromContext; override it with SetTraceFunc.
func NewSessionGateway(gw Gateway, tbl *session.Table) *SessionGateway {
	return &SessionGateway{gw: gw, tbl: tbl, trace: TraceFromContext}
}

// SetTraceFunc overrides how the per-call session trace id is derived (e.g. read it
// off a request-scoped struct instead of the context). A nil fn restores the
// default. Set it before serving traffic.
func (g *SessionGateway) SetTraceFunc(fn func(context.Context) string) {
	if fn == nil {
		fn = TraceFromContext
	}
	g.trace = fn
}

// Model reports the wrapped gateway's model id (provenance passthrough).
func (g *SessionGateway) Model() string { return g.gw.Model() }

// Complete adjudicates the turn through the shared session control plane, then —
// only on a proceed verdict — dials the wrapped gateway and records the returned
// token usage. The contract:
//
//   - gated (Decide.Proceed=false)  -> (nil, ErrSessionGated wrapping the Reason);
//     the base gateway is NEVER dialed, so a stopped/exhausted session costs zero
//     model calls — the whole point of a session boundary gate.
//   - proceeded                     -> the base gateway's (completion, error) as-is;
//     on a nil-error non-nil completion the reported usage is debited so the NEXT
//     Decide observes budget exhaustion (DebitUsage's own boundary discipline).
//
// The base error is returned verbatim (not debited) — usage is only meaningful for
// a turn that actually produced a completion.
func (g *SessionGateway) Complete(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	trace := g.trace(ctx)
	if v := g.tbl.Decide(trace); !v.Proceed {
		return nil, fmt.Errorf("%w: %s", ErrSessionGated, v.Reason)
	}
	comp, err := g.gw.Complete(ctx, msgs, tools, opts...)
	if err != nil {
		return comp, err
	}
	if comp != nil {
		// Debit the completed turn's usage against the shared Table (the debitSession
		// twin). The Table stays price-blind — CostMicroCents is left 0 so an unpriced
		// in-process turn honestly leaves a configured spend budget untouched (Usage
		// doc); a host that prices turns can compose that above this wrap.
		g.tbl.DebitUsage(trace, session.Usage{
			OutputTokens:  comp.Usage.CompletionTokens,
			ContextTokens: comp.Usage.PromptTokens,
		})
	}
	return comp, nil
}

// Sessions exposes the shared session control-plane table (read via Get/Snapshot).
func (g *SessionGateway) Sessions() *session.Table { return g.tbl }
