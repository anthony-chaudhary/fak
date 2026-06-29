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
	"time"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// RunOption configures an optional behavior of RunArm / Run. The zero set of options
// is the historical behavior; each option opts into one capability (today: a session
// drive-state table). It is the variadic-options idiom so adding a capability never
// breaks an existing positional call site.
type RunOption func(*runConfig)

// runConfig is the resolved option set for one RunArm invocation. The zero value is
// the historical loop (nil table => permissive Decide => no per-turn gate; nil route
// => Engine left unset => kernel default for every tool call).
type runConfig struct {
	table *session.Table
	gate  *SessionGate
	trace string
	route *modelroute.Manifest
	spec  *abi.Speculator
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

// routeToolEngine returns the engine route to bind to abi.ToolCall.Engine for one tool
// call under this run's optional routing manifest, or "" for the kernel default. It
// classifies the call into a Subject{Aspect: AspectToolCall, Tool: tool} and returns the
// matched Plan.Primary() for a single-model PICK; a nil manifest or an ENSEMBLE plan
// returns "" (the kernel default — an ensemble fan-out is a separate dispatch concern,
// #597, never collapsed to one member here). It mirrors the gateway's routeEngine
// exactly so the agent loop and the gateway can never diverge on what a call routes to.
func (c runConfig) routeToolEngine(tool string) string {
	if c.route == nil {
		return ""
	}
	d := c.route.Route(modelroute.Subject{Aspect: modelroute.AspectToolCall, Tool: tool})
	if d.Plan.IsEnsemble() {
		return ""
	}
	return d.Plan.Primary()
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
// PAUSED is a non-terminal hold: gateTurn returns proceed=false with the PAUSED reason
// so the loop stops THIS run cleanly (the harness loop is single-shot; a long-lived
// gateway loop would instead wait and re-Decide — that wait belongs to the gateway
// integration, #555, not the A/B harness). DRAINING/STOPPED/budget-exhausted return
// proceed=false with a terminal reason.
func (c runConfig) gateTurn(ctx contextLike) (maxTokens int, proceed bool, reason string) {
	// Function-shaped gate (the gateway native loop): prefer it when wired. It carries
	// the same Decide semantics as the table, projected onto primitives.
	if c.gate != nil && c.gate.Decide != nil {
		mt, proceed, gap, reason := c.gate.Decide(c.trace)
		if !proceed {
			return 0, false, reason
		}
		if gap > 0 {
			select {
			case <-ctx.Done():
				return 0, false, reason
			case <-time.After(time.Duration(gap) * time.Millisecond):
			}
		}
		return mt, true, ""
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
			continue // quarantined/screened operator input — drop, keep draining
		}
		if len(msg.Body.Inline) == 0 {
			continue
		}
		if out != "" {
			out += "\n"
		}
		out += string(msg.Body.Inline)
	}
	return out
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
