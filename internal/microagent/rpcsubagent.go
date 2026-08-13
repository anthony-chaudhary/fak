package microagent

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// rpcsubagent.go — the RPC-tool subagent spine (#2931, epic #2908 "what fak does
// better than Hermes").
//
// Hermes' pitch is "write Python scripts that call tools via RPC, collapsing
// multi-step pipelines into zero-context-cost turns": the orchestrator never
// pays context for the intermediate tool chatter. The power is real; the gap is
// that in Hermes those RPC tool calls run OUTSIDE the command-approval flow —
// isolation is by convention. fak adjudicates every tool call. This spine offers
// the SAME collapse WITHOUT dropping containment: an RPC subagent whose every
// intermediate call is still governed by the same capability floor and journaled.
//
// The two properties this path holds TOGETHER (the whole point — one is worthless
// without the other):
//
//  1. ADJUDICATED + JOURNALED. Every intermediate tool call is routed through the
//     SAME kernel floor the served gateway fronts and `fak guard` wraps — the
//     ToolExec seam, which decides BEFORE dispatch (a denied call costs zero
//     execution at every isolation level, #2018). The REAL verdict the floor
//     returns is then recorded as a durable, hash-chained journal row (DECIDE for
//     an allow, DENY for a refusal — the exact events the kernel's own emit path
//     fans to a registered journal), tagged with the subagent's id. Containment is
//     not traded away for the collapse: a denied intermediate call is refused and
//     journaled, and never runs.
//
//  2. ZERO ORCHESTRATOR-CONTEXT COST. The intermediate tool chatter (each call and
//     its result) is appended ONLY to the subagent's OWN bounded Context, never to
//     the orchestrator's. RunScript never touches the orchestrator's context; it
//     hands back only a bounded collapsed summary (RPCResult.Collapsed) the
//     orchestrator MAY fold. So the orchestrator pays context for the final
//     collapsed turn, not the whole intermediate pipeline — the "collapse a
//     multi-step pipeline into a zero-context-cost turn" claim, measured as a
//     before/after readout on the orchestrator's token count.
//
// Transport note. "Over RPC" in Hermes is the wire between the script and the
// gateway. Here the transport is the in-process ToolExec seam — the direct-call
// twin of the served JSON-RPC gateway (sessiongateway.go documents the same
// twinning for the session control plane). The two witnessed properties are
// transport-independent: adjudication is a property of the SEAM every call passes
// through, and the context collapse is a property of WHERE the chatter is
// appended — neither depends on whether the hop is a socket or a Go method call.
//
// # Generation intent (gen/second-next, #2931)
//
// An architectural OPTION behind the microagent import boundary — nothing in the
// default `fak serve` / `fak guard` / dispatch path constructs an RPCSubagent.
// Closing evidence for the generation frame:
//
//   - Promotion evidence: TestRPCSubagentCollapsesUnderFloor witnesses, on ONE
//     run, that a 3-call script is adjudicated (each step VerdictAllow) AND
//     journaled (a DECIDE row per call, tagged with the subagent id) AND that the
//     orchestrator's context is unchanged by the intermediate chatter while the
//     subagent's own context carries it (the collapse is a measured saving, not a
//     tautology); TestRPCSubagentDeniedCallIsContainedAndJournaled witnesses that
//     a denied intermediate call never executes and lands a DENY row. Promote once
//     the #2001 RunArm extraction drives a real agent loop as the script producer
//     and a density measurement (#2033) confirms the orchestrator-context saving is
//     the binding cost the collapse buys.
//   - Demotion / retirement criteria: retire this spine if the served gateway
//     grows an in-process RPC entry a subagent can drive directly (the collapse
//     then rides that path and this adapter buys nothing), or if a footprint
//     measurement shows a real orchestrator would fold the full intermediate
//     transcript anyway (no collapse to bank), or if #2018's floor cannot be held
//     over the transport a production RPC subagent actually needs.
//   - Invalidating assumption: the collapse assumes the orchestrator needs only a
//     BOUNDED summary of the pipeline, not its intermediate transcript. If a real
//     loop must fold each intermediate result back to make progress (the pipeline
//     is not summarizable), the zero-context-cost property does not hold and the
//     orchestrator pays for the chatter after all — the honest boundary this spine
//     refuses to fake.

// ErrNilExec is returned by NewRPCSubagent when the adjudicating ToolExec seam is
// nil: there is no unadjudicated RPC-subagent path (the #2018 floor invariant).
var ErrNilExec = errors.New("microagent: NewRPCSubagent requires the adjudicating ToolExec seam (nil *ToolExec)")

// RPCStep is one intermediate tool call's outcome. Verdict is the REAL kernel-floor
// verdict the call was adjudicated under (always set for an adjudicated call); Ran
// reports whether the backend actually executed it (false for a refused call).
type RPCStep struct {
	Tool    string      // the logical tool name the floor adjudicated
	Verdict abi.Verdict // the kernel-floor verdict (the same one journaled)
	Ran     bool        // true iff the backend executed the call (never true for a refusal)
	Stdout  []byte      // captured stdout of an executed call (nil for a refusal)
	Err     error       // ErrActionDenied for a refusal; a dispatch error otherwise; nil on a clean run
}

// Allowed reports whether the floor allowed this step.
func (s RPCStep) Allowed() bool { return s.Verdict.Kind == abi.VerdictAllow && s.Err == nil }

// RPCResult is what the subagent hands back to the ORCHESTRATOR: the per-step audit
// trail plus the single BOUNDED collapsed summary the orchestrator may fold. It
// deliberately does NOT carry the intermediate transcript — that is the zero-
// context-cost collapse: the orchestrator gets Collapsed, not the chatter.
type RPCResult struct {
	Steps              []RPCStep // one per script action, in order
	Collapsed          string    // the bounded summary the orchestrator may fold (the collapsed turn)
	Allowed            int       // count of adjudicated+executed calls
	Denied             int       // count of floor-refused calls (contained, never executed)
	Errored            int       // count of allowed-but-not-dispatched calls (config faults, not decisions)
	IntermediateTokens int       // bounded child-context tokens retained after the script
	FoldedTokens       int       // standalone estimate for the Collapsed message
	SavedTokens        int       // max(IntermediateTokens-FoldedTokens, 0)
}

// RPCSubagent runs a fixed multi-call tool SCRIPT on behalf of an orchestrator,
// over the adjudicating ToolExec seam, keeping the two properties above together.
// It is NOT safe for concurrent use — one RPCSubagent belongs to one pipeline run.
type RPCSubagent struct {
	id   string           // the subagent id, tagged onto every journal row (its TraceID)
	exec *ToolExec        // the adjudicating seam every call passes through
	ctx  *Context         // the subagent's OWN bounded transcript (the chatter lands here)
	jrnl *journal.Journal // the durable decision journal (nil => journaling degrades to silence)
	seq  uint64           // per-subagent call counter, the journal join key (ToolCall.SeqNo)
}

// NewRPCSubagent builds a subagent that drives a script through exec, capping its
// own transcript at ctxCap tokens (0 => DefaultContextCap) and journaling each
// call's decision to jrnl. A nil exec is refused loud (there is no unadjudicated
// path); a nil jrnl is valid and degrades journaling to silence (matching
// JournalSink's nil contract), so a caller that only wants the collapse can omit it.
func NewRPCSubagent(id string, exec *ToolExec, ctxCap int, jrnl *journal.Journal) (*RPCSubagent, error) {
	if exec == nil {
		return nil, ErrNilExec
	}
	return &RPCSubagent{id: id, exec: exec, ctx: NewContext(ctxCap), jrnl: jrnl}, nil
}

// Context exposes the subagent's OWN bounded transcript — where the intermediate
// tool chatter lands. The witness reads Tokens() off it to prove the chatter is
// real and contained here, NOT billed to the orchestrator.
func (s *RPCSubagent) Context() *Context { return s.ctx }

// RunScript runs the script action by action, ADJUDICATING each call through the
// ToolExec floor and JOURNALING the floor's real decision, while appending every
// intermediate call+result to the SUBAGENT's context only. It returns the per-step
// trail and a bounded collapsed summary. It NEVER touches an orchestrator context:
// the zero-context-cost collapse is structural — the orchestrator receives only
// RPCResult.Collapsed and chooses whether to fold it.
func (s *RPCSubagent) RunScript(ctx context.Context, script []ToolAction) RPCResult {
	out := RPCResult{}
	for _, act := range script {
		res, err := s.exec.Run(ctx, act)
		step := RPCStep{Tool: act.Tool, Verdict: res.Verdict, Ran: res.Ran, Stdout: res.Stdout, Err: err}
		switch {
		case errors.Is(err, ErrActionDenied):
			// Contained: the floor refused BEFORE dispatch, so the call never ran.
			// Record the REAL deny verdict as a DENY row and keep the refusal in the
			// subagent's own transcript — never the orchestrator's.
			s.journalDecision(act, res.Verdict, abi.EvDeny)
			out.Denied++
			s.ctx.Append("assistant", fmt.Sprintf("call %s refused: %s", act.Tool, abi.ReasonName(res.Verdict.Reason)))
		case err == nil && res.Verdict.Kind == abi.VerdictAllow:
			// Adjudicated + executed: record the REAL allow as a DECIDE row and append
			// the call + its result to the subagent's transcript (the collapsed-out chatter).
			s.journalDecision(act, res.Verdict, abi.EvDecide)
			out.Allowed++
			s.ctx.Append("assistant", "call "+act.Tool)
			s.ctx.Append("tool", string(res.Stdout))
		default:
			// Allowed by the floor but the backend could not dispatch (no program,
			// unregistered tool, malformed args): a config fault, not an adjudication
			// outcome — surfaced on the step, never journaled as a decision.
			out.Errored++
		}
		out.Steps = append(out.Steps, step)
	}
	out.Collapsed = fmt.Sprintf("pipeline %q: %d/%d calls allowed, %d denied", s.id, out.Allowed, len(script), out.Denied)
	out.IntermediateTokens = s.ctx.Tokens()
	fold := NewContext(max(out.IntermediateTokens+1, 1))
	fold.Append("user", "goal")
	beforeFold := fold.Tokens()
	fold.Append("tool", out.Collapsed)
	out.FoldedTokens = fold.Tokens() - beforeFold
	if out.IntermediateTokens > out.FoldedTokens {
		out.SavedTokens = out.IntermediateTokens - out.FoldedTokens
	}
	return out
}

// journalDecision records the floor's REAL decision for one intermediate call as a
// durable, hash-chained journal row tagged with the subagent id. It reuses the SAME
// call envelope the floor adjudicated (toolCall), so the row's tool + args digest
// are the adjudicated ones, and emits the production event kind (EvDecide for an
// allow, EvDeny for a refusal) — the row is derived from the real verdict, never
// stubbed. A nil journal makes this a no-op.
func (s *RPCSubagent) journalDecision(act ToolAction, v abi.Verdict, kind abi.EventKind) {
	if s.jrnl == nil {
		return
	}
	call, err := toolCall(act)
	if err != nil {
		return // an unmarshalable action never reached the floor either
	}
	s.seq++
	call.TraceID = s.id
	call.SeqNo = s.seq
	vv := v
	s.jrnl.Emit(abi.Event{Kind: kind, Call: call, Verdict: &vv})
}
