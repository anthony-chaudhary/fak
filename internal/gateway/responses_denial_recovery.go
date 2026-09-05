package gateway

import (
	"context"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// #5212 — a guard denial is not a completed task.
//
// On the Responses wire (the wire Codex speaks) an all-denied turn used to be
// rendered as a `completed` response whose ONLY output item was a `message`
// carrying the guard's own remediation prose. Codex reads that shape exactly as
// the model authoring a final answer: it serializes the note as
// `response_item/message role=assistant` and then emits `task_complete` — while
// the user's requested work sits untouched. Repeating "keep going" reproduces it,
// because the next turn hits the same gate and renders the same shape.
//
// Two distinct things are wrong in that render, and both are addressed here:
//
//  1. The refusal never reached the MODEL as a result. The proposed call was
//     dropped from the output and summarized as prose, so the model was never
//     handed a structured "this call did not run, and here is why" it could adapt
//     to WITHIN the same turn. recoverDeniedResponsesTurn hands it exactly that —
//     one tool result per refused call, keyed by the call_id the model itself
//     proposed — and re-samples ONCE, so the turn gets a second actuation
//     opportunity before the client ever sees a response.
//
//  2. When recovery cannot produce one, the turn is no longer dressed as a normal
//     completion. The response carries `status=incomplete` plus
//     `incomplete_details.reason=fak_guard_denied`, and the in-band note leads with
//     a typed BLOCKED_BY_GUARD banner naming each unresolved call_id — the explicit
//     blocked state a status consumer can tell apart from `done`.
//
// The recovery sample is bounded to exactly one per HTTP turn: a model that answers
// a refusal with another refused call lands on the typed blocked state rather than
// looping the gateway.

// deniedGuardIncompleteReason is the Responses `incomplete_details.reason` a turn
// carries when the capability floor refused every proposed call and the bounded
// recovery sample produced no allowed alternative. It is deliberately fak-namespaced:
// the wire's own reasons (max_output_tokens, content_filter) describe the provider,
// this one describes the kernel.
const deniedGuardIncompleteReason = "fak_guard_denied"

// denialRecoveryEnabled reports whether the #5212 recovery sample is armed. ON by
// default — a denial-only turn rendered as a completed answer is the defect, so the
// recovery is the default behavior. Config.DenialRecoveryOff stands it down for a deploy
// that would rather pay no second sample and take the typed blocked state directly.
//
// Declared on the gateway's Config surface rather than read from the ambient process
// environment: whether a refused turn gets a second chance is behavior, not a credential,
// and internal/envconfiglint ratchets exactly that distinction (CONFIG_NOT_ENV). A setting
// read out of the environment is invisible to the host that constructs the Server and
// cannot differ between two gateways in one process; a Config field is neither.
func (s *Server) denialRecoveryEnabled() bool {
	return !s.denialRecoveryOff
}

// turnIsDenialOnly reports whether the adjudicated turn has nothing left to show the
// client except the guard's own remediation prose: every proposed call was refused,
// nothing survived, and the model authored no text of its own that would stand as a
// real answer.
//
// The three exclusions matter. A surviving call means the turn still actuates. A vDSO
// served-inline answer IS real content (the read was answered, not refused). A body the
// guard blanked counts as denial-only precisely BECAUSE the model's prose was withheld —
// what would reach the client is the kernel's text and nothing else.
func turnIsDenialOnly(kept []agent.ToolCall, dropped int, content string, bodyRefused bool, servedText string) bool {
	if len(kept) > 0 || dropped == 0 || servedText != "" {
		return false
	}
	if bodyRefused {
		return true
	}
	return strings.TrimSpace(content) == ""
}

// deniedRecoveryMessages builds the continuation the recovery sample runs on: the
// turn's own history, then the assistant turn carrying the calls it proposed, then one
// tool result per REFUSED call keyed by that call's id.
//
// This is the shape both wires already use for a normal tool round-trip, which is the
// point: the model reads a refusal the same way it reads any other tool outcome, so it
// can pick a different tool or different arguments without being told a special dialect.
// Only refused calls get a synthetic result — an admitted call cannot be in a
// denial-only turn by construction, and fabricating a result for one would tell the
// model something ran that did not.
func deniedRecoveryMessages(base []agent.Message, proposed agent.Message, adjs []ToolAdjudication) []agent.Message {
	refused := refusedByCallID(adjs)
	if len(refused) == 0 {
		return nil
	}
	calls := make([]agent.ToolCall, 0, len(proposed.ToolCalls))
	results := make([]agent.Message, 0, len(proposed.ToolCalls))
	for _, tc := range proposed.ToolCalls {
		adj, ok := refused[tc.ID]
		if !ok {
			continue
		}
		AttachOperatorRemedyMetadata(&adj)
		calls = append(calls, tc)
		results = append(results, agent.Message{
			Role:       agent.RoleTool,
			ToolCallID: tc.ID,
			Name:       tc.Function.Name,
			Content:    deniedToolResult(adj),
		})
	}
	if len(calls) == 0 {
		return nil
	}
	out := make([]agent.Message, 0, len(base)+1+len(results))
	out = append(out, base...)
	out = append(out, agent.Message{
		Role:      agent.RoleAssistant,
		Content:   proposed.Content,
		ToolCalls: calls,
	})
	return append(out, results...)
}

// refusedByCallID indexes this turn's refusals by the call id the model proposed, so a
// result can be attached to the exact call it answers. An adjudication with no call id
// is skipped rather than guessed at: a mis-keyed tool result is worse than a missing one
// (it answers a call the model never made).
func refusedByCallID(adjs []ToolAdjudication) map[string]ToolAdjudication {
	refused := make(map[string]ToolAdjudication, len(adjs))
	for _, a := range adjs {
		if a.Admitted || strings.TrimSpace(a.ToolCallID) == "" {
			continue
		}
		refused[a.ToolCallID] = a
	}
	return refused
}

// deniedToolResult renders one refusal as the tool result the model reads. It states the
// three things the model needs and cannot infer: that nothing ran (so it must not narrate
// the call as done), why (the structured reason/disposition), and that the ORIGINAL task
// is still open (so a refusal is not mistaken for the work being finished). The actionable
// half is the same remedy text every other refusal surface renders, via the one registry.
//
// Invariant (#11504): agent-facing text must never emit runnable shell commands (such as
// `fak guard allow`) in backticks or otherwise, preventing automated agents from looping
// in recursive self-modification attempts.
func deniedToolResult(a ToolAdjudication) string {
	var b strings.Builder
	b.WriteString("[fak] FAK_DENIED — the kernel refused this tool call; it did not run, nothing executed, and no state changed.")
	b.WriteString(" call_id=" + a.ToolCallID)
	if tool := strings.TrimSpace(a.Tool); tool != "" {
		b.WriteString(" tool=" + tool)
	}
	b.WriteString(" verdict=" + strings.TrimSpace(a.Verdict.Kind))
	b.WriteString(" reason=" + reasonOrKind(a.Verdict))
	if disp := strings.TrimSpace(a.Verdict.Disposition); disp != "" {
		b.WriteString(" disposition=" + disp)
	}
	notes, _ := renderRefusalNotes(a)
	if notes == "" {
		notes = errorAffordance(reasonOrKind(a.Verdict))
	}
	if notes != "" {
		b.WriteString(" " + notes)
	}
	b.WriteString(" The requested task remains OPEN — this refusal ended one call, not the task." +
		" Continue in this same turn: pick an allowed tool or allowed arguments and keep working." +
		" Report completion only once the work itself is done.")
	return stripExecutableSecurityCommands(b.String())
}

// recoverDeniedResponsesTurn re-samples ONCE with every refusal handed back as a
// structured tool result, returning the recovered completion. ok is false when recovery
// is unavailable or pointless: the gate is stood down, no refusal carried a call id to
// answer, or the second sample errored — each of which falls through to the typed
// blocked state rather than failing the request.
//
// The recovery sample runs on the SAME served session turn, so it is admitted, budgeted,
// and debited exactly like the first: a recovery cannot outrun a session's token budget.
func (s *Server) recoverDeniedResponsesTurn(ctx context.Context, turn servedSessionTurn, base []agent.Message, proposed agent.Message, adjs []ToolAdjudication, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, bool) {
	if !s.denialRecoveryEnabled() {
		return nil, false
	}
	msgs := deniedRecoveryMessages(base, proposed, adjs)
	if len(msgs) == 0 {
		return nil, false
	}
	comp, err := s.completeServed(ctx, turn, msgs, tools, opts...)
	if err != nil || comp == nil {
		if err != nil {
			s.logf("gateway: denial recovery sample failed, falling back to the blocked state: %v", err)
		}
		return nil, false
	}
	return comp, true
}

// turnAdjudications concatenates the pre-recovery refusals with the adjudications of
// whichever sample the client actually receives, so ONE HTTP turn reports every decision
// the kernel made inside it. A refused call that a recovery sample later routed around
// still happened, and dropping it from the wire extension would hide the very refusal
// this turn recovered from. Returns second unchanged when no recovery ran (the common
// case), so the ordinary path allocates nothing.
func turnAdjudications(first, second []ToolAdjudication) []ToolAdjudication {
	if len(first) == 0 {
		return second
	}
	out := make([]ToolAdjudication, 0, len(first)+len(second))
	out = append(out, first...)
	return append(out, second...)
}

// foldRecoveryUsage sums the two samples one recovered turn spent, so the client is
// billed the truth: a recovery is a second call to the provider, and reporting only the
// second would under-count the turn.
//
// Only the headline counters fold. The per-request detail subobjects (cached-token
// breakdowns, DeepSeek hit/miss) describe ONE request's cache shape, so a summed
// prompt_tokens paired with one sample's cached_tokens would read as a measured
// relationship that was never measured. They are dropped instead, which a consumer reads
// as "unknown" rather than as a fabricated number.
func foldRecoveryUsage(first, second agent.Usage) agent.Usage {
	return agent.Usage{
		PromptTokens:             first.PromptTokens + second.PromptTokens,
		CompletionTokens:         first.CompletionTokens + second.CompletionTokens,
		TotalTokens:              first.TotalTokens + second.TotalTokens,
		CacheReadInputTokens:     first.CacheReadInputTokens + second.CacheReadInputTokens,
		CacheCreationInputTokens: first.CacheCreationInputTokens + second.CacheCreationInputTokens,
	}
}

// blockedByGuardNote is the typed BLOCKED/NEEDS_OPERATOR banner for a turn that ran the
// recovery sample and still produced neither an allowed actuation nor a model-authored
// answer. It leads the in-band note so the FIRST thing any reader (model, TUI, or status
// consumer) sees is that the turn was interrupted rather than finished, and it names each
// unresolved call id with its tool and reason as the evidence for that claim.
func blockedByGuardNote(adjs []ToolAdjudication) string {
	unresolved := make([]string, 0, len(adjs))
	seen := make(map[string]struct{}, len(adjs))
	for _, a := range adjs {
		if a.Admitted {
			continue
		}
		id := strings.TrimSpace(a.ToolCallID)
		if id == "" {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		entry := id + "(" + strings.TrimSpace(a.Tool) + "/" + reasonOrKind(a.Verdict)
		if disp := strings.TrimSpace(a.Verdict.Disposition); disp != "" {
			entry += "/" + disp
		}
		unresolved = append(unresolved, entry+")")
	}
	sort.Strings(unresolved)
	note := "[fak] BLOCKED_BY_GUARD needs_operator=true — this turn was interrupted by the capability floor," +
		" not finished. The requested work is still OPEN: every tool call was refused and the recovery" +
		" attempt produced no allowed alternative. Treat this as a blocked turn, never as a completed task."
	if len(unresolved) > 0 {
		note += " unresolved_calls=" + strings.Join(unresolved, ",")
	}
	return note
}
