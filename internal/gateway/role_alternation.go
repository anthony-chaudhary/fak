package gateway

// role_alternation.go — issue #2848: message-array well-formedness as a kernel
// adjudication, plus the persist-guard that keeps a harness-authored turn out of
// an interactive conversation's standing history.
//
// Hermes requires strict message-role alternation — never two same-role messages
// in a row, never a synthetic user message injected mid-turn (`AGENTS.md`) —
// because a violation costs twice: it bursts the provider's prompt cache (the
// prefix no longer matches), and it CORRUPTS the conversation. The canonical
// failure is the "curator-takeover" bug in Hermes' `agent/background_review.py`:
// a background review appended its own harness user-message to drive the review,
// that message got persisted into the real conversation's history, and every
// later turn re-read it as a standing operator instruction. The harness quietly
// took over the operator's conversation. Hermes enforces the rule by convention —
// a line in a contributor doc, policed in review.
//
// fak sits in front of the wire and sees the decoded message array, so it can
// make the rule STRUCTURAL instead of conventional. Two halves, both pure:
//
//   - [CheckRoleAlternation] adjudicates the array's shape (alternation, no
//     synthetic user turn spliced into an open tool exchange, no system turn
//     spliced in after the head) and [RepairRoleAlternation] fixes what is
//     losslessly fixable. Together they answer the two distinct questions the
//     caller has to keep apart: REPAIR the array, or REJECT the request.
//   - [PersistableTurns] is the persist-guard: a turn tagged [OriginHarness] may
//     ride a single request, but it can never be written back into the
//     interactive conversation's standing history. That is the curator-takeover
//     bug class, killed structurally rather than by convention.
//
// Note the array this adjudicates is fak's DECODED []agent.Message, where an
// Anthropic user turn carrying tool_result blocks has already been fanned out
// into RoleTool messages (see agent/anthropic_server.go). So on this array a run
// of RoleTool messages is one logical turn, while two adjacent user messages are
// a genuine violation rather than an artifact of the wire shape.
//
// Everything here is deterministic: array in, verdict out. No clock, no I/O, no
// network — the same messages always yield the same verdict and the same repair.

import (
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// AlternationFlawKind names ONE way a decoded message array is mal-formed. The
// tokens are stable strings so a refusal or a witness can carry the reason
// verbatim instead of re-deriving prose.
type AlternationFlawKind string

const (
	// FlawSameRoleStacked is two adjacent messages with the same speaking role
	// (user→user or assistant→assistant): the literal "never two same-role
	// messages in a row" rule. A run of RoleTool messages is NOT this flaw — it
	// is the several results of one assistant turn's parallel tool calls, which
	// is a single logical turn.
	FlawSameRoleStacked AlternationFlawKind = "SAME_ROLE_STACKED"
	// FlawSyntheticUserMidTurn is a user message spliced into an OPEN tool
	// exchange — the assistant asked for tool calls and some results are still
	// owed, yet a user turn appears before they arrive. No operator types there;
	// the turn is synthetic, injected by a harness. This is the shape the
	// curator-takeover bug wrote into the conversation.
	FlawSyntheticUserMidTurn AlternationFlawKind = "SYNTHETIC_USER_MID_TURN"
	// FlawSystemNotAtHead is a system message anywhere but index 0. The system
	// turn is the array's stable head; one spliced in later is a prompt rebuild
	// mid-conversation, which no repair can undo without changing what was asked.
	FlawSystemNotAtHead AlternationFlawKind = "SYSTEM_NOT_AT_HEAD"
)

// AlternationFlaw locates one violation in the array: what went wrong, where,
// and the role that was speaking, so a caller can render an actionable refusal
// instead of "malformed request".
type AlternationFlaw struct {
	// Kind is the stable violation token.
	Kind AlternationFlawKind
	// Index is the position of the OFFENDING message in the array as supplied.
	Index int
	// Role is the offending message's role.
	Role string
	// Detail is a one-line human-readable account of the violation.
	Detail string
}

// RoleAlternationVerdict is the adjudication result for one message array.
type RoleAlternationVerdict struct {
	// OK is true when the array is well-formed and may go to the provider as-is.
	OK bool
	// Flaws lists every violation found, in array order. Empty when OK.
	Flaws []AlternationFlaw
	// Repairable reports whether [RepairRoleAlternation] can turn this array into
	// a well-formed one WITHOUT losing content. It is only meaningful on a
	// verdict returned by RepairRoleAlternation, which sets it by re-adjudicating
	// its own output — so it can never claim a repair that did not hold. A
	// CheckRoleAlternation verdict leaves it false: checking does not attempt a
	// repair, and guessing would be the "repair vs reject" confusion this seam
	// exists to keep apart.
	Repairable bool
}

// CheckRoleAlternation adjudicates a decoded message array's well-formedness.
// It answers only "is this shape legal" — it never mutates the array and never
// decides the policy response. A caller that wants the array fixed calls
// [RepairRoleAlternation]; a caller that wants to refuse reads Flaws and
// renders them.
func CheckRoleAlternation(msgs []agent.Message) RoleAlternationVerdict {
	var flaws []AlternationFlaw
	// pending counts tool results still owed by the most recent assistant turn.
	// While it is positive the tool exchange is OPEN, and a user turn appearing
	// there was injected by something other than the operator.
	pending := 0
	for i, m := range msgs {
		switch m.Role {
		case agent.RoleSystem:
			if i != 0 {
				flaws = append(flaws, AlternationFlaw{
					Kind: FlawSystemNotAtHead, Index: i, Role: m.Role,
					Detail: "system message at index " + strconv.Itoa(i) + " rebuilds the prompt head mid-conversation",
				})
			}
		case agent.RoleAssistant:
			if i > 0 && msgs[i-1].Role == agent.RoleAssistant {
				flaws = append(flaws, AlternationFlaw{
					Kind: FlawSameRoleStacked, Index: i, Role: m.Role,
					Detail: "assistant message at index " + strconv.Itoa(i) + " follows another assistant message",
				})
			}
			pending = len(m.ToolCalls)
		case agent.RoleTool:
			if pending > 0 {
				pending--
			}
		case agent.RoleUser:
			if pending > 0 {
				flaws = append(flaws, AlternationFlaw{
					Kind: FlawSyntheticUserMidTurn, Index: i, Role: m.Role,
					Detail: "user message at index " + strconv.Itoa(i) + " is spliced into an open tool exchange (" + strconv.Itoa(pending) + " result(s) still owed)",
				})
			} else if i > 0 && msgs[i-1].Role == agent.RoleUser {
				flaws = append(flaws, AlternationFlaw{
					Kind: FlawSameRoleStacked, Index: i, Role: m.Role,
					Detail: "user message at index " + strconv.Itoa(i) + " follows another user message",
				})
			}
		}
	}
	return RoleAlternationVerdict{OK: len(flaws) == 0, Flaws: flaws}
}

// RepairRoleAlternation attempts to make a mal-formed array well-formed without
// losing content, and reports whether it succeeded. It returns the array to send
// and the verdict for the array as SUPPLIED.
//
// The two questions a caller must keep apart are answered by two fields: Flaws
// says what was wrong, Repairable says whether the returned array is safe to
// send. When Repairable is false the repair did NOT hold and the returned array
// is the original, untouched — that request must be REJECTED, not sent.
//
// Repairs, in order:
//
//   - a synthetic user turn spliced into an open tool exchange is DROPPED. It is
//     not the operator's turn, so removing it loses nothing the conversation
//     owns — this is precisely the curator-takeover repair.
//   - two stacked same-role messages are MERGED into one, content joined by a
//     blank line. Merging is attempted only when it is lossless: a message
//     carrying tool calls, a tool-call id, thinking blocks, reasoning content or
//     a speaker name cannot be folded into its neighbour without dropping
//     structure the provider requires, so such a stack is left in place and the
//     re-adjudication below reports the array unrepairable.
//
// A system turn spliced in after the head is never repaired: dropping it would
// silently change the instructions, and moving it would silently change their
// order. Such an array is always a reject.
func RepairRoleAlternation(msgs []agent.Message) ([]agent.Message, RoleAlternationVerdict) {
	verdict := CheckRoleAlternation(msgs)
	original := append([]agent.Message(nil), msgs...)
	if verdict.OK {
		verdict.Repairable = true
		return original, verdict
	}
	repaired := mergeStackedRoles(dropSplicedUserTurns(msgs))
	// Re-adjudicate our own output: Repairable is only ever set from a fresh
	// verdict on the repaired array, so this seam cannot report a repair that did
	// not actually hold.
	verdict.Repairable = CheckRoleAlternation(repaired).OK
	if !verdict.Repairable {
		return original, verdict
	}
	return repaired, verdict
}

// dropSplicedUserTurns removes every user message that appears while a tool
// exchange is still open. It walks the same pending-results state machine
// CheckRoleAlternation uses, so the two agree by construction.
func dropSplicedUserTurns(msgs []agent.Message) []agent.Message {
	out := make([]agent.Message, 0, len(msgs))
	pending := 0
	for _, m := range msgs {
		switch m.Role {
		case agent.RoleAssistant:
			pending = len(m.ToolCalls)
		case agent.RoleTool:
			if pending > 0 {
				pending--
			}
		case agent.RoleUser:
			if pending > 0 {
				continue // spliced mid-exchange: not the operator's turn.
			}
		}
		out = append(out, m)
	}
	return out
}

// mergeStackedRoles folds adjacent same-role user/assistant messages into one,
// but only where the fold is lossless. An unmergeable stack is left standing so
// the caller's re-adjudication refuses the array instead of silently dropping
// structure the provider needs.
func mergeStackedRoles(msgs []agent.Message) []agent.Message {
	out := make([]agent.Message, 0, len(msgs))
	for _, m := range msgs {
		if n := len(out); n > 0 {
			prev := &out[n-1]
			if prev.Role == m.Role && (m.Role == agent.RoleUser || m.Role == agent.RoleAssistant) && foldable(*prev) && foldable(m) {
				prev.Content = joinTurnContent(prev.Content, m.Content)
				continue
			}
		}
		out = append(out, m)
	}
	return out
}

// foldable reports whether a message is plain enough to merge into a neighbour.
// Anything carrying structure the provider round-trips by identity — tool calls,
// a tool-call id, thinking blocks and their signature, reasoning content, or a
// speaker name — is not foldable: merging it would drop that structure.
func foldable(m agent.Message) bool {
	return len(m.ToolCalls) == 0 &&
		m.FunctionCall == nil &&
		m.ToolCallID == "" &&
		m.Name == "" &&
		m.Thinking == "" &&
		m.ThinkingSignature == "" &&
		len(m.RedactedThinking) == 0 &&
		m.ReasoningContent == ""
}

// joinTurnContent joins two merged turns' text with a blank line, tolerating an
// empty side so a merge never introduces leading or trailing blank lines.
func joinTurnContent(a, b string) string {
	if strings.TrimSpace(a) == "" {
		return b
	}
	if strings.TrimSpace(b) == "" {
		return a
	}
	return a + "\n\n" + b
}

// TurnOrigin names WHO authored a message — the one bit the persist-guard turns
// on.
type TurnOrigin int

const (
	// OriginInteractive is a real turn of the operator's own conversation: what
	// the operator typed, and what the model answered them. Only these belong in
	// the conversation's standing history.
	OriginInteractive TurnOrigin = iota
	// OriginHarness is a turn the HARNESS authored to drive some background
	// errand — a review prompt, a curator instruction, a summarisation ask. It
	// may ride the single request it was built for, but it is not part of the
	// operator's conversation and must never be written back into it. A harness
	// turn that IS written back becomes a standing instruction every later turn
	// re-reads: the curator-takeover bug.
	OriginHarness
)

// String renders the origin as the token an operator surface or witness prints.
func (o TurnOrigin) String() string {
	if o == OriginHarness {
		return "harness"
	}
	return "interactive"
}

// TaggedTurn is a message carrying its authorship. It is the small amount of
// plumbing the persist-guard needs: the message array itself has no room for the
// bit, and a harness turn is indistinguishable from an operator turn once the
// tag is lost — which is exactly how the curator-takeover bug persisted.
type TaggedTurn struct {
	// Message is the turn itself.
	Message agent.Message
	// Origin is who authored it. The zero value is OriginInteractive, so an
	// untagged turn is treated as the operator's own — the conservative default
	// for a real conversation, where dropping a genuine turn would be the worse
	// failure.
	Origin TurnOrigin
}

// InteractiveTurn tags a message as part of the operator's own conversation.
func InteractiveTurn(m agent.Message) TaggedTurn {
	return TaggedTurn{Message: m, Origin: OriginInteractive}
}

// HarnessTurn tags a message as harness-authored: usable for the request being
// built, never persistable into the conversation's standing history.
func HarnessTurn(m agent.Message) TaggedTurn {
	return TaggedTurn{Message: m, Origin: OriginHarness}
}

// Persistable reports whether this turn may enter the interactive conversation's
// standing history.
func (t TaggedTurn) Persistable() bool { return t.Origin == OriginInteractive }

// PersistableTurns is the persist-guard: it returns the messages that may be
// written back into the interactive conversation's standing history, dropping
// every harness-authored turn. Run the history through this before persisting it
// and the curator-takeover class cannot happen — a background errand's prompt
// rides its own request and then falls away, instead of becoming a standing
// instruction the operator never wrote.
//
// The result is a fresh slice (empty, never nil, when nothing is persistable),
// so the caller cannot alias the tagged input.
func PersistableTurns(turns []TaggedTurn) []agent.Message {
	out := make([]agent.Message, 0, len(turns))
	for _, t := range turns {
		if t.Persistable() {
			out = append(out, t.Message)
		}
	}
	return out
}
