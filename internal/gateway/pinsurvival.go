package gateway

import (
	"bytes"
	"encoding/json"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// pinsurvival.go — the SURVIVAL-CLASS gate on the compaction path (#2421).
//
// # What it is for
//
// The compaction saga upstream is a list of state lost because the summarizer had no contract:
// plan mode, tool schemas, safety instructions each eaten in turn and each individually restored
// afterwards by hand. The repair is not a better summarizer — it is to stop asking a summarizer
// what may be dropped. Every page in the compactor's eviction domain gets a KIND stamped by the
// kernel, the kind fixes a survival class (ctxplan.ClassOf), and the compaction outcome is
// checked against those classes before it reaches the wire. What survives becomes a verified
// property of the body rather than a hope about the drop heuristic.
//
// # The eviction domain is messages[], and only messages[]
//
// A survival class is a claim about what a PLAN MAY EVICT, so it is only meaningful over the
// pages a plan can actually reach. The byte-level compactor rewrites the messages[] array and
// nothing else — the top-level `system` and `tools` blocks sit outside it and are forwarded
// verbatim on every path (that is precisely the cached head the whole design protects). Typing
// them here would be decorative: a class on a page no plan can evict cannot fail, so it cannot
// witness anything. They are therefore out of scope for this gate, not unclassified by oversight.
//
// # How the kind is assigned (deterministically, and never by the model)
//
//	the ACTIVE STEER        a [fak:goal]-marked turn      -> KindActiveSteer      (PINNED)
//	the CONTINUATION SEED   the last user turn            -> KindContinuationSeed (PINNED)
//	a CAS-BACKED RESULT     a turn carrying tool blocks   -> KindCASResult        (REPLAYABLE)
//	AGED PROSE              everything else               -> KindTranscriptProse  (EVICTABLE)
//
// Each rule reads STRUCTURE (a wire marker fak itself writes, a role, a block type), never model
// prose, so no amount of text a model emits can move its own turn into the protected set.
//
// The steer rule shares agent.IsGoalPinnedMessage with the compactor that hoists the marked turn,
// so the pinned set is exactly the set the compactor already promises to preserve — which is why
// this gate is INERT on today's traffic and fires only when a plan would genuinely eat one.
//
// The originating first user turn is deliberately REPLAYABLE, not PINNED: when the compactor
// tombstones it, it embeds a content-addressed `id=<hex>` handle and hands the gateway the full
// bytes to stash, so fak_context_restore pages the whole task back in. That is the definition of
// replayable — recoverable exactly, on demand — and pinning it instead would refuse every
// compaction on the un-marked sessions the tombstone path exists to serve.
//
// # Fail-safe in both directions
//
// A body this file cannot classify (non-JSON, no messages[]) leaves the gate inert and the
// compactor unchanged, so an unparseable request is never made worse. A body it CAN classify is
// only ever made more conservative: the gate's two outcomes are "compact as planned" and "forward
// the body unchanged with PIN_EVICT_REFUSED". It can refuse a compaction; it can never author a
// byte.

// anthropicSurvivalPages types the eviction domain of an outbound Anthropic body.
//
// It returns one ctxplan.Page per messages[] element (in wire order) and, alongside, the VERBATIM
// element bytes of every page that classes PINNED — the exact byte strings a compacted body must
// still contain for the pinned set to have survived. ok is false when the body has no classifiable
// eviction domain, which leaves the gate inert.
//
// Costs are the compactor's own ~4-chars-per-token currency over the element's raw bytes, so the
// pinned floor and the compaction budget are denominated the same way and can be compared without
// a conversion that could quietly disagree.
func anthropicSurvivalPages(raw []byte) (pages []ctxplan.Page, pinned [][]byte, ok bool) {
	var doc struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil || len(doc.Messages) == 0 {
		return nil, nil, false
	}
	// The continuation seed is the LAST user turn — the state the next turn actually continues
	// from. Resolved in a first pass because it is a property of the array, not of one element.
	lastUser := -1
	for i, el := range doc.Messages {
		if anthropicElementRole(el) == "user" {
			lastUser = i
		}
	}
	pages = make([]ctxplan.Page, 0, len(doc.Messages))
	for i, el := range doc.Messages {
		kind := ctxplan.KindTranscriptProse
		switch {
		case agent.IsGoalPinnedMessage(el):
			kind = ctxplan.KindActiveSteer
		case i == lastUser:
			kind = ctxplan.KindContinuationSeed
		case anthropicElementHasToolBlocks(el):
			kind = ctxplan.KindCASResult
		}
		p := ctxplan.Page{ID: "msg:" + strconv.Itoa(i), Kind: kind, Tokens: (len(el) + 3) / 4}
		pages = append(pages, p)
		if p.Class() == ctxplan.ClassPinned {
			pinned = append(pinned, el)
		}
	}
	return pages, pinned, true
}

// anthropicElementRole extracts a messages[] element's role, or "" when it does not parse.
func anthropicElementRole(el json.RawMessage) string {
	var m struct {
		Role string `json:"role"`
	}
	if json.Unmarshal(el, &m) != nil {
		return ""
	}
	return m.Role
}

// anthropicElementHasToolBlocks reports whether an element carries tool_use / tool_result content
// — the CAS-backed pages whose full bytes the store can page back in. A byte scan rather than a
// content decode: content is a polymorphic block array, and the two type tokens are unambiguous
// inside a JSON object where they can only appear as a `"type"` value or inside prose that the
// classification treats no differently (a false REPLAYABLE only ever costs residency, never the
// pinned guarantee, because the pinned rules are checked first).
func anthropicElementHasToolBlocks(el json.RawMessage) bool {
	return bytes.Contains(el, []byte(`"tool_result"`)) || bytes.Contains(el, []byte(`"tool_use"`))
}

// pinnedPagesSurvive reports whether every pinned page's verbatim bytes are still present in body.
// This is the post-condition that makes the guarantee KERNEL-VERIFIED rather than inherited from
// the drop heuristic's good intentions: the compactor splices on original bytes, so a page it kept
// is byte-identical and a page it dropped is simply absent.
func pinnedPagesSurvive(body []byte, pinned [][]byte) bool {
	for _, p := range pinned {
		if !bytes.Contains(body, p) {
			return false
		}
	}
	return true
}

// compactWithSurvivalClasses runs the byte-level compactor UNDER the survival-class contract. It
// is a drop-in for agent.CompactAnthropicHistoryWithOptions and returns the same pair, so the
// caller's metric, restore-handle, and fired/reason handling is unchanged.
//
// Three steps, in order:
//
//  1. REFUSE A BUDGET THAT CANNOT HOLD THE PINNED SET. If the pinned floor alone exceeds the
//     budget, no honest plan exists — every drop that reaches the budget evicts something that
//     must survive — so the body is forwarded UNCHANGED with PIN_EVICT_REFUSED. This is the
//     "rather than a lossy compaction" half: the operator gets a refusal they can act on (raise
//     the budget, shed load elsewhere) instead of damage that surfaces several turns later.
//
//     It is charged against the CONFIGURED budget, not opts.Budget. The early-firing ramp lowers
//     opts.Budget as a firing optimisation; a survival contract that moved with an optimisation's
//     transient value would refuse and admit the same session on alternating turns.
//
//  2. VERIFY THE FIRED PLAN. A compaction that fired must still carry every pinned page's bytes.
//
//  3. RETRY AGAINST THE EVICTABLE SET ONLY. When it does not, the plan is re-run with the pinned
//     floor ADDED to the budget — which is what "evict from the evictable set only" means in the
//     budget language this compactor speaks: the tokens the pinned pages occupy stop being
//     available for the drop to claim, so the kept window has to reach past them. If that retry
//     also fails to preserve the pinned set, the refusal stands and the body is forwarded
//     unchanged. Losing the standing instruction is not a cheaper outcome than a longer prompt.
func (s *Server) compactWithSurvivalClasses(raw []byte, opts agent.CompactOptions) ([]byte, agent.CompactOutcome) {
	pages, pinned, ok := anthropicSurvivalPages(raw)
	if !ok {
		return agent.CompactAnthropicHistoryWithOptions(raw, opts)
	}
	plan := ctxplan.PlanEviction(pages, s.compactHistoryBudget)
	if plan.Refusal != "" {
		return raw, agent.CompactOutcome{Reason: agent.CompactReasonPinEvictRefused}
	}
	out, outcome := agent.CompactAnthropicHistoryWithOptions(raw, opts)
	if outcome.Reason != agent.CompactReasonNone || pinnedPagesSurvive(out, pinned) {
		return out, outcome
	}
	retry := opts
	retry.Budget = opts.Budget + plan.PinnedTokens
	if out2, outcome2 := agent.CompactAnthropicHistoryWithOptions(raw, retry); outcome2.Reason == agent.CompactReasonNone && pinnedPagesSurvive(out2, pinned) {
		return out2, outcome2
	}
	return raw, agent.CompactOutcome{Reason: agent.CompactReasonPinEvictRefused}
}
