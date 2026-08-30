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
	labels, _, invalidRetention := anthropicRetentionMetadata(raw)
	// The continuation seed is the LAST user turn — the state the next turn actually continues
	// from. Resolved in a first pass because it is a property of the array, not of one element.
	lastUser := -1
	for i, el := range doc.Messages {
		if anthropicElementRole(el) == "user" {
			lastUser = i
		}
	}
	retentionByPage := make(map[string][]ctxplan.RetentionAnnotation, len(labels))
	knownPage := make(map[string]bool, len(doc.Messages))
	for i := range doc.Messages {
		knownPage["msg:"+strconv.Itoa(i)] = true
	}
	invalidTarget := false
	for _, label := range labels {
		if !knownPage[label.PageID] {
			invalidTarget = true
			continue
		}
		retentionByPage[label.PageID] = append(retentionByPage[label.PageID], ctxplan.RetentionAnnotation{
			Intent: label.Intent, Source: label.Source, ReasonCode: label.ReasonCode,
		})
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
		id := "msg:" + strconv.Itoa(i)
		p := ctxplan.Page{ID: id, Kind: kind, Tokens: (len(el) + 3) / 4, Retention: retentionByPage[id]}
		pages = append(pages, p)
		if p.Class() == ctxplan.ClassPinned {
			pinned = append(pinned, el)
		}
	}
	if invalidTarget || invalidRetention {
		// Keep the parser signature compatible with the survival gate while making malformed
		// metadata and unknown stable addresses fail closed in PlanEviction. The invalid marker is
		// bounded metadata and never enters the provider body.
		pages[0].Retention = append(pages[0].Retention, ctxplan.RetentionAnnotation{})
	}
	return pages, pinned, true
}

type anthropicRetentionLabel struct {
	PageID     string                  `json:"page_id"`
	Intent     ctxplan.RetentionIntent `json:"intent"`
	Source     string                  `json:"source"`
	ReasonCode string                  `json:"reason_code,omitempty"`
}

// anthropicRetentionMetadata locates the optional kernel-only transport without conflating a
// malformed value with absence. Presence controls provider stripping; invalidity is projected as
// a bounded planner annotation so malformed metadata refuses atomically before any eviction.
func anthropicRetentionMetadata(raw []byte) (labels []anthropicRetentionLabel, present, invalid bool) {
	var doc map[string]json.RawMessage
	if json.Unmarshal(raw, &doc) != nil {
		return nil, false, false
	}
	fakRaw, ok := doc["fak"]
	if !ok {
		return nil, false, false
	}
	var fak map[string]json.RawMessage
	if json.Unmarshal(fakRaw, &fak) != nil {
		return nil, false, false
	}
	retentionRaw, ok := fak["retention"]
	if !ok {
		return nil, false, false
	}
	present = true
	trimmed := bytes.TrimSpace(retentionRaw)
	if len(trimmed) == 0 || trimmed[0] != '[' || json.Unmarshal(retentionRaw, &labels) != nil {
		return nil, true, true
	}
	return labels, true, false
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

func pagesHaveRetention(pages []ctxplan.Page) bool {
	for _, p := range pages {
		if len(p.Retention) != 0 {
			return true
		}
	}
	return false
}

// stripAnthropicRetention removes the kernel-only label transport before a body reaches the
// provider. Unannotated requests never call it and retain the existing byte-for-byte path.
func stripAnthropicRetention(raw []byte) []byte {
	var doc map[string]json.RawMessage
	if json.Unmarshal(raw, &doc) != nil {
		return raw
	}
	fakRaw, ok := doc["fak"]
	if !ok {
		return raw
	}
	var fak map[string]json.RawMessage
	if json.Unmarshal(fakRaw, &fak) != nil {
		return raw
	}
	delete(fak, "retention")
	if len(fak) == 0 {
		delete(doc, "fak")
	} else if encoded, err := json.Marshal(fak); err == nil {
		doc["fak"] = encoded
	}
	encoded, err := json.Marshal(doc)
	if err != nil {
		return raw
	}
	return encoded
}

// compactAnthropicByPlan applies the planner's exact stable page addresses. It keeps each
// message's role and object position but replaces evicted content with a bounded tombstone, which
// preserves Anthropic's role alternation while removing the resident page bytes. This path is
// opt-in: only requests carrying retention metadata use it; unannotated callers retain the legacy
// byte compactor unchanged.
func compactAnthropicByPlan(raw []byte, plan ctxplan.EvictionPlan, budget int) ([]byte, agent.CompactOutcome, bool) {
	if len(plan.Evict) == 0 {
		return raw, agent.CompactOutcome{Reason: agent.CompactReasonUnderBudget}, true
	}
	var doc map[string]json.RawMessage
	if json.Unmarshal(raw, &doc) != nil {
		return nil, agent.CompactOutcome{}, false
	}
	var messages []json.RawMessage
	if json.Unmarshal(doc["messages"], &messages) != nil {
		return nil, agent.CompactOutcome{}, false
	}
	evicted := make(map[string]bool, len(plan.Evict))
	for _, id := range plan.Evict {
		evicted[id] = true
	}
	for i, message := range messages {
		id := "msg:" + strconv.Itoa(i)
		if !evicted[id] {
			continue
		}
		// Anthropic tool_use/tool_result blocks form a provider-validated history. The planner's
		// message addresses do not yet encode those pair edges, so independently rewriting either
		// side could manufacture an invalid request. Refuse this annotated compaction atomically.
		if anthropicElementHasToolBlocks(message) {
			return nil, agent.CompactOutcome{}, false
		}
		var object map[string]json.RawMessage
		if json.Unmarshal(message, &object) != nil {
			return nil, agent.CompactOutcome{}, false
		}
		tombstone, _ := json.Marshal("[fak: context page " + id + " evicted]")
		object["content"] = tombstone
		encoded, err := json.Marshal(object)
		if err != nil {
			return nil, agent.CompactOutcome{}, false
		}
		messages[i] = encoded
	}
	encodedMessages, err := json.Marshal(messages)
	if err != nil {
		return nil, agent.CompactOutcome{}, false
	}
	doc["messages"] = encodedMessages
	out, err := json.Marshal(doc)
	if err != nil {
		return nil, agent.CompactOutcome{}, false
	}
	// Enforce the budget against the body we will actually return. Tombstones retain message
	// framing, so planned page-token removal is only a candidate calculation, not a postcondition.
	outPages, _, ok := anthropicSurvivalPages(out)
	if !ok {
		return nil, agent.CompactOutcome{}, false
	}
	residentTokens := 0
	for _, page := range outPages {
		residentTokens += page.Tokens
	}
	if residentTokens > budget {
		return nil, agent.CompactOutcome{}, false
	}
	shedTokens := 0
	if len(raw) > len(out) {
		shedTokens = (len(raw) - len(out) + 3) / 4
	}
	return out, agent.CompactOutcome{Reason: agent.CompactReasonNone, Dropped: len(plan.Evict), ShedTokens: shedTokens}, true
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
//
// trace keys the CONTINUATION CONTRACT (#2422) this boundary emits when a compaction actually
// fires: the plan computed here is projected onto the wire record the next completed turn reports
// on both channels (an in-band `[fak]` note and the `fak.compaction` extension). It is recorded
// from the plan this compaction ran under rather than re-derived downstream, so the contract cannot
// disagree with the boundary it describes. "" (tests, non-session callers) records nothing.
func (s *Server) compactWithSurvivalClasses(raw []byte, opts agent.CompactOptions, trace string) ([]byte, agent.CompactOutcome) {
	pages, pinned, ok := anthropicSurvivalPages(raw)
	if !ok {
		return agent.CompactAnthropicHistoryWithOptions(raw, opts)
	}
	_, retentionPresent, _ := anthropicRetentionMetadata(raw)
	annotated := pagesHaveRetention(pages)
	providerRaw := raw
	if retentionPresent {
		providerRaw = stripAnthropicRetention(raw)
	}
	plan := ctxplan.PlanEviction(pages, s.compactHistoryBudget)
	if plan.Refusal != "" {
		reason := plan.Refusal
		if reason == ctxplan.ReasonPinEvictRefused {
			reason = agent.CompactReasonPinEvictRefused
		}
		return providerRaw, agent.CompactOutcome{Reason: reason}
	}
	// announce records this boundary's continuation contract (#2422) for the next completed turn to
	// report. It fires ONLY on a return that actually hands back a compacted body — a bail and a
	// PIN_EVICT_REFUSED both forward the body unchanged, so no boundary was crossed and announcing
	// one would tell the model a loss that did not happen. It reads the PRE-compaction elements,
	// which are exactly the pages the returned body no longer holds.
	announce := func() {
		s.noteCompactionContract(trace, compactionContractFrom(pages, anthropicMessageElements(raw), plan))
	}
	if annotated {
		out, outcome, applied := compactAnthropicByPlan(providerRaw, plan, s.compactHistoryBudget)
		if !applied {
			return providerRaw, agent.CompactOutcome{Reason: agent.CompactReasonWindowNoDrop}
		}
		if outcome.Reason == agent.CompactReasonNone {
			announce()
		}
		return out, outcome
	}
	out, outcome := agent.CompactAnthropicHistoryWithOptions(providerRaw, opts)
	if outcome.Reason != agent.CompactReasonNone {
		return out, outcome
	}
	if pinnedPagesSurvive(out, pinned) {
		announce()
		return out, outcome
	}
	retry := opts
	retry.Budget = opts.Budget + plan.PinnedTokens
	if out2, outcome2 := agent.CompactAnthropicHistoryWithOptions(providerRaw, retry); outcome2.Reason == agent.CompactReasonNone && pinnedPagesSurvive(out2, pinned) {
		announce()
		return out2, outcome2
	}
	return providerRaw, agent.CompactOutcome{Reason: agent.CompactReasonPinEvictRefused}
}
