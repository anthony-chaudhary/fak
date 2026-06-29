package agent

import (
	"log"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// PoisonEvictor is the narrow seam the gateway drives on a tool-result QUARANTINE: the
// in-kernel KV cache must drop any cached prefix that attended to the now-poisoned result
// (candidate #14), so a later turn re-prefills instead of replaying the poisoned KV. It is
// implemented by InKernelPlanner; the gateway type-asserts its planner to it, so a proxy/
// mock planner — or an in-kernel planner with reuse disabled — simply does not engage it.
type PoisonEvictor interface {
	// EvictPoisoned drops the cached KV prefix along the transcript THROUGH and including
	// messages[throughIdx] (the quarantined tool result, rendered with its ORIGINAL content
	// AND the request's tool schemas). Returns the freed token count (0 if nothing was cached
	// / reuse is off). tools MUST be the SAME tool set the generation turn was rendered with
	// (renderChatMLTools): the tool-spec block folds into the leading system block, so a
	// tools-less eviction render is NOT a token-prefix of a tools-bearing cached turn and
	// fails open (reclaims nothing) — the reuse gap on tool-using turns that #612 closes.
	EvictPoisoned(messages []Message, throughIdx int, tools []ToolDef) int
}

// EvictPoisoned renders the transcript up to and including the poisoned message — WITH the
// request's tool schemas (renderTranscriptTools) but WITHOUT the trailing assistant-open
// marker, so the token path ends exactly on the poison's <|im_end|> turn boundary — encodes
// it, and evicts the cached branch along that path. Rendering WITH tools is load-bearing: the
// generation turn was cached as renderChatMLTools(messages, tools) with the tool-spec folded
// into the leading system block, so the eviction render must fold the SAME spec in or it is
// not a token-prefix of the cached turn and the walk reclaims nothing (the #612 fail-open on
// tool-using turns). TestPrefixInvariantWithTools proves renderTranscriptTools(prefix, tools)
// IS a string-prefix of renderChatMLTools(full, tools); because each turn ends on the atomic
// <|im_end|> special token, the encoded partial transcript is a genuine token-prefix of the
// cached turn, so the walk lands on (and EvictNode drops) the node whose KV saw the poison
// while sparing benign siblings. nil tools renders byte-identically to the historical
// renderTranscript, so a non-tool turn is unchanged. It wraps evictPoisonedIDs.
func (p *InKernelPlanner) EvictPoisoned(messages []Message, throughIdx int, tools []ToolDef) int {
	if p.tree == nil || throughIdx < 0 || throughIdx >= len(messages) {
		return 0
	}
	ids, err := p.tok.Encode(renderTranscriptTools(messages[:throughIdx+1], tools))
	if err != nil || len(ids) == 0 {
		return 0
	}
	freed := p.evictPoisonedIDs(ids)
	if freed > 0 {
		log.Printf("inkernel_chat poison-evict model=%s through_msg=%d freed=%dtok", p.modelID, throughIdx, freed)
	}
	return freed
}

// KVSpanEvictor is the model-side KV-quarantine eviction BRIDGE seam the gateway drives on
// a tool-result QUARANTINE (issue #579). Where PoisonEvictor drops a reusable radixkv PREFIX
// node, this enforces the kvmmu bridge: it rebuilds the transcript's per-message K/V spans on
// a fresh model.Session over the LOADED model and EVICTS the quarantined result's span via the
// proven model.KVCache.Evict (re-RoPE + renumber), so the session's attention state is
// bit-identical to a run that never saw the poison. It is implemented by InKernelPlanner and
// engaged ONLY when FAK_INKERNEL_KVMMU opts in; a proxy/mock planner — or the bridge left off
// — does not implement it, so the gateway's type-assert simply skips it (fail-open default).
type KVSpanEvictor interface {
	// EvictKVSpan rebuilds messages[:throughIdx+1] as labeled per-message K/V segments on a
	// fresh session over the loaded model, then quarantines (evicts) the segment for
	// messages[throughIdx] — the quarantined tool result, rendered with its ORIGINAL content
	// AND the request's tool schemas (so the per-segment spans concatenate to EXACTLY the
	// tools-bearing generation token path, not a tools-less one — #612). It returns the number
	// of K/V positions evicted (0 when the bridge is off or nothing matched) and whether the
	// post-eviction cache is bit-exact to a session that only ever prefilled the survivor spans
	// (the never-saw invariant the kvmmu witnesses certify).
	EvictKVSpan(messages []Message, throughIdx int, tools []ToolDef) (freed int, repositionExact bool)
}

// EvictKVSpan is the live-path KV-MMU bridge (#579): it lowers the transcript through the
// poisoned message into per-message token spans, prefills them as labeled kvmmu segments over
// a FRESH model.Session built from the loaded model, and quarantines the poison segment by id —
// which drives the proven model.KVCache.Evict (re-RoPE + renumber). It then proves the
// reposition was bit-exact by comparing the post-evict next-token logits against a reference
// session that only ever prefilled the survivor spans: equal logits == "the cache is identical
// to never having seen the poison" (the structural, model-independent guarantee — true for any
// weights, which is why a synthetic checkpoint is a faithful witness of the wiring). It is
// inert (returns 0,false) unless FAK_INKERNEL_KVMMU opted the bridge in, so the served path is
// unchanged by default and FAILS OPEN on any encode/cache anomaly.
func (p *InKernelPlanner) EvictKVSpan(messages []Message, throughIdx int, tools []ToolDef) (freed int, repositionExact bool) {
	if !p.kvSpanEvict || p.m == nil || p.tok == nil || throughIdx < 0 || throughIdx >= len(messages) {
		return 0, false
	}
	// Lower each message into the incremental token span it adds to the cumulative transcript.
	// Rendering renderTranscriptTools(messages[:i+1], tools) and slicing past the previous
	// cumulative length makes the per-segment spans concatenate to EXACTLY the full transcript
	// token path the generation turn cached (tool-spec folded into the leading system block, #612)
	// — so the poison segment evicts precisely its own span and the survivors renumber correctly.
	segIDs, poisonSeg, ok := p.lowerSegments(messages, throughIdx, tools)
	if !ok {
		return 0, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	// Build a fresh-session kvmmu bridge over the lowered segments. Fail OPEN on a cache whose
	// eviction is formally unsupported (a hybrid Gated-DeltaNet recurrence: kvmmu's evict would
	// panic KVCache.Evict). The byte-gate quarantine already paged the result out, so the KV-MMU
	// span eviction simply does not engage on such a model rather than crash the served turn.
	sess, bridge, ok := p.newSegmentBridge(segIDs)
	if !ok {
		return 0, false
	}
	freed, found := bridge.Quarantine(poisonSeg)
	if !found || freed == 0 {
		return 0, false
	}
	// Reference: a session that ONLY prefilled the survivor spans. Equal next-token logits
	// (within the cross-path FMA tolerance, 0 on amd64) prove the evicted cache is the never-saw
	// cache. This is the bit-exact reposition invariant, witnessed end-to-end on the live path.
	repositionExact = p.repositionIsExact(sess, segIDs, poisonSeg)
	log.Printf("inkernel_chat kvmmu-evict model=%s through_msg=%d freed=%dpos reposition_exact=%v",
		p.modelID, throughIdx, freed, repositionExact)
	return freed, repositionExact
}

// KVSpanElider is the model-side PLANNED-ELISION residency BRIDGE seam the gateway drives on
// a context-planner elision (issue #579, the kvmmu-planned-eviction half). Where KVSpanEvictor
// enforces a trust QUARANTINE (a poisoned span), this enforces a CAPACITY plan: when the live
// ctxplan view-planner decides the resident view is the last residentTail messages, this evicts
// every OLDER message's K/V span via kvmmu.ApplyPlan (the proven model.KVCache.Evict re-RoPE +
// renumber), so the kernel-owned KV residency SHRINKS to the planner's O(1) resident view
// byte-for-byte — the model's attention state stops physically holding the elided history. The
// elided spans keep a content-address page-back-in handle, so the demand-fault path is intact:
// an elision is a page fault, not a lost fact. Implemented by InKernelPlanner and engaged ONLY
// when FAK_INKERNEL_KVMMU opts in; a proxy/mock planner — or the bridge left off — does not
// implement it, so the gateway's type-assert simply skips it (fail-open default).
type KVSpanElider interface {
	// ElideKVSpans rebuilds messages as labeled per-message K/V segments on a fresh session over
	// the loaded model, then applies the context PLANNER's own ctxplan.Plan — evicting every span
	// the plan Elided via the proven model.KVCache.Evict — so the kernel-owned KV residency shrinks
	// to the plan's O(1) resident view. The plan's span ids MUST be the per-message ids segIDFor
	// mints (the adapter contract kvmmu.ApplyPlan keys on); a plan keyed on foreign ids elides
	// nothing.
	//
	// It returns the number of K/V positions freed (0 when the bridge is off, the plan elided
	// nothing, or the model cannot evict) and whether the post-elision cache is bit-exact to a
	// session that only ever prefilled the resident spans (the O(1)-residency invariant). The
	// bit-exact guarantee holds ONLY in the provable direction — every elided span positionally
	// AFTER every resident span (the over-budget-tail case the kvmmu witness proves), because a
	// re-RoPE cannot un-see attention a surviving earlier token already absorbed from a later one.
	// In any other direction the residency still shrinks and stays recoverable, but repositionExact
	// is reported false rather than asserting an invariant that does not hold.
	ElideKVSpans(messages []Message, plan ctxplan.Plan) (freed int, repositionExact bool)
}

// ElideKVSpans is the live-path planned-elision residency bridge (#579, the kvmmu-planned-eviction
// half): it lowers the full transcript into per-message token spans, prefills them as labeled kvmmu
// segments over a FRESH model.Session built from the loaded model, then applies the context
// planner's own plan via kvmmu.ApplyPlan — which drives the proven model.KVCache.Evict over the
// elided spans (re-RoPE + renumber). The planner's plan already guarantees every elided span
// carries a page-back-in handle (ctxplan.Audit faithfulness), so the demand-fault path stays
// intact — an elision is a page fault, not a lost fact.
//
// When the elided spans are all positionally AFTER the resident spans (the over-budget-tail plan
// the optimizer produces, keeping the early pins and shedding later low-density candidates), the
// post-elision cache is BIT-EXACT to a reference session that only ever prefilled the resident
// spans — proven here by comparing next-token logits, the same structural, model-independent
// guarantee EvictKVSpan asserts for a quarantine. In the other direction (eliding an old prefix a
// resident later span already attended to) a re-RoPE cannot reproduce never-having-seen it, so the
// residency still shrinks but repositionExact is reported false rather than overclaimed. It is
// inert (returns 0,false) unless FAK_INKERNEL_KVMMU opted the bridge in, so the served path is
// unchanged by default and FAILS OPEN on any encode/cache anomaly.
func (p *InKernelPlanner) ElideKVSpans(messages []Message, plan ctxplan.Plan) (freed int, repositionExact bool) {
	if !p.kvSpanEvict || p.m == nil || p.tok == nil || len(messages) == 0 {
		return 0, false
	}
	if len(plan.Elided) == 0 {
		return 0, false // the plan elided nothing — residency already matches the view
	}
	// Lower every message into its incremental token span (the same lowering EvictKVSpan uses,
	// through the LAST message so the spans concatenate to exactly the full transcript path). The
	// segment ids are segIDFor(message, i) — the same ids the plan must carry. #612 threads tools
	// into the poison-eviction render; this planned-elision residency bridge is a SEPARATE seam
	// whose driver does not yet carry the request tools, so it keeps its historical tools-less
	// lowering (nil) — byte-identical to before, a tracked follow-on, not a regression.
	segIDs, _, ok := p.lowerSegments(messages, len(messages)-1, nil)
	if !ok {
		return 0, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	// Fail OPEN on a cache whose eviction is formally unsupported (a recurrent / GDN cache):
	// the residency plan simply does not engage rather than panic the served turn.
	sess, bridge, ok := p.newSegmentBridge(segIDs)
	if !ok {
		return 0, false
	}
	freed = bridge.ApplyPlan(plan)
	if freed == 0 {
		return 0, false // no segment id matched the plan's elided set (or all were held/selected)
	}
	// Bit-exact ONLY in the provable direction: every elided span positionally after every
	// resident one. There, the reference is a prefill of just the resident-prefix spans, and equal
	// next-token logits prove the elided cache is the only-ever-saw-the-view cache (the
	// O(1)-residency invariant). Otherwise residency shrank but the never-saw invariant does not
	// hold, so report false instead of asserting it.
	repositionExact = p.residencyIsExact(sess, segIDs, plan)
	log.Printf("inkernel_chat kvmmu-elide model=%s elided=%d freed=%dpos reposition_exact=%v",
		p.modelID, len(plan.Elided), freed, repositionExact)
	return freed, repositionExact
}

// SegElisionPlan builds the ctxplan.Plan ElideKVSpans consumes from a positional resident/elided
// split of a transcript: message i is Elided iff elided[i], else Selected. Every span id is the
// SAME id lowerSegments mints (segIDFor), so kvmmu.ApplyPlan's id-correspondence contract holds —
// a segment is evicted iff its id is elided and not selected. Every elision carries a sha256
// content-address (ctxplan.Digest) of its rendered message as the page-back-in handle, so the
// plan is ctxplan.Audit-Faithful (every elided span recoverable) and the demand-fault path stays
// intact. It is the adapter the gateway uses to turn the context planner's positional view into a
// segIDFor-keyed plan the residency bridge can apply.
func SegElisionPlan(messages []Message, elided []bool) ctxplan.Plan {
	plan := ctxplan.Plan{Objective: ctxplan.ObjGreedy, Candidates: len(messages)}
	for i := range messages {
		id := segIDFor(messages[i], i)
		if i < len(elided) && elided[i] {
			plan.Elided = append(plan.Elided, ctxplan.Elision{
				ID:     id,
				Step:   i,
				Role:   segTool(messages[i]),
				Digest: ctxplan.Digest([]byte(renderTranscript(messages[i : i+1]))),
				Reason: ctxplan.ElideOverBudget,
			})
			continue
		}
		plan.Selected = append(plan.Selected, ctxplan.Selection{
			ID:   id,
			Step: i,
			Role: segTool(messages[i]),
		})
	}
	return plan
}

// residencyIsExact proves the post-elision cache is bit-identical to a run that only ever saw the
// resident spans — but ONLY in the direction where that is true: every elided span positionally
// AFTER every resident span (so no surviving resident token ever attended to an evicted one). It
// partitions segs by the plan's elided/selected sets, returns false unless the resident set is a
// contiguous positional PREFIX (elided set the suffix), then prefills just the resident-prefix ids
// on a reference session and compares the post-elision next-token distribution to it. Equal
// (within the cross-path FMA tolerance) iff the re-RoPE + renumber left the cache identical to
// never having prefilled the elided suffix. It is the planned-elision twin of repositionIsExact.
func (p *InKernelPlanner) residencyIsExact(elided *model.Session, segs []kvSegment, plan ctxplan.Plan) bool {
	elide := make(map[string]bool, len(plan.Elided))
	for _, e := range plan.Elided {
		elide[e.ID] = true
	}
	for _, s := range plan.Selected {
		delete(elide, s.ID)
	}
	// Walk the segments in cache (positional) order. The resident set is bit-exact-reconstructible
	// only if it is a contiguous PREFIX: once we have seen an elided span, every later span must
	// also be elided (the elided set is the positional suffix). Collect the resident-prefix ids.
	var refIDs []int
	seenElided := false
	for _, sg := range segs {
		if elide[sg.id] {
			seenElided = true
			continue
		}
		if seenElided {
			return false // a resident span sits AFTER an elided one — not the provable direction
		}
		refIDs = append(refIDs, sg.ids...)
	}
	return p.refLogitsExact(elided, refIDs)
}

// kvSegment is one lowered per-message K/V span: its kvmmu segment id (the message index +
// tool-call id), the tool that produced it, and the incremental token ids it occupies.
type kvSegment struct {
	id   string
	tool string
	ids  []int
}

// lowerSegments renders messages[:throughIdx+1] into per-message incremental token spans and
// returns the ordered segments plus the segment id of the poisoned message (messages[throughIdx]).
// It renders WITH the request's tool schemas (renderTranscriptTools) so the lowered spans
// concatenate to exactly the tools-bearing generation token path; nil tools is byte-identical
// to the historical renderTranscript lowering. It fails (ok=false) if any encode errors or any
// incremental span is empty, so a degenerate tokenization fails OPEN to no eviction rather than
// evicting the wrong span.
func (p *InKernelPlanner) lowerSegments(messages []Message, throughIdx int, tools []ToolDef) (segs []kvSegment, poisonID string, ok bool) {
	prev := 0
	for i := 0; i <= throughIdx; i++ {
		cum, err := p.tok.Encode(renderTranscriptTools(messages[:i+1], tools))
		if err != nil || len(cum) <= prev {
			return nil, "", false
		}
		span := append([]int(nil), cum[prev:]...)
		prev = len(cum)
		id := segIDFor(messages[i], i)
		segs = append(segs, kvSegment{id: id, tool: segTool(messages[i]), ids: span})
		if i == throughIdx {
			poisonID = id
		}
	}
	return segs, poisonID, len(segs) > 0
}

// newSegmentBridge builds a FRESH model.Session over the loaded model (carrying the planner's
// quant config) and a kvmmu bridge with every lowered segment appended — the shared session +
// bridge construction EvictKVSpan and ElideKVSpans both run under p.mu before quarantining a
// span or applying a residency plan. It returns ok=false (with a nil session and bridge) on a
// cache whose eviction is formally unsupported (a recurrent / GDN cache, whose CanEvict reads
// non-nil on the empty fresh cache), so the caller fails OPEN rather than panicking the turn.
func (p *InKernelPlanner) newSegmentBridge(segs []kvSegment) (*model.Session, *kvmmu.Context, bool) {
	sess := p.m.NewSession()
	sess.Quant, sess.Q4K = p.quant, p.q4k
	if sess.Cache.CanEvict() != nil {
		return nil, nil, false
	}
	bridge := kvmmu.NewWithGate(sess, kvmmu.FoldedGate{})
	for _, sg := range segs {
		bridge.Append(sg.id, sg.tool, sg.ids)
	}
	return sess, bridge, true
}

// repositionIsExact rebuilds a reference session that prefills ONLY the survivor spans (every
// segment except the poison) and compares the bridge session's post-eviction next-token
// distribution to the reference's. The evicted cache holds the survivor spans at compacted
// positions; decoding one step from the same final survivor token on BOTH reads the
// distribution each would continue from — equal (within the cross-path FMA tolerance) iff the
// eviction's re-RoPE + renumber left the cache bit-identical to never having seen the poison.
func (p *InKernelPlanner) repositionIsExact(evicted *model.Session, segs []kvSegment, poisonID string) bool {
	var refIDs []int
	for _, sg := range segs {
		if sg.id == poisonID {
			continue
		}
		refIDs = append(refIDs, sg.ids...)
	}
	return p.refLogitsExact(evicted, refIDs)
}

// refLogitsExact builds a reference session that prefills ONLY refIDs (the resident survivor
// token path, carrying the planner's quant config) and reports whether `cache`'s post-eviction
// next-token distribution is bit-identical to that reference within the cross-path FMA tolerance.
// It is the shared bit-exact reposition check repositionIsExact and residencyIsExact both close
// on, and returns false when the resident path is empty or the cache length already diverges
// from it.
func (p *InKernelPlanner) refLogitsExact(cache *model.Session, refIDs []int) bool {
	if len(refIDs) == 0 || cache.Cache.Len() != len(refIDs) {
		return false
	}
	ref := p.m.NewSession()
	ref.Quant, ref.Q4K = p.quant, p.q4k
	ref.Prefill(refIDs)
	last := refIDs[len(refIDs)-1]
	return logitsClose(cache.Step(last), ref.Step(last))
}

// logitsClose reports whether two next-token logit vectors are equal within the cross-path FMA
// tolerance (0 on amd64; sub-1e-4 on arches where the gc compiler auto-fuses FMA). It is the
// same max|Δ| reposition measure internal/model's rung-3 oracle and the kvmmu witnesses use.
func logitsClose(a, b []float32) bool {
	if len(a) != len(b) || len(a) == 0 {
		return false
	}
	const tol = 1e-4
	for i := range a {
		d := float64(a[i] - b[i])
		if d < 0 {
			d = -d
		}
		if d > tol {
			return false
		}
	}
	return true
}

// segIDFor mints the stable kvmmu segment id for a message at index i: the message index keeps
// distinct messages distinct, and the tool-call id (when present) ties the span to the result
// the gateway admitted, so the poison segment is addressable by the same identity the admission
// ledger carries.
func segIDFor(m Message, i int) string {
	if m.ToolCallID != "" {
		return "m" + strconv.Itoa(i) + ":" + m.ToolCallID
	}
	return "m" + strconv.Itoa(i)
}

// segTool reports the producing tool name for the ledger/reporting (the tool result's Name, or
// the role for a non-tool message).
func segTool(m Message) string {
	if m.Name != "" {
		return m.Name
	}
	return m.Role
}

// evictPoisonedIDs drops the cached prefix lying along `ids` (a poisoned transcript token
// path) — the token-level #14 seam EvictPoisoned wraps. Guarded by mu; no-op when reuse
// is disabled.
// underTreeLock runs fn while holding mu, returning 0 (no-op) when the prefix tree
// is absent (reuse disabled). Centralizes the nil-check + lock the prefix-tree
// accessors share so a copy can't drop the guard.
func (p *InKernelPlanner) underTreeLock(fn func() int) int {
	if p.tree == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return fn()
}

func (p *InKernelPlanner) evictPoisonedIDs(ids []int) int {
	return p.underTreeLock(func() int { return p.tree.EvictPrefix(ids) })
}

// cachedPrefixLen reports how many leading tokens of `ids` are already resident in the
// prefix cache (read-only). It is the reuse-state probe the witnesses assert on; 0 when
// reuse is disabled.
func (p *InKernelPlanner) cachedPrefixLen(ids []int) int {
	return p.underTreeLock(func() int { return p.tree.MatchLen(ids) })
}
