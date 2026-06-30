package agent

// anthropic_compact.go — the cache-prefix-preserving history rewrite for the flagship
// `fak guard -- claude` Anthropic passthrough (the deferred "#555 req.Raw transform").
//
// The passthrough forwards the inbound /v1/messages body to the real Anthropic API
// BYTE-FOR-BYTE so the client's prompt-cache prefix survives → a real cache hit
// (messages.go: WithRawRequestBody / messages_stream_passthrough.go: StreamAnthropicRaw).
// That byte-faithfulness is also why the existing context planner (CtxViewPlanner /
// maybePlanMessages) — which rewrites the DECODED []Message — never reaches this route:
// it would force a re-serialize that reorders JSON keys and destroys the cached prefix.
//
// CompactAnthropicHistory is the byte-level alternative. It shrinks the OUTBOUND body by
// dropping OLD whole turns, but it does so by SPLICING on the original bytes so the cached
// prefix is copied verbatim (a memcpy), never re-marshalled. The load-bearing invariant:
//
//	The protected prefix = every whole message up to AND INCLUDING the message that holds
//	the FIRST cache_control breakpoint (the STABLE cached head the provider reuses every
//	turn). Whole MIDDLE messages between it and the recent kept window may be dropped/stubbed
//	— that middle is the un-cacheable span the provider re-bills anyway. On ANY ambiguity the
//	function returns its input UNCHANGED (identity). Anchoring on the FIRST breakpoint (not
//	the last) is what lets compaction fire on real Claude Code traffic, which marks both the
//	static head AND recent turns (see firstBreakpointMessage).
//
// Protecting at WHOLE-MESSAGE granularity (rather than at the block where the breakpoint
// sits) is the trick that keeps the splice a pure byte copy: a content array is never
// split, so a partially-cached message is never re-serialized. It costs a little
// compaction headroom (the breakpoint's own message is always kept) but makes the cache
// guarantee a bytes.Equal, not a hope.
//
// This is a REQUEST-side transform only. It touches the bytes sent upstream; it never
// touches the decoded req.Messages the kernel adjudicates, so admitInboundResults and
// adjudicateProposed still see the FULL history — the trust boundary is unchanged.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
)

// compactStubPrefix marks the synthetic message that stands in for the dropped turns, so
// the model (and a human reading the wire) sees that earlier turns were compacted rather
// than silently lost. It is emitted as a user-role text message between the protected
// prefix and the kept recent window.
const compactStubPrefix = "[fak] compacted "

// Compaction bail-reason vocabulary — the closed set of identity-return causes, surfaced on
// CompactOutcome so the gateway can label a metric and an operator can see WHY compaction did
// nothing (silence must not read as success). CompactReasonNone means the body was rewritten.
const (
	CompactReasonNone           = ""             // FIRED: a rewrite happened (Dropped/ShedTokens meaningful)
	CompactReasonUnderBudget    = "under_budget" // budget<=0, or the compactible suffix already fits
	CompactReasonNonJSON        = "non_json"     // body is not a JSON object
	CompactReasonNoMsgsKey      = "no_messages_key"
	CompactReasonTooFewMsgs     = "too_few_msgs"   // < 3 messages — nothing safe to drop
	CompactReasonNoBreakpoint   = "no_breakpoint"  // no cache_control to anchor the protected prefix
	CompactReasonCachedSpan     = "cached_span"    // candidate drop would delete cache_control-marked history
	CompactReasonWindowNoDrop   = "window_no_drop" // the kept window swallowed the whole suffix
	CompactReasonSpliceFailed   = "splice_failed"
	CompactReasonRedecodeFail   = "redecode_failed" // the spliced body failed to re-decode
	CompactReasonPrefixMismatch = "prefix_mismatch" // the splice changed the protected prefix bytes
	// CompactReasonBurstUnprofitable is the head-anchored bail (CompactAnchorHead only): the drop
	// would fire, but bursting the recent breakpoint's cached suffix does not repay within the
	// remaining session horizon (CacheBurstPaysBack == false), so the warm cache hit is kept over a
	// smaller prompt. The firstbp default never returns this (it never bursts) — see #1407/#1408.
	CompactReasonBurstUnprofitable = "burst_unprofitable"
)

// CompactOutcome is the observable verdict of one compaction attempt. Reason==CompactReasonNone
// means FIRED — Dropped (whole messages stubbed out) and ShedTokens (estimated tokens removed
// from the outbound body, same ~4-chars/token currency as the budget) are then meaningful. Any
// other Reason means the body was returned unchanged (identity), and Dropped/ShedTokens are 0.
type CompactOutcome struct {
	Reason     string
	Dropped    int
	ShedTokens int
	// Diagnostic split, populated on the under_budget bail (the silent common case). The
	// protected prefix is everything THROUGH the cache_control anchor; the suffix is the
	// compactible span after it. AnchorStarved is true when the lever bailed under_budget
	// DESPITE a protected prefix that already exceeds the budget — i.e. the anchor swallowed
	// the conversation, so compaction structurally cannot fire no matter how long the session
	// grows. That is the signal that distinguishes a BENIGN idle (a genuinely short session)
	// from the anchored-near-the-end dormancy on real Claude Code traffic (#1407), which the
	// bare under_budget reason cannot tell apart. Zero/false on every other outcome.
	ProtectedPrefixTokens int
	SuffixTokens          int
	AnchorStarved         bool
}

// CompactAnchor selects where the protected (verbatim-copied) prefix ends.
type CompactAnchor int

const (
	// CompactAnchorFirstBP protects every message THROUGH the first messages[] cache_control
	// breakpoint (the warm-cache-safe default): only the middle after it is compactible. On real
	// Claude Code traffic whose only message breakpoint is RECENT, this anchors near the end and
	// the lever stays idle — the #1407 dormancy, surfaced by the AnchorStarved diagnostic (#1409).
	CompactAnchorFirstBP CompactAnchor = iota
	// CompactAnchorHead re-anchors the protected prefix on the stable provider head — a top-level
	// system/tools cache_control breakpoint serialized BEFORE messages[] — making the WHOLE message
	// array compactible. This is what lets compaction fire on real traffic (#1407), but a fire
	// bursts the recent message breakpoint's cached suffix, so it is gated on CacheBurstPaysBack
	// economics (#1408). Opt-in: a warm-cache session must keep CompactAnchorFirstBP's byte-identical
	// body, so the firing path only chooses this when the operator asks AND the burst repays.
	CompactAnchorHead
)

// defaultCacheReadMult / defaultCacheWriteMult mirror the gateway cache_pricing multipliers
// (CacheReadMultiplier / CacheWrite5mMultiplier) for the head-anchored burst gate, WITHOUT
// importing the gateway (the agent package must not depend on it — that would be an import
// cycle). Used when CompactOptions leaves the multipliers unset.
const (
	defaultCacheReadMult  = 0.1
	defaultCacheWriteMult = 1.25
)

// CompactOptions parameterizes CompactAnthropicHistoryWithOptions. The zero value (Anchor
// CompactAnchorFirstBP, no horizon) reproduces CompactAnthropicHistoryWithOutcome exactly, so
// the default firing path is byte-for-byte unchanged.
type CompactOptions struct {
	Budget int           // resident-token target for the compactible span (<=0 ⇒ identity)
	Anchor CompactAnchor // where the protected prefix ends (default: first breakpoint)
	// Session horizon for the head-anchored burst gate. Consulted only when Anchor==CompactAnchorHead
	// AND the head re-anchor actually engages (a stable head precedes messages[]). TotalTurns<=0 ⇒
	// unknown horizon ⇒ CacheBurstPaysBack is conservative (no fire unless the burst has no penalty).
	TotalTurns  int
	CurrentTurn int
	ReadMult    float64 // provider cache-read price multiplier (<=0 ⇒ defaultCacheReadMult)
	WriteMult   float64 // provider cache-write price multiplier (<=0 ⇒ defaultCacheWriteMult)
}

// CompactAnthropicHistory rewrites an outbound Anthropic /v1/messages body so the byte range
// from the start through the protected prefix (the FIRST cache_control breakpoint message — the
// stable cached head) is copied VERBATIM, and whole middle messages between it and the recent
// kept window are dropped (replaced by one stub) to bring the compactible span under budget (a
// resident-token target, ~4 chars/token to match EstimateAnthropicTokens).
//
// It returns raw UNCHANGED — the fail-safe identity — whenever it cannot prove the rewrite is
// both cache-safe and well-formed (see the CompactReason* vocabulary). The prefix bytes of a
// non-identity result are guaranteed equal to the input's prefix bytes. This is the byte-only
// wrapper; CompactAnthropicHistoryWithOutcome additionally reports WHY it bailed / how much it
// shed, for observability.
func CompactAnthropicHistory(raw []byte, budget int) []byte {
	out, _ := CompactAnthropicHistoryWithOutcome(raw, budget)
	return out
}

// CompactAnthropicHistoryWithOutcome is CompactAnthropicHistory plus the observable outcome
// (fired vs the labeled bail reason, and the dropped-turn / shed-token counts on a fire). The
// gateway uses it to emit the compaction metric family; the byte-level guarantees are identical.
// anchorCompactablePrefix is the shared front half of both byte-level rewrites
// (CompactAnthropicHistoryWithOutcome and CompactAnthropicHistoryToView), anchoring on the
// warm-cache-safe default (the FIRST messages[] breakpoint). It is anchorCompactablePrefixMode
// with CompactAnchorFirstBP, so the ctxview twin and the byte-only wrapper are unchanged.
func anchorCompactablePrefix(raw []byte, minElems int) (elems []json.RawMessage, spans []elementSpan, pfxEnd int, bail CompactOutcome, ok bool) {
	return anchorCompactablePrefixMode(raw, minElems, CompactAnchorFirstBP)
}

// anchorCompactablePrefixMode decodes the request object, finds the messages[] array (with exact
// byte spans), requires at least minElems elements, and anchors the protected prefix. On any
// ambiguity it returns ok=false with the labeled fail-safe CompactOutcome the caller should
// return verbatim. pfxEnd is the index of the last protected message (-1 when no message is
// protected, so every message is compactible).
//
// In CompactAnchorFirstBP (the default) pfxEnd is the FIRST messages[] cache_control breakpoint
// (or -1 when only a top-level `system` breakpoint exists). In CompactAnchorHead, when the stable
// provider head — a top-level system/tools cache_control breakpoint serialized BEFORE messages[]
// — exists, pfxEnd is forced to -1 so the WHOLE message array is compactible (#1407/#1408). Head
// re-anchoring engages ONLY when the head genuinely precedes messages[] in the byte stream: that
// is exactly what keeps the dominant cached head's byte prefix stable across the drop. A head
// serialized AFTER messages[] would have its prefix shifted by the drop (bursting the dominant
// cache), so head mode falls back to the first-breakpoint anchor there — never silently bursting
// the head. The firstbp condition below is left byte-identical to the pre-#1408 behavior.
func anchorCompactablePrefixMode(raw []byte, minElems int, anchor CompactAnchor) (elems []json.RawMessage, spans []elementSpan, pfxEnd int, bail CompactOutcome, ok bool) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, nil, 0, CompactOutcome{Reason: CompactReasonNonJSON}, false // not a JSON object — leave it alone
	}
	msgsRaw, hasMsgs := obj["messages"]
	if !hasMsgs {
		return nil, nil, 0, CompactOutcome{Reason: CompactReasonNoMsgsKey}, false
	}
	elems, spans, decoded := decodeArrayElements(raw, msgsRaw)
	if !decoded || len(elems) < minElems {
		return nil, nil, 0, CompactOutcome{Reason: CompactReasonTooFewMsgs}, false // nothing safe to compact
	}
	pfxEnd = firstBreakpointMessage(elems)
	headReanchored := false
	if anchor == CompactAnchorHead && stableHeadBeforeMessages(raw, obj, spans) {
		pfxEnd = -1 // whole message array compactible; the stable cached head is the top-level system/tools block
		headReanchored = true
	}
	// firstbp / fallback condition, unchanged from pre-#1408: a pfxEnd<0 that did NOT come from a
	// proven head re-anchor needs a `system` breakpoint to be a valid anchor, else there is none.
	if pfxEnd < 0 && !headReanchored && !rawHasCacheControl(obj["system"]) {
		return nil, nil, 0, CompactOutcome{Reason: CompactReasonNoBreakpoint}, false // no anchor — identity
	}
	return elems, spans, pfxEnd, CompactOutcome{}, true
}

// stableHeadBeforeMessages reports whether a top-level system/tools cache_control breakpoint —
// the stable cached head the provider reuses every turn — is serialized BEFORE the messages[]
// array in raw. Only then does dropping middle messages leave the head's byte prefix unchanged,
// so the dominant cache read survives a head-anchored fire (a head serialized after messages[]
// would have its prefix shifted by the drop). The value bytes of obj[key] are a verbatim slice
// of raw, so bytes.Index locates them — the same anchor technique decodeArrayElements uses.
func stableHeadBeforeMessages(raw []byte, obj map[string]json.RawMessage, spans []elementSpan) bool {
	if len(spans) == 0 {
		return false
	}
	msgsStart := spans[0].start
	for _, key := range []string{"system", "tools"} {
		v, ok := obj[key]
		if !ok || !rawHasCacheControl(v) {
			continue
		}
		if off := bytes.Index(raw, v); off >= 0 && off < msgsStart {
			return true
		}
	}
	return false
}

// CompactAnthropicHistoryWithOutcome is the observable form on the default (warm-cache-safe)
// first-breakpoint anchor. It is CompactAnthropicHistoryWithOptions with CompactAnchorFirstBP and
// no horizon — byte-for-byte identical to the pre-#1408 behavior, so every existing caller and
// test is unchanged. The gateway uses it to emit the compaction metric family.
func CompactAnthropicHistoryWithOutcome(raw []byte, budget int) ([]byte, CompactOutcome) {
	return CompactAnthropicHistoryWithOptions(raw, CompactOptions{Budget: budget, Anchor: CompactAnchorFirstBP})
}

// CompactAnthropicHistoryWithOptions is the parameterized core of the cache-prefix-preserving
// history rewrite. With CompactAnchorFirstBP (the default) it protects through the first messages[]
// breakpoint and only sheds the middle after it — the warm-cache-safe behavior every existing
// caller relies on. With CompactAnchorHead it re-anchors on the stable system/tools head (when that
// head precedes messages[]), making the whole message array compactible so the lever can fire on
// real Claude Code traffic (#1407); because such a fire bursts the recent breakpoint's cached
// suffix, it is gated on CacheBurstPaysBack economics and only fires when the burst repays within
// the session horizon (#1408). All byte-level guarantees (verbatim protected prefix, re-decode +
// prefix-equality proof, fail-safe identity on any ambiguity) are identical across both anchors.
func CompactAnthropicHistoryWithOptions(raw []byte, opts CompactOptions) ([]byte, CompactOutcome) {
	budget := opts.Budget
	if budget <= 0 || len(raw) == 0 {
		return raw, CompactOutcome{Reason: CompactReasonUnderBudget}
	}
	// Anchor the protected prefix per opts.Anchor: the FIRST cache_control breakpoint message (the
	// warm-cache-safe default that lets compaction fire on multi-breakpoint traffic marking BOTH the
	// static head and recent turns), or — in head mode when a stable system/tools head precedes
	// messages[] — the empty prefix (pfxEnd=-1, whole array compactible). See anchorCompactablePrefixMode.
	elems, spans, pfxEnd, bail, ok := anchorCompactablePrefixMode(raw, 3, opts.Anchor)
	if !ok {
		return raw, bail
	}
	// When only `system` holds the cache, the protected message prefix is empty (-1):
	// every message is compactible. Otherwise it ends at the FIRST breakpoint message (the
	// cached head); everything after it up to the recent kept window is compactible middle.

	// 2. Is the compactible suffix already under budget? Then there is nothing to do — but record
	//    the token split so the caller can tell a BENIGN idle (a short session) from an
	//    ANCHOR-STARVED one. On real Claude Code traffic the only message breakpoint is a RECENT
	//    one, so pfxEnd lands near the end, the protected prefix swallows ~the whole conversation,
	//    and this suffix is structurally tiny no matter how long the session grows — the lever can
	//    never fire (#1407). AnchorStarved flags exactly that: under_budget WITH a protected prefix
	//    already past the budget. pfxEnd<0 (system-only anchor) makes the whole array compactible,
	//    so prefixTokens is 0 and under_budget there is genuinely benign.
	suffixTokens := 0
	for i := pfxEnd + 1; i < len(elems); i++ {
		suffixTokens += len(elems[i]) / 4
	}
	if suffixTokens <= budget {
		prefixTokens := 0
		for i := 0; i <= pfxEnd && i < len(elems); i++ {
			prefixTokens += len(elems[i]) / 4
		}
		return raw, CompactOutcome{
			Reason:                CompactReasonUnderBudget,
			ProtectedPrefixTokens: prefixTokens,
			SuffixTokens:          suffixTokens,
			AnchorStarved:         prefixTokens > budget,
		}
	}

	// 3. Choose the kept recent window: walk from the END accumulating tokens until the
	//    budget is met, then snap the window start to a clean turn boundary that does not
	//    orphan a tool_result (a user turn carrying tool_result blocks needs the assistant
	//    tool_use turn before it). Everything between pfxEnd+1 and keepStart is dropped.
	keepStart := chooseKeptWindow(elems, pfxEnd+1, budget)
	if keepStart <= pfxEnd+1 || keepStart >= len(elems) {
		return raw, CompactOutcome{Reason: CompactReasonWindowNoDrop} // nothing drops / empty window
	}

	// 3b. Role alternation (F7): the synthetic stub is one message inserted BETWEEN the
	//     protected prefix (ends at pfxEnd) and the kept window (starts at keepStart). Anthropic
	//     rejects two consecutive same-role messages (400), and the stub's content is text — so
	//     it must carry a role that alternates with BOTH neighbors. The stub role is the opposite
	//     of the prefix's last message role; then we snap keepStart so the kept window's first
	//     message alternates with the stub (i.e. matches the prefix-last role). Dropping one more
	//     message flips the kept-first role, so this is always reachable while keepStart stays a
	//     real drop. When pfxEnd<0 (system-only breakpoint) there is no preceding message: the
	//     stub leads the array, so it must alternate only with the kept window — pick the opposite
	//     of the kept-first role.
	prefixLastRole := ""
	if pfxEnd >= 0 {
		prefixLastRole = messageRole(elems[pfxEnd])
	}
	stubRole := "user"
	if prefixLastRole == "user" {
		stubRole = "assistant"
	}
	if prefixLastRole == "" { // system-only: alternate against the kept window head instead
		if messageRole(elems[keepStart]) == "user" {
			stubRole = "assistant"
		} else {
			stubRole = "user"
		}
	}
	// Snap the kept window so its first message alternates with the stub. If it collides,
	// drop one more message (flipping the kept-first role); never cross back over pfxEnd+1.
	if keepStart < len(elems) && messageRole(elems[keepStart]) == stubRole {
		if keepStart+1 < len(elems) {
			keepStart++
		} else {
			return raw, CompactOutcome{Reason: CompactReasonWindowNoDrop} // can't fix alternation — fail safe
		}
	}
	if keepStart <= pfxEnd+1 {
		return raw, CompactOutcome{Reason: CompactReasonWindowNoDrop} // snap swallowed the drop — identity
	}
	// A kept window that opens on a user tool_result still needs its assistant tool_use; the
	// snap above only moves the boundary FORWARD, so re-assert the orphan guard once more.
	if messageHasToolResult(elems[keepStart]) {
		return raw, CompactOutcome{Reason: CompactReasonWindowNoDrop} // would orphan a tool_result — fail safe
	}
	if rangeHasCacheControl(elems, pfxEnd+1, keepStart) {
		// A cache_control-bearing message is provider-warm history. Dropping it may shrink
		// the prompt, but it also intentionally bursts the cached suffix after the first
		// changed byte. Without an explicit horizon/economics gate, the conservative action is
		// identity: keep the provider's cache hit over a smaller prompt.
		return raw, CompactOutcome{Reason: CompactReasonCachedSpan}
	}
	dropped := keepStart - (pfxEnd + 1)

	// shedTokens: the estimated tokens removed from the outbound body — the sum over the dropped
	// MIDDLE [pfxEnd+1, keepStart), minus the stub's own ~cost. Same ~4-chars/token currency as
	// the budget and the provider input_tokens, so it is the CLAIMED-savings half of the
	// billing-truth comparison (vs the provider's cache_read on the same turn).
	shedTokens := 0
	for i := pfxEnd + 1; i < keepStart; i++ {
		shedTokens += len(elems[i]) / 4
	}
	if shedTokens -= compactStubTokenCost(dropped); shedTokens < 0 {
		shedTokens = 0
	}

	// 3c. Head-anchored economics gate (#1407/#1408): when head re-anchoring engaged (pfxEnd<0 under
	//     CompactAnchorHead), the drop deliberately bursts the recent breakpoint's cached suffix — the
	//     stable head itself stays byte-stable because it precedes messages[] (see
	//     anchorCompactablePrefixMode). Fire only when CacheBurstPaysBack approves: the cached middle we
	//     shed forever (a per-turn read saving) must repay the one-time cold re-write of the invalidated
	//     suffix within the remaining session horizon. An unknown horizon (TotalTurns<=0) returns false,
	//     so a caller that cannot supply the horizon never bursts the cache. The firstbp default never
	//     reaches this branch (pfxEnd>=0 there, or Anchor!=Head), so its body is unchanged.
	if opts.Anchor == CompactAnchorHead && pfxEnd < 0 {
		readMult, writeMult := opts.ReadMult, opts.WriteMult
		if readMult <= 0 {
			readMult = defaultCacheReadMult
		}
		if writeMult <= 0 {
			writeMult = defaultCacheWriteMult
		}
		droppedCachedTokens, invalidatedSuffixTokens := headBurstEconomics(elems, pfxEnd+1, keepStart)
		if !CacheBurstPaysBack(opts.TotalTurns, opts.CurrentTurn, droppedCachedTokens, invalidatedSuffixTokens, readMult, writeMult) {
			return raw, CompactOutcome{Reason: CompactReasonBurstUnprofitable, SuffixTokens: suffixTokens}
		}
	}

	// 4. Splice on ORIGINAL bytes. The prefix span [0, spans[pfxEnd].end) (or just the
	//    array-open when pfxEnd<0) is copied verbatim; then the stub; then the kept
	//    elements verbatim; then the verbatim tail from the array close onward.
	out, ok := spliceCompacted(raw, spans, pfxEnd, keepStart, len(elems), dropped, stubRole)
	// 5. Prove it: the spliced body must still decode AND keep the protected prefix bytes
	//    intact, or we ship identity rather than a broken/cache-busting body.
	if outcome, good := compactSpliceVerdict(raw, out, ok, spans, pfxEnd); !good {
		return raw, outcome
	}
	return out, CompactOutcome{Reason: CompactReasonNone, Dropped: dropped, ShedTokens: shedTokens}
}

// headBurstEconomics prices a head-anchored drop for the CacheBurstPaysBack gate.
// droppedCachedTokens is the cached middle [dropStart, keepStart) we shed for good — each future
// turn no longer re-reads it (the per-turn saving). invalidatedSuffixTokens is the cached span the
// drop bursts ONCE: from the first kept message through the last SURVIVING cache_control breakpoint,
// whose byte prefix the drop shifts so the provider must cold-write it again. Bytes beyond the last
// breakpoint were never cached, so they are not counted as invalidated. Same ~4-chars/token currency
// as the budget and the provider input_tokens. dropStart is pfxEnd+1 (0 in head mode).
func headBurstEconomics(elems []json.RawMessage, dropStart, keepStart int) (droppedCachedTokens, invalidatedSuffixTokens int) {
	if dropStart < 0 {
		dropStart = 0
	}
	for i := dropStart; i < keepStart && i < len(elems); i++ {
		droppedCachedTokens += len(elems[i]) / 4
	}
	lastBp := lastBreakpointMessage(elems)
	if lastBp >= keepStart {
		for i := keepStart; i <= lastBp && i < len(elems); i++ {
			invalidatedSuffixTokens += len(elems[i]) / 4
		}
	}
	return droppedCachedTokens, invalidatedSuffixTokens
}

// elementSpan is the [start,end) byte range of one messages[] element within the original
// body, where start points at the element's first byte and end just past its last.
type elementSpan struct{ start, end int }

// spliceVerdict is the shared "prove it" outcome for a byte-spliced request body, mapped
// by each caller onto its own (Compact|Elide)Reason vocabulary.
type spliceVerdict int

const (
	spliceVerdictOK             spliceVerdict = iota // re-decodes AND protected prefix is byte-identical
	spliceVerdictRedecodeFail                        // spliced body no longer parses as a Messages request
	spliceVerdictPrefixMismatch                      // protected cache prefix bytes changed (would burst the cache)
)

// verifySplicedBody is the shared post-splice proof both the compaction and elision rewrites
// run: the result must still decode as a valid Messages request, and the protected prefix
// bytes (through spans[pfxEnd], or just the array open when pfxEnd<0) must be byte-identical
// to the input. Either failing is a splice bug, not a reason to ship a broken / cache-busting
// body — the caller returns identity with its own labeled reason.
func verifySplicedBody(raw, out []byte, spans []elementSpan, pfxEnd int) spliceVerdict {
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		return spliceVerdictRedecodeFail
	}
	prefixEnd := arrayContentStart(spans) // byte offset just inside `[` when only `system` is cached
	if pfxEnd >= 0 {
		prefixEnd = spans[pfxEnd].end
	}
	if prefixEnd > len(out) || !bytes.Equal(raw[:prefixEnd], out[:prefixEnd]) {
		return spliceVerdictPrefixMismatch
	}
	return spliceVerdictOK
}

// compactSpliceVerdict maps a post-splice (out, ok) result onto the CompactOutcome reason
// vocabulary shared by the compaction and ctxplan-view rewrites. It returns (outcome,
// false) — the identity reason the caller ships — when the splice itself failed (!ok), the
// body no longer re-decodes, or the protected prefix bytes changed; and (zero, true) when
// the spliced body is proven safe to ship. Behaviorally identical to inlining the
// `if !ok { …SpliceFailed } switch verifySplicedBody(…) { …RedecodeFail / …PrefixMismatch }`
// guard at each call site.
func compactSpliceVerdict(raw, out []byte, ok bool, spans []elementSpan, pfxEnd int) (CompactOutcome, bool) {
	if !ok {
		return CompactOutcome{Reason: CompactReasonSpliceFailed}, false
	}
	switch verifySplicedBody(raw, out, spans, pfxEnd) {
	case spliceVerdictRedecodeFail:
		return CompactOutcome{Reason: CompactReasonRedecodeFail}, false
	case spliceVerdictPrefixMismatch:
		return CompactOutcome{Reason: CompactReasonPrefixMismatch}, false
	}
	return CompactOutcome{}, true
}

// decodeArrayElements returns each messages[] element's raw bytes (json.RawMessage) and its
// absolute byte span within raw, using a streaming decoder + InputOffset so the spans are
// exact anchors for byte-splicing (never a fragile string search). msgsRaw must be the
// `messages` value as it appears in raw (json.Unmarshal of an object preserves the value
// bytes verbatim, so a sub-search for it is reliable). ok is false on any decode error.
func decodeArrayElements(raw []byte, msgsRaw json.RawMessage) (elems []json.RawMessage, spans []elementSpan, ok bool) {
	// Find where msgsRaw sits in raw so element offsets are absolute. json.RawMessage is a
	// verbatim slice of the input, so a single LastIndex of its bytes locates it; the
	// `"messages"` key value is unique enough in practice, and we re-verify with a prefix
	// byte-equality at the end, so a wrong guess can only produce identity, never breakage.
	base := bytes.Index(raw, msgsRaw)
	if base < 0 {
		return nil, nil, false
	}
	dec := json.NewDecoder(bytes.NewReader(msgsRaw))
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, false
	}
	if d, isDelim := tok.(json.Delim); !isDelim || d != '[' {
		return nil, nil, false
	}
	for dec.More() {
		// InputOffset() before Decode points just past the previous token (the `[` or a
		// prior element's `}`), so it sits BEFORE this element's leading `,`/whitespace.
		// Advance past both to the element's first significant byte for a clean start.
		startRel := int(dec.InputOffset())
		for startRel < len(msgsRaw) && (isJSONSpace(msgsRaw[startRel]) || msgsRaw[startRel] == ',') {
			startRel++
		}
		var el json.RawMessage
		if err := dec.Decode(&el); err != nil {
			return nil, nil, false
		}
		endRel := int(dec.InputOffset())
		elems = append(elems, el)
		spans = append(spans, elementSpan{start: base + startRel, end: base + endRel})
	}
	return elems, spans, true
}

// arrayContentStart returns the absolute byte offset just inside the messages `[` — the
// fallback protected-prefix end when only `system` holds the cache (no message breakpoint).
// It is the start of the first element minus any element-leading bytes; we use the first
// span's start, which already points at the first element's first byte.
func arrayContentStart(spans []elementSpan) int {
	if len(spans) == 0 {
		return 0
	}
	return spans[0].start
}

// lastBreakpointMessage returns the index of the last messages[] element whose content
// carries a cache_control breakpoint, or -1 if none does.
func lastBreakpointMessage(elems []json.RawMessage) int {
	last := -1
	for i, el := range elems {
		if messageHasCacheControl(el) {
			last = i
		}
	}
	return last
}

// firstBreakpointMessage returns the index of the FIRST messages[] element whose content
// carries a cache_control breakpoint, or -1 if none does. This is the anchor for the protected
// prefix: the earliest message breakpoint marks the stable cached HEAD the provider reuses
// every turn (the growing-conversation layout marks the static head AND recent turns; only the
// head's prefix is byte-stable across turns). Anchoring here — not on the last breakpoint —
// is what lets compaction drop the un-cacheable MIDDLE on real multi-breakpoint traffic.
func firstBreakpointMessage(elems []json.RawMessage) int {
	for i, el := range elems {
		if messageHasCacheControl(el) {
			return i
		}
	}
	return -1
}

func rangeHasCacheControl(elems []json.RawMessage, start, end int) bool {
	if start < 0 {
		start = 0
	}
	if end > len(elems) {
		end = len(elems)
	}
	for i := start; i < end; i++ {
		if messageHasCacheControl(elems[i]) {
			return true
		}
	}
	return false
}

func messageHasCacheControl(el json.RawMessage) bool {
	var m struct {
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(el, &m) != nil {
		return false
	}
	return rawHasCacheControl(m.Content)
}

// CacheBurstBreakEvenTurns prices an explicit cache-burst rewrite. If a compaction would
// delete already cache_control-marked tokens, the immediate penalty is the cached suffix that
// must be written cold again; the future saving is only the provider's discounted read cost for
// the deleted cached tokens. It returns the minimum future turns needed to repay that burst.
// A return of 0 means there is no one-time suffix penalty; MaxInt means the rewrite never
// breaks even under the supplied multipliers.
func CacheBurstBreakEvenTurns(droppedCachedTokens, invalidatedSuffixTokens int, readMult, writeMult float64) int {
	if invalidatedSuffixTokens <= 0 {
		return 0
	}
	perTurnSaving := float64(droppedCachedTokens) * readMult
	oneTimePenalty := float64(invalidatedSuffixTokens) * (writeMult - readMult)
	if oneTimePenalty <= 0 {
		return 0
	}
	if perTurnSaving <= 0 {
		return int(^uint(0) >> 1)
	}
	return int(math.Ceil(oneTimePenalty / perTurnSaving))
}

// CacheBurstPaysBack reports whether an explicit cache-burst rewrite has enough future
// turns left in this session to repay itself. currentTurn is 1-based and "now": in a
// 50-turn session at currentTurn=20, there are 30 future turns left (21..50). Unknown or
// exhausted horizons return false unless the burst has no one-time penalty.
func CacheBurstPaysBack(totalTurns, currentTurn, droppedCachedTokens, invalidatedSuffixTokens int, readMult, writeMult float64) bool {
	breakEven := CacheBurstBreakEvenTurns(droppedCachedTokens, invalidatedSuffixTokens, readMult, writeMult)
	if breakEven == 0 {
		return true
	}
	if totalTurns <= 0 || currentTurn <= 0 {
		return false
	}
	remainingTurns := totalTurns - currentTurn
	return remainingTurns >= breakEven
}

// rawHasCacheControl reports whether a `system` or message `content` value (a bare string,
// a single block object, or an array of blocks) carries a cache_control breakpoint on any
// block. A bare string has no blocks → no breakpoint.
func rawHasCacheControl(content json.RawMessage) bool {
	if len(content) == 0 {
		return false
	}
	// Array of blocks (the common Claude Code shape).
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(content, &blocks) == nil {
		for _, b := range blocks {
			if _, ok := b["cache_control"]; ok {
				return true
			}
		}
		return false
	}
	// A single block object.
	var block map[string]json.RawMessage
	if json.Unmarshal(content, &block) == nil {
		_, ok := block["cache_control"]
		return ok
	}
	return false
}

// chooseKeptWindow walks the messages from the END accumulating ~token cost until it
// reaches budget, then snaps the window start UP to a clean boundary that does not orphan a
// tool_result: a user turn whose content carries tool_result blocks must keep the assistant
// turn before it (the tool_use). It returns the index of the first KEPT message, clamped so
// the window never starts before the first compactible message (compactStart).
func chooseKeptWindow(elems []json.RawMessage, compactStart, budget int) int {
	keep := len(elems)
	acc := 0
	for i := len(elems) - 1; i >= compactStart; i-- {
		acc += len(elems[i]) / 4
		if acc > budget {
			break
		}
		keep = i
	}
	// Don't orphan a tool_result: if the first kept message is a user turn bearing
	// tool_result blocks, pull the preceding (assistant tool_use) message into the window.
	// Guard keep < len(elems): a budget so small that even the last message exceeds it leaves
	// keep == len(elems) (an empty window), and elems[keep] would be out of range — the caller
	// treats an empty window as identity.
	for keep > compactStart && keep < len(elems) && messageHasToolResult(elems[keep]) {
		keep--
	}
	if keep < compactStart {
		keep = compactStart
	}
	return keep
}

// messageHasToolResult reports whether a messages[] element is a user turn carrying at
// least one tool_result block — the case whose matching assistant tool_use turn must not be
// dropped from under it.
func messageHasToolResult(el json.RawMessage) bool {
	var m struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(el, &m) != nil || m.Role != "user" {
		return false
	}
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(m.Content, &blocks) != nil {
		return false
	}
	for _, b := range blocks {
		if t, ok := b["type"]; ok {
			var s string
			if json.Unmarshal(t, &s) == nil && s == "tool_result" {
				return true
			}
		}
	}
	return false
}

// messageRole returns a messages[] element's role ("user"/"assistant"), or "" if it cannot be
// parsed. Used to keep the synthetic stub alternating with its neighbors (Anthropic rejects two
// consecutive same-role messages).
func messageRole(el json.RawMessage) string {
	var m struct {
		Role string `json:"role"`
	}
	if json.Unmarshal(el, &m) != nil {
		return ""
	}
	return m.Role
}

// compactStubTokenCost estimates the synthetic stub message's own ~token cost (the same
// ~4-chars/token basis the budget uses), so the reported shed is NET of the message we add
// back. The stub text is fixed apart from the drop count, so this is a close estimate.
func compactStubTokenCost(dropped int) int {
	stub := fmt.Sprintf("%s%d earlier turn(s) to stay within the context budget; their detail is omitted from this request.", compactStubPrefix, dropped)
	return (len(stub) + len(`{"role":"assistant","content":""}`)) / 4
}

// spliceCompacted assembles the rewritten body from original byte spans: the verbatim
// protected prefix (through the breakpoint message, or just the array open when pfxEnd<0),
// then a synthetic stub message naming the drop count, then the verbatim kept elements,
// then the verbatim tail from the array close onward. It never re-serializes a protected or
// kept element, so their bytes (and thus the cached prefix) are preserved exactly. ok is
// false if the stub cannot be marshalled (it never realistically fails).
func spliceCompacted(raw []byte, spans []elementSpan, pfxEnd, keepStart, n, dropped int, stubRole string) ([]byte, bool) {
	if stubRole != "user" && stubRole != "assistant" {
		stubRole = "user"
	}
	stub := map[string]any{
		"role":    stubRole,
		"content": fmt.Sprintf("%s%d earlier turn(s) to stay within the context budget; their detail is omitted from this request.", compactStubPrefix, dropped),
	}
	stubBytes, err := json.Marshal(stub)
	if err != nil {
		return nil, false
	}

	// prefixEnd: byte just past the last protected element, or the first element's start
	// (the array's content head) when no message is protected.
	prefixEnd := arrayContentStart(spans)
	if pfxEnd >= 0 {
		prefixEnd = spans[pfxEnd].end
	}
	// tailStart: byte at the first kept element's start. Everything from arrayClose (the
	// `]`) onward rides along inside the kept-elements + tail copy, so we copy from the
	// first kept element through end of body.
	keptFrom := spans[keepStart].start
	bodyTail := raw[spans[n-1].end:] // from just past the last element to EOF (the `]` + trailing keys)

	var b bytes.Buffer
	b.Grow(len(raw))
	b.Write(raw[:prefixEnd]) // verbatim protected prefix (includes `[` and any kept-element-leading bytes up to prefixEnd)
	// Separator before the stub: a comma only if at least one protected element preceded it.
	if pfxEnd >= 0 {
		b.WriteByte(',')
	}
	b.Write(stubBytes)
	b.WriteByte(',')
	b.Write(raw[keptFrom:spans[n-1].end]) // verbatim kept elements (keepStart..n-1)
	b.Write(bodyTail)                     // verbatim `]` + any trailing top-level keys
	return b.Bytes(), true
}

// isJSONSpace reports whether b is JSON insignificant whitespace.
func isJSONSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

// --- ctxplan-view twin (#927, the deferred #555 req.Raw transform) ---------------

// ctxViewStubPrefix marks a message the ctxplan planned view ELIDED from the passthrough
// body (replaced in place by a same-role stub), so the model sees the turn was planned
// out rather than silently lost. Distinct from compactStubPrefix so an operator can tell
// a ctxview elision from a compaction drop in the forwarded bytes.
const ctxViewStubPrefix = "[fak] ctxview-elided "

// CompactAnthropicHistoryToView is the ctxplan-view twin of CompactAnthropicHistory
// (#927 — the deferred #555 req.Raw step the buffered maybePlanMessages path could not
// reach). Where compaction drops a contiguous suffix of OLD whole turns, this
// materializes the planner's O(1) RESIDENT SET onto the passthrough body: each
// messages[] element whose text content the planner did NOT select as resident — and
// which sits beyond the protected cache_control prefix — is REPLACED IN PLACE by a
// same-role stub, while resident messages keep their ORIGINAL bytes (cache_control and
// all) and the protected prefix is copied VERBATIM.
//
// Replacing (not dropping) is the key simplification over compaction's contiguous-suffix
// constraint: a same-role stub preserves the message COUNT and the user/assistant role
// alternation EXACTLY as the original, so Anthropic accepts the body no matter which
// non-contiguous middle turns the forecast shed. It is fail-safe identity on any
// ambiguity: non-JSON, no messages[], no cache_control anchor, a would-be-elided message
// that carries its own cache_control (would burst the cached suffix), content fak cannot
// confidently match (tool_use/tool_result blocks — always kept), or a splice that fails
// to re-decode or alters the protected prefix bytes.
//
// planned is the planner's rendered resident view (CtxViewPlanner.RenderTurn). A message
// element is resident when its extracted text content equals a planned message's content
// — the planner pages each resident span's bytes verbatim, so content equality is the
// faithful signal. This is a REQUEST-side transform only: it touches the bytes sent
// upstream; it never touches the decoded req.Messages the kernel adjudicates.
func CompactAnthropicHistoryToView(raw []byte, planned []Message) ([]byte, CompactOutcome) {
	if len(raw) == 0 || len(planned) == 0 {
		return raw, CompactOutcome{Reason: CompactReasonUnderBudget}
	}
	// Anchor the protected prefix on the FIRST cache_control breakpoint message (or a
	// system-level breakpoint), exactly as compaction does — the stable cached head.
	elems, spans, pfxEnd, bail, ok := anchorCompactablePrefix(raw, 1)
	if !ok {
		return raw, bail
	}

	// Build the resident content set from the planned view. The planner renders each
	// resident span's bytes verbatim, so a message element whose text content appears in
	// this set is resident and keeps its original bytes.
	resident := make(map[string]bool, len(planned))
	for _, m := range planned {
		if m.Content != "" {
			resident[m.Content] = true
		}
	}
	if len(resident) == 0 {
		return raw, CompactOutcome{Reason: CompactReasonUnderBudget}
	}

	// Walk the messages AFTER the protected prefix. A non-resident element becomes a
	// same-role stub; a resident element (or one whose content cannot be cleanly
	// extracted — e.g. tool_use/tool_result blocks) keeps its original bytes.
	var stubIdx []int
	for i := pfxEnd + 1; i < len(elems); i++ {
		content, ok := elementTextContent(elems[i])
		if !ok {
			continue // cannot confidently match → keep verbatim (fail-safe)
		}
		if resident[content] {
			continue // resident → keep verbatim
		}
		if messageHasCacheControl(elems[i]) {
			// Eliding a cache_control-bearing message would burst the cached suffix — the
			// same conservative posture as compaction's CompactReasonCachedSpan.
			return raw, CompactOutcome{Reason: CompactReasonCachedSpan}
		}
		stubIdx = append(stubIdx, i)
	}
	if len(stubIdx) == 0 {
		return raw, CompactOutcome{Reason: CompactReasonUnderBudget} // nothing to elide
	}

	out, ok := spliceToView(raw, spans, pfxEnd, stubIdx, elems)
	// Prove it: re-decode + protected-prefix byte-equality (shared with compaction/elision).
	if outcome, good := compactSpliceVerdict(raw, out, ok, spans, pfxEnd); !good {
		return raw, outcome
	}
	shedTokens := 0
	for _, i := range stubIdx {
		shedTokens += len(elems[i]) / 4
	}
	if stubCost := ctxViewStubTokenCost(len(stubIdx)); shedTokens > stubCost {
		shedTokens -= stubCost
	} else {
		shedTokens = 0
	}
	return out, CompactOutcome{Reason: CompactReasonNone, Dropped: len(stubIdx), ShedTokens: shedTokens}
}

// spliceToView assembles the rewritten body from original byte spans: the verbatim
// protected prefix (through the breakpoint message, or just the array open when pfxEnd<0),
// then each suffix element either copied verbatim (resident / unmatchable) or emitted as
// a same-role stub (elided), then the verbatim tail from the array close onward. Because
// each stub carries the SAME role as the message it replaces, the message count and the
// user/assistant alternation are identical to the original — Anthropic accepts the body.
// ok is false only if a stub cannot be marshalled (it never realistically fails).
func spliceToView(raw []byte, spans []elementSpan, pfxEnd int, stubIdx []int, elems []json.RawMessage) ([]byte, bool) {
	stubSet := make(map[int]bool, len(stubIdx))
	for _, i := range stubIdx {
		stubSet[i] = true
	}
	prefixEnd := arrayContentStart(spans)
	if pfxEnd >= 0 {
		prefixEnd = spans[pfxEnd].end
	}
	n := len(elems)
	var b bytes.Buffer
	b.Grow(len(raw))
	b.Write(raw[:prefixEnd]) // verbatim protected prefix
	first := pfxEnd < 0      // no preceding element when the prefix is empty (system-only cache)
	for i := pfxEnd + 1; i < n; i++ {
		if !first {
			b.WriteByte(',')
		}
		first = false
		if stubSet[i] {
			role := messageRole(elems[i])
			if role != "user" && role != "assistant" {
				role = "user"
			}
			stubBytes, err := json.Marshal(map[string]any{
				"role":    role,
				"content": ctxViewStubPrefix + "turn detail omitted from the planned resident view.",
			})
			if err != nil {
				return nil, false
			}
			b.Write(stubBytes)
		} else {
			b.Write(raw[spans[i].start:spans[i].end]) // verbatim element
		}
	}
	b.Write(raw[spans[n-1].end:]) // verbatim `]` + any trailing top-level keys
	return b.Bytes(), true
}

// elementTextContent extracts the matchable text content of one messages[] element: a
// bare-string content yields the string; a content array of ONLY text blocks yields the
// concatenated text. Any other shape (tool_use / tool_result blocks, or unparseable
// content) returns ok=false so the caller keeps the element verbatim — never stubbing a
// message it cannot confidently identify as elided. This is also what keeps tool_use ↔
// tool_result pairings intact: a message carrying either block type is always kept.
func elementTextContent(el json.RawMessage) (string, bool) {
	var m struct {
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(el, &m) != nil {
		return "", false
	}
	// Bare string content (the common simple-prompt shape).
	var s string
	if json.Unmarshal(m.Content, &s) == nil {
		return s, true
	}
	// Array of blocks — clean only if every block is text.
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(m.Content, &blocks) != nil {
		return "", false
	}
	text := ""
	for _, blk := range blocks {
		if traw, ok := blk["type"]; ok {
			var bt string
			if json.Unmarshal(traw, &bt) == nil && bt != "text" {
				return "", false // tool_use / tool_result / unknown — keep verbatim
			}
		}
		if traw, ok := blk["text"]; ok {
			var ts string
			if json.Unmarshal(traw, &ts) == nil {
				text += ts
			}
		}
	}
	return text, true
}

// ctxViewStubTokenCost estimates the total ~token cost of the same-role stubs that replace
// the elided messages (the same ~4-chars/token basis the budget uses), so the reported
// shed is NET of the bytes added back.
func ctxViewStubTokenCost(stubbed int) int {
	stub := ctxViewStubPrefix + "turn detail omitted from the planned resident view."
	perStub := (len(stub) + len(`{"role":"user","content":""}`)) / 4
	return perStub * stubbed
}
