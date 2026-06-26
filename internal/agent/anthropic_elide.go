package agent

// anthropic_elide.go — oversized tool_result elision, the bounded-loss sibling of the
// cache-prefix-preserving history compaction in anthropic_compact.go.
//
// Compaction DROPS whole old middle turns. Elision is gentler and complementary: it keeps
// every turn's structure but SHRINKS an individual oversized tool_result body — a scrolled-past
// file dump, a long command output, a giant search result — to a bounded head+tail form. A
// coding agent (the flagship `fak guard -- claude` use case) accumulates these constantly, and
// most of them are never read again after the turn that produced them.
//
// The load-bearing invariant is the SAME cache guarantee compaction makes, and it is enforced
// the same way — by SPLICING on the original bytes so the protected prefix is copied verbatim
// (a memcpy), never re-marshalled:
//
//	Only a tool_result that lives STRICTLY AFTER the protected prefix (the FIRST cache_control
//	breakpoint message — the stable cached head the provider reuses every turn), is OUTSIDE the
//	recent working-set window (the last elideRecentKeepMsgs messages), and sits in a message
//	that carries NO cache_control of its own may be shrunk. That band is the un-cacheable middle
//	the provider re-bills every turn anyway. Because no cache_control-bearing message is ever
//	touched, elision is strictly cache-NON-bursting: it never changes a single byte that any
//	breakpoint caches. On ANY ambiguity the function returns its input UNCHANGED (identity), and
//	a non-identity result is proven to (a) still decode as a valid Messages request and (b) keep
//	the protected prefix bytes byte-identical to the input.
//
// Like compaction this is a REQUEST-side transform only: it touches the bytes sent upstream,
// never the decoded req.Messages the kernel adjudicates, so the trust boundary is unchanged.
// It IS lossy (the middle of an old large result is replaced by a marker) — that is why it is
// off by default and gated on this working-set guard; the loss is bounded to old, un-cached,
// non-resident results and never drops a result entirely (head+tail survive).

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
)

// elideRecentKeepMsgs is the recent working-set window elision never shrinks: the last N
// messages (≈ the current turn plus the one before it) are left byte-for-byte intact even if
// they carry an oversized tool_result, because the model is most likely still reasoning over
// them. Combined with the cache_control-skip, this keeps the model's live working set whole.
const elideRecentKeepMsgs = 4

// elideMarkerf renders the in-band notice that stands in for the omitted middle of a shrunk
// tool_result, so the model (and a human reading the wire) sees that detail was elided rather
// than silently truncated.
func elideMarkerf(omittedRunes int) string {
	return fmt.Sprintf("\n\n…[fak: %d characters of older tool_result output elided to stay within the context budget; head and tail are preserved]…\n\n", omittedRunes)
}

// Elision bail-reason vocabulary — the closed set of identity-return causes, mirrored on
// ElideOutcome so a caller can label a metric and an operator can see WHY elision did nothing
// (silence must not read as success). ElideReasonNone means the body was rewritten.
const (
	ElideReasonNone           = ""                // FIRED: a rewrite happened (Elided/ShedBytes meaningful)
	ElideReasonOff            = "off"             // threshold<=0 or empty body — disabled
	ElideReasonNonJSON        = "non_json"        // body is not a JSON object
	ElideReasonNoMsgsKey      = "no_messages_key" // no "messages" key
	ElideReasonTooFewMsgs     = "too_few_msgs"    // messages[] could not be decoded / nothing to scan
	ElideReasonNoBreakpoint   = "no_breakpoint"   // no cache_control anchor — cannot know the cache boundary
	ElideReasonUnderThreshold = "under_threshold" // no oversized eligible tool_result found
	ElideReasonSpliceFailed   = "splice_failed"   // the edits overlapped or fell out of range
	ElideReasonRedecodeFail   = "redecode_failed" // the spliced body failed to re-decode
	ElideReasonPrefixMismatch = "prefix_mismatch" // the splice changed the protected prefix bytes
)

// ElideOutcome is the observable verdict of one elision attempt. Reason==ElideReasonNone means
// FIRED — Elided (number of tool_result bodies shrunk) and ShedBytes (raw bytes removed from the
// outbound body) are then meaningful. Any other Reason means the body was returned unchanged
// (identity), and Elided/ShedBytes are 0.
type ElideOutcome struct {
	Reason    string
	Elided    int
	ShedBytes int
}

// ElideAnthropicResults shrinks oversized tool_result bodies in an outbound Anthropic
// /v1/messages body to a bounded head+tail form, byte-splicing on the original bytes so the
// cached prefix is preserved verbatim. It returns raw UNCHANGED whenever it cannot prove the
// rewrite is both cache-safe and well-formed. This is the byte-only wrapper;
// ElideAnthropicResultsWithOutcome additionally reports WHY it bailed / how much it shed.
func ElideAnthropicResults(raw []byte, threshold int) []byte {
	out, _ := ElideAnthropicResultsWithOutcome(raw, threshold)
	return out
}

// ElideAnthropicResultsWithOutcome is ElideAnthropicResults plus the observable outcome. threshold
// is the byte size above which a single tool_result text payload is shrunk; the documented
// candidate is gateway.DocumentedElideResultBytes. The byte-level guarantees are identical to the
// wrapper.
func ElideAnthropicResultsWithOutcome(raw []byte, threshold int) ([]byte, ElideOutcome) {
	if threshold <= 0 || len(raw) == 0 {
		return raw, ElideOutcome{Reason: ElideReasonOff}
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, ElideOutcome{Reason: ElideReasonNonJSON}
	}
	msgsRaw, ok := obj["messages"]
	if !ok {
		return raw, ElideOutcome{Reason: ElideReasonNoMsgsKey}
	}
	elems, spans, ok := decodeArrayElements(raw, msgsRaw)
	if !ok || len(elems) < 2 {
		return raw, ElideOutcome{Reason: ElideReasonTooFewMsgs}
	}

	// Anchor the protected prefix on the FIRST cache_control breakpoint message — the stable
	// cached head — exactly as compaction does. Require a cache anchor (a message breakpoint or
	// one in `system`); without it we cannot know the cache boundary and must not touch the body.
	pfxEnd := firstBreakpointMessage(elems)
	sysHasCC := rawHasCacheControl(obj["system"])
	if pfxEnd < 0 && !sysHasCC {
		return raw, ElideOutcome{Reason: ElideReasonNoBreakpoint}
	}

	// The eligible band: strictly after the protected prefix, before the recent working-set
	// window, and never a cache_control-bearing message. Every byte we touch is in the un-cached,
	// re-billed middle — so the splice is strictly cache-non-bursting.
	lastEligible := len(elems) - elideRecentKeepMsgs // exclusive
	var edits []spliceEdit
	shed := 0
	for i := pfxEnd + 1; i < lastEligible; i++ {
		if i < 0 || messageHasCacheControl(elems[i]) {
			continue
		}
		es, sh := collectResultElisionEdits(spans[i].start, elems[i], threshold)
		edits = append(edits, es...)
		shed += sh
	}
	if len(edits) == 0 {
		return raw, ElideOutcome{Reason: ElideReasonUnderThreshold}
	}

	out, ok := applySpliceEdits(raw, edits)
	if !ok {
		return raw, ElideOutcome{Reason: ElideReasonSpliceFailed}
	}
	// Prove it: the result must still decode as a valid Messages request, and the protected
	// prefix bytes must be byte-identical to the input. Either failing is a splice bug, not a
	// reason to ship a broken / cache-busting body — fall back to identity.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		return raw, ElideOutcome{Reason: ElideReasonRedecodeFail}
	}
	prefixEnd := arrayContentStart(spans) // byte just inside `[` when only `system` is cached
	if pfxEnd >= 0 {
		prefixEnd = spans[pfxEnd].end
	}
	if prefixEnd > len(out) || !bytes.Equal(raw[:prefixEnd], out[:prefixEnd]) {
		return raw, ElideOutcome{Reason: ElideReasonPrefixMismatch}
	}
	return out, ElideOutcome{Reason: ElideReasonNone, Elided: len(edits), ShedBytes: shed}
}

// spliceEdit is one byte-range replacement within the original body: [start,end) → repl.
type spliceEdit struct {
	start, end int
	repl       []byte
}

// collectResultElisionEdits scans one messages[] element (a user turn that may carry
// tool_result blocks) and returns a splice edit for each oversized tool_result text payload,
// shrinking it to head+tail. msgBase is the element's absolute start byte in the original body;
// every returned edit span is absolute. A non-user message, a non-array content, or a content
// shape it cannot parse yields no edits (the conservative identity for that message).
func collectResultElisionEdits(msgBase int, el json.RawMessage, threshold int) (edits []spliceEdit, shed int) {
	var m struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(el, &m) != nil || m.Role != "user" || len(m.Content) == 0 || m.Content[0] != '[' {
		return nil, 0
	}
	blocks, blockSpans, ok := decodeArrayElements(el, m.Content) // spans relative to el
	if !ok {
		return nil, 0
	}
	for j, blk := range blocks {
		var b struct {
			Type    string          `json:"type"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(blk, &b) != nil || b.Type != "tool_result" || len(b.Content) == 0 {
			continue
		}
		blkBase := msgBase + blockSpans[j].start
		switch b.Content[0] {
		case '"': // content is a bare JSON string
			if e, sh, ok := elideStringValue(blkBase, blk, b.Content, threshold); ok {
				edits = append(edits, e)
				shed += sh
			}
		case '[': // content is an array of blocks (the common Claude Code shape) — shrink each oversized text block
			inner, innerSpans, ok := decodeArrayElements(blk, b.Content) // spans relative to blk
			if !ok {
				continue
			}
			for k, ib := range inner {
				var tb struct {
					Type string          `json:"type"`
					Text json.RawMessage `json:"text"`
				}
				if json.Unmarshal(ib, &tb) != nil || tb.Type != "text" || len(tb.Text) == 0 || tb.Text[0] != '"' {
					continue
				}
				if e, sh, ok := elideStringValue(blkBase+innerSpans[k].start, ib, tb.Text, threshold); ok {
					edits = append(edits, e)
					shed += sh
				}
			}
		}
	}
	return edits, shed
}

// elideStringValue builds the splice edit that shrinks one JSON string value (val, a verbatim
// slice of container) to head+tail, if val exceeds threshold bytes and the shrunk form is
// strictly shorter. containerBase is container's absolute start byte in the original body. ok is
// false (no edit) when the value is not oversized, cannot be decoded, or would not actually save.
func elideStringValue(containerBase int, container, val json.RawMessage, threshold int) (spliceEdit, int, bool) {
	if len(val) <= threshold {
		return spliceEdit{}, 0, false
	}
	rel := bytes.Index(container, val)
	if rel < 0 {
		return spliceEdit{}, 0, false
	}
	var s string
	if json.Unmarshal(val, &s) != nil {
		return spliceEdit{}, 0, false
	}
	shrunk := elideHeadTail(s, threshold)
	newVal, err := json.Marshal(shrunk)
	if err != nil || len(newVal) >= len(val) {
		return spliceEdit{}, 0, false
	}
	start := containerBase + rel
	return spliceEdit{start: start, end: start + len(val), repl: newVal}, len(val) - len(newVal), true
}

// elideHeadTail returns the head+tail-shrunk form of s: the first ~3/4·threshold and last
// ~1/4·threshold characters, joined by an in-band elision marker. Slicing is on rune boundaries
// so the result is always valid UTF-8. If s is not meaningfully longer than head+tail it is
// returned unchanged (the caller's "strictly shorter" guard then drops the edit).
func elideHeadTail(s string, threshold int) string {
	head := threshold * 3 / 4
	tail := threshold - head
	r := []rune(s)
	if len(r) <= head+tail {
		return s
	}
	omitted := len(r) - head - tail
	return string(r[:head]) + elideMarkerf(omitted) + string(r[len(r)-tail:])
}

// applySpliceEdits applies disjoint byte-range replacements to raw, building the result forward.
// Edits are sorted by start; an overlap or an out-of-range span returns ok=false (identity at the
// caller). Bytes outside every edit span — including the whole protected prefix — are copied
// verbatim, so the cache guarantee is a bytes.Equal, not a hope.
func applySpliceEdits(raw []byte, edits []spliceEdit) ([]byte, bool) {
	sort.Slice(edits, func(i, j int) bool { return edits[i].start < edits[j].start })
	out := make([]byte, 0, len(raw))
	prev := 0
	for _, e := range edits {
		if e.start < prev || e.end > len(raw) || e.start > e.end {
			return nil, false // overlap or out of range — bail
		}
		out = append(out, raw[prev:e.start]...)
		out = append(out, e.repl...)
		prev = e.end
	}
	out = append(out, raw[prev:]...)
	return out, true
}
