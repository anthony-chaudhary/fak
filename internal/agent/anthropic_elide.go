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
// The cache guarantee is the SAME one compaction makes, and it is enforced the same way — by
// SPLICING on the original bytes so the protected prefix is copied verbatim (a memcpy), never
// re-marshalled:
//
//	Only a tool_result that lives STRICTLY AFTER the protected prefix (the FIRST cache_control
//	breakpoint message — the stable cached HEAD the provider reuses every turn), is OUTSIDE the
//	recent working-set window (the last elideRecentKeepMsgs messages), and whose message carries
//	NO cache_control reachable by the shrinker may be shrunk. The guarantee, proven before ship,
//	is that the FIRST-breakpoint (head) cached prefix stays BYTE-IDENTICAL, so the dominant cache
//	hit survives. Editing the middle DOES shift the bytes a LATER breakpoint caches — exactly as
//	compaction's middle-drop does — so those later (recent-turn) breakpoints cascade-burst and the
//	provider's read walks back to the byte-identical head; the shed middle is re-billed once. This
//	is a cascade trade, NOT "never touches a cached byte." On ANY ambiguity the function returns
//	its input UNCHANGED (identity), and a non-identity result is proven to (a) still decode as a
//	valid Messages request and (b) keep the head prefix bytes byte-identical to the input.
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
	ElideReasonNone       = ""                // FIRED: a rewrite happened (Elided/ShedBytes meaningful)
	ElideReasonOff        = "off"             // threshold<=0 or empty body — disabled
	ElideReasonNonJSON    = "non_json"        // body is not a JSON object
	ElideReasonNoMsgsKey  = "no_messages_key" // no "messages" key
	ElideReasonTooFewMsgs = "too_few_msgs"    // < 2 messages — nothing to scan (benign, high-volume)
	// ElideReasonDecodeFailed is the STRUCTURAL messages[] failure, split out of too_few_msgs so a
	// present-but-undecodable `messages` value is not counted in the benign short-request bucket.
	// Mirrors CompactReasonDecodeFailed; same wire token, so the three subsystems agree.
	ElideReasonDecodeFailed   = "decode_failed"
	ElideReasonNoBreakpoint   = "no_breakpoint"   // no cache_control anchor — cannot know the cache boundary
	ElideReasonUnderThreshold = "under_threshold" // no oversized eligible tool_result found
	ElideReasonSpliceFailed   = "splice_failed"   // the edits overlapped or fell out of range
	ElideReasonRedecodeFail   = "redecode_failed" // the spliced body failed to re-decode
	ElideReasonPrefixMismatch = "prefix_mismatch" // the splice changed the protected prefix bytes
	// ElideReasonMalformedResult is the semantic-well-formedness bail: the spliced body is valid
	// JSON and re-decodes through fak's OWN permissive decoder, but it lands a message-content
	// shape the real Anthropic Messages API rejects with `400 … malformed` — an empty `text` value,
	// an empty message `content` array, or a `tool_result` with empty content. fak's decoder drops
	// those silently (it accumulates text with a strings.Builder), so the re-decode guard alone
	// would ship a body the provider 400s. We return identity instead; the input was well-formed by
	// construction, since elision only ever rewrites a single non-empty string VALUE.
	ElideReasonMalformedResult = "malformed_result"
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
// cached head prefix is preserved verbatim. It returns raw UNCHANGED whenever it cannot prove the
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
	if !ok {
		// Structural: `messages` is present but is not a decodable JSON array.
		return raw, ElideOutcome{Reason: ElideReasonDecodeFailed}
	}
	if len(elems) < 2 {
		return raw, ElideOutcome{Reason: ElideReasonTooFewMsgs}
	}

	// Anchor the protected prefix on the FIRST cache_control breakpoint message, using the DEEP
	// detector — one nesting level deeper than compaction's, because the shrinker descends into
	// tool_result.content and must not anchor PAST a breakpoint nested there. Require a cache
	// anchor or we cannot know the cache boundary and must not touch the body.
	pfxEnd := firstBreakpointForElide(elems)
	if pfxEnd < 0 {
		// No message-level breakpoint. Fall back to a system-only cache anchor ONLY if `system`
		// carries a breakpoint AND its bytes precede the messages array (so the array head is a
		// sound protected-prefix end). If `system` sat AFTER messages on the wire, the
		// system-anchored cached prefix would span the messages, and editing one could burst it —
		// which we cannot prove safe, so bail. (Real clients send system before messages; this
		// guard makes the byte order, not the convention, load-bearing.)
		sysRaw, ok := obj["system"]
		if !ok || !rawHasCacheControl(sysRaw) || bytes.Index(raw, sysRaw) >= spans[0].start {
			return raw, ElideOutcome{Reason: ElideReasonNoBreakpoint}
		}
	}

	// The eligible band: strictly after the protected prefix, before the recent working-set
	// window, and never a message with cache_control reachable by the shrinker. Editing here keeps
	// the head prefix byte-identical (proven below); later breakpoints cascade-burst, as documented.
	lastEligible := len(elems) - elideRecentKeepMsgs // exclusive
	var edits []spliceEdit
	shed := 0
	for i := pfxEnd + 1; i < lastEligible; i++ {
		if i < 0 || messageHasCacheControlForElide(elems[i]) {
			continue
		}
		es, sh := collectResultElisionEdits(spans[i].start, elems[i], threshold)
		edits = append(edits, es...)
		shed += sh
	}

	// Cross-turn dedup (anthropic_elide_crossturn.go) is a distinct dedup LEVEL layered on the same
	// splice transform: fold a later tool_result line-span that appeared verbatim in a strictly-
	// earlier one to a one-line pointer. It rebuilds the whole value, so where it and head+tail
	// elision target the SAME value their byte spans coincide; prefer the dedup fold (a lossless
	// relocation to the run's earliest home) and drop the superseded head+tail edit. Its edits, like
	// the head+tail ones, all lie strictly after the protected prefix, so the cached head is safe.
	if ctEdits, ctShed := collectCrossTurnDedupEdits(raw, elems, spans, pfxEnd, lastEligible); len(ctEdits) > 0 {
		merged := make([]spliceEdit, 0, len(edits)+len(ctEdits))
		for _, e := range edits {
			if spliceEditOverlapsAny(e, ctEdits) {
				shed -= (e.end - e.start) - len(e.repl) // superseded by a dedup fold on the same value
				continue
			}
			merged = append(merged, e)
		}
		edits = append(merged, ctEdits...)
		shed += ctShed
	}

	if len(edits) == 0 {
		return raw, ElideOutcome{Reason: ElideReasonUnderThreshold}
	}

	out, ok := applySpliceEdits(raw, edits)
	if !ok {
		return raw, ElideOutcome{Reason: ElideReasonSpliceFailed}
	}
	// Prove it: re-decode + protected-prefix byte-equality + Anthropic semantic well-formedness
	// (shared with compaction — verifySplicedBody). Any of these failing is a splice bug, not a
	// reason to ship a broken / cache-busting / provider-400-ing body — fall back to identity. The
	// semantic check (spliceVerdictMalformedResult) is the load-bearing addition: the re-decode
	// through fak's OWN permissive decoder passes an empty-text/empty-content body the real
	// Anthropic API rejects with 400, so without it a bad splice would ship and 400 the session.
	switch verifySplicedBody(raw, out, spans, pfxEnd) {
	case spliceVerdictRedecodeFail:
		return raw, ElideOutcome{Reason: ElideReasonRedecodeFail}
	case spliceVerdictPrefixMismatch:
		return raw, ElideOutcome{Reason: ElideReasonPrefixMismatch}
	case spliceVerdictMalformedResult:
		return raw, ElideOutcome{Reason: ElideReasonMalformedResult}
	}
	return out, ElideOutcome{Reason: ElideReasonNone, Elided: len(edits), ShedBytes: shed}
}

// messageHasCacheControlForElide reports whether a message carries a cache_control breakpoint at
// ANY depth the shrinker can reach: on a top-level content block (messageHasCacheControl) OR on a
// block nested inside a tool_result's content array (which collectResultElisionEdits descends
// into). The shallow messageHasCacheControl misses the nested case — so using it for the anchor or
// the per-message skip would let elision shrink, and mis-anchor PAST, a real breakpoint and burst
// the head cache. This deeper predicate closes that gap for the elision path only; compaction's
// shallow detector is left untouched.
func messageHasCacheControlForElide(el json.RawMessage) bool {
	if messageHasCacheControl(el) {
		return true
	}
	var m struct {
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(el, &m) != nil || len(m.Content) == 0 || m.Content[0] != '[' {
		return false
	}
	var blocks []json.RawMessage
	if json.Unmarshal(m.Content, &blocks) != nil {
		return false
	}
	for _, blk := range blocks {
		var b struct {
			Type    string          `json:"type"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(blk, &b) == nil && b.Type == "tool_result" && len(b.Content) > 0 && rawHasCacheControl(b.Content) {
			return true
		}
	}
	return false
}

// firstBreakpointForElide is firstBreakpointMessage with the deep detector: the index of the
// first message carrying a cache_control breakpoint at any depth the shrinker can reach, or -1.
func firstBreakpointForElide(elems []json.RawMessage) int {
	for i, el := range elems {
		if messageHasCacheControlForElide(el) {
			return i
		}
	}
	return -1
}

// spliceEdit is one byte-range replacement within the original body: [start,end) → repl.
type spliceEdit struct {
	start, end int
	repl       []byte
}

// collectResultElisionEdits scans one messages[] element (a user turn that may carry tool_result
// blocks) and returns a splice edit for each oversized tool_result text payload, shrinking it to
// head+tail. msgBase is the element's absolute start byte in the original body; every returned
// edit span is absolute. Value byte ranges are located by KEY via objectValueSpan (never a
// bytes.Index over the whole block, which a sibling field with identical bytes — e.g. a
// tool_use_id equal to the content — would mis-locate). A non-user message, a non-array content,
// a tool_result that itself carries cache_control, or a shape it cannot parse yields no edits.
func collectResultElisionEdits(msgBase int, el json.RawMessage, threshold int) (edits []spliceEdit, shed int) {
	var m struct {
		Role string `json:"role"`
	}
	if json.Unmarshal(el, &m) != nil || m.Role != "user" {
		return nil, 0
	}
	forEachToolResultBlock(msgBase, el, func(blk json.RawMessage, blkBase int) {
		// Defense in depth: never shrink a tool_result that carries cache_control on the block
		// itself or on any block nested in its content (the per-message skip already excludes such
		// a message, but keep this self-contained too).
		if rawHasCacheControl(blk) || toolResultContentHasCacheControl(blk) {
			return
		}
		cStart, cEnd, ok := objectValueSpan(blk, "content")
		if !ok {
			return
		}
		cVal := blk[cStart:cEnd]
		switch {
		case len(cVal) > 0 && cVal[0] == '"': // content is a bare JSON string
			if e, sh, ok := elideStringEdit(blkBase+cStart, cVal, threshold); ok {
				edits = append(edits, e)
				shed += sh
			}
		case len(cVal) > 0 && cVal[0] == '[': // content is an array of blocks — shrink each oversized text block
			inner, innerSpans, ok := arrayElementSpans(cVal) // spans relative to cVal (base 0)
			if !ok {
				return
			}
			for k, ib := range inner {
				var tb struct {
					Type string `json:"type"`
				}
				if json.Unmarshal(ib, &tb) != nil || tb.Type != "text" {
					continue
				}
				tStart, tEnd, ok := objectValueSpan(ib, "text") // exact text-value span within ib
				if !ok {
					continue
				}
				if e, sh, ok := elideStringEdit(blkBase+cStart+innerSpans[k].start+tStart, ib[tStart:tEnd], threshold); ok {
					edits = append(edits, e)
					shed += sh
				}
			}
		}
	})
	return edits, shed
}

// arrayElementSpans decodes a JSON array's elements with spans RELATIVE TO arr (base 0). Unlike
// decodeArrayElements, which locates the array inside a larger container by bytes.Index, the
// caller has already located arr exactly — so re-searching it (and risking a byte-identical
// sibling) is both unnecessary and unsafe. This keeps every value location key-exact.
func arrayElementSpans(arr []byte) ([]json.RawMessage, []elementSpan, bool) {
	return decodeArrayElements(arr, arr) // bytes.Index(arr, arr) == 0, so spans are arr-relative
}

// forEachToolResultBlock locates el's content ARRAY by KEY (exact, never a bytes.Index a
// byte-identical sibling field could mis-locate) and invokes fn once per tool_result block in it,
// passing the raw block and blkBase — the block's absolute start byte in the body (msgBase-relative).
// It is the shared scanning prologue of collectResultElisionEdits, collectToolReferenceEdits, and
// collectEmptyContentEdits; a missing content key, bare-string content, or an unparseable array
// yields no calls.
func forEachToolResultBlock(msgBase int, el json.RawMessage, fn func(blk json.RawMessage, blkBase int)) {
	mcStart, mcEnd, ok := objectValueSpan(el, "content")
	if !ok {
		return
	}
	mContent := el[mcStart:mcEnd]
	if len(mContent) == 0 || mContent[0] != '[' {
		return
	}
	blocks, blockSpans, ok := arrayElementSpans(mContent) // spans relative to mContent (base 0)
	if !ok {
		return
	}
	for j, blk := range blocks {
		var b struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(blk, &b) != nil || b.Type != "tool_result" {
			continue
		}
		fn(blk, msgBase+mcStart+blockSpans[j].start) // absolute start of blk in the body
	}
}

// toolResultContentHasCacheControl reports whether a tool_result block's content array carries a
// cache_control breakpoint on any nested block.
func toolResultContentHasCacheControl(blk json.RawMessage) bool {
	var b struct {
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(blk, &b) != nil || len(b.Content) == 0 {
		return false
	}
	return rawHasCacheControl(b.Content)
}

// elideStringEdit builds the splice edit that shrinks one JSON string VALUE (valBytes, located
// at absolute byte offset valAbs) to head+tail, if it exceeds threshold bytes and the shrunk form
// is strictly shorter. ok is false (no edit) when the value is not a string, not oversized, cannot
// be decoded, or would not actually save.
func elideStringEdit(valAbs int, valBytes []byte, threshold int) (spliceEdit, int, bool) {
	if len(valBytes) <= threshold || len(valBytes) == 0 || valBytes[0] != '"' {
		return spliceEdit{}, 0, false
	}
	var s string
	if json.Unmarshal(valBytes, &s) != nil {
		return spliceEdit{}, 0, false
	}
	shrunk := elideHeadTail(s, threshold)
	// Source-side invariant guard (belt-and-suspenders for the post-splice semantic check): the
	// shrunk value must never be the empty string, since an empty `text`/tool_result value is
	// exactly the shape the Anthropic API 400s. elideHeadTail joins head+marker+tail and the marker
	// is non-empty, so this is unreachable in practice — but keep it load-bearing so a future
	// threshold/marker change cannot silently ship an empty value past this line.
	if shrunk == "" {
		return spliceEdit{}, 0, false
	}
	newVal, err := json.Marshal(shrunk)
	if err != nil || len(newVal) >= len(valBytes) {
		return spliceEdit{}, 0, false
	}
	return spliceEdit{start: valAbs, end: valAbs + len(valBytes), repl: newVal}, len(valBytes) - len(newVal), true
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

// objectValueSpan returns the [start,end) byte span (relative to obj) of the VALUE for the given
// top-level key in a JSON object, or ok=false if absent / not an object. It walks the object with
// a streaming decoder so the span is exact even when a sibling value has identical bytes — a plain
// bytes.Index over the object would mis-locate it (the tool_use_id == content corruption vector).
func objectValueSpan(obj json.RawMessage, key string) (start, end int, ok bool) {
	return objectValueSpanImpl(obj, key, false)
}

// objectValueSpanLastWins is objectValueSpan but, on a duplicate top-level key, returns the
// LAST matching key's value span instead of the first — matching json.Unmarshal-into-a-map
// (last-key-wins). The top-level cachebp routing (decodeTopLevelArray) reads obj[key] with those
// same map semantics, so it must anchor the splice on the SAME value bytes the decoder keeps, or
// an adversarial duplicate key would splice one value while the rest of the planner reasons about
// another (#3773 confusion risk).
func objectValueSpanLastWins(obj json.RawMessage, key string) (start, end int, ok bool) {
	return objectValueSpanImpl(obj, key, true)
}

// objectValueSpanImpl is the shared streaming walk behind objectValueSpan (first match) and
// objectValueSpanLastWins (last match). lastWins keeps scanning after a hit so the final
// occurrence wins; otherwise it returns on the first hit.
func objectValueSpanImpl(obj json.RawMessage, key string, lastWins bool) (start, end int, ok bool) {
	dec := json.NewDecoder(bytes.NewReader(obj))
	tok, err := dec.Token()
	if err != nil {
		return 0, 0, false
	}
	if d, isDelim := tok.(json.Delim); !isDelim || d != '{' {
		return 0, 0, false
	}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return 0, 0, false
		}
		k, isStr := kt.(string)
		if !isStr {
			return 0, 0, false
		}
		// InputOffset is now just past the key token, before the ':' and the value. Skip
		// whitespace and the single ':' separator to the value's first significant byte.
		vStart := int(dec.InputOffset())
		for vStart < len(obj) && (isJSONSpace(obj[vStart]) || obj[vStart] == ':') {
			vStart++
		}
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return 0, 0, false
		}
		vEnd := int(dec.InputOffset())
		// Trim any trailing whitespace the decoder may have left before the ','/'}' delimiter.
		for vEnd > vStart && isJSONSpace(obj[vEnd-1]) {
			vEnd--
		}
		if k == key {
			start, end, ok = vStart, vEnd, true
			if !lastWins {
				return start, end, ok
			}
		}
	}
	return start, end, ok
}

// applySpliceEdits applies disjoint byte-range replacements to raw, building the result forward.
// Edits are sorted by start; an overlap or an out-of-range span returns ok=false (identity at the
// caller). Bytes outside every edit span — including the whole head prefix — are copied verbatim,
// so the cache guarantee is a bytes.Equal, not a hope.
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
