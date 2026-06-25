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
	CompactReasonWindowNoDrop   = "window_no_drop" // the kept window swallowed the whole suffix
	CompactReasonSpliceFailed   = "splice_failed"
	CompactReasonRedecodeFail   = "redecode_failed" // the spliced body failed to re-decode
	CompactReasonPrefixMismatch = "prefix_mismatch" // the splice changed the protected prefix bytes
)

// CompactOutcome is the observable verdict of one compaction attempt. Reason==CompactReasonNone
// means FIRED — Dropped (whole messages stubbed out) and ShedTokens (estimated tokens removed
// from the outbound body, same ~4-chars/token currency as the budget) are then meaningful. Any
// other Reason means the body was returned unchanged (identity), and Dropped/ShedTokens are 0.
type CompactOutcome struct {
	Reason     string
	Dropped    int
	ShedTokens int
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
func CompactAnthropicHistoryWithOutcome(raw []byte, budget int) ([]byte, CompactOutcome) {
	if budget <= 0 || len(raw) == 0 {
		return raw, CompactOutcome{Reason: CompactReasonUnderBudget}
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, CompactOutcome{Reason: CompactReasonNonJSON} // not a JSON object — leave it alone
	}
	msgsRaw, ok := obj["messages"]
	if !ok {
		return raw, CompactOutcome{Reason: CompactReasonNoMsgsKey}
	}
	elems, spans, ok := decodeArrayElements(raw, msgsRaw)
	if !ok || len(elems) < 3 {
		return raw, CompactOutcome{Reason: CompactReasonTooFewMsgs} // nothing safe to compact
	}

	// 1. Anchor the protected prefix on the FIRST cache_control breakpoint message — the
	//    STABLE cached head the provider reuses every turn — NOT the last. This is the fix
	//    for the silent no-op on real Claude Code traffic: the growing-conversation cache
	//    layout (Anthropic prompt-caching docs: up to 4 breakpoints) marks the static head
	//    AND a RECENT turn, so the LAST breakpoint sits near the end. Anchoring on it would
	//    "protect" the whole conversation and compact NOTHING. The span we want to drop is
	//    the un-cacheable MIDDLE — the turns between the cached head and the recent kept
	//    window — which the provider re-bills every turn anyway (it is beyond the recent
	//    breakpoint's 20-block backward lookback). Removing it shifts the recent breakpoint's
	//    position, so the provider's cache read walks back to the HEAD breakpoint (the cache
	//    cascade), where the bytes are byte-identical → the dominant cache hit survives. A
	//    breakpoint in `system` (or no message breakpoint) still protects the array head; we
	//    require a breakpoint somewhere or we cannot know the cache boundary and must not
	//    touch the body.
	pfxEnd := firstBreakpointMessage(elems)
	sysHasCC := rawHasCacheControl(obj["system"])
	if pfxEnd < 0 && !sysHasCC {
		return raw, CompactOutcome{Reason: CompactReasonNoBreakpoint} // no anchor — identity
	}
	// When only `system` holds the cache, the protected message prefix is empty (-1):
	// every message is compactible. Otherwise it ends at the FIRST breakpoint message (the
	// cached head); everything after it up to the recent kept window is compactible middle.

	// 2. Is the compactible suffix already under budget? Then there is nothing to do.
	suffixTokens := 0
	for i := pfxEnd + 1; i < len(elems); i++ {
		suffixTokens += len(elems[i]) / 4
	}
	if suffixTokens <= budget {
		return raw, CompactOutcome{Reason: CompactReasonUnderBudget}
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

	// 4. Splice on ORIGINAL bytes. The prefix span [0, spans[pfxEnd].end) (or just the
	//    array-open when pfxEnd<0) is copied verbatim; then the stub; then the kept
	//    elements verbatim; then the verbatim tail from the array close onward.
	out, ok := spliceCompacted(raw, spans, pfxEnd, keepStart, len(elems), dropped, stubRole)
	if !ok {
		return raw, CompactOutcome{Reason: CompactReasonSpliceFailed}
	}

	// 5. Prove it: the result must still decode as a valid Messages request, and the
	//    protected prefix bytes must be byte-identical to the input. Either failing is a
	//    bug in the splice, not a reason to ship a broken/cache-busting body — fall back.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		return raw, CompactOutcome{Reason: CompactReasonRedecodeFail}
	}
	prefixEnd := arrayContentStart(spans) // byte offset just inside `[`
	if pfxEnd >= 0 {
		prefixEnd = spans[pfxEnd].end
	}
	if prefixEnd > len(out) || !bytes.Equal(raw[:prefixEnd], out[:prefixEnd]) {
		return raw, CompactOutcome{Reason: CompactReasonPrefixMismatch}
	}
	return out, CompactOutcome{Reason: CompactReasonNone, Dropped: dropped, ShedTokens: shedTokens}
}

// elementSpan is the [start,end) byte range of one messages[] element within the original
// body, where start points at the element's first byte and end just past its last.
type elementSpan struct{ start, end int }

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
		var m struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(el, &m) != nil {
			continue
		}
		if rawHasCacheControl(m.Content) {
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
		var m struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(el, &m) != nil {
			continue
		}
		if rawHasCacheControl(m.Content) {
			return i
		}
	}
	return -1
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
