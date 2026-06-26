package agent

// anthropic_cachebp.go — the OFFENSIVE half of kernel cache-prefix control (#806), the
// sibling of the DEFENSIVE half in anthropic_compact.go.
//
// The defensive half (CompactAnthropicHistory) ANCHORS on a cache_control breakpoint the
// client already placed and never busts it. But a client that talks to the Anthropic
// passthrough with NO cache_control at all (a raw OpenAI-shaped caller, a minimal SDK, a
// hand-rolled request) leaves provider prefix caching entirely on the table: the stable
// system+tools head is re-prefilled every turn at full price. The offensive half PLACES a
// breakpoint on that stable head so the provider caches it — turning a coin-flip into a
// near-guarantee for callers that never asked for it.
//
// Where the breakpoint lands. Anthropic's prompt cache is positional and orders the prefix
// tools → system → messages: a cache_control breakpoint marks the end of a cacheable span
// that includes everything BEFORE it in that order. So a breakpoint on the LAST `system`
// block caches tools+system (the maximal stable head); when there is no `system` array we
// fall back to the last `tools` entry (caches tools). The volatile message tail is left
// uncached, exactly where it should be.
//
// The same fail-safe-identity discipline as anthropic_compact.go governs every step:
//
//   - If a cache_control breakpoint already exists ANYWHERE in the body, do nothing and
//     return the input unchanged. We never override a layout the client (or a smarter peer
//     stage) already chose, and we never risk busting a cache that is already working.
//   - The breakpoint is spliced into the target block on the ORIGINAL bytes (a comma + one
//     key inserted before the block's closing `}`); no sibling block, and nothing before the
//     target, is ever re-marshalled — so the bytes upstream of the new breakpoint are
//     byte-identical to the input.
//   - The result must re-decode as a valid Messages request; on ANY ambiguity the function
//     returns its input UNCHANGED.
//
// Like CompactAnthropicHistory this is a REQUEST-side transform on the wire bytes only. It
// never touches the decoded req.Messages the kernel adjudicates, so the trust boundary is
// unchanged — it only adds a caching hint to the bytes forwarded upstream.

import (
	"bytes"
	"encoding/json"
)

// cacheControlBreakpoint is the byte sequence spliced into a target block to mark it as a
// cache prefix boundary — an ephemeral (5-minute) breakpoint, the Anthropic default tier.
const cacheControlBreakpoint = `"cache_control":{"type":"ephemeral"}`

// Breakpoint-placement bail vocabulary — the closed set of outcomes, mirroring CompactReason*.
// BreakpointReasonNone means a breakpoint was PLACED (the body was rewritten); every other
// value means the body was returned unchanged (identity).
const (
	BreakpointReasonNone         = ""               // PLACED: a breakpoint was spliced onto the stable head
	BreakpointReasonNonJSON      = "non_json"        // body is empty or not a JSON object
	BreakpointReasonAlreadySet   = "already_set"     // a cache_control already exists — respect the existing layout
	BreakpointReasonNoStableHead = "no_stable_head"  // no system[] or tools[] block to anchor on
	BreakpointReasonSpliceFailed = "splice_failed"   // the target block is not a spliceable object
	BreakpointReasonRedecodeFail = "redecode_failed" // the spliced body failed to re-decode as a request
)

// BreakpointOutcome is the observable verdict of one placement attempt. Reason==BreakpointReasonNone
// means PLACED — Target ("system" or "tools") then names where the breakpoint landed. Any other
// Reason means the body was returned unchanged (identity) and Target is empty.
type BreakpointOutcome struct {
	Reason string
	Target string // "system" | "tools" — which head block carries the new breakpoint (on a placement)
}

// PlaceAnthropicCacheBreakpoint splices a cache_control breakpoint onto the stable system+tools
// head of an outbound Anthropic /v1/messages body so the provider caches it, when the body
// carries no breakpoint of its own. It returns the input UNCHANGED on any ambiguity (see the
// BreakpointReason* vocabulary). This is the byte-only wrapper; PlaceAnthropicCacheBreakpointWithOutcome
// additionally reports WHY it bailed / where it landed, for observability.
func PlaceAnthropicCacheBreakpoint(raw []byte) []byte {
	out, _ := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	return out
}

// PlaceAnthropicCacheBreakpointWithOutcome is PlaceAnthropicCacheBreakpoint plus the observable
// outcome (placed-and-where vs the labeled bail reason). The byte-level guarantees are identical:
// the bytes before the new breakpoint are byte-identical to the input, and the result re-decodes
// as a valid request — or the input is returned unchanged.
func PlaceAnthropicCacheBreakpointWithOutcome(raw []byte) ([]byte, BreakpointOutcome) {
	if len(raw) == 0 {
		return raw, BreakpointOutcome{Reason: BreakpointReasonNonJSON}
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, BreakpointOutcome{Reason: BreakpointReasonNonJSON} // not a JSON object — leave it alone
	}

	// 1. If a cache_control breakpoint already exists ANYWHERE in the body, respect it — never
	//    override a working layout. A bare substring scan is deliberately conservative: a false
	//    positive (the literal inside some string value) only means we DON'T place, which is the
	//    fail-safe direction. The common Claude Code shape already marks its head + recent turns,
	//    so this stage targets precisely the callers that left caching on the table.
	if bytes.Contains(raw, []byte("cache_control")) {
		return raw, BreakpointOutcome{Reason: BreakpointReasonAlreadySet}
	}

	// 2. Pick the stable-head target: the LAST `system` block (caches tools+system) if `system`
	//    is a non-empty array, else the LAST `tools` entry (caches tools). No array head ⇒ there
	//    is no stable span we can safely cache without touching the volatile message tail — bail.
	target := "system"
	elems, spans, ok := decodeArrayElements(raw, obj["system"])
	if !ok || len(elems) == 0 {
		target = "tools"
		elems, spans, ok = decodeArrayElements(raw, obj["tools"])
	}
	if !ok || len(elems) == 0 {
		return raw, BreakpointOutcome{Reason: BreakpointReasonNoStableHead}
	}
	last := spans[len(spans)-1]

	// 3. Splice the breakpoint into the last block on the ORIGINAL bytes: everything before the
	//    block is copied verbatim, the breakpoint key is inserted before the block's closing `}`,
	//    and the tail is copied verbatim. No other block is re-marshalled.
	spliced, ok := spliceCacheControlIntoObject(raw[last.start:last.end])
	if !ok {
		return raw, BreakpointOutcome{Reason: BreakpointReasonSpliceFailed}
	}
	var b bytes.Buffer
	b.Grow(len(raw) + len(spliced) - (last.end - last.start))
	b.Write(raw[:last.start])
	b.Write(spliced)
	b.Write(raw[last.end:])
	out := b.Bytes()

	// 4. Prove it: the result must re-decode as a valid Messages request, and every byte before
	//    the rewritten block must be byte-identical to the input (the cache prefix upstream of the
	//    new breakpoint is untouched). Either failing is a splice bug, not a reason to ship a
	//    malformed/cache-busting body — fall back to identity.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		return raw, BreakpointOutcome{Reason: BreakpointReasonRedecodeFail}
	}
	if !bytes.Equal(raw[:last.start], out[:last.start]) {
		return raw, BreakpointOutcome{Reason: BreakpointReasonRedecodeFail}
	}
	return out, BreakpointOutcome{Reason: BreakpointReasonNone, Target: target}
}

// spliceCacheControlIntoObject returns obj with a cache_control breakpoint key inserted before
// its closing `}`, preserving every existing byte. obj must be a single JSON object (`{...}`);
// ok is false otherwise. An empty object `{}` gets the lone key (no leading comma); a non-empty
// object gets a leading comma so the existing keys are kept verbatim.
func spliceCacheControlIntoObject(obj []byte) ([]byte, bool) {
	if len(obj) < 2 || obj[0] != '{' || obj[len(obj)-1] != '}' {
		return nil, false
	}
	hasContent := false
	for _, c := range obj[1 : len(obj)-1] {
		if !isJSONSpace(c) {
			hasContent = true
			break
		}
	}
	var b bytes.Buffer
	b.Grow(len(obj) + len(cacheControlBreakpoint) + 1)
	if hasContent {
		b.Write(obj[:len(obj)-1]) // everything up to (not incl.) the closing `}`
		b.WriteByte(',')
		b.WriteString(cacheControlBreakpoint)
		b.WriteByte('}')
	} else { // empty object — drop interior whitespace, no leading comma
		b.WriteByte('{')
		b.WriteString(cacheControlBreakpoint)
		b.WriteByte('}')
	}
	return b.Bytes(), true
}
