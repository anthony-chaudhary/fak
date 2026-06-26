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
//   - A candidate head span that carries a self-evidently per-request token — a sub-day timestamp
//     or a UUID/nonce (headValueIsVolatile) — is NOT byte-stable across turns, so anchoring a
//     breakpoint on it would pay the provider's cache-WRITE premium for a prefix doomed to miss. We
//     step DOWN from a volatile tools+system head to caching just the stable tools head, and bail to
//     identity (BreakpointReasonVolatileHead) when no stable span remains. This is the fail-safe,
//     single-body half of #806 bullet 2 (keep the stable spans byte-stable); the aggressive form
//     (STRIP/normalize the volatile token in place) needs a redaction spec + soak and is deferred.
//
// Like CompactAnthropicHistory this is a REQUEST-side transform on the wire bytes only. It
// never touches the decoded req.Messages the kernel adjudicates, so the trust boundary is
// unchanged — it only adds a caching hint to the bytes forwarded upstream.

import (
	"bytes"
	"encoding/json"
	"regexp"
)

// cacheControlBreakpoint is the byte sequence spliced into a target block to mark it as a
// cache prefix boundary — an ephemeral (5-minute) breakpoint, the Anthropic default tier.
const cacheControlBreakpoint = `"cache_control":{"type":"ephemeral"}`

// Breakpoint-placement bail vocabulary — the closed set of outcomes, mirroring CompactReason*.
// BreakpointReasonNone means a breakpoint was PLACED (the body was rewritten); every other
// value means the body was returned unchanged (identity).
const (
	BreakpointReasonNone         = ""                // PLACED: a breakpoint was spliced onto the stable head
	BreakpointReasonNonJSON      = "non_json"        // body is empty or not a JSON object
	BreakpointReasonAlreadySet   = "already_set"     // a cache_control already exists — respect the existing layout
	BreakpointReasonNoStableHead = "no_stable_head"  // no system[] or tools[] block to anchor on
	BreakpointReasonVolatileHead = "volatile_head"   // every cacheable head span carries a per-request token
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

	// 2. Pick the stable-head target, preferring the MAXIMAL stable span. The breakpoint marks the
	//    end of a positional prefix the provider caches (order: tools → system → messages), so the
	//    LAST `system` block caches tools+system and the LAST `tools` entry caches tools alone. A
	//    span is only worth anchoring if it is BYTE-STABLE across turns: a per-request token in it
	//    (a sub-day timestamp, a UUID/nonce — headValueIsVolatile) changes the very prefix this
	//    breakpoint secures, so we'd pay the provider's cache-WRITE premium for a prefix doomed to
	//    miss. So we step DOWN from a volatile tools+system head to caching just the stable tools,
	//    and bail to identity when no cacheable span is byte-stable (#806 bullet 2, fail-safe form).
	sysElems, sysSpans, sysOK := decodeArrayElements(raw, obj["system"])
	sysOK = sysOK && len(sysElems) > 0
	toolElems, toolSpans, toolOK := decodeArrayElements(raw, obj["tools"])
	toolOK = toolOK && len(toolElems) > 0
	toolsVolatile := headValueIsVolatile(obj["tools"])

	var target string
	var spans []elementSpan
	switch {
	case sysOK && !toolsVolatile && !headValueIsVolatile(obj["system"]):
		target, spans = "system", sysSpans // maximal stable head: cache tools+system
	case toolOK && !toolsVolatile:
		target, spans = "tools", toolSpans // system absent or volatile, tools stable: cache tools alone
	case sysOK || toolOK:
		// There IS an array head, but every cacheable span carries a volatility signature — anchoring
		// here would bust the prefix it caches, so leave the body unchanged (the fail-safe direction).
		return raw, BreakpointOutcome{Reason: BreakpointReasonVolatileHead}
	default:
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

// Volatility signatures — the per-request token SHAPES that, sitting in a cache-prefix head, change
// the bytes between turns and bust the prefix a breakpoint is meant to secure. Only UNAMBIGUOUS
// shapes are listed, because a false positive merely SKIPS a cache (fail-safe) while a false
// negative caches a busting span (the harm). Single-body detection cannot see a value-only nonce
// that looks like an ordinary word, nor reordered-key JSON (which needs two turns to observe); those
// remain for the aggressive strip/normalize follow-up — the full form of #806 bullet 2.
var (
	// volUUID matches a canonical UUID/GUID (8-4-4-4-12 hex) — the standard nonce / request-id shape.
	volUUID = regexp.MustCompile(`[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}`)
	// volDateTime matches an ISO-8601 date with a TIME-OF-DAY component (a `T`/space then HH:MM):
	// sub-day resolution changes faster than the 5-minute ephemeral cache TTL. A date-ONLY token
	// (2026-06-26) lacks the trailing HH:MM and is intentionally NOT matched — it is byte-stable
	// across a session's turns and is the common "Today's date is ..." head shape we WANT to cache.
	volDateTime = regexp.MustCompile(`[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt ][0-9]{2}:[0-9]{2}`)
)

// headValueIsVolatile reports whether a candidate cache-prefix value (the `system` or `tools` JSON
// value, raw) carries a self-evidently per-request token. It scans the raw bytes, so it sees a
// token embedded anywhere in the head — a UUID in a tool description, a timestamp in a system block.
// An empty or absent value is not volatile.
func headValueIsVolatile(v json.RawMessage) bool {
	if len(v) == 0 {
		return false
	}
	return volUUID.Match(v) || volDateTime.Match(v)
}
