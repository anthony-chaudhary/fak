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
//     (STRIP/normalize the volatile token in place) is the spec-governed, opt-in redaction retry
//     in anthropic_cachebp_redact.go (#2191) — default OFF until its soak + ablation row land.
//
// Like CompactAnthropicHistory this is a REQUEST-side transform on the wire bytes only. It
// never touches the decoded req.Messages the kernel adjudicates, so the trust boundary is
// unchanged — it only adds a caching hint to the bytes forwarded upstream.

import (
	"bytes"
	"encoding/json"
	"regexp"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// cacheControlBreakpoint is the byte sequence spliced into a target block to mark it as a
// cache prefix boundary — an ephemeral (5-minute) breakpoint, the Anthropic default tier.
const cacheControlBreakpoint = `"cache_control":{"type":"ephemeral"}`

const cacheControlTTL1h = `"ttl":"1h"`

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
	Reason          string
	Target          string // "system" | "tools" — which head block carries the new breakpoint (on a placement)
	Rewritten       bool   // true when M2 hoisted volatile system blocks behind the cacheable anchor
	MovedVolatile   int
	PredictedUplift int64

	// Redaction witness (#2191): set only when the volatile_head refusal path ran with the
	// FAK_CACHEBP_REDACT lever in play. Redacted=true means the placement above happened on a
	// spec-normalized head (RedactedUUID/RedactedTimestamp count the tokens replaced); on a
	// refusal, RedactReason labels why the redaction retry did not convert it.
	Redacted          bool
	RedactedUUID      int
	RedactedTimestamp int
	RedactReason      string
}

const (
	TTLUpgradeReasonNone               = "" // UPGRADED: ttl:"1h" was spliced into a stable-head cache_control object.
	TTLUpgradeReasonNonJSON            = "non_json"
	TTLUpgradeReasonNoStableBreakpoint = "no_stable_breakpoint"    // no cache_control on system/tools; message-tail breakpoints are not stable head.
	TTLUpgradeReasonAlready1h          = "already_1h"              // the stable-head breakpoint is already on the 1h tier.
	TTLUpgradeReasonTTLAlreadySet      = "ttl_already_set"         // another ttl value exists; respect the caller's choice.
	TTLUpgradeReasonVolatileHead       = "volatile_head"           // the candidate head carries an obvious per-request token.
	TTLUpgradeReasonVolatileMessage    = "volatile_message_prefix" // the candidate message prefix carries an obvious per-request token.
	TTLUpgradeReasonSpliceFailed       = "splice_failed"
	TTLUpgradeReasonRedecodeFail       = "redecode_failed"
)

// TTLUpgradeOutcome reports whether UpgradeAnthropicStableCacheTTL1h changed the existing
// stable-head cache_control object. Reason==TTLUpgradeReasonNone means Target was upgraded.
type TTLUpgradeOutcome struct {
	Reason string
	Target string // "system" | "tools" | "messages"

	// Split counts make the head-only versus message-prefix ablation inspectable.
	UpgradedHeadBreakpoints    int
	UpgradedMessageBreakpoints int

	// Redaction witness (#2191) — same contract as BreakpointOutcome's redaction fields.
	Redacted          bool
	RedactedUUID      int
	RedactedTimestamp int
	RedactReason      string
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
// as a valid request — or the input is returned unchanged. A volatile_head refusal gets ONE
// spec-governed redaction retry (anthropic_cachebp_redact.go, opt-in via FAK_CACHEBP_REDACT);
// with the lever off (the default) the refusal is returned exactly as before.
func PlaceAnthropicCacheBreakpointWithOutcome(raw []byte) ([]byte, BreakpointOutcome) {
	out, oc := placeAnthropicCacheBreakpointOnce(raw)
	if oc.Reason != BreakpointReasonVolatileHead {
		return out, oc
	}
	return retryPlaceWithRedactedHead(raw, oc)
}

// placeAnthropicCacheBreakpointOnce is one un-retried placement pass — the whole original
// pipeline (already-set guard, stable-head pick, M2 hoist, splice-and-prove). The redaction
// retry above composes AFTER its volatile_head refusal, so hoisting keeps first claim and the
// default path is byte-for-byte the pre-#2191 behavior.
func placeAnthropicCacheBreakpointOnce(raw []byte) ([]byte, BreakpointOutcome) {
	if len(raw) == 0 {
		return raw, BreakpointOutcome{Reason: BreakpointReasonNonJSON}
	}

	// 1. If a cache_control breakpoint already exists ANYWHERE in the body, respect it — never
	//    override a working layout. The scan is deliberately conservative: a false positive (the
	//    literal inside some string value) only means we DON'T place, which is the fail-safe
	//    direction. The common Claude Code shape already marks its head + recent turns, so this
	//    stage targets precisely the callers that left caching on the table. This scan runs BEFORE
	//    the whole-body json.Unmarshal below: the dominant default body already carries
	//    cache_control, so returning here skips decoding + COPYING the entire messages array into a
	//    map[string]json.RawMessage (a full-request-body allocation this stage would otherwise pay on
	//    every wire only to discard). The skipSpace/'{' + json.Valid check keeps this EXACTLY the old
	//    decode-first behavior: json.Unmarshal into a map accepts only a well-formed JSON object, so a
	//    malformed or non-object body carrying the literal still bails NonJSON (identity either way).
	//
	//    #3774: the detector is bodyHasCacheControlKey, not a bare bytes.Contains. A client whose key
	//    spells an ASCII letter/underscore as a JSON \uXXXX escape — cache_control — decodes to
	//    the SAME cache_control key yet slips past a byte-literal scan, so placement would splice a
	//    SECOND (semantic) breakpoint over the client's own layout, the exact override this guard
	//    exists to prevent. bodyHasCacheControlKey EXTENDS the scan to the escaped form (only in the
	//    fail-safe direction: it can add a skip, never remove one) and keeps the byte-literal hot path
	//    allocation-free. This also aligns the placement guard with the upgrade path, whose
	//    rawHasCacheControl already decodes escaped keys via json.Unmarshal.
	if bodyHasCacheControlKey(raw) {
		if t := skipSpace(raw); len(t) > 0 && t[0] == '{' && json.Valid(raw) {
			return raw, BreakpointOutcome{Reason: BreakpointReasonAlreadySet}
		}
		return raw, BreakpointOutcome{Reason: BreakpointReasonNonJSON}
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, BreakpointOutcome{Reason: BreakpointReasonNonJSON} // not a JSON object — leave it alone
	}

	// 2. Pick the stable-head target, preferring the MAXIMAL stable span. The breakpoint marks the
	//    end of a positional prefix the provider caches (order: tools → system → messages), so the
	//    LAST `system` block caches tools+system and the LAST `tools` entry caches tools alone. A
	//    span is only worth anchoring if it is BYTE-STABLE across turns: a per-request token in it
	//    (a sub-day timestamp, a UUID/nonce — headValueIsVolatile) changes the very prefix this
	//    breakpoint secures, so we'd pay the provider's cache-WRITE premium for a prefix doomed to
	//    miss. So we step DOWN from a volatile tools+system head to caching just the stable tools,
	//    and bail to identity when no cacheable span is byte-stable (#806 bullet 2, fail-safe form).
	// Anchor the head arrays by KEY (decodeTopLevelArray), never bytes.Index on their value bytes:
	// a `system`/`tools` array can be byte-identical to a message content array, and a first-
	// occurrence search would then splice the breakpoint into the wrong array (#3773).
	sysElems, sysSpans, sysOK := decodeTopLevelArray(raw, "system")
	sysOK = sysOK && len(sysElems) > 0
	toolElems, toolSpans, toolOK := decodeTopLevelArray(raw, "tools")
	toolOK = toolOK && len(toolElems) > 0
	toolsVolatile := headValueIsVolatile(obj["tools"])

	if toolsVolatile {
		// The provider prefix order is tools → system → messages. A volatile tools[] value sits
		// ahead of every possible system anchor, so no system rewrite can make tools+system stable.
		if sysOK || toolOK {
			return raw, BreakpointOutcome{Reason: BreakpointReasonVolatileHead}
		}
		return raw, BreakpointOutcome{Reason: BreakpointReasonNoStableHead}
	}

	if sysOK {
		if plan, ok := planAnthropicSystemAnchor(sysElems); ok {
			if plan.rewritten {
				out, ok := rewriteSystemArrayWithBreakpoint(raw, sysElems, plan)
				if !ok {
					return raw, BreakpointOutcome{Reason: BreakpointReasonSpliceFailed}
				}
				if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
					return raw, BreakpointOutcome{Reason: BreakpointReasonRedecodeFail}
				}
				return out, BreakpointOutcome{
					Reason:          BreakpointReasonNone,
					Target:          "system",
					Rewritten:       true,
					MovedVolatile:   plan.recommendation.MovedVolatile,
					PredictedUplift: plan.recommendation.PredictedUplift,
				}
			}
			out, reason := placeAndValidateAtSpan(raw, sysSpans[plan.anchorOriginal])
			if reason != BreakpointReasonNone {
				return raw, BreakpointOutcome{Reason: reason}
			}
			return out, BreakpointOutcome{
				Reason:          BreakpointReasonNone,
				Target:          "system",
				MovedVolatile:   plan.recommendation.MovedVolatile,
				PredictedUplift: plan.recommendation.PredictedUplift,
			}
		}
	}

	var target string
	var spans []elementSpan
	switch {
	case toolOK:
		target, spans = "tools", toolSpans // system absent or fully volatile, tools stable: cache tools alone
	case sysOK:
		// There IS a system head, but every system block carries a volatility signature and no
		// stable tools prefix is available, so leave the body unchanged (the fail-safe direction).
		return raw, BreakpointOutcome{Reason: BreakpointReasonVolatileHead}
	default:
		return raw, BreakpointOutcome{Reason: BreakpointReasonNoStableHead}
	}
	last := spans[len(spans)-1]

	// 3. Splice the breakpoint onto the last block on the ORIGINAL bytes and prove it (re-decodes
	//    as a request; the cache prefix upstream of the new breakpoint is byte-identical to the
	//    input). See placeAndValidateAtSpan for the placement-and-proof step and its bail reasons.
	out, reason := placeAndValidateAtSpan(raw, last)
	if reason != BreakpointReasonNone {
		return raw, BreakpointOutcome{Reason: reason}
	}
	return out, BreakpointOutcome{Reason: BreakpointReasonNone, Target: target}
}

// bodyHasCacheControlKey reports whether raw carries a cache_control breakpoint anywhere — the
// already-set guard's detector (#3774). The dominant Claude Code body marks its head with a
// LITERAL cache_control, so the byte-literal scan answers first and the hot path stays
// allocation-free. Only when the literal is ABSENT but the body contains a JSON `\u` escape do we
// pay for a semantic re-scan: a client whose key spells an ASCII letter or underscore as \uXXXX
// (e.g. cache_control) decodes to the same cache_control key and would otherwise evade the
// scan, letting placement splice a SECOND breakpoint over the client's layout. We only ever EXTEND
// the scan — a false positive here merely skips placement (fail-safe), never a double-mark.
// cache_control is pure ASCII, so a \uXXXX escape is the sole way to hide it from the literal scan:
// JSON's other escapes (\" \\ \/ \n ...) cannot spell a letter or underscore.
func bodyHasCacheControlKey(raw []byte) bool {
	if bytes.Contains(raw, []byte("cache_control")) {
		return true
	}
	if !bytes.Contains(raw, []byte(`\u`)) {
		return false // no unicode escape — the byte-literal scan above was already exact
	}
	return bytes.Contains(jsonUnescapeASCII(raw), []byte("cache_control"))
}

// jsonUnescapeASCII returns raw with every JSON \uXXXX escape denoting an ASCII byte (code point
// <= 0x7F) replaced by that byte, so an escaped key like cache_control collapses to its
// literal form for the already-set scan. Any other escape (\\ \" \/ \n, or a \uXXXX above 0x7F)
// is consumed and replaced by a single space: a backslash, quote, or non-ASCII rune can never be
// part of the pure-ASCII cache_control token, and consuming BOTH bytes of a two-char escape stops
// an escaped backslash (\\) from being misread as the start of a following \u. This is a scan aid
// for a substring test, not a general JSON unescaper: it does not validate surrogate pairs.
func jsonUnescapeASCII(raw []byte) []byte {
	out := make([]byte, 0, len(raw))
	for i := 0; i < len(raw); {
		c := raw[i]
		if c != '\\' || i+1 >= len(raw) {
			out = append(out, c)
			i++
			continue
		}
		if raw[i+1] == 'u' && i+6 <= len(raw) {
			if cp, ok := parseHex4(raw[i+2 : i+6]); ok {
				if cp <= 0x7F {
					out = append(out, byte(cp))
				} else {
					out = append(out, ' ')
				}
				i += 6
				continue
			}
		}
		// Any other (or malformed) escape: consume both bytes, emit a non-matching placeholder.
		out = append(out, ' ')
		i += 2
	}
	return out
}

// parseHex4 parses exactly four hex digits into a code point; ok is false on any non-hex byte.
func parseHex4(b []byte) (cp int, ok bool) {
	if len(b) != 4 {
		return 0, false
	}
	for _, c := range b {
		cp <<= 4
		switch {
		case c >= '0' && c <= '9':
			cp |= int(c - '0')
		case c >= 'a' && c <= 'f':
			cp |= int(c-'a') + 10
		case c >= 'A' && c <= 'F':
			cp |= int(c-'A') + 10
		default:
			return 0, false
		}
	}
	return cp, true
}

// UpgradeAnthropicStableCacheTTL1h upgrades an EXISTING stable-head cache_control breakpoint
// (system first, then tools) from the default 5-minute ephemeral tier to the 1-hour tier by
// splicing `"ttl":"1h"` into the cache_control object. Message-tail breakpoints are ignored:
// those cache volatile conversation history, not the stable provider head #1850 targets.
//
// The edit is deliberately narrower than placement: it never moves a breakpoint and never
// re-marshals the body. Bytes before the cache_control object are copied verbatim; the only change
// is inside that existing metadata object. On ambiguity, an existing non-1h ttl, or an obviously
// volatile head, the body is returned unchanged. A volatile_head refusal gets ONE spec-governed
// redaction retry (anthropic_cachebp_redact.go, opt-in via FAK_CACHEBP_REDACT); with the lever
// off (the default) the refusal is returned exactly as before.
func UpgradeAnthropicStableCacheTTL1h(raw []byte) ([]byte, TTLUpgradeOutcome) {
	return upgradeAnthropicStableCacheTTL1h(raw, false)
}

// UpgradeAnthropicStableCacheTTL1hWithMessagePrefixes extends the original
// head-only transform across eligible message-prefix breakpoints while preserving
// Anthropic's longer-before-shorter TTL ordering.
func UpgradeAnthropicStableCacheTTL1hWithMessagePrefixes(raw []byte) ([]byte, TTLUpgradeOutcome) {
	return upgradeAnthropicStableCacheTTL1h(raw, true)
}

// UpgradeAnthropicStableCacheTTL1hHeadOnly is the explicit ablation baseline for
// the original managed-cache behavior. It upgrades system/tools breakpoints but
// leaves message-prefix breakpoints on their caller-selected tier.
func UpgradeAnthropicStableCacheTTL1hHeadOnly(raw []byte) ([]byte, TTLUpgradeOutcome) {
	return upgradeAnthropicStableCacheTTL1h(raw, false)
}

func upgradeAnthropicStableCacheTTL1h(raw []byte, includeMessages bool) ([]byte, TTLUpgradeOutcome) {
	out, oc := upgradeAnthropicStableCacheTTL1hOnce(raw, includeMessages)
	if oc.Reason != TTLUpgradeReasonVolatileHead {
		return out, oc
	}
	return retryUpgradeWithRedactedHead(raw, oc, includeMessages)
}

// upgradeAnthropicStableCacheTTL1hOnce is one un-retried upgrade pass — the original edit.
//
// The upgrade is ALL-OR-NOTHING over the head arrays. Anthropic accepts only non-increasing
// TTLs in prefix-processing order (tools → system → messages), so upgrading one stable-head
// breakpoint while an earlier one stays on the default 5m tier is a per-turn HTTP 400
// ("a ttl='1h' cache_control block must not come after a ttl='5m' cache_control block") —
// witnessed live with Claude Code 2.1.x, which marks BOTH the last tool and system blocks
// (#5363). So EVERY tools/system breakpoint is upgraded together; message-tail breakpoints
// stay 5m, which is legal because they FOLLOW the 1h head (descending order). Any refusal —
// a volatile head, an explicit caller ttl on any head breakpoint, a splice ambiguity —
// refuses the WHOLE body (identity), never a partial edit that would 400 upstream.
func upgradeAnthropicStableCacheTTL1hOnce(raw []byte, includeMessages bool) ([]byte, TTLUpgradeOutcome) {
	if len(raw) == 0 {
		return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonNonJSON}
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonNonJSON}
	}

	// One eligible splice target: the absolute span of an existing cache_control object.
	type ttlSplice struct {
		abs int    // absolute offset of the cache_control object value in raw
		cc  []byte // the cache_control object bytes
	}
	var splices []ttlSplice
	marked := 0         // head breakpoints seen (any ttl state)
	already1h := 0      // head breakpoints already on the 1h tier
	primaryTarget := "" // the deepest marked head array — "system" outranks "tools", matching the historical Target label

	toolsVolatile := headValueIsVolatile(obj["tools"])
	for _, head := range []struct {
		key               string
		inheritedVolatile bool
	}{
		// system is scanned first so primaryTarget keeps the historical "deepest anchor
		// wins" label; splice ORDER is by absolute offset below, so scan order is
		// label-only. A volatile tools[] value sits ahead of every system block in the
		// provider's prefix order, so system inherits tools volatility.
		{"system", toolsVolatile},
		{"tools", false},
	} {
		// Anchor the head array by its KEY, never bytes.Index on the value bytes: a head
		// array byte-identical to a message content array would otherwise upgrade the ttl
		// inside the wrong occurrence (#3773).
		elems, spans, ok := decodeTopLevelArray(raw, head.key)
		if !ok || len(elems) == 0 {
			continue
		}
		for i := len(elems) - 1; i >= 0; i-- {
			el := elems[i]
			if !rawHasCacheControl(el) {
				continue
			}
			marked++
			if primaryTarget == "" {
				primaryTarget = head.key
			}
			if head.inheritedVolatile || anyHeadElementVolatile(elems[:i+1]) {
				return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonVolatileHead, Target: head.key}
			}
			ccStart, ccEnd, ok := objectValueSpan(el, "cache_control")
			if !ok {
				return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonSpliceFailed, Target: head.key}
			}
			cc, ttl, ok := parseEphemeralCacheControl(el, ccStart, ccEnd)
			if !ok {
				return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonSpliceFailed, Target: head.key}
			}
			switch ttl {
			case "1h":
				already1h++
			case "":
				splices = append(splices, ttlSplice{abs: spans[i].start + ccStart, cc: cc})
			default:
				// The caller chose a ttl on a head breakpoint; a mixed-tier edit around it
				// could invert the required ordering. Respect the whole layout.
				return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonTTLAlreadySet, Target: head.key}
			}
		}
	}

	// Messages follow the tools/system head in Anthropic's cache-prefix order. A
	// cache_control on a content block therefore marks a stable message prefix:
	// later turns append after it without changing the marked bytes. Upgrade every
	// eligible marked prefix, not merely the last one, because Anthropic requires a
	// longer TTL to precede a shorter TTL. The head scan above either upgraded every
	// earlier head breakpoint or refused the entire edit, so this extension cannot
	// manufacture an invalid 5m-before-1h layout.
	messageSplices := 0
	if messages, messageSpans, ok := decodeTopLevelArray(raw, "messages"); includeMessages && ok {
		for mi, message := range messages {
			contentStart, contentEnd, ok := objectValueSpan(message, "content")
			if !ok {
				continue
			}
			content := message[contentStart:contentEnd]
			blocks, blockSpans, ok := arrayElementSpans(content)
			if !ok {
				continue // string content cannot carry a block cache_control
			}
			for bi, block := range blocks {
				if !rawHasCacheControl(block) {
					continue
				}
				marked++
				primaryTarget = "messages"
				// Refuse when any byte in the prefix through this breakpoint has a
				// self-evident per-request token. This is deliberately the same
				// conservative UUID/timestamp vocabulary used for the head.
				if anyHeadElementVolatile(messages[:mi]) || anyHeadElementVolatile([]json.RawMessage{message[:contentStart+blockSpans[bi].end]}) {
					return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonVolatileMessage, Target: "messages"}
				}
				ccStart, ccEnd, ok := objectValueSpan(block, "cache_control")
				if !ok {
					return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonSpliceFailed, Target: "messages"}
				}
				cc, ttl, ok := parseEphemeralCacheControl(block, ccStart, ccEnd)
				if !ok {
					return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonSpliceFailed, Target: "messages"}
				}
				switch ttl {
				case "1h":
					already1h++
				case "":
					abs := messageSpans[mi].start + contentStart + blockSpans[bi].start + ccStart
					splices = append(splices, ttlSplice{abs: abs, cc: cc})
					messageSplices++
				default:
					return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonTTLAlreadySet, Target: "messages"}
				}
			}
		}
	}

	if marked == 0 {
		return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonNoStableBreakpoint}
	}
	if len(splices) == 0 {
		return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonAlready1h, Target: primaryTarget}
	}
	headSplices := len(splices) - messageSplices
	// Apply back-to-front so earlier absolute offsets stay valid as later bytes grow.
	sort.Slice(splices, func(i, j int) bool { return splices[i].abs > splices[j].abs })
	out := raw
	for _, sp := range splices {
		next, ok := spliceTTL1hIntoObject(out, sp.abs, sp.cc)
		if !ok {
			return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonSpliceFailed, Target: primaryTarget}
		}
		out = next
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		return raw, TTLUpgradeOutcome{Reason: TTLUpgradeReasonRedecodeFail, Target: primaryTarget}
	}
	return out, TTLUpgradeOutcome{Reason: TTLUpgradeReasonNone, Target: primaryTarget, UpgradedHeadBreakpoints: headSplices, UpgradedMessageBreakpoints: messageSplices}
}

func parseEphemeralCacheControl(raw []byte, start, end int) (cc []byte, ttl string, ok bool) {
	cc = raw[start:end]
	var parsed struct {
		Type string `json:"type"`
		TTL  string `json:"ttl"`
	}
	if json.Unmarshal(cc, &parsed) != nil || parsed.Type != "ephemeral" {
		return nil, "", false
	}
	return cc, parsed.TTL, true
}

func anyHeadElementVolatile(elems []json.RawMessage) bool {
	for _, el := range elems {
		if headValueIsVolatile(el) {
			return true
		}
	}
	return false
}

func spliceTTL1hIntoObject(raw []byte, objAbs int, obj []byte) ([]byte, bool) {
	if len(obj) < 2 || obj[0] != '{' || obj[len(obj)-1] != '}' {
		return nil, false
	}
	insert := objAbs + len(obj) - 1
	var b bytes.Buffer
	b.Grow(len(raw) + len(cacheControlTTL1h) + 1)
	b.Write(raw[:insert])
	if objectHasContent(obj) {
		b.WriteByte(',')
	}
	b.WriteString(cacheControlTTL1h)
	b.Write(raw[insert:])
	return b.Bytes(), true
}

func objectHasContent(obj []byte) bool {
	for _, c := range obj[1 : len(obj)-1] {
		if !isJSONSpace(c) {
			return true
		}
	}
	return false
}

type anthropicSystemAnchorPlan struct {
	order          []int
	anchorOriginal int
	rewritten      bool
	recommendation cachemeta.LayoutRecommendation
}

func planAnthropicSystemAnchor(elems []json.RawMessage) (anthropicSystemAnchorPlan, bool) {
	segs := make([]cachemeta.PromptSegment, 0, len(elems))
	nonVol := make([]int, 0, len(elems))
	vol := make([]int, 0, len(elems))
	for i, el := range elems {
		kind := cachemeta.SegStable
		if headValueIsVolatile(el) {
			kind = cachemeta.SegVolatile
			vol = append(vol, i)
		} else {
			nonVol = append(nonVol, i)
		}
		segs = append(segs, cachemeta.PromptSegment{
			Kind:    kind,
			Tokens:  estimatedPromptTokens(el),
			Content: append([]byte(nil), el...),
		})
	}
	if len(nonVol) == 0 {
		return anthropicSystemAnchorPlan{}, false
	}
	order := append(append([]int(nil), nonVol...), vol...)
	return anthropicSystemAnchorPlan{
		order:          order,
		anchorOriginal: nonVol[len(nonVol)-1],
		rewritten:      !sameIntOrder(order),
		recommendation: cachemeta.RecommendLayout(segs),
	}, true
}

func estimatedPromptTokens(raw json.RawMessage) int64 {
	n := int64(len(raw) / 4)
	if n <= 0 {
		return 1
	}
	return n
}

func sameIntOrder(order []int) bool {
	for i, v := range order {
		if i != v {
			return false
		}
	}
	return true
}

func placeCacheControlAtSpan(raw []byte, span elementSpan) ([]byte, bool) {
	spliced, ok := spliceCacheControlIntoObject(raw[span.start:span.end])
	if !ok {
		return nil, false
	}
	var b bytes.Buffer
	b.Grow(len(raw) + len(spliced) - (span.end - span.start))
	b.Write(raw[:span.start])
	b.Write(spliced)
	b.Write(raw[span.end:])
	return b.Bytes(), true
}

// placeAndValidateAtSpan splices a cache_control breakpoint onto the block at span and PROVES the
// result three ways, returning BreakpointReasonNone (with the rewritten bytes) only when all hold:
// the splice must succeed (else BreakpointReasonSpliceFailed), the body must re-decode as a valid
// Messages request (else BreakpointReasonRedecodeFail), and every byte BEFORE the block must be
// byte-identical to the input — the cache prefix upstream of the new breakpoint is untouched (else
// BreakpointReasonRedecodeFail). On any failure the INPUT bytes are returned unchanged alongside
// the labeled reason. It is the shared placement-and-proof step the stable system-anchor and the
// tools-anchor placements both close on (the rewritten-system path proves differently — it MOVES
// bytes, so it cannot assert the byte-identical-prefix invariant — and stays inline).
func placeAndValidateAtSpan(raw []byte, span elementSpan) ([]byte, string) {
	out, ok := placeCacheControlAtSpan(raw, span)
	if !ok {
		return raw, BreakpointReasonSpliceFailed
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		return raw, BreakpointReasonRedecodeFail
	}
	if !bytes.Equal(raw[:span.start], out[:span.start]) {
		return raw, BreakpointReasonRedecodeFail
	}
	return out, BreakpointReasonNone
}

func rewriteSystemArrayWithBreakpoint(raw []byte, elems []json.RawMessage, plan anthropicSystemAnchorPlan) ([]byte, bool) {
	// Locate the system value by KEY, not bytes.Index on its bytes: a system array byte-identical
	// to a message content array would otherwise rewrite the wrong occurrence (#3773).
	start, end, ok := objectValueSpanLastWins(raw, "system")
	if !ok {
		return nil, false
	}
	var sys bytes.Buffer
	sys.WriteByte('[')
	for pos, idx := range plan.order {
		if pos > 0 {
			sys.WriteByte(',')
		}
		el := []byte(elems[idx])
		if idx == plan.anchorOriginal {
			spliced, ok := spliceCacheControlIntoObject(el)
			if !ok {
				return nil, false
			}
			el = spliced
		}
		sys.Write(el)
	}
	sys.WriteByte(']')

	var out bytes.Buffer
	out.Grow(len(raw) + sys.Len() - (end - start))
	out.Write(raw[:start])
	out.Write(sys.Bytes())
	out.Write(raw[end:])
	return out.Bytes(), true
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

// HeadValueIsVolatile is headValueIsVolatile exported for the TOON wire (#3067): the
// gateway sources toon.Decide's Volatile signal from THIS check — the same per-request-
// token evidence (UUID/nonce, sub-day timestamp) the breakpoint planner above uses —
// so the VOLATILE_SPAN skip is wired to the real volatility state, not a re-implementation.
func HeadValueIsVolatile(v json.RawMessage) bool { return headValueIsVolatile(v) }

// ClassifyVolatileHead is the NAMED counterpart to HeadValueIsVolatile (#3341): where the
// bool only says a cache-prefix head IS volatile, this returns the per-CLASS diagnosis
// (uuid / iso8601 / jwt / hex_hash counts) plus an operator warning line, so a silently
// collapsed cache hit-rate surfaces as WHICH volatile class sits in the system prompt
// rather than a bare `volatile_head` counter — and it names JWTs and hex hashes, which the
// bool check misses. It scans the same raw head bytes and is read-only: it never rewrites
// the head (the M2 anchor still owns reordering). An empty/absent value yields an empty,
// stable report.
func ClassifyVolatileHead(v json.RawMessage) cachemeta.VolatileReport {
	return cachemeta.ClassifyVolatile(v)
}
