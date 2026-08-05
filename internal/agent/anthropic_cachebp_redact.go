package agent

// anthropic_cachebp_redact.go — the volatile-head REDACTION spec + engine (#2191, #2181):
// the "aggressive form" of #806 bullet 2 that anthropic_cachebp.go deferred behind "needs a
// redaction spec + soak". It converts the volatile_head dead-end — a system/tools head whose
// per-request token (a UUID, a sub-day timestamp) permanently refuses breakpoint placement and
// the managed-cache TTL upgrade — into an upgradable condition, by normalizing exactly those
// tokens to byte-stable placeholders and re-running the refused transform on the stable head.
//
// # The redaction spec
//
// Vocabulary rule: the redaction vocabulary IS the refusal vocabulary. A token class may be
// normalized here if and only if headValueIsVolatile refuses on it — the engine reuses the
// SAME compiled patterns (volUUID, volDateTime), never a re-implementation, so the two can
// not drift (the #3774 detector-parity discipline). Classes cachemeta.ClassifyVolatile can
// NAME but the refusal check does not act on (jwt, hex_hash) are therefore out of scope on
// both sides: they never blocked a cache, so "converting" them would change model-visible
// bytes with zero cache benefit. A class enters this vocabulary only by first entering the
// refusal vocabulary.
//
// The closed vocabulary, and each class's exact normalization:
//
//   - uuid — a canonical UUID/GUID (8-4-4-4-12 hex, volUUID). Replaced in full by the fixed
//     placeholder "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx": same length and shape, self-evidently
//     a redaction (never plausible-but-wrong data), and deliberately NON-hex so the detector
//     does not re-match it (idempotence by construction).
//   - iso8601 sub-day — an ISO-8601 date carrying a time-of-day (volDateTime), extended right
//     to consume any trailing seconds, fractional seconds, and zone suffix (redactDateTime is
//     a strict right-extension of volDateTime: it fires exactly where the detector fires, and
//     additionally swallows the sub-minute bytes the detector's match would leave behind as
//     residual per-request noise). The DATE and its original [Tt ] separator are preserved —
//     they are byte-stable across a session's turns and are the "Today's date is ..." content
//     we want cached — and the sub-day remainder becomes the placeholder "hh:mm". A date-ONLY
//     token is not volatile and is never touched.
//
// Semantic identity: a placeholder is self-evidently redacted, never a plausible wrong value
// (a zeroed "00:00" would be misinformation; "hh:mm" is visibly a redaction). The only bytes
// changed are per-request tokens whose VALUE the model cannot meaningfully rely on across a
// cached prefix anyway — the same bytes that were already busting the prefix every turn.
//
// Where it applies (prefer hoist over strip, #2181): redaction runs ONLY as a retry after the
// existing pipeline refuses volatile_head. The M2 volatile-hoist (planAnthropicSystemAnchor)
// keeps first claim on placement — a head with at least one stable block anchors or hoists
// WITHOUT redaction, so tokens that can be moved out of the cached span are moved, never
// stripped. Redaction converts only the residue: a volatile tools value (which no hoist can
// help — tools sit ahead of every system anchor in the provider's positional prefix), or a
// system head with no stable block at all, or a client-placed breakpoint whose covered span
// carries a volatile token (the TTL-upgrade refusal). Both top-level head values ("system",
// "tools") that the detector flags are normalized uniformly: the head is one candidate cache
// prefix, and a uniformly stable head lets the retry anchor the MAXIMAL span.
//
// The refusal vocabulary (what stays a dead-end, closed):
//
//   - redact_disabled — the lever is off (the default posture; see below).
//   - stable_head — nothing in the head is volatile; there is nothing to convert (identity).
//   - no_head — the body is not a JSON object, or a flagged head value cannot be located by
//     the same last-wins key walk the decoder uses (identity; never guess at spans).
//   - residual_volatile — the normalized head STILL trips headValueIsVolatile (identity; the
//     belt-and-braces post-proof that vocabulary parity actually held on this body).
//   - redecode_failed — the normalized body no longer decodes as a Messages request
//     (identity; a redaction may only ever produce a body at least as well-formed).
//   - redact_unconverted — redaction produced a stable head but the retried transform still
//     refused (identity; the ORIGINAL refusal stands, witnessed).
//
// Byte-safety bar (same as the TTL splice): identity on ANY ambiguity, and every decision
// witnessed. The engine proves, before returning a rewritten body: bytes outside the flagged
// head-value spans are byte-identical by construction (the spans come from the last-wins key
// walk of #3773, so they are the SAME value bytes the decoder keeps); the result re-decodes
// as a valid Messages request; and the decoder-visible head is now detector-stable. The
// rewrite is idempotent (placeholders never re-match) and deterministic: two turns differing
// only in their volatile token VALUES normalize to IDENTICAL head bytes — which is precisely
// what makes the provider prefix cacheable again. (The date byte kept in the iso8601 rule
// rolls once per day: one cache miss at midnight, by design.)
//
// Default posture: OFF. The lever is FAK_CACHEBP_REDACT (mirroring the FAK_CTXPLAN_SEAM
// pattern; distinct from FAK_WIRE_REDACT, the compliance/PII wirescreen redactor — THAT one
// holds originals in the CAS for authorized restore, this one drops per-request noise for
// cache stability and restores nothing). Default-on must be earned by the soak + ablation row
// the issues require (#2191 acceptance 3, #2181 shadow-mode soak); until then every
// conversion is opt-in and per-attempt witnessed via the Redacted*/RedactReason fields on
// BreakpointOutcome and TTLUpgradeOutcome.

import (
	"bytes"
	"encoding/json"
	"os"
	"regexp"
	"sort"
	"strings"
)

// CacheBPRedactEnvVar is the opt-in lever for volatile-head redaction. Unset/anything else
// leaves the transforms exactly as they were: volatile_head remains a labeled identity bail.
const CacheBPRedactEnvVar = "FAK_CACHEBP_REDACT"

// cacheBPRedactEnabled reports whether the operator armed the redaction retry. Read at the
// refusal site only, so the default path never pays the env lookup.
func cacheBPRedactEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(CacheBPRedactEnvVar))) {
	case "1", "on", "true":
		return true
	}
	return false
}

// Redaction bail vocabulary — closed, mirroring the BreakpointReason*/TTLUpgradeReason*
// discipline. RedactReasonNone means the head was normalized (the body was rewritten); every
// other value means identity.
const (
	RedactReasonNone         = ""                   // NORMALIZED: volatile head tokens replaced by stable placeholders
	RedactReasonDisabled     = "redact_disabled"    // the FAK_CACHEBP_REDACT lever is off
	RedactReasonStableHead   = "stable_head"        // nothing volatile in the head — nothing to convert
	RedactReasonNoHead       = "no_head"            // not a JSON object, or a flagged head value span cannot be located
	RedactReasonResidual     = "residual_volatile"  // post-proof: the normalized head still trips the detector
	RedactReasonRedecodeFail = "redecode_failed"    // post-proof: the normalized body no longer decodes as a request
	RedactReasonUnconverted  = "redact_unconverted" // redaction held but the retried transform still refused
)

// RedactOutcome is the witness of one redaction attempt: what was normalized (per class), or
// why the body was left alone.
type RedactOutcome struct {
	Reason            string
	RedactedUUID      int // uuid-class tokens replaced
	RedactedTimestamp int // iso8601-sub-day-class tokens replaced
}

// redactDateTime is the iso8601-class REPLACER pattern: volDateTime (the detector) extended
// right to also consume optional seconds, fractional seconds, and a zone suffix. It fires at
// exactly the detector's match sites (same mandatory prefix) — the extension only widens the
// match so the sub-minute bytes cannot survive as residual per-request noise.
var redactDateTime = regexp.MustCompile(
	`[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt ][0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]+)?)?(?:[Zz]|[+-][0-9]{2}:?[0-9]{2})?`)

const (
	redactedUUIDPlaceholder = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" // non-hex: the detector cannot re-match it
	redactedTimePlaceholder = "hh:mm"                                // non-digit: volDateTime cannot re-match it
)

// redactVolatileHead normalizes every volatile top-level head value ("system", "tools") of an
// Anthropic Messages body to the spec's stable placeholders. It returns the input UNCHANGED
// with a labeled reason on any ambiguity (see the RedactReason* vocabulary); RedactReasonNone
// means the returned body carries a detector-stable head, re-decodes as a valid request, and
// is byte-identical to the input outside the flagged head-value spans.
func redactVolatileHead(raw []byte) ([]byte, RedactOutcome) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, RedactOutcome{Reason: RedactReasonNoHead}
	}
	var targets []elementSpan
	for _, key := range []string{"system", "tools"} {
		if !headValueIsVolatile(obj[key]) {
			continue
		}
		// Locate the value by the SAME last-wins key walk the decoder's map semantics keep
		// (#3773): on a duplicate key we must normalize the value the planner reasons about.
		start, end, ok := objectValueSpanLastWins(raw, key)
		if !ok {
			return raw, RedactOutcome{Reason: RedactReasonNoHead}
		}
		targets = append(targets, elementSpan{start: start, end: end})
	}
	if len(targets) == 0 {
		return raw, RedactOutcome{Reason: RedactReasonStableHead}
	}
	sort.Slice(targets, func(i, j int) bool { return targets[i].start < targets[j].start })

	var oc RedactOutcome
	var b bytes.Buffer
	b.Grow(len(raw))
	last := 0
	for _, t := range targets {
		if t.start < last || t.end > len(raw) {
			return raw, RedactOutcome{Reason: RedactReasonNoHead} // overlapping/out-of-range spans: never guess
		}
		b.Write(raw[last:t.start])
		b.Write(redactVolatileTokens(raw[t.start:t.end], &oc))
		last = t.end
	}
	b.Write(raw[last:])
	out := b.Bytes()

	// Post-proof, on the DECODER's view of the rewritten body: the head must now be stable,
	// and the body must still decode as a Messages request. Any failure is identity.
	var obj2 map[string]json.RawMessage
	if err := json.Unmarshal(out, &obj2); err != nil {
		return raw, RedactOutcome{Reason: RedactReasonRedecodeFail}
	}
	if headValueIsVolatile(obj2["system"]) || headValueIsVolatile(obj2["tools"]) {
		return raw, RedactOutcome{Reason: RedactReasonResidual}
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		return raw, RedactOutcome{Reason: RedactReasonRedecodeFail}
	}
	return out, oc
}

// redactVolatileTokens applies the spec's per-class normalizations to one head-value span,
// counting replacements into oc. Timestamps are normalized before UUIDs so the counts stay
// legible; the classes cannot overlap (a UUID has no ':', a timestamp no hex letters), so the
// order does not change the output.
func redactVolatileTokens(v []byte, oc *RedactOutcome) []byte {
	out := redactDateTime.ReplaceAllFunc(v, func(m []byte) []byte {
		oc.RedactedTimestamp++
		// m opens with the 10 date bytes and the original [Tt ] separator — both byte-stable
		// across a session — then the sub-day remainder this class exists to drop.
		return append(append(make([]byte, 0, 11+len(redactedTimePlaceholder)), m[:11]...), redactedTimePlaceholder...)
	})
	out = volUUID.ReplaceAllFunc(out, func([]byte) []byte {
		oc.RedactedUUID++
		return []byte(redactedUUIDPlaceholder)
	})
	return out
}

// retryPlaceWithRedactedHead is the placement half of the conversion: called only after
// placeAnthropicCacheBreakpointOnce refused volatile_head, it normalizes the head per the
// spec and retries placement on the stable bytes. The identity fallback ALWAYS returns the
// caller's original refusal (witnessed via RedactReason) — a failed conversion never ships a
// half-redacted body.
func retryPlaceWithRedactedHead(raw []byte, refused BreakpointOutcome) ([]byte, BreakpointOutcome) {
	if !cacheBPRedactEnabled() {
		refused.RedactReason = RedactReasonDisabled
		return raw, refused
	}
	red, roc := redactVolatileHead(raw)
	if roc.Reason != RedactReasonNone {
		refused.RedactReason = roc.Reason
		return raw, refused
	}
	out, oc := placeAnthropicCacheBreakpointOnce(red)
	if oc.Reason != BreakpointReasonNone {
		refused.RedactReason = RedactReasonUnconverted
		return raw, refused
	}
	oc.Redacted = true
	oc.RedactedUUID = roc.RedactedUUID
	oc.RedactedTimestamp = roc.RedactedTimestamp
	return out, oc
}

// retryUpgradeWithRedactedHead is the TTL-upgrade half: called only after
// upgradeAnthropicStableCacheTTL1hOnce refused volatile_head, it normalizes the head and
// retries the upgrade — converting the permanent managed-cache dead-end (#2191) into a
// witnessed 1h-tier upgrade on the stabilized head. Same identity discipline as placement.
func retryUpgradeWithRedactedHead(raw []byte, refused TTLUpgradeOutcome, includeMessages bool) ([]byte, TTLUpgradeOutcome) {
	if !cacheBPRedactEnabled() {
		refused.RedactReason = RedactReasonDisabled
		return raw, refused
	}
	red, roc := redactVolatileHead(raw)
	if roc.Reason != RedactReasonNone {
		refused.RedactReason = roc.Reason
		return raw, refused
	}
	out, oc := upgradeAnthropicStableCacheTTL1hOnce(red, includeMessages)
	if oc.Reason != TTLUpgradeReasonNone {
		refused.RedactReason = RedactReasonUnconverted
		return raw, refused
	}
	oc.Redacted = true
	oc.RedactedUUID = roc.RedactedUUID
	oc.RedactedTimestamp = roc.RedactedTimestamp
	return out, oc
}
