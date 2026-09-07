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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// compactStubPrefix marks the synthetic message that stands in for the dropped turns, so
// the model (and a human reading the wire) sees that earlier turns were compacted rather
// than silently lost. It is emitted as a user-role text message between the protected
// prefix and the kept recent window.
const compactStubPrefix = "[fak] compacted "

// compactGoalMarker is the wire sentinel that PINS a message across compaction: a message in the
// compactible middle whose text content carries this marker is HOISTED out of the drop range
// verbatim rather than laundered into the stub, so a session's standing goal / instruction
// survives a compaction it would otherwise fall into. It is the byte-level passthrough counterpart
// of the decoded planner's RoleGoal GC-root pin (chat.go RoleGoal, ctxplan_session.go): the
// flagship `fak guard -- claude` passthrough forwards raw bytes and never sees the decoded
// RoleGoal, so on this surface the goal is marked in the message TEXT instead. A host injects it
// (e.g. a harness /goal that prefixes the instruction) or a user types it at the start of a
// standing instruction. Absent from every message, compaction is byte-for-byte unchanged.
const compactGoalMarker = "[fak:goal]"

// compactTombstonePrefix leads the excerpt line the stub carries when this compaction is about to
// drop the session's ORIGINATING task (the first user turn) and no [fak:goal] pin covers it. It is
// the automatic, lossy counterpart to the verbatim goal pin: even an UNMARKED task leaves a trace
// in the stub instead of being laundered into a bare turn count (the model-switch symptom where a
// resuming model finds only "[fak] compacted N earlier turn(s) ... detail is omitted" and no longer
// knows what it was asked to do). It is kept DISTINCT from compactStubPrefix so an operator (or a
// test) can tell the count sentinel from the task excerpt, and so it does not read as a second
// "[fak] compacted " match. Because it sits in the stub CONTENT — after the protected prefix — the
// cached prefix bytes are untouched; the tombstone only widens the un-cached span the provider
// re-bills anyway.
const compactTombstonePrefix = "[fak] originating task (compacted): "

// compactTombstoneCap bounds the originating-task excerpt (in runes) so the tombstone stays
// low-volume — it records WHAT the dropped context was, not a replay of it. Long tasks are trimmed
// with an ellipsis; the point is orientation, not fidelity (the verbatim [fak:goal] pin is the
// fidelity path).
const compactTombstoneCap = 240

// compactMediaTombstone is the generic excerpt for a dropped originating turn that carries an image
// (or other non-text media) instead of text. There is nothing to excerpt, but the turn must still
// leave an orientation line and — via the accompanying bytes — a fak_context_restore handle, so a
// resuming model knows the session opened with a media turn it can page back in, rather than seeing
// only a bare turn count. It stands where the text excerpt would; the id=<hex> handle is appended
// by compactStubContent exactly as for a text tombstone.
const compactMediaTombstone = "[media turn — image or non-text content]"

// compactRestoreIDField is the token in the tombstone stub that carries the CALLABLE handle for the
// dropped originating task: a content-address (sha256 hex, the ctxplan.Digest scheme) a resuming
// model can present to fak_context_restore to page the full task bytes back in. The lossy excerpt is
// orientation; this handle is the recovery edge — the excerpt says WHAT was dropped, the id says how
// to get ALL of it. It is only emitted when the compaction path can back the handle with the bytes
// (the gateway stashes digest→bytes on a fired tombstone); a bare byte-level caller with no CAS to
// populate leaves it empty and the stub stays excerpt-only. Kept as a labelled `id=<hex>` field
// inside the tombstone line (not a second line) so it rides the same low-volume, cache-untouched
// stub and a human/test can read the handle without parsing.
const compactRestoreIDField = "id="

// originatingTaskDigestID content-addresses the originating-task bytes with the SAME sha256-hex
// scheme as ctxplan.Digest / recall / blob / memq, so a compaction-minted handle and a
// ctxplan/recall handle are interchangeable recovery addresses. Kept local (a two-line helper) so
// the byte-level agent package depends on no mechanism package — the address is a pure function of
// the bytes, computable here with nothing but crypto/sha256.
func originatingTaskDigestID(taskBytes []byte) string {
	sum := sha256.Sum256(taskBytes)
	return hex.EncodeToString(sum[:])
}

// Compaction bail-reason vocabulary — the closed set of identity-return causes, surfaced on
// CompactOutcome so the gateway can label a metric and an operator can see WHY compaction did
// nothing (silence must not read as success). CompactReasonNone means the body was rewritten.
const (
	CompactReasonNone        = ""             // FIRED: a rewrite happened (Dropped/ShedTokens meaningful)
	CompactReasonUnderBudget = "under_budget" // budget<=0, or the compactible suffix already fits
	CompactReasonNonJSON     = "non_json"     // body is not a JSON object
	CompactReasonNoMsgsKey   = "no_messages_key"
	CompactReasonTooFewMsgs  = "too_few_msgs" // < minElems messages — nothing safe to drop (benign, high-volume)
	// CompactReasonDecodeFailed is the STRUCTURAL messages[] failure: the key is present but its
	// value does not decode as a JSON array of elements (decodeArrayElements returned ok=false).
	// It is deliberately NOT folded into too_few_msgs: that bucket is the benign short-request
	// idle and is expected to be large, so a structural failure counted there raises no suspicion.
	// Split out, decode_failed>0 is assertable as fak-fault the way prefix_mismatch>0 already is.
	// On well-formed traffic it is close to unreachable by construction — msgsRaw comes from the
	// json.Unmarshal of the same raw one line above, so the bytes.Index base cannot miss and the
	// document is already proven valid JSON; the one live path is a client sending `messages` as a
	// non-array (null, an object). This is attribution hygiene for defensive code, not a live fault.
	CompactReasonDecodeFailed   = "decode_failed"
	CompactReasonNoBreakpoint   = "no_breakpoint"  // no cache_control to anchor the protected prefix
	CompactReasonCachedSpan     = "cached_span"    // candidate drop would delete cache_control-marked history
	CompactReasonWindowNoDrop   = "window_no_drop" // the kept window swallowed the whole suffix
	CompactReasonSpliceFailed   = "splice_failed"
	CompactReasonRedecodeFail   = "redecode_failed" // the spliced body failed to re-decode
	CompactReasonPrefixMismatch = "prefix_mismatch" // the splice changed the protected prefix bytes
	CompactReasonMalformedBody  = "malformed_body"  // the spliced body decodes for fak but is Anthropic-invalid (empty text/content) → would 400
	// CompactReasonBurstUnprofitable is the head-anchored bail (CompactAnchorHead only): the drop
	// would fire, but bursting the recent breakpoint's cached suffix does not repay within the
	// remaining session horizon (CacheBurstPaysBack == false), so the warm cache hit is kept over a
	// smaller prompt. The firstbp default never returns this (it never bursts) — see #1407/#1408.
	CompactReasonBurstUnprofitable = "burst_unprofitable"
	// CompactReasonPinEvictRefused is the SURVIVAL-CLASS refusal (#2421): the compaction would
	// have evicted a page whose kind classes it PINNED (the active steer, the live continuation
	// seed, a standing system invariant — ctxplan.ClassPinned), so the body is forwarded UNCHANGED
	// rather than compacted lossily. It is the one bail whose cause is a CONTRACT rather than an
	// economics or a structural limit: every other reason says "the drop was not worth it" or "the
	// drop could not be built", while this one says "the drop was refused".
	//
	// Two properties are deliberate. It is emitted by the GATEWAY's compaction path
	// (compactAnthropicRawWithReason), which owns the page classification, not by this package's
	// byte splicer — it is registered here because this is the package that OWNS the bail
	// vocabulary the gateway's metric labels and Prometheus HELP enumerate. And its token is
	// SCREAMING_CASE where its siblings are lower_snake, because it is the same string the repo's
	// refusal vocabulary registers (dos.toml [reasons.PIN_EVICT_REFUSED]) and the same string the
	// planner returns (ctxplan.ReasonPinEvictRefused): one token from planner to wire to operator,
	// with no translation table in between to drift.
	CompactReasonPinEvictRefused = "PIN_EVICT_REFUSED"
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
	ProtectedPrefixTokens      int
	SuffixTokens               int
	InducedCacheCreationTokens int `json:"induced_cache_creation_tokens,omitempty"`
	AnchorStarved              bool
	// Restore handle for a tombstoned originating task. On a FIRED compaction that drops the
	// session's first user turn (the automatic tombstone path — see originatingTaskExcerptAndBytes),
	// RestoreID is the content-address (sha256 hex, the ctxplan.Digest scheme) embedded in the stub,
	// and RestoreBytes is the FULL raw JSON of that dropped turn. A gateway with a per-session CAS
	// stashes RestoreID→RestoreBytes so fak_context_restore(id) can page the task back in; a
	// byte-level caller with no CAS ignores them and the stub embeds no id (compactStubContent leaves
	// the handle out when the caller passes an empty id). All are zero on every non-tombstone
	// outcome — the goal-pin path preserves the task verbatim and mints no handle. RestoreExcerpt is
	// the same bounded orientation line embedded in the stub, carried alongside so a stashing gateway
	// need not re-derive it from the bytes.
	RestoreID      string
	RestoreExcerpt string
	RestoreBytes   []byte

	PositiveResidue        string
	ResidueRestoreID       string
	ResidueRestoreBytes    []byte
	ResidueBytesDropped    int
	PositiveAssertionsKept int
	// SolvencyForced marks a FIRED head-anchored compaction that the cache economics REFUSED and
	// the context-solvency override fired anyway (see CompactOptions.SolvencyFloorTokens). It is
	// a deliberately unprofitable burst — bounded one-time cost, paid to keep the session inside
	// its window — so an operator (and the cache-value ledger) must not read it as a profitable
	// fire. False on every economics-approved fire and on every bail.
	SolvencyForced bool
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
	// system/tools cache_control breakpoint, wherever it serializes (real Claude Code puts it
	// AFTER messages[]; the provider cache is keyed on the semantic tools→system→messages
	// hierarchy, not JSON key order) — making the WHOLE message array compactible. This is what
	// lets compaction fire on real traffic (#1407), but a fire bursts the recent message
	// breakpoint's cached suffix, so it is gated on CacheBurstPaysBack economics (#1408): it only
	// fires when a known session horizon repays the burst, or when the caller OBSERVED the
	// suffix's cache already cold (CompactOptions.ColdCache — a zero-penalty burst).
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
	// ColdCache: the caller OBSERVED that this session's message-span cache entries have already
	// expired (e.g. the trace idled past the provider's message-breakpoint TTL since its last
	// served turn). An expired suffix re-bills cold this turn whether or not we compact, so the
	// head-anchored burst gate prices the one-time invalidation at ZERO and can fire without a
	// session horizon — the exact cold case #1407 says the lever was built for. Never set this
	// from a guess: a false cold claim converts a warm cache read into a cold re-write.
	ColdCache bool

	// RestoreStash optionally receives content-addressed drops for CAS storage.
	RestoreStash func(id, excerpt string, body []byte)

	// PositiveResidue opts into conservative positive-state extraction. It is off by default.
	PositiveResidue bool
	// MinHorizonMargin is the fed-back fire/bail threshold (#2817): the EXTRA predicted headroom,
	// in future turns, the head-anchored burst must clear OVER its break-even before firing. The
	// gate fires iff remainingTurns >= breakEven + MinHorizonMargin, so a positive margin bails the
	// thin-headroom fires whose realized net most often goes negative when the session ends earlier
	// than predicted. The zero value is today's plain gate (fire whenever the burst pays back at
	// all), keeping the default firing path byte-for-byte unchanged. Its value is learned OFFLINE by
	// rsiloop.TuneFirePolicy over a corpus of scored per-fire receipts (rsiloop.CompactionFireObs →
	// ScoreCompactionFire) and fed back here; it gates ONLY on this ex-ante horizon feature, never on
	// the ex-post net (feeding the net back would be circular). Negative values are treated as 0. It
	// does NOT relax the penalty-free short-circuit: a burst with no one-time penalty (breakEven 0,
	// e.g. ColdCache) still fires horizon-free regardless of the margin, since there is no estimation
	// error to hedge.
	MinHorizonMargin int

	// Context-solvency override — the OCCUPANCY axis of the head-anchored burst gate, and the
	// answer to "compaction is enabled, fires, and the window still fills up".
	//
	// CacheBurstPaysBack prices a fire in CACHE DOLLARS: fire only when the per-turn read saving
	// repays the one-time cold re-write within the remaining horizon. That objective function has
	// no term for RUNNING OUT OF WINDOW, so the gate refuses hardest exactly where refusing is
	// most expensive. Measured over 3191 real served turns in .dispatch-runs (224 traces), the
	// fire rate INVERTS against occupancy — 33.4% at 96-110k, 33.9% at 110-125k, 24.7% at
	// 125-140k, 14.3% at 140-155k, 3.4% at 155-170k, 0.0% above 170k — because breakEven ≈ 11.5 ×
	// (invalidatedSuffix / droppedMiddle) degrades monotonically as a session deepens (Claude
	// Code's last breakpoint sits ever further toward the tail, growing the invalidated span
	// faster than the droppable middle). The result is a ONE-WAY LATCH: of the traces that ever
	// fired, 100% never fired again, running a median 9-turn (max 16) un-compacted tail over
	// which resident rose a median +33.8k (max +53.3k) — straight into PROMPT_TOO_LONG.
	//
	// A burst that never repays in dollars still pays for itself if it keeps the session alive:
	// the penalty is ONE-TIME and bounded (a cold re-write of the invalidated suffix) while
	// hitting the context wall costs the whole session. So above a floor the gate stops asking
	// whether the burst is PROFITABLE and asks only whether it is NECESSARY.
	//
	// ResidentTokens is the caller's OBSERVED resident window occupancy for this trace (the same
	// input+cached currency the coordinator meters); SolvencyFloorTokens is the occupancy at or
	// above which solvency overrides the economics. BOTH must be positive to arm the override —
	// either one unset leaves the gate byte-for-byte on its pure-economics behavior, so every
	// existing caller, ablation row and test is unchanged. The override can ONLY convert a
	// burst_unprofitable BAIL into a fire: it never suppresses a fire, never fires below the
	// budget line (it sits downstream of the under_budget bail), and relaxes no correctness guard
	// — role alternation, the orphaned-tool_result guards, the cached-span refusal and the splice
	// verification all still run and still fail safe to identity. A forced fire is reported as
	// CompactOutcome.SolvencyForced so it is never mistaken for a profitable one.
	ResidentTokens      int
	SolvencyFloorTokens int
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
// provider head — a top-level system/tools cache_control breakpoint — exists ANYWHERE in the
// object, pfxEnd is forced to -1 so the WHOLE message array is compactible (#1407/#1408). The
// head's serialization position does not matter: the provider's prompt cache is keyed on the
// SEMANTIC prompt hierarchy (tools → system → messages), never on the raw JSON key order, and the
// splice rewrites bytes only inside the messages[] array span — verifySplicedBody proves both the
// pre-messages prefix AND the post-messages tail survive verbatim. (An earlier serialized-
// before-messages engagement condition made head mode unreachable on real Claude Code traffic,
// which serializes messages[] BEFORE system/tools — live capture 2026-07-02.) The firstbp
// condition below is left byte-identical to the pre-#1408 behavior.
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
	if !decoded {
		// Structural: `messages` is present but is not a decodable JSON array. Distinct from the
		// benign short-request bail below — see CompactReasonDecodeFailed.
		return nil, nil, 0, CompactOutcome{Reason: CompactReasonDecodeFailed}, false
	}
	if len(elems) < minElems {
		return nil, nil, 0, CompactOutcome{Reason: CompactReasonTooFewMsgs}, false // nothing safe to compact
	}
	pfxEnd = firstBreakpointMessage(elems)
	headReanchored := false
	if anchor == CompactAnchorHead && stableHeadMarked(obj) {
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

// stableHeadMarked reports whether the request carries a stable provider-head cache_control
// breakpoint — a top-level system/tools block the provider reuses every turn. The head's byte
// POSITION in the request object deliberately does not gate engagement: the provider's prompt
// cache is keyed on the semantic prompt hierarchy (tools → system → messages), not on the raw
// JSON key order, so a marked head anchors a head-mode fire wherever it happens to serialize.
// Byte safety is proven separately — the splice rewrites bytes only inside the messages[] array
// span, and verifySplicedBody proves the pre-messages prefix AND the post-messages tail (which
// carries system/tools on real Claude Code traffic, whose key order is messages BEFORE
// system/tools — live capture 2026-07-02) survive verbatim. The earlier serialized-before-
// messages condition made head mode structurally unreachable on that flagship traffic (#1407).
func stableHeadMarked(obj map[string]json.RawMessage) bool {
	return rawHasCacheControl(obj["system"]) || rawHasCacheControl(obj["tools"])
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
// caller relies on. With CompactAnchorHead it re-anchors on the stable system/tools head (wherever
// it serializes — see stableHeadMarked), making the whole message array compactible so the lever
// can fire on real Claude Code traffic (#1407); because such a fire bursts the recent breakpoint's
// cached suffix, it is gated on CacheBurstPaysBack economics and only fires when the burst repays
// within the session horizon, or costs nothing because the caller observed that cache already cold
// (#1408, CompactOptions.ColdCache). All byte-level guarantees (verbatim protected prefix + body
// tail, re-decode proof, fail-safe identity on any ambiguity) are identical across both anchors.
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
	suffixTokens := estimatedElementTokens(elems, pfxEnd+1, len(elems))
	if suffixTokens <= budget {
		prefixTokens := estimatedElementTokens(elems, 0, pfxEnd+1)
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

	// Goal pin (#845, the passthrough half): if a goal-marked message sits in the compactible
	// middle, hoist it out verbatim so the drop cannot launder the session's active goal. Absent
	// any goal marker this is a no-op — lastGoalPinInRange returns -1 and the contiguous path below
	// runs byte-for-byte unchanged (the "no goal ⇒ no change" invariant, mirroring pins()).
	if lastGoalPinInRange(elems, pfxEnd+1, keepStart) >= 0 {
		return compactWithGoalPin(raw, elems, spans, pfxEnd, keepStart, opts, suffixTokens)
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
	// Stub role: the opposite of the prefix's last message role, so it alternates with its LEFT
	// neighbor. When pfxEnd<0 the stub has NO left neighbor — it LEADS the array — so it must be a
	// user turn (Anthropic 400s on a leading non-user message). prefixLastRole=="" leaves stubRole at
	// its "user" default, which is exactly the leading-user requirement; the snap below then forces
	// the kept window head to alternate (an assistant turn) on the stub's RIGHT. (Picking the stub
	// role to alternate with the kept head instead — the old system-only branch — could seat a
	// leading ASSISTANT stub whenever the kept head was a user turn, an invalid body.)
	stubRole := "user"
	if prefixLastRole == "user" {
		stubRole = "assistant"
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
	// The head guards above only inspect the window BOUNDARY. They miss the interior orphan that
	// produced the live "400 upstream rejected as malformed" failures: a tool_result deeper in the
	// kept window whose matching assistant tool_use is in the middle we are about to drop. Anthropic
	// 400s any body with a tool_result that has no preceding tool_use. Checked here, AFTER the
	// alternation snap has settled keepStart, because the snap can move the boundary and strand a
	// pair the pre-snap window did not. Fail safe to identity — compaction re-fires next turn.
	if keptWindowOrphansToolUse(elems, pfxEnd+1, keepStart) {
		return raw, CompactOutcome{Reason: CompactReasonWindowNoDrop} // would orphan an interior tool_result — fail safe
	}
	if rangeHasCacheControl(elems, pfxEnd+1, keepStart) {
		// A cache_control-bearing message is provider-warm history. Dropping it may shrink
		// the prompt, but it also intentionally bursts the cached suffix after the first
		// changed byte. Without an explicit horizon/economics gate, the conservative action is
		// identity: keep the provider's cache hit over a smaller prompt.
		return raw, CompactOutcome{Reason: CompactReasonCachedSpan}
	}
	dropped := keepStart - (pfxEnd + 1)

	// Tombstone the originating task: when this drop would launder the session's first user turn
	// (no user turn survives in the protected prefix, and this is the no-goal path — a marked goal
	// is hoisted verbatim by compactWithGoalPin above, never here), embed a bounded excerpt of it in
	// the stub content so a model resuming past the compaction sees WHAT it was asked to do instead
	// of a bare turn count. Empty ⇒ the stub is byte-identical to before, so every case the pin
	// already covered is unchanged. We also content-address the FULL dropped turn (restoreID/
	// restoreBytes) so a gateway with a per-session CAS can back a fak_context_restore handle — the
	// excerpt is orientation, the handle is the recovery edge that pages the whole task back in.
	tombstone, taskBytes := originatingTaskExcerptAndBytes(elems, pfxEnd+1, keepStart)
	restoreID := ""
	if priorExcerpt, priorID, ok := existingOriginatingTaskTombstone(elems, pfxEnd+1, keepStart); ok {
		// Re-compaction must carry the original task handle forward rather than minting a
		// digest for the synthetic compaction stub. The original bytes already live in
		// the gateway restore stash; preserving their handle keeps them addressable at
		// arbitrary depth without duplicating those bytes in every CompactOutcome.
		tombstone, restoreID, taskBytes = priorExcerpt, priorID, nil
	} else if len(taskBytes) > 0 {
		restoreID = originatingTaskDigestID(taskBytes)
	}
	positiveResidue := positiveResidueResult{}
	if opts.PositiveResidue {
		positiveResidue = extractPositiveResidue(elems[pfxEnd+1 : keepStart])
	}

	// shedTokens: the estimated tokens removed from the outbound body — the sum over the dropped
	// MIDDLE [pfxEnd+1, keepStart), minus the stub's own ~cost. Same ~4-chars/token currency as
	// the budget and the provider input_tokens, so it is the CLAIMED-savings half of the
	// billing-truth comparison (vs the provider's cache_read on the same turn).
	shedTokens := 0
	for i := pfxEnd + 1; i < keepStart; i++ {
		shedTokens += estimateElementTokens(elems[i])
	}
	if shedTokens -= compactStubTokenCost(dropped, tombstone, restoreID, positiveResidue.Text, positiveResidue.RestoreID); shedTokens < 0 {
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
	solvencyForced := false
	// Nothing survives the dropped middle here (excludeIdx -1), and this path DOES grant the
	// solvency-floor escape below.
	switch headBurstGate(opts, elems, pfxEnd, keepStart, -1, true) {
	case headBurstRefuse:
		return raw, CompactOutcome{Reason: CompactReasonBurstUnprofitable, SuffixTokens: suffixTokens}
	case headBurstForced:
		// Unprofitable in cache dollars, but the trace has climbed to the solvency floor, where the
		// question stops being "is this profitable?" and becomes "can we afford not to?". A forced
		// fire is labeled on the outcome so it is never counted as a profitable one.
		solvencyForced = true
	}

	// 4. Splice on ORIGINAL bytes. The prefix span [0, spans[pfxEnd].end) (or just the
	//    array-open when pfxEnd<0) is copied verbatim; then the stub; then the kept
	//    elements verbatim; then the verbatim tail from the array close onward.
	// 5. Prove it: the spliced body must still decode AND keep the protected prefix bytes
	//    intact, or we ship identity rather than a broken/cache-busting body (spliceProven).
	out, refusal, good := spliceProven(raw, spans, pfxEnd, func() ([]byte, bool) {
		return spliceCompacted(raw, spans, pfxEnd, keepStart, len(elems), dropped, stubRole, tombstone, restoreID, positiveResidue.Text, positiveResidue.RestoreID)
	})
	if !good {
		return raw, refusal
	}
	inducedCacheCreation := 0
	if opts.Anchor == CompactAnchorHead && !opts.ColdCache {
		inducedCacheCreation = invalidatedSuffixSpanTokens(elems, keepStart)
	}
	return out, CompactOutcome{
		Reason: CompactReasonNone, Dropped: dropped, ShedTokens: shedTokens,
		InducedCacheCreationTokens: inducedCacheCreation,
		RestoreID:                  restoreID, RestoreExcerpt: tombstone, RestoreBytes: taskBytes,
		PositiveResidue: positiveResidue.Text, ResidueRestoreID: positiveResidue.RestoreID,
		ResidueRestoreBytes: positiveResidue.RestoreBytes, ResidueBytesDropped: positiveResidue.DroppedBytes,
		PositiveAssertionsKept: positiveResidue.AssertionsKept,
		SolvencyForced:         solvencyForced,
	}
}

// estimatedElementTokens totals a bounded half-open message range. Keeping the
// boundary normalization here makes protected-prefix and compactible-suffix
// accounting use the same rules when the selected anchor precedes messages[].
func estimatedElementTokens(elems []json.RawMessage, start, end int) int {
	start = max(start, 0)
	end = min(end, len(elems))
	if start >= end {
		return 0
	}
	total := 0
	for _, elem := range elems[start:end] {
		total += estimateElementTokens(elem)
	}
	return total
}

// headBurstVerdict is what the head-anchored cache-economics gate decided.
type headBurstVerdict int

const (
	// headBurstNotApplicable: not head-anchored, or a protected prefix exists — no burst, so no gate.
	headBurstNotApplicable headBurstVerdict = iota
	// headBurstPays: the burst repays inside the remaining session horizon.
	headBurstPays
	// headBurstForced: unprofitable, but the solvency floor forces the drop anyway.
	headBurstForced
	// headBurstRefuse: unprofitable — do not compact.
	headBurstRefuse
)

// headBurstGate runs the head-anchored economics gate (#1407/#1408) both compaction rewrites apply
// before they drop a middle. When head re-anchoring engaged (pfxEnd<0 under CompactAnchorHead) the
// drop deliberately bursts the recent breakpoint's cached suffix — the stable head itself stays
// byte-stable because it precedes messages[] (see anchorCompactablePrefixMode) — so the drop may
// fire only if the per-turn read saving from the cached middle we shed forever repays the one-time
// cold re-write of the invalidated suffix within the remaining session horizon. An unknown horizon
// (TotalTurns<=0) returns false, so a caller that cannot supply the horizon never bursts the cache.
// The firstbp default never reaches this branch (pfxEnd>=0 there, or Anchor!=Head).
//
// The two callers' divergences are parameters, not branches:
//   - excludeIdx names one message inside the dropped middle that SURVIVES the rewrite — the
//     goal-pin path re-inserts its hoisted goal verbatim, so those tokens are re-read every future
//     turn and are NOT part of the per-turn saving; crediting them would over-credit the saving and
//     fire a burst the true economics reject. Pass -1 when nothing survives.
//   - allowSolvencyOverride is whether an unprofitable burst may still fire because the trace
//     reached the solvency floor (CompactOptions.SolvencyFloorTokens). Only the main path grants it;
//     the goal-pin path passes false, reproducing its existing refuse-outright behaviour exactly.
func headBurstGate(opts CompactOptions, elems []json.RawMessage, pfxEnd, keepStart, excludeIdx int, allowSolvencyOverride bool) headBurstVerdict {
	if opts.Anchor != CompactAnchorHead || pfxEnd >= 0 {
		return headBurstNotApplicable
	}
	readMult, writeMult := opts.ReadMult, opts.WriteMult
	if readMult <= 0 {
		readMult = defaultCacheReadMult
	}
	if writeMult <= 0 {
		writeMult = defaultCacheWriteMult
	}
	droppedCachedTokens, invalidatedSuffixTokens := headBurstEconomics(elems, pfxEnd+1, keepStart)
	if excludeIdx >= 0 {
		if droppedCachedTokens -= estimateElementTokens(elems[excludeIdx]); droppedCachedTokens < 0 {
			droppedCachedTokens = 0
		}
	}
	if opts.ColdCache {
		// Observed-cold (never guessed): the suffix's cache entries have already expired, so this
		// turn re-bills them at the cold write rate with or without the drop — the burst carries no
		// marginal penalty and the gate fires horizon-free (breakEven 0).
		invalidatedSuffixTokens = 0
	}
	if CacheBurstPaysBackWithMargin(opts.TotalTurns, opts.CurrentTurn, droppedCachedTokens, invalidatedSuffixTokens, readMult, writeMult, opts.MinHorizonMargin) {
		return headBurstPays
	}
	if allowSolvencyOverride && compactionSolvencyOverride(opts) {
		return headBurstForced
	}
	return headBurstRefuse
}

// beginCompactSplice opens the byte-level splice both compaction rewrites share: it resolves the
// protected-prefix end (just past the last protected element, or the array's content head when no
// message is protected), locates the first kept element and the verbatim body tail, and returns a
// buffer already holding the prefix bytes verbatim. Nothing here is re-serialized, which is exactly
// what keeps the cached prefix byte-identical. bodyTail runs from just past the last element to EOF
// (the `]` plus any trailing top-level keys), so the caller copies it last.
func beginCompactSplice(raw []byte, spans []elementSpan, pfxEnd, keepStart, n int) (b *bytes.Buffer, keptFrom int, bodyTail []byte) {
	prefixEnd := arrayContentStart(spans)
	if pfxEnd >= 0 {
		prefixEnd = spans[pfxEnd].end
	}
	b = &bytes.Buffer{}
	b.Grow(len(raw))
	b.Write(raw[:prefixEnd]) // verbatim protected prefix (includes `[` when pfxEnd<0)
	return b, spans[keepStart].start, raw[spans[n-1].end:]
}

// compactionSolvencyOverride reports whether context SOLVENCY overrides the head-anchored burst
// gate's cache economics for this attempt: the caller supplied a real observed occupancy AND armed
// a floor, and the occupancy has reached it. Both zero values disarm it, so a caller that cannot
// observe occupancy never forces a burst. See CompactOptions.SolvencyFloorTokens for why an
// unprofitable burst is still the right trade this high in the window.
func compactionSolvencyOverride(opts CompactOptions) bool {
	return opts.SolvencyFloorTokens > 0 && opts.ResidentTokens >= opts.SolvencyFloorTokens
}

// headBurstEconomics prices a head-anchored drop for the CacheBurstPaysBack gate.
// droppedCachedTokens is the cached middle [dropStart, keepStart) we shed for good — each future
// turn no longer re-reads it (the per-turn saving). invalidatedSuffixTokens is the cached span the
// drop bursts ONCE: from the first kept message through the last SURVIVING cache_control breakpoint,
// whose byte prefix the drop shifts so the provider must cold-write it again. Bytes beyond the last
// breakpoint were never cached, so they are not counted as invalidated. Same ~4-chars/token currency
// as the budget and the provider input_tokens. dropStart is pfxEnd+1 (0 in head mode).
func invalidatedSuffixSpanTokens(elems []json.RawMessage, keepStart int) int {
	if keepStart < 0 {
		keepStart = 0
	}
	invalidated := 0
	for i := keepStart; i < len(elems); i++ {
		invalidated += estimateElementTokens(elems[i])
	}
	return invalidated
}

func headBurstEconomics(elems []json.RawMessage, dropStart, keepStart int) (droppedCachedTokens, invalidatedSuffixTokens int) {
	if dropStart < 0 {
		dropStart = 0
	}
	for i := dropStart; i < keepStart && i < len(elems); i++ {
		droppedCachedTokens += estimateElementTokens(elems[i])
	}
	lastBp := lastBreakpointMessage(elems)
	if lastBp >= keepStart {
		for i := keepStart; i <= lastBp && i < len(elems); i++ {
			invalidatedSuffixTokens += estimateElementTokens(elems[i])
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
	spliceVerdictOK              spliceVerdict = iota // re-decodes AND protected prefix is byte-identical
	spliceVerdictRedecodeFail                         // spliced body no longer parses as a Messages request
	spliceVerdictPrefixMismatch                       // protected cache prefix bytes changed (would burst the cache)
	spliceVerdictMalformedResult                      // decodes for fak, but violates an Anthropic semantic rule (empty text/content) → the API 400s it
)

// verifySplicedBody is the shared post-splice proof both the compaction and elision rewrites
// run: the result must still decode as a valid Messages request, the protected prefix bytes
// (through spans[pfxEnd], or just the array open when pfxEnd<0) must be byte-identical to the
// input, and the body TAIL — every byte after the last messages[] element, i.e. the `]` plus
// every trailing top-level key — must survive verbatim. The tail proof is what makes a
// head-anchored fire safe on the real Claude Code key order (messages BEFORE system/tools):
// there the marked stable head lives in the tail, not the prefix. Any failure is a splice bug,
// not a reason to ship a broken / cache-busting body — the caller returns identity with its own
// labeled reason.
//
// The re-decode alone is NOT enough: DecodeAnthropicMessagesRequest is fak's OWN permissive
// decoder — it accumulates text with a strings.Builder and silently drops empty blocks, so an
// `{"type":"text","text":""}`, an empty message `content` array, or a `tool_result` with empty
// content all decode CLEANLY here yet are rejected by the real Anthropic Messages API with
// `400 … request … malformed` (text must be non-empty; content must be non-empty). A splice that
// lands the body in one of those shapes is JSON-valid but SEMANTICALLY malformed upstream, so we
// scan the spliced result for that shape and return spliceVerdictMalformedResult rather than ship
// a body the provider will 400 — the caller falls back to identity, which was well-formed by
// construction (we never edit anything but a single string VALUE, so identity cannot introduce
// the empty; only a bad splice can).
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
	if len(spans) > 0 {
		if tail := raw[spans[len(spans)-1].end:]; !bytes.HasSuffix(out, tail) {
			return spliceVerdictPrefixMismatch // the tail carries the head on the CC shape — same burst class
		}
	}
	if bodyHasEmptyBlockSemantics(out) {
		return spliceVerdictMalformedResult
	}
	return spliceVerdictOK
}

// bodyHasEmptyBlockSemantics reports whether a /v1/messages body contains a message-content shape
// the Anthropic Messages API rejects with 400 even though it is valid JSON that fak's permissive
// decoder accepts: a `text` block whose value is the empty string, a message whose `content`
// array is empty, or a `tool_result` whose `content` is empty (empty string or empty array). It
// is deliberately CONSERVATIVE — it only flags shapes Anthropic is known to reject and treats
// anything it cannot parse as "not empty" (unparseable is the re-decode guard's job, not this
// one's), so it can never turn a well-formed splice into a false identity return. It reads only
// the outbound bytes; it never touches the decoded req the kernel adjudicates.
func bodyHasEmptyBlockSemantics(body []byte) bool {
	var obj struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if json.Unmarshal(body, &obj) != nil {
		return false // unparseable → the re-decode guard owns it; do not flag here
	}
	for _, m := range obj.Messages {
		c := skipSpace(m.Content)
		if len(c) == 0 {
			continue // a bare-string content that decoded to nothing is a separate (empty-message) concern
		}
		if c[0] == '"' {
			continue // a bare-string message content is out of scope (only block arrays carry the empty-text hazard)
		}
		if c[0] != '[' {
			continue
		}
		var blocks []json.RawMessage
		if json.Unmarshal(c, &blocks) != nil {
			continue
		}
		if len(blocks) == 0 {
			return true // empty content array — the API rejects a message with no blocks
		}
		for _, blk := range blocks {
			var b struct {
				Type    string          `json:"type"`
				Text    *string         `json:"text"`
				Content json.RawMessage `json:"content"`
			}
			if json.Unmarshal(blk, &b) != nil {
				continue
			}
			switch b.Type {
			case "text":
				// text must be present and non-empty; the marker guarantees a shrunk value is
				// never "", so a "" here means the splice landed an empty value the API will 400.
				if b.Text != nil && *b.Text == "" {
					return true
				}
			case "tool_result":
				if toolResultContentIsEmpty(b.Content) {
					return true
				}
			}
		}
	}
	return false
}

// toolResultContentIsEmpty reports whether a tool_result's `content` value is one the Anthropic
// API treats as empty: absent, an empty string, an empty array, or an array of blocks that are
// themselves all empty-text. A tool_result with no usable content is a 400. Non-empty or
// unparseable content is treated as "not empty" (conservative, same rationale as the caller).
func toolResultContentIsEmpty(content json.RawMessage) bool {
	c := skipSpace(content)
	if len(c) == 0 {
		return true // no content at all
	}
	if c[0] == '"' {
		var s string
		return json.Unmarshal(c, &s) == nil && s == ""
	}
	if c[0] != '[' {
		return false // a non-string, non-array content we do not model → leave it
	}
	var blocks []json.RawMessage
	if json.Unmarshal(c, &blocks) != nil {
		return false
	}
	if len(blocks) == 0 {
		return true // empty content array
	}
	for _, blk := range blocks {
		var b struct {
			Type string  `json:"type"`
			Text *string `json:"text"`
		}
		if json.Unmarshal(blk, &b) != nil {
			return false // an opaque block counts as content
		}
		if b.Type != "text" || b.Text == nil || *b.Text != "" {
			return false // any non-empty (or non-text) block means there IS content
		}
	}
	return true // every block was an empty-text block → effectively empty content
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
	case spliceVerdictMalformedResult:
		return CompactOutcome{Reason: CompactReasonMalformedBody}, false
	}
	return CompactOutcome{}, true
}

// spliceProven binds a rewrite to its proof, which is the rule every compacting path here
// obeys: run the splice, and hand back the rewritten body ONLY if compactSpliceVerdict
// says it still re-decodes and left the protected prefix bytes untouched. Otherwise the
// ORIGINAL body comes back with the verdict's identity reason — a compaction that cannot
// prove itself ships nothing rather than a broken or cache-busting body. The two are bound
// into one call so a future splicer cannot ship unproven by forgetting the check. good
// false means "return (out, refusal) verbatim"; good true leaves the success outcome to
// the caller, which alone knows what it dropped and shed.
func spliceProven(raw []byte, spans []elementSpan, pfxEnd int, splice func() ([]byte, bool)) (out []byte, refusal CompactOutcome, good bool) {
	spliced, ok := splice()
	if outcome, proven := compactSpliceVerdict(raw, spliced, ok, spans, pfxEnd); !proven {
		return raw, outcome, false
	}
	return spliced, CompactOutcome{}, true
}

// decodeArrayElements returns each messages[] element's raw bytes (json.RawMessage) and its
// absolute byte span within raw, using a streaming decoder + InputOffset so the spans are
// exact anchors for byte-splicing (never a fragile string search). msgsRaw must be the
// `messages` value as it appears in raw (json.Unmarshal of an object preserves the value
// bytes verbatim, so a sub-search for it is reliable). ok is false on any decode error.
//
// Locating the base with bytes.Index is only sound when msgsRaw is unlikely to be byte-identical
// to a sibling value; the head arrays (`system`, `tools`) CAN collide with a message content
// array, so those callers must anchor by KEY via decodeTopLevelArray instead (#3773).
func decodeArrayElements(raw []byte, msgsRaw json.RawMessage) (elems []json.RawMessage, spans []elementSpan, ok bool) {
	// Find where msgsRaw sits in raw so element offsets are absolute. json.RawMessage is a
	// verbatim slice of the input, so a single Index of its bytes locates it; the
	// `"messages"` key value is unique enough in practice, and we re-verify with a prefix
	// byte-equality at the end, so a wrong guess can only produce identity, never breakage.
	base := bytes.Index(raw, msgsRaw)
	if base < 0 {
		return nil, nil, false
	}
	return decodeArrayElementsAt(msgsRaw, base)
}

// decodeTopLevelArray locates the value of top-level object key `key` by a streaming KEY walk
// (objectValueSpanLastWins) and decodes it as a JSON array with absolute spans. Anchoring by key
// — never bytes.Index on the value bytes, which finds the FIRST byte-identical occurrence — is
// what stops a head splice (`system`/`tools`) from misrouting into a byte-identical sibling array,
// e.g. a user message whose content array equals the system array: the #3773 vector. Last-key-wins
// matches the map[string]json.RawMessage the callers read obj[key] with, so an adversarial
// duplicate top-level key anchors the same value the decoder keeps. ok is false when the key is
// absent, its value is not a JSON object member, or the value is not a JSON array.
func decodeTopLevelArray(raw []byte, key string) (elems []json.RawMessage, spans []elementSpan, ok bool) {
	start, end, found := objectValueSpanLastWins(raw, key)
	if !found {
		return nil, nil, false
	}
	return decodeArrayElementsAt(raw[start:end], start)
}

// decodeArrayElementsAt is the shared core: it decodes the JSON array arrRaw and reports each
// element's byte span offset by base — arrRaw's absolute start within the request body. The base
// comes from bytes.Index (decodeArrayElements) or a key walk (decodeTopLevelArray). ok is false on
// any decode error or if arrRaw is not a JSON array.
func decodeArrayElementsAt(arrRaw json.RawMessage, base int) (elems []json.RawMessage, spans []elementSpan, ok bool) {
	dec := json.NewDecoder(bytes.NewReader(arrRaw))
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
		for startRel < len(arrRaw) && (isJSONSpace(arrRaw[startRel]) || arrRaw[startRel] == ',') {
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

// chooseKeptWindow walks the messages from the END accumulating ~token cost until it
// reaches budget, then snaps the window start UP to a clean boundary that does not orphan a
// tool_result: a user turn whose content carries tool_result blocks must keep the assistant
// turn before it (the tool_use). It returns the index of the first KEPT message, clamped so
// the window never starts before the first compactible message (compactStart).
func chooseKeptWindow(elems []json.RawMessage, compactStart, budget int) int {
	keep := len(elems)
	acc := 0
	for i := len(elems) - 1; i >= compactStart; i-- {
		acc += estimateElementTokens(elems[i])
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

// keptWindowOrphansToolUse reports whether dropping [compactStart, keepStart) would orphan a
// tool_result: i.e. some tool_result in the KEPT window [keepStart, len) references a tool_use_id
// whose defining assistant tool_use sits in the dropped range and appears in NO kept message. The
// head-only guards in chooseKeptWindow / the post-snap re-assert only rescue a tool_result at the
// window boundary; this catches the INTERIOR case — a tool_result deeper in the kept window whose
// tool_use is laundered by the drop — which produced the live "400 upstream rejected as malformed"
// failures (Anthropic requires every tool_result reference a preceding tool_use). The caller treats
// a true result as a reason to fall back to identity (skip this compaction) rather than ship an
// orphaned body; compaction re-fires next turn, so skipping one drop is cheap and always safe.
func keptWindowOrphansToolUse(elems []json.RawMessage, compactStart, keepStart int) bool {
	if keepStart < 0 || keepStart >= len(elems) {
		return false
	}
	want := keptResultToolUseIDs(elems, keepStart)
	if len(want) == 0 {
		return false
	}
	// Any kept message that itself supplies a wanted id means that id is NOT orphaned — remove it
	// from the wanted set. (A tool_use and its result can both be inside the kept window.)
	for i := keepStart; i < len(elems); i++ {
		clearProvidedToolUseIDs(elems[i], want)
	}
	if len(want) == 0 {
		return false
	}
	// A wanted id is orphaned unless some DROPPED message supplies it AND that message is kept —
	// but dropped messages are, by definition, not kept. So any id still in `want` that is defined
	// only within [compactStart, keepStart) is orphaned. Confirm at least one wanted id is actually
	// defined in the dropped range (else it was never in this request at all — leave it to fak's
	// existing decode, not our concern) before declaring an orphan.
	for i := compactStart; i < keepStart && i < len(elems); i++ {
		if messageProvidesToolUseID(elems[i], want) {
			return true
		}
	}
	return false
}

// keptResultToolUseIDs collects the set of tool_use_id values referenced by tool_result blocks in
// the kept window elems[keepStart:]. These are the ids whose defining assistant tool_use must NOT
// be dropped, or Anthropic rejects the compacted body. Returns nil when the window references none.
func keptResultToolUseIDs(elems []json.RawMessage, keepStart int) map[string]bool {
	var ids map[string]bool
	for i := keepStart; i < len(elems); i++ {
		walkTypedBlocks(elems[i], "user", "tool_result", func(b map[string]json.RawMessage) bool {
			var id string
			if raw, ok := b["tool_use_id"]; ok && json.Unmarshal(raw, &id) == nil && id != "" {
				if ids == nil {
					ids = map[string]bool{}
				}
				ids[id] = true
			}
			return true
		})
	}
	return ids
}

// walkTypedBlocks decodes one messages[] element and hands `visit` every content block of type
// blockType carried by a `role` turn, stopping early the first time visit returns false. An
// element that will not parse, one with a different role, or content that is not a block array
// is skipped silently: the pairing scans below are a safety check on the compacted body, and a
// shape this decoder does not recognize must never fail the request. Those scans differ only in
// the role, the block type, and what they do with each block, so the decode-and-walk that every
// one of them would otherwise re-spell lives here once.
func walkTypedBlocks(el json.RawMessage, role, blockType string, visit func(b map[string]json.RawMessage) bool) {
	var m struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(el, &m) != nil || m.Role != role {
		return
	}
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(m.Content, &blocks) != nil {
		return
	}
	for _, b := range blocks {
		t, ok := b["type"]
		if !ok {
			continue
		}
		var s string
		if json.Unmarshal(t, &s) != nil || s != blockType {
			continue
		}
		if !visit(b) {
			return
		}
	}
}

// clearProvidedToolUseIDs deletes from want every id that an assistant tool_use block in el defines
// (an id whose tool_use is itself in the kept window is not orphaned). It is the in-window
// bookkeeping half of keptWindowOrphansToolUse.
func clearProvidedToolUseIDs(el json.RawMessage, want map[string]bool) {
	walkTypedBlocks(el, "assistant", "tool_use", func(b map[string]json.RawMessage) bool {
		var id string
		if raw, ok := b["id"]; ok && json.Unmarshal(raw, &id) == nil {
			delete(want, id)
		}
		return true
	})
}

// messageProvidesToolUseID reports whether an assistant messages[] element carries a tool_use block
// whose id is in want — i.e. dropping this message would orphan a kept tool_result. It is the
// tool_use-side mirror of messageHasToolResult.
func messageProvidesToolUseID(el json.RawMessage, want map[string]bool) bool {
	provides := false
	walkTypedBlocks(el, "assistant", "tool_use", func(b map[string]json.RawMessage) bool {
		var id string
		if raw, ok := b["id"]; ok && json.Unmarshal(raw, &id) == nil && want[id] {
			provides = true
			return false
		}
		return true
	})
	return provides
}

// messageHasToolResult reports whether a messages[] element is a user turn carrying at
// least one tool_result block — the case whose matching assistant tool_use turn must not be
// dropped from under it.
func messageHasToolResult(el json.RawMessage) bool {
	has := false
	walkTypedBlocks(el, "user", "tool_result", func(map[string]json.RawMessage) bool {
		has = true
		return false
	})
	return has
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

// compactStubContent builds the synthetic stub's text: the drop-count sentinel, plus — when a
// tombstone excerpt is supplied — a second line carrying a bounded excerpt of the compacted
// originating task AND, when restoreID is set, a callable content-address handle (id=<hex>) a
// resuming model can present to fak_context_restore to page the full task back in. compactStubBytes
// and compactStubTokenCost BOTH derive from this one function so the shed-token estimate can never
// drift from the bytes actually emitted. An empty tombstone yields the exact pre-tombstone text (the
// byte-identical default); an empty restoreID (a bare byte-level caller with no CAS to back the
// handle) yields the excerpt-only tombstone line.
func compactStubContent(dropped int, tombstone, restoreID, positiveResidue, residueRestoreID string) string {
	base := fmt.Sprintf("%s%d earlier turn(s) to stay within the context budget; their detail is omitted from this request.", compactStubPrefix, dropped)
	if tombstone == "" && positiveResidue == "" {
		return base
	}
	// The tombstone line: the orientation excerpt, plus — when the compaction path can back a
	// handle with the bytes (restoreID set) — a labelled content-address the model can present to
	// fak_context_restore to page the full task back in. The id rides INSIDE the tombstone line
	// (before the excerpt) so a resuming model reads "what + how to recover" in one place and the
	// stub stays a single low-volume, cache-untouched addition.
	if tombstone != "" {
		line := compactTombstonePrefix
		if restoreID != "" {
			line += compactRestoreIDField + restoreID + " "
		}
		base += "\n" + line + strconv.Quote(tombstone)
	}
	if positiveResidue != "" {
		line := "[fak] positive residual state: "
		if residueRestoreID != "" {
			line += "residue_id=" + residueRestoreID + " "
		}
		base += "\n" + line + positiveResidue
	}
	return base
}

// compactStubTokenCost estimates the synthetic stub message's own ~token cost (the same
// ~4-chars/token basis the budget uses), so the reported shed is NET of the message we add
// back. tombstone must be the SAME excerpt passed to the stub, so the estimate tracks the bytes.
func compactStubTokenCost(dropped int, tombstone, restoreID, positiveResidue, residueRestoreID string) int {
	stub := compactStubContent(dropped, tombstone, restoreID, positiveResidue, residueRestoreID)
	return (len(stub) + len(`{"role":"assistant","content":""}`)) / 4
}

// compactStubBytes marshals the synthetic stub message that stands in for the dropped middle
// turns. Shared by the contiguous drop (spliceCompacted) and the goal-pinned drop
// (spliceCompactedWithGoal) so both emit a byte-identical stub for the same (role, count,
// tombstone). tombstone is the bounded originating-task excerpt (empty on the goal path, where the
// pin already preserves the task verbatim). An out-of-range stubRole falls back to "user".
func compactStubBytes(stubRole string, dropped int, tombstone, restoreID, positiveResidue, residueRestoreID string) ([]byte, error) {
	if stubRole != "user" && stubRole != "assistant" {
		stubRole = "user"
	}
	return json.Marshal(map[string]any{
		"role":    stubRole,
		"content": compactStubContent(dropped, tombstone, restoreID, positiveResidue, residueRestoreID),
	})
}

// originatingTaskDigest returns a bounded, single-line, text-only excerpt of the session's FIRST
// user turn WHEN this compaction is about to drop it — i.e. the first user turn falls in the dropped
// range [dropStart, keepStart) and no user turn survives ahead of it in the protected prefix. It is
// the automatic tombstone for an UNMARKED originating task: the verbatim [fak:goal] pin covers a
// MARKED standing goal, but a plain first task (the common case) would otherwise be laundered into a
// bare turn count. Returns "" — leaving the stub byte-identical to the pre-tombstone default — when
// the task already survives (protected or kept), when there is no user turn in range, or when the
// turn is not pure text (a tool_result/image turn, which is never an originating task anyway).
func originatingTaskDigest(elems []json.RawMessage, dropStart, keepStart int) string {
	excerpt, _ := originatingTaskExcerptAndBytes(elems, dropStart, keepStart)
	return excerpt
}

// existingOriginatingTaskTombstone finds a restore handle already carried by a
// synthetic compaction stub in the span about to be dropped. Re-compaction treats
// that handle as the originating-task identity instead of tombstoning the stub.
func existingOriginatingTaskTombstone(elems []json.RawMessage, dropStart, keepStart int) (excerpt, id string, ok bool) {
	if dropStart < 0 {
		dropStart = 0
	}
	if keepStart > len(elems) {
		keepStart = len(elems)
	}
	for i := dropStart; i < keepStart; i++ {
		text, textOK := elementTextContent(elems[i])
		if !textOK || !strings.Contains(text, compactStubPrefix) {
			continue
		}
		tombstoneAt := strings.Index(text, compactTombstonePrefix)
		if tombstoneAt < 0 {
			continue
		}
		rest := text[tombstoneAt+len(compactTombstonePrefix):]
		if !strings.HasPrefix(rest, compactRestoreIDField) {
			continue
		}
		rest = rest[len(compactRestoreIDField):]
		space := strings.IndexByte(rest, ' ')
		if space <= 0 {
			continue
		}
		id = rest[:space]
		if len(id) != 64 || strings.Trim(id, "0123456789abcdef") != "" {
			continue
		}
		quoted := strings.TrimSpace(rest[space+1:])
		if decoded, err := strconv.Unquote(quoted); err == nil {
			excerpt = decoded
		} else {
			excerpt = quoted
		}
		return excerpt, id, true
	}
	return "", "", false
}

// originatingTaskExcerptAndBytes is originatingTaskDigest that ALSO returns the FULL raw bytes of the
// dropped originating-task turn, so the caller can content-address them (originatingTaskDigestID) and
// stash them behind the restore handle. The excerpt is the lossy orientation trace embedded in the
// stub; the bytes are what a fak_context_restore(id) pages back in verbatim. Both are empty in exactly
// the same cases originatingTaskDigest returns "" (task survives, no user turn in range, non-text
// turn), so a caller that gets an empty excerpt also gets nil bytes and mints no handle. The returned
// slice aliases the input element (the caller stashes it before any mutation, and the byte functions
// only ever read it), matching the verbatim-copy discipline of the surrounding splice.
func originatingTaskExcerptAndBytes(elems []json.RawMessage, dropStart, keepStart int) (string, []byte) {
	firstUser := -1
	for i := 0; i < len(elems) && i < keepStart; i++ {
		if messageRole(elems[i]) == "user" {
			firstUser = i
			break
		}
	}
	if firstUser < dropStart || firstUser >= keepStart {
		return "", nil // no user turn, or the task survives in the protected prefix / kept window
	}
	text, ok := elementTextContent(elems[firstUser])
	if !ok {
		// The originating turn is not pure text (an image or tool block). There is no text to
		// excerpt, but a MEDIA turn must still leave a recovery edge rather than vanishing into a
		// bare turn count: return a generic marker excerpt AND the full bytes so the caller mints a
		// content-address handle and stashes them, letting a resuming model page the original media
		// turn back in via fak_context_restore. A non-media non-text turn (a tool_result-only turn)
		// still keeps the bare stub — those are recoverable from the tool's own re-run, not lost
		// context the way a pasted image is.
		if messageContentHasImage(elems[firstUser]) {
			return compactMediaTombstone, elems[firstUser]
		}
		return "", nil
	}
	// Strip a leading [fak:goal] marker a preamble may carry (defensive — a marked goal is hoisted,
	// not dropped, so this path rarely sees one), then collapse all whitespace runs to single spaces
	// so the excerpt is one low-volume line, then bound it.
	text = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(text), compactGoalMarker))
	text = strings.Join(strings.Fields(text), " ")
	if text == "" {
		return "", nil
	}
	return truncateRunes(text, compactTombstoneCap), elems[firstUser]
}

// truncateRunes shortens s to at most limit runes, appending a single ellipsis rune when it trims,
// so a long originating task cannot blow the tombstone's low-volume budget. Rune-based so it never
// splits a multi-byte character.
func truncateRunes(s string, limit int) string {
	r := []rune(s)
	if len(r) <= limit {
		return s
	}
	return string(r[:limit]) + "…"
}

// spliceCompacted assembles the rewritten body from original byte spans: the verbatim
// protected prefix (through the breakpoint message, or just the array open when pfxEnd<0),
// then a synthetic stub message naming the drop count, then the verbatim kept elements,
// then the verbatim tail from the array close onward. It never re-serializes a protected or
// kept element, so their bytes (and thus the cached prefix) are preserved exactly. ok is
// false if the stub cannot be marshalled (it never realistically fails).
func spliceCompacted(raw []byte, spans []elementSpan, pfxEnd, keepStart, n, dropped int, stubRole, tombstone, restoreID, positiveResidue, residueRestoreID string) ([]byte, bool) {
	stubBytes, err := compactStubBytes(stubRole, dropped, tombstone, restoreID, positiveResidue, residueRestoreID)
	if err != nil {
		return nil, false
	}

	// Everything from arrayClose (the `]`) onward rides along inside the kept-elements + tail copy,
	// so the splice copies from the first kept element through end of body.
	b, keptFrom, bodyTail := beginCompactSplice(raw, spans, pfxEnd, keepStart, n)
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
