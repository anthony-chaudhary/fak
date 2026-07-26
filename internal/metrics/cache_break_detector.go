package metrics

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// cache_break_detector.go — the mid-conversation prefix-mutation DETECTOR (#2847,
// Track C of epic #2834): the live producer cache_break.go's counter was landed
// waiting for.
//
// Hermes' single most important cost invariant is "Prompt Caching Must Not Break"
// (its `AGENTS.md`): do not alter past context, change toolsets, or rebuild the
// system prompt mid-conversation. Hermes enforces it by ASKING CONTRIBUTORS NOT TO
// — a convention policed in code review. One stray mutation silently busts the
// cache and multiplies cost, and no runtime signal fires. fak sits in front of the
// wire and sees every request, so the invariant can be STRUCTURAL instead: hash the
// turn's stable prefix, compare it against the session's established prefix, and
// price the divergence.
//
//   - TurnPrefix is the stable prefix in wire order — system prompt, tool schema,
//     already-sent history head — each with its token length, because the cost of a
//     break depends on WHERE in that order the divergence started.
//   - DigestTurnPrefix is the per-turn prefix-hash: one content-free digest per
//     component. The detector retains digests and a length, never prompt bytes, so
//     arming it across a whole fleet costs no transcript retention.
//   - CacheBreakDetector.Observe scores one turn against the session baseline and
//     returns the verdict: whether the prefix broke, the CLOSED cause, the measured
//     induced cache_creation, and the "cache broken here, +N tokens" witness.
//   - CacheBreakPolicy is the deny/warn/off lever. warn witnesses and lets the
//     mutation through; deny witnesses and REFUSES it, so the established prefix
//     survives and the cost is avoided rather than incurred. They are deliberately
//     not the same mode: only deny keeps the baseline, and only warn advances it.
//   - Report / Avoided are the two folds that keep that distinction honest in the
//     output — cost actually incurred vs cost the deny policy prevented.
//
// CAUSE ATTRIBUTION is by first divergence in wire order, because every byte after
// a divergence is cold regardless: a changed system prompt is rebuilt_prompt, an
// unchanged system with a changed tool schema is toolset_change, and an unchanged
// system+tools with a rewritten history head is altered_turn. A history head that
// merely GREW (the baseline is still a byte-prefix of it) is an ordinary appended
// turn, not a break — that is the false-positive guard the contract names, and it
// is what keeps a normal conversation from flagging on every turn.
//
// provider_quirk is deliberately NOT reachable from this detector: a provider-side
// eviction leaves the prefix bytes IDENTICAL, so it is attributed by the cache-read
// observation path (internal/metrics/provider_cache.go), never by byte divergence.
// Claiming it here would be a fabricated cause.
//
// COST is the induced cache_creation on the diverged turn: the tokens of the NEW
// turn's prefix from the first divergent component through the end, which is the
// span the provider must now write to cache. Per cache_break.go's own contract that
// figure is a cold-rebuild UPPER bound, not a settled invoice.
//
// This package stays pure — no engine, no kernel import — so the seam is
// unit-testable. The gateway lowers a verdict onto its live Prometheus surface with
// the seam that already exists there (internal/gateway/metrics_observe.go):
//
//	if v := det.Observe(turn); v.Broken {
//		m.recordCacheBreak(v.Event.Cause, v.Event.CostTokens)
//	}
//
// Generation intent: gen/next foundation (Track C of #2834, Hermes-evidence epic
// #2908) — the same classification cache_break.go carries, since the cache/context
// program map (docs/notes/GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md) puts
// live provider/cache metrics at gen/next until a live caller corroborates them.
//   - Promotion evidence (toward "now"): the gateway request path calls Observe per
//     turn and feeds recordCacheBreak, and a witnessed CostTokens is corroborated
//     against the provider-relayed cache_creation tokens for the same turn
//     (internal/metrics/provider_cache.go) — at which point cache_break.go's counter
//     stops being a fixture and CheckBudget can gate a real regression.
//   - Demotion / retirement evidence: if the deny mode is never armed on a real
//     serve path, or corroboration shows the wire-order cost model consistently
//     over-states the provider's actual cache_creation, the priced half earns
//     nothing and should be reduced to a bare stable/mutated bit.
//   - Invalidating assumption: that a turn's stable prefix is exactly
//     system+tools+history-head. A provider that varies its cache breakpoint, or a
//     harness that legitimately rewrites history head under compaction, breaks that
//     framing — compaction rewrites are a SANCTIONED prefix change and must call
//     Reset, or they will be witnessed as altered_turn and priced as a break.

// CacheBreakPolicy is the closed deny/warn/off lever for what the detector does
// with a divergence it finds. The contract names deny and warn as distinct modes;
// they differ structurally, not just in log level (see CacheBreakDetector.Observe).
type CacheBreakPolicy string

const (
	// CacheBreakPolicyOff disarms the detector: Observe does no work, keeps no
	// baseline, and never witnesses. It is the default for a caller that has not
	// opted in, so landing the detector costs an unarmed session nothing.
	CacheBreakPolicyOff CacheBreakPolicy = "off"
	// CacheBreakPolicyWarn detects and witnesses the break but LETS THE MUTATION
	// THROUGH: the mutated prefix becomes the new baseline and the induced
	// cache_creation is really paid, so it lands in Report.
	CacheBreakPolicyWarn CacheBreakPolicy = "warn"
	// CacheBreakPolicyDeny detects, witnesses, and REFUSES the mutation: the
	// established baseline is KEPT, so the warm prefix survives and the cost is
	// avoided rather than incurred — it lands in Avoided, not Report.
	CacheBreakPolicyDeny CacheBreakPolicy = "deny"
)

// ParseCacheBreakPolicy maps an operator string onto the closed lever. Anything
// outside the set — including empty — folds to CacheBreakPolicyOff, so an
// unrecognized value can never silently arm a denying gate.
func ParseCacheBreakPolicy(s string) CacheBreakPolicy {
	switch CacheBreakPolicy(s) {
	case CacheBreakPolicyWarn:
		return CacheBreakPolicyWarn
	case CacheBreakPolicyDeny:
		return CacheBreakPolicyDeny
	default:
		return CacheBreakPolicyOff
	}
}

// TurnPrefix is one turn's stable prefix in WIRE ORDER: the system prompt, then
// the tool schema, then the already-sent history head that must not be altered.
// The order is load-bearing — a divergence in an earlier component makes every
// later one cold too, which is how the induced cost is derived.
//
// The three *Tokens fields are the caller's measured token lengths for the same
// three spans; they are what the induced cache_creation is priced from. A caller
// that cannot measure them still gets a correct cause, with a zero cost.
type TurnPrefix struct {
	// System is the system-prompt bytes: the span a mid-conversation prompt
	// rebuild changes.
	System string
	// Tools is the serialized tool schema. It is order-sensitive on purpose: a
	// reordered schema is a real cache break at the provider, not a no-op.
	Tools string
	// HistoryHead is the already-sent prior context. Growth here is an ordinary
	// appended turn; a rewrite of what was already sent is the break.
	HistoryHead string

	SystemTokens  int64
	ToolsTokens   int64
	HistoryTokens int64
}

// PrefixDigest is the per-turn prefix-hash: one content-free digest per stable
// component plus a combined digest over all three. It carries no prompt bytes, so
// it is safe to log, journal, or compare across processes.
type PrefixDigest struct {
	System   string `json:"system"`
	Tools    string `json:"tools"`
	History  string `json:"history"`
	Combined string `json:"combined"`
}

// digest is the shared content-free hash: sha256, truncated to 64 bits of hex.
// Truncation keeps a witness line readable; 64 bits is far past collision risk for
// the handful of prefixes one conversation establishes.
func digest(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:8])
}

// DigestTurnPrefix computes the per-turn prefix-hash over system + tools +
// history-head. Components are hashed separately so a divergence is ATTRIBUTABLE:
// a single whole-prefix hash would prove only that something moved, which is the
// signal Hermes' convention already implicitly has and cannot act on.
func DigestTurnPrefix(t TurnPrefix) PrefixDigest {
	d := PrefixDigest{
		System:  digest(t.System),
		Tools:   digest(t.Tools),
		History: digest(t.HistoryHead),
	}
	// Length-delimited so a byte moving across a component boundary cannot
	// produce the same combined digest as the unmoved prefix.
	d.Combined = digest(fmt.Sprintf("%d:%s|%d:%s|%d:%s",
		len(t.System), t.System, len(t.Tools), t.Tools, len(t.HistoryHead), t.HistoryHead))
	return d
}

// CacheBreakVerdict is the answer for one observed turn: whether the established
// prefix broke, why, what it cost, and what the policy did about it.
type CacheBreakVerdict struct {
	// Broken is true when this turn's stable prefix diverged from the session's
	// established prefix. A first turn, an identical prefix, and a pure history
	// append are all false.
	Broken bool
	// Cause is the closed attribution, valid only when Broken.
	Cause CacheBreakCause `json:"cause"`
	// Policy is the lever the detector was carrying for this observation.
	Policy CacheBreakPolicy `json:"policy"`
	// Denied is true when the policy REFUSED the mutation. The established prefix
	// is kept, so CostTokens is the cost AVOIDED rather than incurred.
	Denied bool `json:"denied"`
	// Event is the witnessed break, ready for FoldCacheBreaks or the gateway's
	// recordCacheBreak seam. Zero-valued when Broken is false.
	Event CacheBreakEvent `json:"event"`
	// Digest is this turn's prefix-hash — the content-free evidence behind the
	// verdict, on a clean turn as well as a broken one.
	Digest PrefixDigest `json:"digest"`
	// Witness is the operator line the contract asks for: "cache broken here,
	// +N tokens (cause=..., policy=...)". Empty when nothing broke.
	Witness string `json:"witness"`
	// Reason is the short human-readable explanation, matching the Reason-field
	// convention used across the cache primitives.
	Reason string `json:"reason"`
	// ObservedTurn is the 1-indexed count of armed Observe calls on this detector.
	ObservedTurn int `json:"observed_turn"`
}

// CacheBreakDetector holds one session's established prefix and scores each turn
// against it. Its retained state is content-free: three digests and the history
// head's byte length, never the prompt itself.
//
// It is a single-session, turn-at-a-time state machine and is not safe for
// concurrent use without external synchronization — the same contract as
// cachemeta.PrefixStabilityTracker.
type CacheBreakDetector struct {
	policy CacheBreakPolicy

	hasBaseline   bool
	baseSystem    string
	baseTools     string
	baseHistory   string
	baseHistoryLn int

	turns    int
	incurred []CacheBreakEvent
	avoided  []CacheBreakEvent
}

// NewCacheBreakDetector builds a detector under the given policy. An unrecognized
// policy folds to off, so a misconfigured lever fails closed (observes nothing)
// rather than denying live traffic.
func NewCacheBreakDetector(policy CacheBreakPolicy) *CacheBreakDetector {
	return &CacheBreakDetector{policy: ParseCacheBreakPolicy(string(policy))}
}

// Policy reports the lever this detector is carrying.
func (d *CacheBreakDetector) Policy() CacheBreakPolicy { return d.policy }

// Reset clears the established prefix: the next Observe re-primes a baseline and
// reports no break. Call it for a SANCTIONED prefix change — an acknowledged
// compaction rewrite or an operator-forced rebuild — so the following turn is not
// witnessed as a surprise altered_turn.
func (d *CacheBreakDetector) Reset() {
	d.hasBaseline = false
	d.baseSystem, d.baseTools, d.baseHistory = "", "", ""
	d.baseHistoryLn = 0
}

// Observe scores one turn's stable prefix against the session's established
// prefix and folds the result.
//
// Under warn, a detected break ADVANCES the baseline: the mutation went to the
// wire, so the mutated prefix is what the next turn must match, and the induced
// cache_creation is really paid (it folds into Report). Under deny, the break is
// refused and the baseline is KEPT: the established prefix is still what the
// session carries, and the cost folds into Avoided instead. That difference is
// the whole point of having two modes rather than one log level.
//
// Advancing the baseline on an allowed break is also what stops a persistently-new
// prefix from counting one break per turn forever — the same last-accepted-baseline
// discipline cachemeta.PrefixStabilityTracker uses.
func (d *CacheBreakDetector) Observe(turn TurnPrefix) CacheBreakVerdict {
	if d.policy == CacheBreakPolicyOff {
		return CacheBreakVerdict{Policy: CacheBreakPolicyOff, Reason: "detector disarmed (policy off)"}
	}

	d.turns++
	v := CacheBreakVerdict{
		Policy:       d.policy,
		Digest:       DigestTurnPrefix(turn),
		ObservedTurn: d.turns,
	}

	if !d.hasBaseline {
		d.adopt(turn)
		v.Reason = "no established prefix yet; this turn becomes the baseline"
		return v
	}

	cause, cost, ok := d.diverge(turn)
	if !ok {
		v.Reason = "stable prefix matches the established prefix"
		if len(turn.HistoryHead) > d.baseHistoryLn {
			v.Reason = "stable prefix intact; history head extended by an appended turn"
		}
		d.adopt(turn)
		return v
	}

	v.Broken = true
	v.Cause = cause
	v.Event = WitnessCacheBreak(cause, cost)
	v.Denied = d.policy == CacheBreakPolicyDeny

	if v.Denied {
		v.Reason = fmt.Sprintf("mid-conversation %s refused; established prefix kept", cause)
		v.Witness = fmt.Sprintf("cache broken here, +%d tokens avoided (cause=%s, policy=deny)", v.Event.CostTokens, cause)
		d.avoided = append(d.avoided, v.Event)
		// Baseline deliberately NOT advanced: the mutation did not reach the wire.
		return v
	}

	v.Reason = fmt.Sprintf("mid-conversation %s allowed; established prefix rebuilt", cause)
	v.Witness = fmt.Sprintf("cache broken here, +%d tokens (cause=%s, policy=warn)", v.Event.CostTokens, cause)
	d.incurred = append(d.incurred, v.Event)
	d.adopt(turn)
	return v
}

// diverge finds the FIRST divergent component in wire order and prices the cold
// span it induces. It reports ok=false when the established prefix survived.
func (d *CacheBreakDetector) diverge(turn TurnPrefix) (CacheBreakCause, int64, bool) {
	switch {
	case digest(turn.System) != d.baseSystem:
		// Everything after a rebuilt system prompt is cold too.
		return CacheBreakRebuiltPrompt, turn.SystemTokens + turn.ToolsTokens + turn.HistoryTokens, true
	case digest(turn.Tools) != d.baseTools:
		return CacheBreakToolsetChange, turn.ToolsTokens + turn.HistoryTokens, true
	case !d.historyPreserved(turn.HistoryHead):
		return CacheBreakAlteredTurn, turn.HistoryTokens, true
	}
	return "", 0, false
}

// historyPreserved reports whether the established history head survived intact:
// either byte-identical, or still a byte-PREFIX of a grown head (an appended
// turn). Anything else is a rewrite of already-sent context.
//
// The check needs only the baseline's digest and length, so the detector never
// retains the transcript to answer it.
func (d *CacheBreakDetector) historyPreserved(head string) bool {
	if len(head) < d.baseHistoryLn {
		// The head shrank: already-sent context was dropped or rewritten.
		return false
	}
	return digest(head[:d.baseHistoryLn]) == d.baseHistory
}

// adopt makes turn the established prefix, retaining digests and a length only.
func (d *CacheBreakDetector) adopt(turn TurnPrefix) {
	d.hasBaseline = true
	d.baseSystem = digest(turn.System)
	d.baseTools = digest(turn.Tools)
	d.baseHistory = digest(turn.HistoryHead)
	d.baseHistoryLn = len(turn.HistoryHead)
}

// Report folds the breaks this session actually PAID for — those a warn policy
// let through — into the operator readout cache_break.go defines. This is the
// report a CheckBudget gate should read: it counts induced cache_creation, not
// mutations the deny policy prevented.
func (d *CacheBreakDetector) Report() CacheBreakReport {
	return FoldCacheBreaks(d.incurred)
}

// Avoided folds the breaks the deny policy REFUSED: the cost the structural gate
// saved versus a harness that, like Hermes, could only ask contributors not to
// mutate the prefix. It is the payoff figure, and it is deliberately separate from
// Report so a budget gate can never read avoided cost as if it had been spent.
func (d *CacheBreakDetector) Avoided() CacheBreakReport {
	return FoldCacheBreaks(d.avoided)
}
