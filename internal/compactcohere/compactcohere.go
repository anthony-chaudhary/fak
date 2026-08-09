package compactcohere

import "time"

// DefaultProviderCacheTTL is the idle expiry the coordinator assumes for the provider
// prompt cache. Anthropic's ephemeral cache_control entries expire after ~5 minutes of
// idle; past that the cached prefix is gone even if the bytes are byte-identical, so the
// next turn re-bills at cache_creation regardless. The 1-hour beta TTL is opt-in and not
// assumed here; a caller using it passes its own ttl.
const DefaultProviderCacheTTL = 5 * time.Minute

// DefaultBailStreakToYield is how many turns in a row fak's own cache-preserving
// compaction must BAIL (it wanted to shed tokens but could not — a real bail, not a
// healthy under-budget no-op) before the coordinator yields the safety net back to the
// harness. While fak is coping the standing posture blocks the harness's cache-destroying
// auto-compaction; once fak has demonstrably stopped coping, suppressing the harness would
// risk a hard context overflow, so the net is handed back. Conservative on purpose.
const DefaultBailStreakToYield = 3

// DefaultResidentCeiling is the resident-token high-water mark (input + cache_creation +
// cache_read for the served turn) above which fak is judged to be NOT bounding the context
// window, regardless of whether its own cut is "firing." ~160k leaves head-room under a 200k
// model's context so the harness net can still act before a hard overflow. A clean fire that
// leaves the window above this is not coping (#3158); a harness rewrite that lets the window
// climb back above it did not durably hold (#3159). Overridable per deployment (the gateway
// reads FAK_CTX_YIELD_CEILING); a non-positive ceiling disables every resident-token term.
const DefaultResidentCeiling int64 = 160_000

// DefaultCeilingStreakToYield is how many consecutive turns the resident window must exceed the
// ceiling before the coordinator forces PostureAllow regardless of the bail streak (#3158). Small
// and conservative: one spike is noise, a sustained streak is fak failing to bound the window.
const DefaultCeilingStreakToYield = 3

// DefaultNonHoldRewritesToReset is how many non-holding harness rewrites (a rewrite after which
// the window climbs back over the ceiling before the next rewrite) accumulate before the
// coordinator escalates from "let the harness keep rewriting" to ActionRecommendReset + a real
// reset (#3159). Two thrash cycles is enough signal that repeating the cache-destroying rewrite
// is not converging; a fresh RESET beats another one.
const DefaultNonHoldRewritesToReset = 2

// PrefixEvent attributes what happened to the cacheable prompt prefix between two
// consecutive served turns on the fak<->harness boundary. Exactly one is returned per
// turn; the order of attribution is fixed (see Classify).
type PrefixEvent string

const (
	// EventStable: the inbound protected prefix is byte-stable and within TTL — the
	// provider cache will hit. fak's happy path (also the first-turn default).
	EventStable PrefixEvent = "stable"
	// EventFakCut: fak's cache-preserving CUT fired this turn; the protected prefix is
	// preserved by construction, so the cache still hits. fak is managing the window.
	EventFakCut PrefixEvent = "fak_cut"
	// EventFakWorldBreak: fak DELIBERATELY broke the prefix this turn (a cachemeta
	// world-witness refutation — a stale span fak must not let the provider re-serve).
	// fak caused it, so it is not harness interference.
	EventFakWorldBreak PrefixEvent = "fak_world_break"
	// EventHarnessRewrite: the HARNESS rewrote its own history (auto-compaction /
	// /compact) — the inbound protected-prefix digest changed in a way fak never causes
	// (fak forwards the inbound protected prefix verbatim). This bursts the provider
	// cache and is the previously-invisible second-compactor event.
	EventHarnessRewrite PrefixEvent = "harness_rewrite"
	// EventColdTTL: the provider prompt cache went cold on an UNCHANGED prefix — the idle
	// gap exceeded the TTL, or the provider billed a cache_creation with no cache_read on
	// a prefix that should have hit (observed cold). A fresh RESET beats another cut (#774).
	EventColdTTL PrefixEvent = "cold_ttl"
)

// TurnObservation is one served turn's coherence-relevant facts — all cheap, content-free,
// and already available on the gateway passthrough. Every field is a digest, a flag, a
// count, or a duration; none carries prompt content, so a shadow log of it is safe.
type TurnObservation struct {
	// InboundPrefixDigest is a digest of the harness-sent protected prefix (the bytes
	// through the first cache_control breakpoint — the stable cached head), taken BEFORE
	// fak's own request-side transforms. The caller hashes the bytes; this package treats
	// the digest as opaque. Empty means "unknown / not captured" (first turn).
	InboundPrefixDigest string
	// FakCompactFired is true when fak's cache-preserving CUT actually rewrote the body
	// this turn (agent.CompactOutcome.Reason == "").
	FakCompactFired bool
	// FakBailReason is non-empty only when fak WANTED to compact but could not — a real
	// labeled bail (prefix_mismatch, cached_span, window_no_drop, splice_failed,
	// redecode_failed). It is empty for a clean fire AND for a healthy "under_budget"
	// no-op (nothing to shed is not a failure). The caller maps CompactOutcome.Reason:
	// "" and "under_budget" -> "", anything else -> that reason.
	FakBailReason string
	// FakWorldBreak is true when fak injected a cachemeta world-coherence break this turn
	// (EvaluatePrefixCoherence / InjectBreakMarker) — fak's own deliberate prefix break.
	FakWorldBreak bool
	// SealedSpanPresent is true when fak sealed/quarantined a span that is part of this
	// turn's context (a ctxmmu/cachemeta SegSealed). It feeds the quarantine-at-risk
	// signal: a seal that precedes a harness rewrite may be folded into the harness summary.
	SealedSpanPresent bool
	// CacheCreationTokens / CacheReadTokens are the provider's OBSERVED counters for this
	// turn (relayed verbatim, never a fak claim). Used to confirm a cold cache on an
	// unchanged prefix. Both zero means the provider reported no counters.
	CacheCreationTokens int64
	CacheReadTokens     int64
	// InputTokens is the provider's OBSERVED non-cached prompt-token count for this turn
	// (relayed verbatim). Together with the cache counters it gives the RESIDENT window size
	// (see residentTokens); on a cold/reset turn the whole prompt bills as input with zero
	// cache_read, so input is the term that keeps the resident estimate honest there.
	InputTokens int64
	// IdleSinceLastTurn is the wall-clock gap since the previous served turn — the
	// prospective TTL signal (idle past TTL ⇒ the ephemeral cache expired).
	IdleSinceLastTurn time.Duration
}

// residentTokens is the served turn's RESIDENT context size — the full prompt the provider
// billed, whether cached or not: input + cache_creation + cache_read. It is the size the
// resident-token ceiling (#3158) and the non-holding-rewrite watch (#3159) compare against.
func (o TurnObservation) residentTokens() int64 {
	return o.InputTokens + o.CacheCreationTokens + o.CacheReadTokens
}

// Classify attributes one served turn's prefix event from prev (the previous turn's
// observation; a zero value means "first turn, nothing to compare") and cur. ttl is the
// assumed provider cache idle expiry; a non-positive ttl falls back to
// DefaultProviderCacheTTL. It is total and side-effect-free.
//
// The attribution order IS the policy:
//  1. fak's own deliberate world-break dominates (fak caused it — not harness interference).
//  2. an inbound protected-prefix digest change is a HARNESS rewrite (fak never changes the
//     inbound protected prefix; it forwards it verbatim), so a change can only be the harness.
//  3. on an unchanged prefix, a cold cache (idle past TTL, or observed cache_creation with no
//     read) is COLD_TTL.
//  4. otherwise a fak cut this turn is FAK_CUT; with nothing else, STABLE.
func Classify(prev, cur TurnObservation, ttl time.Duration) PrefixEvent {
	if ttl <= 0 {
		ttl = DefaultProviderCacheTTL
	}
	if cur.FakWorldBreak {
		return EventFakWorldBreak
	}
	havePrev := prev.InboundPrefixDigest != ""
	if havePrev && cur.InboundPrefixDigest != "" && cur.InboundPrefixDigest != prev.InboundPrefixDigest {
		return EventHarnessRewrite
	}
	// Cold detection only makes sense relative to a previous turn whose prefix we expected
	// to still be warm.
	if havePrev && (coldByIdle(cur, ttl) || coldByCounters(cur)) {
		return EventColdTTL
	}
	if cur.FakCompactFired {
		return EventFakCut
	}
	return EventStable
}

// coldByIdle reports the prospective TTL signal: the idle gap since the last served turn
// exceeded the assumed cache TTL, so the ephemeral prefix has expired regardless of bytes.
func coldByIdle(cur TurnObservation, ttl time.Duration) bool {
	return cur.IdleSinceLastTurn > ttl
}

// coldByCounters reports the CONFIRMED cold signal from the provider's own counters: it
// billed a cache_creation with zero cache_read on a prefix Classify has already determined
// did not change — i.e. it rebuilt what should have hit, so the cache expired early. Both
// counters zero (no provider signal) is not a cold verdict.
func coldByCounters(cur TurnObservation) bool {
	return cur.CacheCreationTokens > 0 && cur.CacheReadTokens == 0
}

// Posture is the STANDING block/allow stance a Claude Code PreCompact hook enforces
// between turns. It is what the actuator reads; the per-turn Decision explains WHY it is
// where it is.
type Posture string

const (
	// PostureBlock: the PreCompact hook should exit 2 — BLOCK the harness's auto-compaction.
	// fak is the cache-preserving context manager and is coping, so the harness's
	// cache-destroying compaction is redundant and harmful. The default standing posture.
	PostureBlock Posture = "block"
	// PostureAllow: the PreCompact hook should exit 0 — ALLOW the harness's auto-compaction.
	// fak's own compaction has bailed for a sustained streak, so the harness is the only
	// thing standing between the session and a hard context overflow; keep it as the net.
	PostureAllow Posture = "allow"
)

// PreCompactExitCode maps a standing posture to the exit code a Claude Code PreCompact
// hook returns: 2 BLOCKS the pending harness auto-compaction, 0 ALLOWS it. This is the one
// dependable lever to suppress Claude Code's auto-compaction (settings/env toggles are
// silently ignored), so the whole coordination reduces to "what exit code does the hook
// return right now" — which is exactly Posture.
func PreCompactExitCode(p Posture) int {
	if p == PostureBlock {
		return 2
	}
	return 0
}

// Action is the per-turn coordination verdict the coordinator surfaces for observability
// and for the actuators downstream of this policy.
type Action string

const (
	// ActionProceed: fak is managing the window; no harness coordination needed this turn.
	ActionProceed Action = "proceed"
	// ActionBlockHarnessCompact: the harness is acting as a cache-destroying second
	// manager; recommend blocking its next auto-compaction (PreCompact exit 2) so fak's
	// cache-preserving compaction stays the single manager.
	ActionBlockHarnessCompact Action = "block_harness_compact"
	// ActionAllowHarnessCompact: fak's own compaction is not coping (bail streak), so the
	// harness net is needed; recommend allowing its auto-compaction (PreCompact exit 0).
	ActionAllowHarnessCompact Action = "allow_harness_compact"
	// ActionRecommendReset: the provider cache went cold; route to the existing
	// gateway.ResetScore / ResetOnBudget path — a fresh RESET beats another cut (#774).
	ActionRecommendReset Action = "recommend_reset"
)

// Decision is the coordinator's per-turn output: the attributed event, the recommended
// action, the standing PreCompact posture, the resident-window size signals (#3158/#3159),
// and the one-shot risk/escalation flags. It is content-free (counts and enums only).
type Decision struct {
	// Event is the attributed prefix event this turn.
	Event PrefixEvent
	// Action is the per-turn coordination recommendation.
	Action Action
	// HarnessPosture is the STANDING block/allow posture a PreCompact hook enforces. It can
	// differ from Action: Action describes THIS turn's event, Posture is the durable stance.
	HarnessPosture Posture
	// QuarantineAtRisk is true when a fak-sealed (quarantined) span coincided with — or
	// preceded, since the last rewrite — a harness rewrite: the poisoned content may have
	// been folded into the harness's summary, surviving the kernel's quarantine. The trust
	// hole this policy exists to surface.
	QuarantineAtRisk bool
	// BurstObserved is true when this turn (will) cost a provider cache_creation burst — a
	// harness rewrite or a cold-TTL rebuild. Lets an operator line read the cost honestly.
	BurstObserved bool
	// ResidentTokens is this turn's OBSERVED resident window size (input + cache_creation +
	// cache_read) — the content-free size signal both #3158 and #3159 turn on. A count, no content.
	ResidentTokens int64
	// OverCeiling is true when ResidentTokens exceeded the configured resident ceiling this
	// turn — fak is not bounding the window (#3158). Feeds the ceiling-yield and the
	// non-holding-rewrite watch.
	OverCeiling bool
	// RewriteNoDrop is true on the turn a previously-watched harness rewrite is judged to have
	// NOT held: the window climbed back over the ceiling before the next rewrite (#3159). The
	// previously-invisible "the summary didn't stick" event.
	RewriteNoDrop bool
	// EscalateReset is the DISTINCT actuation signal: true only when the non-holding-rewrite
	// streak reached the escalation threshold this turn (#3159). The actuator arms a real reset
	// on this flag alone — NOT on Action==ActionRecommendReset, which cold_ttl also emits
	// (shadow-only), so cold_ttl never actuates a live reset through this path.
	EscalateReset bool
	// Reason is a short, content-free explanation of the verdict.
	Reason string
}

// Coordinator carries the rolling, content-free state across a session's served turns and
// emits a Decision per turn plus the standing PreCompact Posture. It is NOT safe for
// concurrent use; one session drives one Coordinator in turn order.
type Coordinator struct {
	ttl                    time.Duration
	bailStreakToYield      int
	residentCeiling        int64
	ceilingStreakToYield   int
	nonHoldRewritesToReset int

	prev               TurnObservation
	havePrev           bool
	fakBailStreak      int
	sealedSinceRewrite bool
	posture            Posture

	// ceilingStreak counts consecutive turns whose resident window exceeded the ceiling (#3158).
	ceilingStreak int
	// watchingRewrite is true after a harness rewrite until the coordinator learns whether it
	// held: it is cleared when the window climbs back over the ceiling (non-holding, #3159) or
	// when the next rewrite supersedes it.
	watchingRewrite bool
	// nonHoldStreak counts non-holding harness rewrites toward the reset escalation (#3159).
	nonHoldStreak int

	// pruneRefusals counts rejected individual prune proposals by stable reason.
	pruneRefusals RefusalCounters
	// resultPrunes counts selected tool-result evictions, including the producer-flagged subset.
	resultPrunes ToolResultPruneCounters
}

// Config is the coordinator's tunable policy. A zero value is valid: New/NewWith fill it with
// the package defaults, and NewConfig applies a default for every non-positive field, so a
// caller can set only the knobs it cares about.
type Config struct {
	// TTL is the assumed provider prompt-cache idle expiry (see DefaultProviderCacheTTL).
	TTL time.Duration
	// BailStreakToYield is the sustained fak-bail streak that yields the net (DefaultBailStreakToYield).
	BailStreakToYield int
	// ResidentCeiling is the resident-token high-water mark (DefaultResidentCeiling). A negative
	// value falls back to the default; pass a very large value to make the resident terms inert.
	ResidentCeiling int64
	// CeilingStreakToYield is the consecutive over-ceiling turns that force a yield (DefaultCeilingStreakToYield).
	CeilingStreakToYield int
	// NonHoldRewritesToReset is the non-holding-rewrite count that escalates to a reset (DefaultNonHoldRewritesToReset).
	NonHoldRewritesToReset int
}

// New returns a Coordinator with the default block-by-default standing posture and the default
// resident-token terms: when fak's cache-preserving compaction is wired and coping, the harness's
// cache-destroying auto-compaction should be suppressed. A non-positive ttl falls back to
// DefaultProviderCacheTTL. Use NewWith to override the bail streak or NewConfig for the full policy.
func New(ttl time.Duration) *Coordinator { return NewWith(ttl, DefaultBailStreakToYield) }

// NewWith is New with an explicit bail-streak-to-yield threshold (turns of sustained fak
// compaction bail before the net is handed back to the harness). A non-positive streak
// falls back to DefaultBailStreakToYield; the resident-token terms use their defaults.
func NewWith(ttl time.Duration, bailStreakToYield int) *Coordinator {
	return NewConfig(Config{TTL: ttl, BailStreakToYield: bailStreakToYield})
}

// NewConfig returns a Coordinator with an explicit policy. Every non-positive field falls back
// to its package default, so a partial Config is safe. The standing posture starts at
// PostureBlock (fak is the single manager until it demonstrably stops coping).
func NewConfig(cfg Config) *Coordinator {
	if cfg.TTL <= 0 {
		cfg.TTL = DefaultProviderCacheTTL
	}
	if cfg.BailStreakToYield <= 0 {
		cfg.BailStreakToYield = DefaultBailStreakToYield
	}
	if cfg.ResidentCeiling <= 0 {
		cfg.ResidentCeiling = DefaultResidentCeiling
	}
	if cfg.CeilingStreakToYield <= 0 {
		cfg.CeilingStreakToYield = DefaultCeilingStreakToYield
	}
	if cfg.NonHoldRewritesToReset <= 0 {
		cfg.NonHoldRewritesToReset = DefaultNonHoldRewritesToReset
	}
	return &Coordinator{
		ttl:                    cfg.TTL,
		bailStreakToYield:      cfg.BailStreakToYield,
		residentCeiling:        cfg.ResidentCeiling,
		ceilingStreakToYield:   cfg.CeilingStreakToYield,
		nonHoldRewritesToReset: cfg.NonHoldRewritesToReset,
		posture:                PostureBlock,
	}
}

// Posture returns the current standing block/allow stance for the PreCompact hook to
// enforce between turns, without advancing state.
func (c *Coordinator) Posture() Posture { return c.posture }

// CheckPrune applies the arithmetic and structural guards at the coordinator's
// prune seam and records refused decisions for observability. Callers still use
// Observe/Classify to decide when cache timing permits compaction; this method
// decides whether the particular proposed cut is profitable and well-formed.
func (c *Coordinator) CheckPrune(spanTokens int, transcript []TranscriptItem, cut int) PruneResult {
	d := CheckPrune(spanTokens, transcript, cut)
	c.pruneRefusals = CountRefusal(c.pruneRefusals, d)
	return d
}

// PruneRefusals reports coordinator-local refused-prune counters by reason.
func (c *Coordinator) PruneRefusals() RefusalCounters { return c.pruneRefusals }

// SelectToolResults applies the producer preference behind the existing prefix-event
// cache-safety gate and records the selected-result counters for observability.
func (c *Coordinator) SelectToolResults(event PrefixEvent, candidates []ToolResultEntry, limit int) ToolResultSelection {
	selection := SelectToolResults(event, candidates, limit)
	c.resultPrunes = CountToolResultPrunes(c.resultPrunes, selection)
	return selection
}

// ToolResultPrunes reports coordinator-local tool-result eviction counters.
func (c *Coordinator) ToolResultPrunes() ToolResultPruneCounters { return c.resultPrunes }

// Observe folds one served turn into the coordinator and returns the per-turn Decision. It
// updates the standing posture (block while fak copes; allow once fak's compaction has
// bailed bailStreakToYield turns in a row) and tracks the quarantine-at-risk signal across
// turns (a seal that precedes a harness rewrite may be in the summary).
func (c *Coordinator) Observe(cur TurnObservation) Decision {
	var prev TurnObservation
	if c.havePrev {
		prev = c.prev
	}
	ev := Classify(prev, cur, c.ttl)

	// Resident-window size is the first-class signal (#3158/#3159): the ceiling term, the
	// held-streak refinement, and the non-holding-rewrite watch all key off it.
	resident := cur.residentTokens()
	overCeiling := c.residentCeiling > 0 && resident > c.residentCeiling
	if overCeiling {
		c.ceilingStreak++
	} else {
		c.ceilingStreak = 0
	}

	// fak's ability to cope: a real bail increments the streak. A clean fire / healthy
	// under-budget no-op (FakBailReason == "") resets it ONLY when the window is under the
	// ceiling — a fire that leaves the window over budget is not coping, so hold the streak
	// rather than crediting it with a reset (#3158).
	if cur.FakBailReason != "" {
		c.fakBailStreak++
	} else if !overCeiling {
		c.fakBailStreak = 0
	}

	// Standing posture: yield the net to the harness when fak has bailed for a streak OR when
	// the resident window has sat over the ceiling for a streak (fak is not bounding it, #3158).
	yieldOnBail := c.fakBailStreak >= c.bailStreakToYield
	yieldOnCeiling := c.ceilingStreakToYield > 0 && c.ceilingStreak >= c.ceilingStreakToYield
	if yieldOnBail || yieldOnCeiling {
		c.posture = PostureAllow
	} else {
		c.posture = PostureBlock
	}

	// A seal anywhere since the last harness rewrite is exposure: if the harness then
	// summarizes, that poisoned span may be carried into the summary.
	if cur.SealedSpanPresent {
		c.sealedSinceRewrite = true
	}

	// Non-holding-rewrite watch (#3159): after a harness rewrite we watch whether the window
	// stays bounded. If a later (non-rewrite) turn is back over the ceiling before the next
	// rewrite, the summary did not durably hold — the previously-invisible "compaction bailed".
	rewriteNoDrop := false
	if c.watchingRewrite && ev != EventHarnessRewrite && overCeiling {
		rewriteNoDrop = true
		c.watchingRewrite = false
		c.nonHoldStreak++
	}

	d := Decision{
		Event:          ev,
		HarnessPosture: c.posture,
		ResidentTokens: resident,
		OverCeiling:    overCeiling,
		RewriteNoDrop:  rewriteNoDrop,
	}
	switch ev {
	case EventHarnessRewrite:
		d.BurstObserved = true
		d.QuarantineAtRisk = c.sealedSinceRewrite
		c.sealedSinceRewrite = false // the prior seals are now either in the summary or dropped
		c.watchingRewrite = true     // watch whether THIS rewrite durably bounds the window (#3159)
		if c.posture == PostureAllow {
			d.Action = ActionAllowHarnessCompact
			d.Reason = "harness rewrote the prefix; fak is not coping (bail or over-ceiling streak) — keep the harness as the context net"
		} else {
			d.Action = ActionBlockHarnessCompact
			d.Reason = "harness rewrote the prefix (cache-destroying); fak's cache-preserving compaction should be the single manager — block the next harness auto-compaction"
		}
	case EventColdTTL:
		d.BurstObserved = true
		d.Action = ActionRecommendReset
		d.Reason = "provider prompt cache went cold (idle past TTL or observed cache_creation with no read); a fresh RESET beats another cut (#774)"
	default: // EventStable, EventFakCut, EventFakWorldBreak
		d.Action = ActionProceed
		d.Reason = string(ev) + ": fak is managing the context window; no harness coordination needed this turn"
	}

	// Escalation (#3159): once non-holding rewrites accumulate to the threshold, repeating the
	// cache-destroying rewrite is not converging — recommend a hard RESET and raise the DISTINCT
	// EscalateReset flag the actuator arms on (never cold_ttl's ActionRecommendReset). Overrides
	// the per-event Action for this turn; the streak resets so the next escalation must re-earn it.
	if c.nonHoldRewritesToReset > 0 && c.nonHoldStreak >= c.nonHoldRewritesToReset {
		d.Action = ActionRecommendReset
		d.EscalateReset = true
		d.Reason = "harness rewrites are not holding (window returned over the ceiling) — recommend a hard reset (#3159)"
		c.nonHoldStreak = 0
	}

	c.prev = cur
	c.havePrev = true
	return d
}
