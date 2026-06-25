package gateway

// reset_score.go -- the per-session compaction-health reset policy (#792, epic #788).
//
// The shipped compaction path is CUT-ONLY: when a session's body grows past budget it
// drops the middle turns and splices on the cached prefix (agent.CompactAnthropicHistory).
// A cut is the right move WHILE the provider's prompt-cache prefix is still landing -- the
// shed tokens are pure savings. But once the cached prefix has gone stale (the provider
// stopped serving cache reads -- TTL expiry, eviction, the client moved its breakpoint), a
// cut no longer helps: the whole body re-prefills uncached either way, and a fresh RESET
// (a clean, human-like new context seeded from a recap, see internal/sessionreset) would
// reclaim more. #774 calls this the cut-vs-reset crossover.
//
// ResetScore is the PURE, testable decision surface the issue asks for: given one session's
// rolling cache-health state it returns "would a reset beat another cut right now?", a 0..1
// score, and a reason -- with hysteresis + a cooldown so a session cannot flap between cut
// and reset. It mutates nothing and calls nothing; the gateway feeds it observed counters and
// logs the verdict in SHADOW mode (recommend-only). The policy is CUT-BY-DEFAULT: it only
// recommends a reset on clear, sustained staleness, never on a single cold turn or an
// unknown-provider read with no signal.

// ResetReason is the closed vocabulary explaining a ResetScore verdict, so a shadow log /
// metric can say WHY without exposing any prompt content.
type ResetReason string

const (
	// ResetReasonHealthy: the cached prefix is still landing -- keep cutting, do not reset.
	ResetReasonHealthy ResetReason = "healthy_cache"
	// ResetReasonStalePrefix: cache reads have cratered while the body keeps growing -- the
	// prefix is stale, so a cut re-prefills uncached anyway; a reset would reclaim more.
	ResetReasonStalePrefix ResetReason = "stale_prefix"
	// ResetReasonDecay: the cache-read ratio has decayed below the floor over the window
	// (a softer, trending form of staleness short of a hard crater).
	ResetReasonDecay ResetReason = "cache_decay"
	// ResetReasonCooldown: a reset WOULD be recommended, but this session reset too
	// recently -- hold (hysteresis), keep cutting until the cooldown elapses.
	ResetReasonCooldown ResetReason = "cooldown"
	// ResetReasonUnknown: not enough signal to judge (no provider cache counters, or too
	// few observed turns) -- cut-by-default, never reset on a guess.
	ResetReasonUnknown ResetReason = "unknown_provider"
)

// Default reset-policy knobs. These are conservative on purpose: the policy must stay
// cut-by-default until shadow evidence supports enabling reset, so the thresholds demand a
// SUSTAINED stale signal, not a single cold turn.
const (
	// DefaultMinObservedTurns is the minimum number of compaction-eligible turns observed
	// before ResetScore will judge anything other than unknown. Below it there is not enough
	// signal to tell a warming cache from a stale one.
	DefaultMinObservedTurns = 4
	// DefaultHealthyReadRatio is the cache_read/input ratio at or above which the prefix is
	// considered healthy (cutting is clearly winning) -- never recommend a reset.
	DefaultHealthyReadRatio = 0.30
	// DefaultStaleReadRatio is the read ratio at or below which, over the window, the prefix
	// is considered stale enough that a reset beats another cut.
	DefaultStaleReadRatio = 0.05
	// DefaultResetCooldownTurns is how many turns must pass after a reset before another is
	// recommended -- the hysteresis that stops cut/reset flapping.
	DefaultResetCooldownTurns = 8
)

// CacheHealthState is one session's rolling, content-free compaction-health snapshot -- the
// pure input to ResetScore. The gateway accumulates it from the provider's OBSERVED cache
// counters (relayed verbatim, never a fak claim) plus its own cut bookkeeping. Every field is
// a count or a ratio; none carries prompt content, so a shadow log of this struct is safe.
type CacheHealthState struct {
	// ObservedTurns is how many compaction-eligible turns have been seen for this session.
	ObservedTurns int
	// RecentReadRatio is the cache_read/input-token ratio over the recent window (0..1). A
	// healthy, still-landing prefix keeps this high; a stale one craters it toward 0.
	RecentReadRatio float64
	// HasProviderSignal is false when the provider reported no cache counters at all for this
	// session (an unknown-provider / no-telemetry case) -- then the ratio is meaningless.
	HasProviderSignal bool
	// ProviderIdleHint is true when the provider exposed an idle/TTL-expiry hint that the
	// cached prefix has aged out. Optional: most providers do not expose it, so it only ever
	// strengthens a stale verdict, never creates one on its own.
	ProviderIdleHint bool
	// TurnsSinceReset is how many turns have elapsed since this session last reset (a large
	// sentinel when it never has). Drives the cooldown.
	TurnsSinceReset int
}

// ResetDecision is the verdict ResetScore returns: a one-bit recommendation, a 0..1 score
// (how strongly the evidence favors a reset), and the closed reason. It is advisory -- the
// gateway logs it in shadow mode; nothing acts on it until shadow evidence supports enabling.
type ResetDecision struct {
	// ShouldReset is the recommendation: true == a reset would beat another cut right now.
	// Cut-by-default means this is false unless the evidence clearly favors a reset AND the
	// cooldown has elapsed.
	ShouldReset bool
	// Score is 0..1: 0 == clearly keep cutting, 1 == clearly reset. It is reported even when
	// ShouldReset is held by the cooldown, so a shadow log shows the pressure building.
	Score float64
	// Reason is why, from the closed ResetReason vocabulary.
	Reason ResetReason
}

// ResetPolicy holds the reset-decision thresholds. The zero value is NOT valid (it would
// have zero thresholds); use DefaultResetPolicy and override fields as needed.
type ResetPolicy struct {
	MinObservedTurns int
	HealthyReadRatio float64
	StaleReadRatio   float64
	CooldownTurns    int
}

// DefaultResetPolicy returns the conservative, cut-by-default knobs.
func DefaultResetPolicy() ResetPolicy {
	return ResetPolicy{
		MinObservedTurns: DefaultMinObservedTurns,
		HealthyReadRatio: DefaultHealthyReadRatio,
		StaleReadRatio:   DefaultStaleReadRatio,
		CooldownTurns:    DefaultResetCooldownTurns,
	}
}

// ResetScore is the pure decision: given a session's cache-health state, should the hybrid
// policy reset (fresh context) instead of cut (drop middle turns)? It is total and
// side-effect-free -- same state in, same decision out -- so the whole policy is unit-tested
// without a gateway, a provider, or a clock.
//
// The order of the rungs IS the policy, cut-by-default first:
//  1. No provider signal or too few turns -> unknown, keep cutting (never reset on a guess).
//  2. The cached prefix is still healthy -> keep cutting (a cut is pure savings here).
//  3. The prefix is stale (read ratio cratered, optionally with an idle hint) -> a reset
//     beats another cut -- BUT if the session reset within the cooldown, hold (hysteresis).
//  4. A softer decay below the floor -> the same reset recommendation, same cooldown gate.
func (p ResetPolicy) ResetScore(s CacheHealthState) ResetDecision {
	// Rung 1: not enough signal -> unknown, cut-by-default.
	if !s.HasProviderSignal || s.ObservedTurns < p.MinObservedTurns {
		return ResetDecision{ShouldReset: false, Score: 0, Reason: ResetReasonUnknown}
	}

	// Rung 2: the prefix is still landing -> keep cutting.
	if s.RecentReadRatio >= p.HealthyReadRatio {
		return ResetDecision{ShouldReset: false, Score: 0, Reason: ResetReasonHealthy}
	}

	// The prefix is degraded. Decide HOW degraded (hard crater vs soft decay) and SCORE it.
	stale := s.RecentReadRatio <= p.StaleReadRatio || s.ProviderIdleHint
	reason := ResetReasonDecay
	if stale {
		reason = ResetReasonStalePrefix
	}
	score := stalenessScore(s.RecentReadRatio, p, s.ProviderIdleHint)

	// Rung 3/4 cooldown gate (hysteresis): a reset is warranted, but if the session reset
	// within the cooldown window, hold and keep cutting so it cannot flap. The score is still
	// reported so a shadow log shows the pressure that is being held back.
	if s.TurnsSinceReset < p.CooldownTurns {
		return ResetDecision{ShouldReset: false, Score: score, Reason: ResetReasonCooldown}
	}

	return ResetDecision{ShouldReset: true, Score: score, Reason: reason}
}

// stalenessScore maps the recent read ratio to a 0..1 reset-pressure score: it is 0 at the
// healthy floor and 1 at (or below) the stale floor, linearly between. An idle hint floors the
// score at 0.5 so an aged-out prefix always reads as meaningful pressure even if the ratio is
// mid-band. The score is a magnitude for the shadow log; the ShouldReset bit is the gate.
func stalenessScore(readRatio float64, p ResetPolicy, idleHint bool) float64 {
	var score float64
	switch {
	case readRatio <= p.StaleReadRatio:
		score = 1
	case readRatio >= p.HealthyReadRatio:
		score = 0
	default:
		span := p.HealthyReadRatio - p.StaleReadRatio
		if span <= 0 {
			score = 1 // degenerate config: any sub-healthy ratio is full pressure
		} else {
			score = (p.HealthyReadRatio - readRatio) / span
		}
	}
	if idleHint && score < 0.5 {
		score = 0.5
	}
	return score
}
