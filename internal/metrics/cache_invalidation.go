package metrics

// cache_invalidation.go — the kernel deferred-vs-now cache-invalidation primitive (#2895),
// the fak-does-it-better answer to Hermes' per-command `/skills install --now`.
//
// Hermes makes each slash command that mutates system-prompt state (skills, tools, memory)
// cache-aware BY DISCIPLINE: every command re-implements a `--now` flag and defaults to
// deferred invalidation. fak decides cache lifetime, so deferred-vs-now is a KERNEL PRIMITIVE
// any state-mutating operation inherits instead of a per-command habit:
//
//   - InvalidationMode is the shared flag: Deferred by default (the change lands next
//     session, no cold prefix rebuild is paid now) with an opt-in Now (`--now`) that pays
//     the rebuild immediately. ParseInvalidationMode maps a bare `--now` bool onto it, so a
//     command inherits the default just by not passing the flag.
//   - WitnessInvalidation witnesses the trade each mutation actually made: for the chosen
//     mode, the cold-rebuild tokens a cold prefill would cost and — the deferral's payoff —
//     how many of those tokens the deferral saved now against the staleness window it opened.
//   - FoldCacheInvalidation folds a session's witnesses into the operator readout the issue
//     asks for: the total cold-rebuild cost AVOIDED by deferring vs the worst staleness
//     window incurred. That readout is the "witnessed report of the cold-rebuild cost avoided
//     by deferring" — the report, not per-command discipline, is what prices the trade.
//
// The cold-rebuild token figure is caller-supplied cost, not invented here: it is the
// prompt-prefix re-prefill an anthropic_cachebp breakpoint would otherwise have kept warm
// (internal/agent/anthropic_cachebp.go, internal/cachemeta prefix accounting). This package
// stays pure — no engine, no kernel import — so the seam is unit-testable and any command
// or report can fold its own mutations through it.
//
// This is NOT the engine-side K/V eviction plan (cachemeta.PlanExternalInvalidations) nor
// the prefix-stability break detector (sibling issue #2894); it is the operator-facing
// deferred-vs-now decision + its priced witness.
//
// Generation intent: gen/next foundation (#2895, Hermes-inspiration epic #2871). One
// prompt/cache seam, moderate risk kept behind an explicit opt-in.
//   - Promotion evidence (toward "now"): a real state-mutating command adopts
//     ParseInvalidationMode for its `--now` flag and emits a FoldCacheInvalidation readout
//     whose TokensSavedByDeferring is corroborated against measured provider cache-write
//     tokens (internal/metrics/provider_cache.go). Two commands sharing the one primitive
//     retires the per-command-flag habit this replaces.
//   - Demotion / retirement evidence: if commands keep hand-rolling their own `--now`
//     parsing instead of inheriting ParseInvalidationMode, or if the cold-rebuild cost cannot
//     be sourced from cache accounting and TokensSavedByDeferring is always zero, the
//     primitive earns nothing and should be retired.
//   - Invalidating assumption: that deferral is free of correctness risk — it is not if a
//     mutation must take effect within the SAME session (a security-relevant tool/skill
//     removal). Such a mutation must pass InvalidationNow; the staleness window this witness
//     prices is a token trade, NOT a licence to defer a change that has to land now.

// InvalidationMode is the kernel primitive any state-mutating operation inherits: whether a
// mutation to cached system-prompt state invalidates the provider prefix cache now or defers
// it to the next session. Deferred is the default; Now is the opt-in.
type InvalidationMode string

const (
	// InvalidationDeferred is the default: the mutation takes effect next session, so no
	// cold prefix rebuild is paid now — you keep the warm cache for the rest of this session.
	InvalidationDeferred InvalidationMode = "deferred"
	// InvalidationNow is the opt-in (`--now`): invalidate immediately and pay the cold
	// rebuild so the change is in effect this turn.
	InvalidationNow InvalidationMode = "now"
)

// ParseInvalidationMode maps a bare `--now` flag onto the mode. Deferred is the default: a
// state mutation never pays a cold rebuild unless the operator explicitly opts in with
// `--now`. This is the one seam a command inherits instead of re-deciding the default.
func ParseInvalidationMode(now bool) InvalidationMode {
	if now {
		return InvalidationNow
	}
	return InvalidationDeferred
}

// IsNow reports whether the mode invalidates immediately.
func (m InvalidationMode) IsNow() bool { return m == InvalidationNow }

// EffectiveNextSession reports whether the change lands next session (the deferred trade)
// rather than this turn. An unknown/zero mode is treated as the deferred default.
func (m InvalidationMode) EffectiveNextSession() bool { return m != InvalidationNow }

// CacheInvalidationWitness is the per-mutation receipt: for the chosen mode, what a cold
// prefix rebuild would cost and — the deferral's payoff — how many of those tokens deferring
// saved now against the staleness window it opened. It is the priced trade for ONE mutation.
type CacheInvalidationWitness struct {
	Mode InvalidationMode `json:"mode"`
	// ColdRebuildTokens is the prompt-prefix re-prefill a cold rebuild costs: paid now under
	// Now, avoided now under Deferred. Caller-supplied from cache accounting.
	ColdRebuildTokens int64 `json:"cold_rebuild_tokens"`
	// TokensSavedByDeferring is the cold-rebuild cost this mutation avoided by deferring
	// (== ColdRebuildTokens when Deferred, 0 when Now).
	TokensSavedByDeferring int64 `json:"tokens_saved_by_deferring"`
	// StalenessTurns is the window the change stays not-yet-in-effect under Deferred
	// (0 under Now). It is the cost side of the deferral trade.
	StalenessTurns int `json:"staleness_turns"`
	// EffectiveNextSession is true when the change lands next session (Deferred) rather
	// than this turn (Now).
	EffectiveNextSession bool `json:"effective_next_session"`
}

// WitnessInvalidation prices one state mutation under the chosen mode. Negative inputs are
// clamped to zero. Under Now the cold rebuild is paid now (no saving, no staleness); under
// Deferred the cold rebuild is avoided now (saving == cost) at the price of the staleness
// window until next session.
func WitnessInvalidation(mode InvalidationMode, coldRebuildTokens int64, stalenessTurns int) CacheInvalidationWitness {
	if coldRebuildTokens < 0 {
		coldRebuildTokens = 0
	}
	if stalenessTurns < 0 {
		stalenessTurns = 0
	}
	w := CacheInvalidationWitness{Mode: mode, ColdRebuildTokens: coldRebuildTokens}
	if mode == InvalidationNow {
		// opt-in: pay the cold rebuild now; the change is in effect this turn, no staleness.
		return w
	}
	// deferred default: avoid the cold rebuild now; the change lands next session.
	w.TokensSavedByDeferring = coldRebuildTokens
	w.StalenessTurns = stalenessTurns
	w.EffectiveNextSession = true
	return w
}

// InvalidationTradeoff witnesses the trade EACH WAY for one mutation, so an operator (or a
// command deciding whether to surface `--now`) can see both costs before choosing.
type InvalidationTradeoff struct {
	Deferred CacheInvalidationWitness `json:"deferred"`
	Now      CacheInvalidationWitness `json:"now"`
}

// WitnessInvalidationTradeoff prices both modes for the same mutation.
func WitnessInvalidationTradeoff(coldRebuildTokens int64, stalenessTurns int) InvalidationTradeoff {
	return InvalidationTradeoff{
		Deferred: WitnessInvalidation(InvalidationDeferred, coldRebuildTokens, stalenessTurns),
		Now:      WitnessInvalidation(InvalidationNow, coldRebuildTokens, stalenessTurns),
	}
}

// RecommendMode returns the mode with the better trade given the operator's tolerance for
// staleness: defer when a cold rebuild would actually cost tokens AND the change tolerates
// the staleness window; otherwise invalidate now. A free rebuild (0 tokens) is never worth
// opening a staleness window, so it recommends Now. This is decision support, not a mandate:
// a mutation that must land this session still passes InvalidationNow regardless.
func (t InvalidationTradeoff) RecommendMode(maxStalenessTurns int) InvalidationMode {
	if t.Deferred.ColdRebuildTokens <= 0 {
		return InvalidationNow
	}
	if t.Deferred.StalenessTurns > maxStalenessTurns {
		return InvalidationNow
	}
	return InvalidationDeferred
}

// CacheInvalidationReport is the operator readout the issue asks for: a session's mutation
// witnesses folded into the total cold-rebuild cost AVOIDED by deferring vs the cost PAID by
// the opt-in immediate mutations, plus the worst staleness window any deferral incurred.
type CacheInvalidationReport struct {
	Mutations int `json:"mutations"`
	Deferred  int `json:"deferred"`
	Immediate int `json:"immediate"`
	// TokensSavedByDeferring is the headline: cold-rebuild tokens the session did not pay
	// because its mutations deferred.
	TokensSavedByDeferring int64 `json:"tokens_saved_by_deferring"`
	// ColdRebuildTokensPaid is the cold-rebuild cost the opt-in `--now` mutations chose to pay.
	ColdRebuildTokensPaid int64 `json:"cold_rebuild_tokens_paid"`
	// MaxStalenessTurns is the worst not-yet-in-effect window any deferred mutation opened.
	MaxStalenessTurns int `json:"max_staleness_turns"`
}

// FoldCacheInvalidation folds per-mutation witnesses into the session readout. It is the
// witnessed report of the cold-rebuild cost avoided by deferring — the report, not
// per-command discipline, prices the trade.
func FoldCacheInvalidation(witnesses []CacheInvalidationWitness) CacheInvalidationReport {
	var r CacheInvalidationReport
	for _, w := range witnesses {
		r.Mutations++
		if w.Mode == InvalidationNow {
			r.Immediate++
			r.ColdRebuildTokensPaid += w.ColdRebuildTokens
			continue
		}
		r.Deferred++
		r.TokensSavedByDeferring += w.TokensSavedByDeferring
		if w.StalenessTurns > r.MaxStalenessTurns {
			r.MaxStalenessTurns = w.StalenessTurns
		}
	}
	return r
}
