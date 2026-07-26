package gateway

// The DEFER_ENABLED_BUT_INERT live watchdog (#3621, cache-verify epic #3569): the
// cold-tool-defer lever (#3232 — the epic's highest-risk lever) marks the cold custom tool
// tail defer_loading:true and injects a tool_search_tool, but it does NOT shrink request
// bytes; its whole payoff is provider-side and therefore only OBSERVABLE on a live run. The
// existing counter family (fak_gateway_tool_defer_cold_total / _turns_total) witnesses that
// fak AUTHORED the transform — it is a pure numerator. Nothing witnessed the inert case: a
// session armed with --defer-cold-tools where the transform ran every turn and bit on none of
// them (a wrong dated tool_search_tool type, a client body already carrying defer_loading, an
// all-hot surface) reads on every surface exactly like a session where the lever was never
// turned on at all, because both render a flat zero.
//
// The fold below closes that gap by pairing the numerator with the denominator
// metrics_observe.go now books (deferStandDownTurns): defer-ELIGIBLE turns that stood down to
// identity. Two live surfaces raise the finding — the /debug/vars cache_attribution
// fak_defer_finding field (cacheAttributionVars) and the guard exit banner (cmd/fak
// formatAuditSummary) — the same pair the sibling UPGRADE_NEVER_FIRED watchdog uses.
//
// Like that sibling this is WITNESSED-only: stand-down rows accrue exclusively past
// maybeDeferColdTools' eligibility gate (lever on, Anthropic passthrough wire, ablation arm
// off), so a lever-off session, a non-Anthropic wire, an ablated A/B arm, or a cold process
// can never trip it — no posture flag needs threading in, and the banner can raise the same
// finding from the exit summary alone.

// deferInertMinTurns is the eligible-turn floor the watchdog keys on: below it a zero-defer
// session is merely short (a turn or two before the client's real tool surface arrives is not
// evidence of anything), at or past it the lever has had several live requests to bite on and
// still never did — the armed-but-inert pathology rather than a cold start. Mirrors the
// sibling upgradeNeverFiredMinTurns floor so the two watchdogs read alike.
const deferInertMinTurns = 3

// DeferAttempts is the total defer-ELIGIBLE turns this session witnessed: turns the transform
// actually fired on (DeferColdTurns) plus every turn it ran and stood down to identity
// (DeferStandDownTurns). This is the denominator of the fired-vs-inert ratio the #3621
// watchdog tracks, and it is nonzero ONLY on a session that armed the lever on an Anthropic
// wire outside the ablation arm.
func (s AdjudicationSummary) DeferAttempts() uint64 {
	return s.DeferColdTurns + s.DeferStandDownTurns
}

// DeferEnabledButInert reports the #3621 finding: the lever was armed for at least
// deferInertMinTurns eligible turns and not one cold tool definition was ever deferred. A
// single deferred def clears it for the session's lifetime; a session below the floor, with
// the lever off, on a non-Anthropic wire, or in the ablation arm never raises it (no eligible
// turns accrue there at all).
func (s AdjudicationSummary) DeferEnabledButInert() bool {
	return s.DeferColdCount == 0 && s.DeferAttempts() >= deferInertMinTurns
}
