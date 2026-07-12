package fleetaccounts

import "time"

// capstate.go is the cap-disambiguation core: it resolves a carried throttle's
// "5-hour rolling cap vs weekly limit" question ONCE into a canonical CapState, and the
// predicates in status.go / capacity.go read that state instead of each re-deriving from
// the raw map[string]any. It layers on the reset-string parser core in soonness.go
// (resetTime / resetIsFuture), mirroring the same split on the Python side
// (fleet_accounts.throttle_is_active / _weekly_throttle_is_active read one _reset_is_future
// core).
//
// The load-bearing contract: with a zero CapObservation and the default policy, CapState's
// Active/WeeklyActive reproduce the legacy throttleIsActive/weeklyThrottleIsActive exactly —
// including the fail-closed rule that a present-but-unparseable reset/weekly stays active.
// The aging and probe-override cycles below are additive: they fire only when a real
// observation (a first-seen timestamp or an OK-probe streak, derived from the probe ledger)
// is supplied, so the passive fold is unchanged.

// CapKind classifies which leg of the cap is holding a seat, once disambiguated.
type CapKind int

const (
	// CapNone: no live cap (the throttle is inactive / provably expired).
	CapNone CapKind = iota
	// CapDaily: the 5-hour rolling reset is what holds the seat (no live weekly leg).
	CapDaily
	// CapWeekly: a weekly limit with a parseable, still-future reset holds the seat.
	CapWeekly
	// CapWeeklyUnknown: a weekly leg is present but its reset string is unparseable, so
	// it holds fail-closed (bounded by aging once a first-seen is known).
	CapWeeklyUnknown
)

// CapState is the canonical, disambiguated view of one carried throttle. The raw Reset/
// Weekly strings and their presence bits are carried verbatim because applyThrottleStatus,
// the JSON serializer, and the capacity fold all propagate the originals for byte-parity.
type CapState struct {
	Kind         CapKind
	Active       bool   // weekly-OR-daily liveness, fail-closed on an unparseable reset
	WeeklyActive bool   // the weekly leg's independent liveness (the fresh-OK suppressor)
	Reset        string // raw daily reset string, verbatim
	HasReset     bool   // the "reset" key was present (mirrors Python None-vs-value)
	Weekly       string // raw weekly reset string, verbatim
	HasWeekly    bool   // the "weekly" key was present
	BlockReason  string // composed "usage limit; resets X; weekly Y"
	AgedOut      bool   // the weekly leg was cleared by elapsed-time aging
	OverriddenBy int    // >0: the weekly leg was cleared by an OK-probe streak of this length
}

// EffectiveFreeUp is the single "when does this seat free up" string, weekly-first — the
// value the capacity fold renders as "blocked-until" and the roster uses to order the
// soonest-to-free. Mirrors capacityFirstNonEmpty(weekly, reset).
func (c CapState) EffectiveFreeUp() string {
	if c.Weekly != "" {
		return c.Weekly
	}
	return c.Reset
}

// CapObservation carries the ledger-derived signals the aging and probe-override cycles
// need. The zero value means "no observation" — which reduces DisambiguateCap to the legacy
// single-shot behavior, so every passive caller keeps today's semantics.
type CapObservation struct {
	FirstSeen    time.Time // start of the current block episode (first flip into a cap)
	HasFirstSeen bool
	OKStreak     int // consecutive fresh-OK probes since the last non-OK verdict
}

// CapPolicy tunes the two disambiguation cycles. Both knobs are single constants, easy to
// retune after observing real recovery behavior.
type CapPolicy struct {
	WeeklyMaxAge   time.Duration // age a still-held weekly out after this long (default 7d)
	OverrideStreak int           // OK probes past the daily reset to overturn a hold (default 2)
}

// DefaultCapPolicy: a weekly holds at most one real weekly window (7 days), and two
// consecutive fresh-OK probes past a passed daily reset are enough to overturn a stale hold.
func DefaultCapPolicy() CapPolicy {
	return CapPolicy{WeeklyMaxAge: 7 * 24 * time.Hour, OverrideStreak: 2}
}

// DisambiguateCap resolves a carried throttle map into a canonical CapState at instant now.
// With obs == CapObservation{} it is a pure function of the reset strings and reproduces the
// legacy throttleIsActive / weeklyThrottleIsActive exactly; a non-zero obs enables the aging
// and probe-override cycles.
func DisambiguateCap(thr map[string]any, obs CapObservation, now time.Time, pol CapPolicy) CapState {
	resetVal, hasReset := thr["reset"]
	weeklyVal, hasWeekly := thr["weekly"]
	reset := asString(resetVal)
	weekly := asString(weeklyVal)

	c := CapState{
		Reset: reset, HasReset: hasReset,
		Weekly: weekly, HasWeekly: hasWeekly,
		BlockReason: capBlockReason(reset, weekly),
	}

	// --- base disambiguation (mirrors throttleIsActive / weeklyThrottleIsActive) ---
	weeklyPresent := weekly != ""
	weeklyFuture := resetIsFuture(weekly, now) // *bool: true future, false expired, nil unknown
	weeklyProvablyPast := weeklyPresent && weeklyFuture != nil && !*weeklyFuture
	resetFuture := resetIsFuture(reset, now)
	resetProvablyPast := resetFuture != nil && !*resetFuture

	// throttleIsActive: a weekly not provably past keeps the throttle active on its own;
	// otherwise defer to the daily reset (unknown daily -> active, fail-closed).
	if weeklyPresent && !weeklyProvablyPast {
		c.Active = true
	} else {
		c.Active = !resetProvablyPast
	}
	// weeklyThrottleIsActive: present weekly, not provably past, and the throttle is live.
	c.WeeklyActive = weeklyPresent && !weeklyProvablyPast && c.Active

	// --- aging cycle: bound a fail-closed weekly hold by elapsed time ---
	// A weekly that is unparseable or nominally future would otherwise hold forever. Once
	// we know when the episode started, release it after WeeklyMaxAge; the daily leg then
	// decides (and, unlike the fail-closed default, an unknown daily no longer walls it).
	if c.WeeklyActive && obs.HasFirstSeen && now.Sub(obs.FirstSeen) >= pol.WeeklyMaxAge {
		c.AgedOut = true
		c.WeeklyActive = false
		c.Active = resetFuture != nil && *resetFuture // hold only if the daily reset is provably future
	}

	// --- probe-override cycle: let accumulated fresh-OK probes overturn a stale hold ---
	// A run of OK probes past a passed daily reset is strong evidence the seat has really
	// recovered; overturn the weekly hold rather than let it suppress every probe forever.
	if c.WeeklyActive && !c.AgedOut && obs.OKStreak >= pol.OverrideStreak && resetProvablyPast {
		c.OverriddenBy = obs.OKStreak
		c.WeeklyActive = false
		c.Active = false
	}

	c.Kind = capKind(c, weeklyFuture)
	return c
}

// capKind labels the leg that is holding, for the capacity/doctor views.
func capKind(c CapState, weeklyFuture *bool) CapKind {
	switch {
	case !c.Active:
		return CapNone
	case c.WeeklyActive && weeklyFuture == nil:
		return CapWeeklyUnknown
	case c.WeeklyActive:
		return CapWeekly
	default:
		return CapDaily
	}
}

// capBlockReason composes the block-reason string exactly as applyThrottleStatus did:
// "usage limit" [+ "; resets <reset>"] [+ "; weekly <weekly>"].
func capBlockReason(reset, weekly string) string {
	reason := "usage limit"
	if reset != "" {
		reason = "usage limit; resets " + reset
	}
	if weekly != "" {
		reason += "; weekly " + weekly
	}
	return reason
}
