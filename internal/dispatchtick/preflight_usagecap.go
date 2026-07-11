package dispatchtick

// preflight_usagecap.go -- the usage_cap ADVISORY preflight term: surface a MEASURED
// fleet-wide usage/weekly-cap exhaustion so an operator (or the dispatch loop) can see
// that fresh spawns are futile until the caps reset -- WITHOUT gating admission.
//
// This term is deliberately NOT a cap term. The documented effective-cap formula
// (docs/safe-to-raise-cap-checklist.md) folds host/seat/at-cap/account/gate/rate_budget
// DOWN into the spawn decision; this one folds NOTHING. It reads a signal and attaches a
// note. Two reasons it must stay advisory-only:
//
//  1. A usage cap is not cleared by backing off concurrency (unlike a transient 429). The
//     seat is quota-exhausted until a wall-clock reset; refusing dispatch fleet-wide would
//     just starve the multi-account goal on the seats that are still healthy. The already
//     -cooled accounts are ALREADY dropped from the servable pool by the cooldown overlay
//     (dispatchApplyAccountCooldown), so the cap decision is already correct -- what is
//     missing is VISIBILITY into WHY the pool is thin.
//
//  2. The signal is authoritative precisely because it does NOT come from the witness
//     classifier. A worker that walls on a usage cap frequently prints a 429/529-shaped
//     error that ClassifyNoCommitReason grades reason=rate_limit (IsAPIError matches the
//     tail before the usage-cap banner does), so the rate_budget term fires on the
//     MISLABEL and tells the loop "transient overload -- route to another provider,"
//     holding a cold-start probe that keeps re-walling and cooling the next fresh seat.
//     The guard/launch layer, by contrast, records the seat's cooldown with Kind
//     "usage-limit" (it saw the reset instant). So the cooldown store -- read here -- is
//     the one place that knows the truth: these accounts are usage-capped until <reset>,
//     and concurrency backoff will not help. This advisory carries that truth forward.
//
// UNIT NOTE (load-bearing): the census is measured in ACCOUNTS, not seats. A claude
// account carries several session seats (DefaultClaudeSessionsPerAccount), so a
// seat-count and an account-count are different units; the arming decision stays entirely
// in accounts (capped accounts vs total accounts) so the two are never mixed. FreeSeats
// is carried only as CONTEXT for the human hint and is never part of Armed.
//
// Pure: state in (an account census), note out. The zero value is not armed, so a caller
// that wires nothing keeps byte-identical behavior.

import (
	"fmt"
	"time"
)

// UsageCapExhaustionSignal is the stable machine-readable label the advisory carries in
// its note. It is NOT a closed-vocabulary REFUSAL token (the advisory refuses nothing) --
// it is an observability tag so a loop or dashboard can match the condition without
// scraping the human hint. It intentionally does not collide with RateLimitBackoff.
const UsageCapExhaustionSignal = "usage_cap_exhaustion"

// DefaultUsageCapAdvisoryMin is the arming floor: the fewest accounts under an active
// usage-limit cooldown that count as a genuine fleet-wide exhaustion rather than ordinary
// churn (one account rotating through its weekly reset). A single capped account on an
// otherwise healthy fleet is noise; the default requires a cluster. The impure shell
// overlays FAK_USAGECAP_ADVISORY_MIN.
const DefaultUsageCapAdvisoryMin = 3

// UsageCapAdvisory carries the MEASURED usage-cap census the advisory renders. Capped is
// the count of routable accounts the cooldown store holds under an active
// Kind="usage-limit" entry at the census instant; Accounts is the total routable-account
// count for this backend; FreeSeats is the servable pool's free-seat count (context
// only); EarliestReset is the soonest reset instant among the capped accounts (zero when
// unknown). All fields are supplied by the impure shell from the authoritative cooldown
// store and seat pool -- never a worker self-report.
type UsageCapAdvisory struct {
	// Capped is the number of accounts under an active usage-limit-kind cooldown now.
	Capped int
	// Accounts is the total number of routable accounts in the pool for this backend.
	Accounts int
	// FreeSeats is the pool's free-seat count, carried only as context for the hint. It
	// is a DIFFERENT unit from Capped/Accounts and is deliberately never used in Armed.
	FreeSeats int
	// EarliestReset is the soonest reset instant among the capped accounts, or the zero
	// time when none was parseable. Used only to cite "hold until ~<reset>".
	EarliestReset time.Time
	// Threshold is the arming floor on Capped; <= 0 means DefaultUsageCapAdvisoryMin.
	Threshold int
	// Now is the census instant, used to render the remaining time to reset. The zero
	// value omits the remaining-time rendering (the absolute instant still renders).
	Now time.Time
}

// threshold resolves the arming floor, defaulting a zero/negative Threshold to the
// built-in DefaultUsageCapAdvisoryMin so the zero-value census stays hermetic.
func (u UsageCapAdvisory) threshold() int {
	if u.Threshold <= 0 {
		return DefaultUsageCapAdvisoryMin
	}
	return u.Threshold
}

// Armed reports whether the census describes a genuine usage-cap exhaustion worth
// surfacing. Two conditions must BOTH hold, and both are measured in ACCOUNTS so the
// comparison never mixes units: the count of usage-capped accounts clears the arming
// floor (a real cluster, not one account rotating through its reset), AND the caps cover
// at least HALF the routable fleet (Capped*2 >= Accounts). The half-fleet condition is
// what keeps the advisory honest on a healthy fleet: three capped accounts out of twenty
// is churn with ample headroom, not exhaustion, and does not arm. It arms only when the
// caps have taken over the majority of the fleet, i.e. the accounts still healthy are the
// exception. Pure: state in, decision out.
func (u UsageCapAdvisory) Armed() bool {
	return u.Accounts > 0 && u.Capped >= u.threshold() && u.Capped*2 >= u.Accounts
}

// Note renders the advisory as a structured record for the preflight output map. It is
// meaningful only when Armed; an unarmed census returns nil so the caller omits the field
// entirely (keeping the common preflight payload byte-identical). The record carries the
// census counts, the machine-readable signal label, the reset instant, an explicit
// advisory_only=true marker, and an actionable human hint that names the mislabel trap so
// a reader does not confuse this with the rate_budget backoff.
func (u UsageCapAdvisory) Note() map[string]any {
	if !u.Armed() {
		return nil
	}
	note := map[string]any{
		"signal":          UsageCapExhaustionSignal,
		"advisory_only":   true,
		"capped_accounts": u.Capped,
		"total_accounts":  u.Accounts,
		"free_seats":      u.FreeSeats,
		"hint":            u.hint(),
	}
	if !u.EarliestReset.IsZero() {
		note["earliest_reset"] = u.EarliestReset.UTC().Format(time.RFC3339)
	}
	return note
}

// hint renders the actionable human sentence. It names the count, the reset horizon, and
// -- critically -- the mislabel trap (a usage cap the witness may grade rate_limit), so
// the operator understands why concurrency backoff will not clear it and that this note
// blocks nothing.
func (u UsageCapAdvisory) hint() string {
	reset := "an upstream reset"
	if !u.EarliestReset.IsZero() {
		reset = "~" + u.EarliestReset.UTC().Format(time.RFC3339)
		if !u.Now.IsZero() {
			if remaining := u.EarliestReset.Sub(u.Now); remaining > 0 {
				reset = fmt.Sprintf("%s (%s away)", reset, remaining.Round(time.Minute))
			}
		}
	}
	return fmt.Sprintf("%d of %d account(s) are in a usage-limit cooldown (a self-recovering weekly/usage cap, not a transient 429), leaving only %d free seat(s); the free seat(s) are likely near their own caps too. A fresh spawn that walls here is a usage cap the witness classifier often mislabels reason=rate_limit -- concurrency backoff will NOT clear it. Advisory only: dispatch is NOT blocked. Consider holding new spawns until %s, or routing new work to a different provider/backend.",
		u.Capped, u.Accounts, u.FreeSeats, reset)
}
