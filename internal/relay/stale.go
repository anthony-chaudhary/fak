// Rung D-stale (issue #1878): the typed baton-stale outcome. The D1 reload verifier
// (#1877) decides fresh|stale; this rung names a stale cursor as a CLOSED relay outcome —
// RELAY_BATON_STALE — carrying the deciding (culprit) claim and the git evidence behind
// it, so a driver (H-track) can route to re-derivation from durable state instead of
// trusting a baton whose progress anchor no longer matches ground truth. It adds no driver
// wiring; it only turns the verifier's verdict into the emittable outcome.
package relay

// ReasonBatonStale is the relay reason token emitted when a baton's ProgressCursor has
// diverged from ground truth (RELAY-REASON-VOCABULARY-2026-07-01.md). It is a member of
// the closed reason vocabulary the successor routes on.
const ReasonBatonStale = "RELAY_BATON_STALE"

// StaleOutcome is the typed result of checking a baton for staleness. When Stale is false
// it is the zero outcome (the cursor still matches git). When Stale is true it carries the
// RELAY_BATON_STALE reason, the Culprit (which cursor anchor diverged — the deciding
// claim), and the git Evidence behind the verdict. Evidence is display-only; it is never
// consumed as progress.
type StaleOutcome struct {
	Stale    bool   `json:"stale"`
	Reason   string `json:"reason,omitempty"`
	Culprit  string `json:"culprit,omitempty"`
	Evidence string `json:"evidence,omitempty"`
}

// CheckBatonStale re-verifies b's ProgressCursor against git through the D1 verifier and,
// on divergence, returns a RELAY_BATON_STALE outcome naming the culprit claim and its git
// evidence. A fresh cursor returns the zero (non-stale) outcome. Pure over the injected
// Resolver: it reads no clock and does I/O only through the resolver.
func CheckBatonStale(b Baton, r Resolver) StaleOutcome {
	rr := VerifyReload(b.ProgressCursor, r)
	if rr.Verdict == ReloadFresh {
		return StaleOutcome{Stale: false}
	}
	return StaleOutcome{
		Stale:    true,
		Reason:   ReasonBatonStale,
		Culprit:  rr.Anchor,
		Evidence: rr.Reason,
	}
}
