package accounts

import (
	"fmt"
	"strings"
	"time"
)

// canary.go gates cooldown EXIT on a successful canary round-trip (#3389,
// borrowed from LMCache's remote-backend health check, borrow id
// H2-canary-recovery). Timer-only recovery — "the window elapsed, so the
// account must be free" — re-admits a seat on hope, not evidence: a weekly cap
// that mis-parsed its reset, an upstream still shedding load, or a revoked
// credential all read as "recovered" the moment ResetAt passes, and the pool
// dispatches straight back into the wall it just backed off from.
//
// The gate is OPT-IN per store: arming it (RequireCanaryExit) flips cooldown
// exit from timer-elapse to probation — an entry whose every signal window has
// elapsed is still reported cooled (CooledDown true, Probation set, Prune
// retains it) until CanaryExit witnesses ONE successful canary round-trip for
// that account. A failed round-trip keeps the account cooled. An unarmed store
// keeps the historical timer-only behavior byte-for-byte, so existing callers
// and the fleet-shared on-disk file are untouched until a caller opts in.
//
// The canary itself is injected, not owned here: production wires a real
// round-trip (the dispatch layer's live probe), tests wire a fake. The store
// only sequences it — never probing while a window still holds, and only
// re-admitting on a witnessed success.

// CooldownCanary is the injectable round-trip probe the probation gate runs
// before re-admitting an account whose cooldown window has elapsed. It receives
// the cooldown store's account key and returns nil only when a REAL round-trip
// against that account succeeded. Any error keeps the account cooled.
type CooldownCanary func(account string) error

// RequireCanaryExit arms the probation gate with probe: from this call on,
// cooldown exit requires a successful canary round-trip (via CanaryExit)
// instead of a mere timer elapse. Passing nil disarms the gate, restoring
// timer-only recovery. The probe is process-local state, never persisted — a
// reloaded store is unarmed until its owner arms it again.
func (s *CooldownStore) RequireCanaryExit(probe CooldownCanary) { s.canary = probe }

// canaryArmed reports whether cooldown exit requires a canary round-trip.
func (s *CooldownStore) canaryArmed() bool { return s.canary != nil }

// CanaryExit attempts to move account out of cooldown through the probation
// gate and reports whether the account is now free of cooldown. It never runs
// the canary early: while any signal window still holds, it returns false
// without probing (the timer half of the latch is not in question). Once every
// window has elapsed, it runs the armed canary exactly once — a nil result
// clears the entry (the account re-enters the pool), an error keeps the
// account cooled in probation and is returned so the caller can log or back
// off. On an unarmed store it simply realizes the timer-only exit (drops the
// elapsed entry), matching what CooledDown already reports.
func (s *CooldownStore) CanaryExit(account string, now time.Time) (bool, error) {
	account = strings.TrimSpace(account)
	if account == "" {
		return true, nil
	}
	e, ok := s.entries[account]
	if !ok {
		return true, nil
	}
	if signals, _ := activeCooldownSignals(e, now); len(signals) > 0 {
		return false, nil
	}
	if s.canaryArmed() {
		if err := s.canary(account); err != nil {
			return false, fmt.Errorf("accounts: canary exit %s: %w", account, err)
		}
	}
	delete(s.entries, account)
	s.syncDegraded(now)
	return true, nil
}
