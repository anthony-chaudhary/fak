package accounts

import (
	"context"
	"time"

	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

// rehydrate.go — the #1183 credential-revalidation RUNG for the staged wake gate
// (internal/rehydrate, #1181). Like internal/leaseref (the lease fence) and internal/recall
// (the recall reverify), this leaf supplies one rung's CHECK; the orchestrator owns WHEN it
// fires. It adds NO new behavior to the existing identity probe — it is a thin, injectable
// gate over it.
//
// Today accounts derives identity from disk but tracks no token expiry (HasCreds is
// file-existence, blind to age), so a stale OAuth token is found reactively on a 401 — the
// ~13-min headless gap operators hit. This rung turns that into a wake-time gate: on a Cold+
// re-entry it checks whether the credential is still fresh given how long the image slept and,
// when it is not, forces a refresh before the first upstream request, refusing with STALE_CRED
// (route to the AUTH / rehome path) when a login wall blocks the refresh rather than failing
// mid-request.

// CredFreshness reports whether a credential last refreshed at lastRefresh is still fresh at
// now, given window (the token's usable lifetime) — the dormancy-gated age check the issue
// asks for: a token stamped older than its window is stale. A zero lastRefresh or a
// non-positive window is treated as stale (fail-closed: re-verify rather than assume fresh).
func CredFreshness(lastRefresh, now time.Time, window time.Duration) bool {
	if lastRefresh.IsZero() || window <= 0 {
		return false
	}
	return now.Sub(lastRefresh) <= window
}

// CredCheck is the injected revalidation seam — the boundary between this rung's staging
// logic and the real OAuth refresh (or a test stub). It reports whether the credential is
// fresh and, when it is not, whether a refresh succeeded in place. The production check reads
// the token's age (e.g. via CredFreshness over the credential file's last-refresh time) and
// attempts the refresh; a test injects a deterministic stub. refreshed is meaningful only
// when fresh is false.
type CredCheck func(ctx context.Context) (fresh bool, refreshed bool)

// NewRehydrateCredRung builds the StaleCred rung from a freshness/refresh check. It fires at
// the canonical StaleCred band (Cold — a token may lapse within the cold gap). On a Cold+
// wake: a fresh credential clears; a stale one that refreshed in place clears (the refresh
// ran before the first request); a stale one that could not refresh (a login wall) refuses
// with STALE_CRED so the caller routes to re-auth instead of failing mid-request. A nil check
// fails closed (refuse — never admit on a credential it cannot vouch for).
func NewRehydrateCredRung(check CredCheck) rehydrate.Rung {
	return rehydrate.NewRung(rehydrate.StaleCred, func(ctx context.Context) rehydrate.Verdict {
		if check == nil {
			return rehydrate.Refuse(rehydrate.StaleCred,
				"credential check unavailable; re-authenticate before the first request")
		}
		fresh, refreshed := check(ctx)
		if fresh || refreshed {
			return rehydrate.Clear()
		}
		return rehydrate.Refuse(rehydrate.StaleCred,
			"credential is past its refresh window and could not refresh; route to re-auth before serving")
	})
}
