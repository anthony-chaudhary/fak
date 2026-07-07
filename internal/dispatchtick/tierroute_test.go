package dispatchtick

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// acct builds a minimal routable account row at a given ModelTier + availability.
func acct(name string, modelTier int, available bool) AccountRow {
	return AccountRow{Account: name, ModelTier: modelTier, Available: available, Kind: "worker"}
}

func tagged(required, optimal modelroute.WorkTier) IssueTier {
	return IssueTier{Required: required, Optimal: optimal, HasTier: true}
}

// Exact: when the optimal tier's account is available, the chooser picks it and
// reports no waste.
func TestTierRouteExactOptimal(t *testing.T) {
	rows := []AccountRow{acct("frontier", 1, true), acct("mid", 2, true), acct("small", 3, true)}
	// Normal implementation: required T1, optimal T1 -> model tier 2.
	res := RouteAccountForTier(rows, "", tagged(modelroute.TierT1, modelroute.TierT1))
	if !res.OK || res.Account.Account != "mid" {
		t.Fatalf("expected mid (tier2) for normal work, got %+v", res)
	}
	if res.OverTier || res.UnderTierRefused {
		t.Fatalf("exact optimal should be neither over nor under: %+v", res)
	}
	if res.FallbackReason != TierReasonOptimalAvailable {
		t.Fatalf("reason = %q, want %q", res.FallbackReason, TierReasonOptimalAvailable)
	}
}

// Cost control: routine work (optimal T2) prefers the cheapest account even when
// frontier and mid seats are free.
func TestTierRouteCheapestForRoutine(t *testing.T) {
	rows := []AccountRow{acct("frontier", 1, true), acct("mid", 2, true), acct("small", 3, true)}
	res := RouteAccountForTier(rows, "", tagged(modelroute.TierT2, modelroute.TierT2))
	if !res.OK || res.Account.Account != "small" {
		t.Fatalf("routine work should pick small (tier3), got %+v", res)
	}
	if res.OverTier {
		t.Fatalf("cheapest-for-routine should not be over-tier: %+v", res)
	}
}

// Upward fallback: routine work whose cheapest tiers are unavailable falls UP to a
// more capable account and reports over-tier waste — never refuses.
func TestTierRouteUpwardFallbackIsWaste(t *testing.T) {
	rows := []AccountRow{acct("frontier", 1, true), acct("mid", 2, false), acct("small", 3, false)}
	res := RouteAccountForTier(rows, "", tagged(modelroute.TierT2, modelroute.TierT2))
	if !res.OK || res.Account.Account != "frontier" {
		t.Fatalf("expected upward fallback to frontier, got %+v", res)
	}
	if !res.OverTier {
		t.Fatalf("using frontier for routine work must flag over-tier waste: %+v", res)
	}
	if res.FallbackReason != TierReasonOverTierFallback {
		t.Fatalf("reason = %q, want %q", res.FallbackReason, TierReasonOverTierFallback)
	}
}

// A capable seat at its session cap is set aside; the chooser spills to the
// next under-cap seat, and refuses with the cap reason when none remains.
func TestTierRouteSkipsAndRefusesAtSessionCap(t *testing.T) {
	t.Setenv(SessionsPerAccountEnv, "")
	full := acct("frontier-full", 1, true)
	full.LiveSessions = DefaultClaudeSessionsPerAccount
	free := acct("frontier-free", 1, true)
	res := RouteAccountForTier([]AccountRow{full, free}, "", tagged(modelroute.TierT0, modelroute.TierT0))
	if !res.OK || res.Account.Account != "frontier-free" {
		t.Fatalf("expected spill to the under-cap frontier seat, got %+v", res)
	}
	res = RouteAccountForTier([]AccountRow{full}, "", tagged(modelroute.TierT0, modelroute.TierT0))
	if res.OK || res.UnderTierRefused || res.FallbackReason != TierReasonAllAtSessionCap {
		t.Fatalf("expected session-cap refusal (not under-tier), got %+v", res)
	}
	if len(res.Blocked) != 1 || !strings.Contains(res.Blocked[0].BlockReason, "at session cap") {
		t.Fatalf("blocked = %+v, want the at-cap seat named", res.Blocked)
	}
}

// Fail closed (the done condition): a fixture whose only available account is
// BELOW the required floor is refused, not launched.
func TestTierRouteUnderTierRefused(t *testing.T) {
	// Ultra-hard work (required T0 -> needs model tier 1) with only a tier-3 seat.
	rows := []AccountRow{acct("frontier", 1, false), acct("small", 3, true)}
	res := RouteAccountForTier(rows, "", tagged(modelroute.TierT0, modelroute.TierT0))
	if res.OK {
		t.Fatalf("under-tier work must be refused, got OK with %+v", res)
	}
	if !res.UnderTierRefused {
		t.Fatalf("expected UnderTierRefused, got %+v", res)
	}
	if res.FallbackReason != TierReasonUnderTierRefused {
		t.Fatalf("reason = %q, want %q", res.FallbackReason, TierReasonUnderTierRefused)
	}
	// The refusal names the too-weak account it turned away.
	if len(res.Blocked) != 1 || res.Blocked[0].Account != "small" {
		t.Fatalf("refusal should list the below-floor account, got %+v", res.Blocked)
	}
}

// A security/release class (required T1 floor, optimal T0) picks the frontier when
// available (optimal) and never falls to routine T2.
func TestTierRouteSecurityFloor(t *testing.T) {
	pol := modelroute.PolicyFor(modelroute.ClassSecurityRelease)
	issue := tagged(pol.RequiredTier, pol.OptimalTier)

	// Frontier available -> chosen (optimal T0).
	rows := []AccountRow{acct("frontier", 1, true), acct("mid", 2, true), acct("small", 3, true)}
	res := RouteAccountForTier(rows, "", issue)
	if !res.OK || res.Account.Account != "frontier" {
		t.Fatalf("security work should pick frontier (optimal T0), got %+v", res)
	}

	// Only a routine seat available -> refused (floor is T1, never T2).
	only3 := []AccountRow{acct("small", 3, true)}
	res2 := RouteAccountForTier(only3, "", issue)
	if res2.OK || !res2.UnderTierRefused {
		t.Fatalf("security work with only a routine seat must refuse, got %+v", res2)
	}
}

// An untagged issue stays conservative: it is treated as frontier-required, so a
// routine-only pool is refused rather than silently down-tiered.
func TestTierRouteUntaggedConservative(t *testing.T) {
	untagged := IssueTier{} // HasTier false
	rows := []AccountRow{acct("small", 3, true)}
	res := RouteAccountForTier(rows, "", untagged)
	if res.OK {
		t.Fatalf("untagged issue must not route to a routine seat, got %+v", res)
	}
	if res.RequiredTier != modelroute.TierT0 {
		t.Fatalf("untagged issue should require T0 (conservative), got %s", res.RequiredTier)
	}

	// With a frontier seat, the untagged issue routes and is flagged conservative.
	rows2 := []AccountRow{acct("frontier", 1, true)}
	res2 := RouteAccountForTier(rows2, "", untagged)
	if !res2.OK || res2.Account.Account != "frontier" {
		t.Fatalf("untagged issue with frontier seat should route to it, got %+v", res2)
	}
}

// No routable accounts at all is a distinct, non-refusal outcome.
func TestTierRouteNoAccounts(t *testing.T) {
	res := RouteAccountForTier(nil, "", tagged(modelroute.TierT1, modelroute.TierT1))
	if res.OK || res.UnderTierRefused {
		t.Fatalf("empty pool is not a floor refusal: %+v", res)
	}
	if res.FallbackReason != TierReasonNoAccounts {
		t.Fatalf("reason = %q, want %q", res.FallbackReason, TierReasonNoAccounts)
	}
}

// The floor invariant: across every tier combo and pool, the chooser never
// returns OK with an account below the required floor.
func TestTierRouteNeverBelowFloor(t *testing.T) {
	pools := [][]AccountRow{
		{acct("a", 1, true)},
		{acct("b", 2, true)},
		{acct("c", 3, true)},
		{acct("a", 1, true), acct("c", 3, true)},
		{acct("a", 1, false), acct("c", 3, true)},
	}
	tiers := []modelroute.WorkTier{modelroute.TierT0, modelroute.TierT1, modelroute.TierT2}
	for _, pool := range pools {
		for _, req := range tiers {
			for _, opt := range tiers {
				res := RouteAccountForTier(pool, "", tagged(req, opt))
				if res.OK && res.Account.ModelTier > modelTierFloorForWork(req) {
					t.Fatalf("chose ModelTier %d below floor for required %s: %+v",
						res.Account.ModelTier, req, res)
				}
			}
		}
	}
}
