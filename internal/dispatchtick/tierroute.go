package dispatchtick

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// ---------------------------------------------------------------------------
// TIER-AWARE ACCOUNT CHOOSER — spend the cheapest account that still meets a
// task's required tier, and FAIL CLOSED when the only accounts available are
// below it.
// ---------------------------------------------------------------------------
//
// This is the "account chooser" node of C5's working path (#3042):
//
//	issue tier metadata  ->  THIS  ->  worker launch
//
// It consumes the tier vocabulary C3 established (modelroute.WorkTier + the
// per-work-class policy). The asymmetry from C3 carries all the way down here:
//   - OVER-TIER is WASTE. Running a scarce frontier account on routine work is
//     allowed but flagged (OverTier + a fallback reason), so cost is visible.
//   - UNDER-TIER is RISK. If no available account meets the required floor, the
//     chooser REFUSES (UnderTierRefused) rather than launch a too-weak worker on
//     high-stakes work. This is the load-bearing default.
//
// THE TWO LADDERS. A unit of WORK has a tier T0..T2 (T0 hardest). An ACCOUNT has
// a ModelTier 1..3 (tier 1 = frontier, the most capable). The two run in the
// SAME direction — the hardest work wants the most capable model — so the
// mapping is monotone and there is no numbering trap ACROSS the ladders (the trap
// C3 warns about is WITHIN the T0<T1<T2 ladder, and stays confined to modelroute).
//
// PURITY: a pure fold over the passed account rows — no I/O, no launch. It is
// dead-safe until the dispatch dry-run and status surface wire it in (the
// remaining C5 step); nothing routes through it yet.

// modelTierFloorForWork maps a WORK tier to the LEAST-capable account ModelTier
// that may still serve it — the risk floor expressed in account-tier units. An
// account qualifies for required work tier R iff its ModelTier <= this floor
// (a lower ModelTier number is MORE capable). T0 -> 1 (frontier only), T1 -> 2,
// T2 -> 3 (any).
func modelTierFloorForWork(t modelroute.WorkTier) int {
	switch t {
	case modelroute.TierT0:
		return 1
	case modelroute.TierT1:
		return 2
	default: // TierT2 and any out-of-range value stay at the routine floor
		return 3
	}
}

// workTierForModelTier is the inverse view: the work tier an account's ModelTier
// optimally serves (1 -> T0, 2 -> T1, 3 -> T2). Used only for reporting the
// chosen account's tier back to the operator.
func workTierForModelTier(modelTier int) modelroute.WorkTier {
	switch modelTier {
	case 1:
		return modelroute.TierT0
	case 2:
		return modelroute.TierT1
	default:
		return modelroute.TierT2
	}
}

// IssueTier is the per-issue tier metadata the dispatcher reads. Required is the
// floor (never violated); Optimal is the ideal target. HasTier distinguishes a
// tagged issue from an untagged one: a missing tier stays CONSERVATIVE (treated
// as frontier-required) rather than silently inferring routine — an untagged
// issue must not be quietly handed to a weak worker.
type IssueTier struct {
	Required modelroute.WorkTier
	Optimal  modelroute.WorkTier
	HasTier  bool
}

// resolve fills a conservative default for an untagged issue: require the
// frontier (T0) so nothing routes below it until C4 tags the issue.
func (it IssueTier) resolve() (required, optimal modelroute.WorkTier, conservative bool) {
	if !it.HasTier {
		return modelroute.TierT0, modelroute.TierT0, true
	}
	return it.Required, it.Optimal, false
}

// Closed-vocabulary fallback reasons — a status surface renders these verbatim.
const (
	TierReasonOptimalAvailable     = "optimal-tier-available"
	TierReasonOverTierFallback     = "over-tier-fallback-optimal-unavailable"
	TierReasonCheaperThanOptimal   = "below-optimal-cheapest-acceptable"
	TierReasonUnderTierRefused     = "under-tier-refused-no-account-meets-floor"
	TierReasonUntaggedConservative = "untagged-issue-conservative-frontier-floor"
	TierReasonNoAccounts           = "no-routable-accounts"
)

// TierRouteResult is the chooser's decision, carrying every field C5 must emit so
// dispatch status is explainable without free text.
type TierRouteResult struct {
	OK               bool                `json:"ok"`
	Account          AccountRow          `json:"-"`
	ChosenModelTier  int                 `json:"chosen_model_tier"`
	ChosenTier       modelroute.WorkTier `json:"chosen_tier"`
	RequiredTier     modelroute.WorkTier `json:"required_tier"`
	OptimalTier      modelroute.WorkTier `json:"optimal_tier"`
	OverTier         bool                `json:"over_tier"`
	UnderTierRefused bool                `json:"under_tier_refused"`
	FallbackReason   string              `json:"fallback_reason"`
	Blocked          []AccountRow        `json:"-"`
}

// RouteAccountForTier chooses an account for an issue's tier metadata. It filters
// rows to the routable, available pool (respecting an optional product filter),
// keeps only accounts that MEET the required floor, and picks the one CLOSEST to
// the optimal tier, breaking ties toward the cheaper (less capable) account so a
// frontier seat is not spent when a cheaper eligible one is equally close.
//
// If no available account meets the floor it returns OK=false with
// UnderTierRefused — the fail-closed default. Over-tier selection (the cheapest
// eligible account is still more capable than optimal) is reported via OverTier
// and a fallback reason, never refused.
func RouteAccountForTier(rows []AccountRow, product string, issue IssueTier) TierRouteResult {
	required, optimal, conservative := issue.resolve()
	res := TierRouteResult{RequiredTier: required, OptimalTier: optimal}
	if conservative {
		res.FallbackReason = TierReasonUntaggedConservative
	}

	floor := modelTierFloorForWork(required)
	optimalModelTier := modelTierFloorForWork(optimal)

	normProduct, _, workers := normalizeWorkerPool(rows, product, "")
	_ = normProduct
	if len(workers) == 0 {
		res.FallbackReason = TierReasonNoAccounts
		return res
	}

	// Eligible = available accounts capable enough for the floor (ModelTier <= floor).
	eligible := []AccountRow{}
	for _, row := range workers {
		if row.Available && row.ModelTier <= floor {
			eligible = append(eligible, row)
		}
	}
	if len(eligible) == 0 {
		res.UnderTierRefused = true
		res.FallbackReason = TierReasonUnderTierRefused
		res.Blocked = belowFloorAccounts(workers, floor)
		return res
	}

	// Choose the account whose ModelTier is closest to the optimal tier; on a tie
	// prefer the cheaper (higher ModelTier number) account, then fall back to the
	// stable account ordering so the choice is deterministic.
	sort.Slice(eligible, func(i, j int) bool {
		di := abs(eligible[i].ModelTier - optimalModelTier)
		dj := abs(eligible[j].ModelTier - optimalModelTier)
		if di != dj {
			return di < dj
		}
		if eligible[i].ModelTier != eligible[j].ModelTier {
			return eligible[i].ModelTier > eligible[j].ModelTier // cheaper first
		}
		return accountRouteLess(eligible[i], eligible[j])
	})

	chosen := eligible[0]
	res.OK = true
	res.Account = chosen
	res.ChosenModelTier = chosen.ModelTier
	res.ChosenTier = workTierForModelTier(chosen.ModelTier)

	switch {
	case chosen.ModelTier < optimalModelTier:
		// More capable than optimal — waste, because the optimal tier had no
		// available account (or was too weak for the floor).
		res.OverTier = true
		if res.FallbackReason == "" || res.FallbackReason == TierReasonUntaggedConservative {
			res.FallbackReason = TierReasonOverTierFallback
		}
	case chosen.ModelTier > optimalModelTier:
		// Cheaper than the ideal but still at/above the required floor — a saving.
		if res.FallbackReason == "" {
			res.FallbackReason = TierReasonCheaperThanOptimal
		}
	default:
		if res.FallbackReason == "" {
			res.FallbackReason = TierReasonOptimalAvailable
		}
	}
	return res
}

// belowFloorAccounts lists the available accounts that were rejected for being
// below the required floor, so a refusal can name what it turned away.
func belowFloorAccounts(workers []AccountRow, floor int) []AccountRow {
	out := []AccountRow{}
	for _, row := range workers {
		if row.Available && row.ModelTier > floor {
			out = append(out, row)
		}
	}
	return out
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
