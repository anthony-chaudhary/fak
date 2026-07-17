package dispatchtick

import (
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
)

// aging_order.go is the OPT-IN anti-starvation ordering path for OrderLaneCandidates (#3588). The
// default picker (defaultLaneScorers) ranks by base priority weight ABSOLUTELY, so a low-priority
// ready issue can be out-weighed by fresher higher-priority arrivals forever — classic starvation.
// internal/dispatchaging.Fold already computes the fair answer (an EFFECTIVE weight = base priority
// + a bounded aging boost, plus a hard starvation deadline), but nothing wired it into the lane
// picker. This file is that seam: behind the default-off FAK_DISPATCH_AGING flag, OrderLaneCandidates
// folds each candidate's wait time (LaneCandidate.ReadySince) through dispatchaging.Fold and consumes
// the per-unit effective weight + starved standing, while keeping the leaf's own NUMERIC recency
// tiebreak so enabling the flag never silently swaps the by-number order for Fold's string-ID one.
//
// # No-regression, two ways
//
//  1. Flag OFF (default): OrderLaneCandidates never calls this file — it returns
//     defaultLaneScorers.Order verbatim, byte-for-byte the pre-aging order (the golden test).
//  2. Flag ON but no wait data: a candidate set that carries no ReadySince (all zero) yields zero
//     wait, zero boost, and no starvation for every unit, so effective weight == base weight and the
//     aged order reproduces the default order exactly (also golden-tested). Aging only reorders when
//     real ReadySince stamps are present, so flipping the flag on a caller that has not yet started
//     stamping ready times is a safe no-op rather than a silent reshuffle.

// FAKDispatchAgingEnv is the env var that opts OrderLaneCandidates into the aging path. Default-off:
// only an explicit truthy value ("1"/"true"/"yes"/"on") enables it; anything else keeps the
// pre-aging order.
const FAKDispatchAgingEnv = "FAK_DISPATCH_AGING"

// agingOrderingEnabled reports whether the opt-in aging path is turned on. Default-off.
func agingOrderingEnabled() bool {
	return agingFlagTruthy(os.Getenv(FAKDispatchAgingEnv))
}

// agingFlagTruthy is the default-off reading of the flag: empty or any unrecognized value is off, so
// the aging path is opt-in and never enabled by accident.
func agingFlagTruthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// agingNowUnix is the wall clock the flag-on path folds against, read only at the gated boundary so
// the pure ordering core (orderLaneCandidatesAged) still takes its clock as data.
func agingNowUnix() int64 { return time.Now().Unix() }

// orderLaneCandidatesAged is the pure aging order: it folds the candidates' wait clocks through
// dispatchaging.Fold and ranks by the fold's verdict — starved units force-served first, then by
// descending EFFECTIVE weight (base + aging boost), then longer wait — with the leaf's existing
// NUMERIC recency tiebreak (numberTiebreak, honoring preferNewest) as the final key. It takes Params
// as data (no clock read), so a test pins a deterministic clock. With zero-wait input (or a
// zero-value Params) every unit is fresh with effective weight == base weight, so the result is the
// default base-weight-then-recency order.
func orderLaneCandidatesAged(cands []LaneCandidate, preferNewest bool, p dispatchaging.Params) []int {
	agingCands := make([]dispatchaging.Candidate, len(cands))
	for i, c := range cands {
		agingCands[i] = dispatchaging.Candidate{
			ID:         strconv.Itoa(c.Number),
			BaseWeight: c.Weight,
			ReadySince: c.ReadySince,
		}
	}
	res := dispatchaging.Fold(agingCands, p)

	// Consume the fold's per-unit verdict keyed by number, so we can re-order the original
	// candidates with a numeric (not string-ID) recency tiebreak.
	type verdict struct {
		eff     int
		wait    int64
		starved bool
	}
	byNumber := make(map[int]verdict, len(res.Order))
	for _, r := range res.Order {
		n, err := strconv.Atoi(r.ID)
		if err != nil {
			continue
		}
		byNumber[n] = verdict{
			eff:     r.EffectiveWeight,
			wait:    r.WaitSeconds,
			starved: r.Standing == dispatchaging.StandingStarved,
		}
	}

	ordered := append([]LaneCandidate(nil), cands...)
	sort.SliceStable(ordered, func(i, j int) bool {
		vi, vj := byNumber[ordered[i].Number], byNumber[ordered[j].Number]
		if vi.starved != vj.starved {
			return vi.starved // force-serve starved units ahead of everything else this tick
		}
		if vi.eff != vj.eff {
			return vi.eff > vj.eff // then descending effective weight (base + aging boost)
		}
		if vi.wait != vj.wait {
			return vi.wait > vj.wait // longer wait first, matching Fold's oldest-first band
		}
		return numberTiebreak(ordered[i].Number, ordered[j].Number, preferNewest)
	})
	return orderedNumbers(len(ordered), func(i int) int { return ordered[i].Number })
}
