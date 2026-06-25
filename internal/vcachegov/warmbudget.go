package vcachegov

import (
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// warmbudget.go is the rate-limit warm-budget scheduler (issue #720, acceptance 2,
// from design §5.5). Warming a prefix spends one request and P input tokens out of
// the tier's RPM/TPM headroom; the scheduler answers two questions:
//
//  1. How many warms can the tier sustain per minute without eating into real
//     traffic's quota? warms/min = min(R − R_real, (X − X_real)/P). The RPM/TPM
//     crossover is P* = (X − X_real)/(R − R_real) (~100 tok on a half-used
//     4000/400k tier), so EVERY realistic anchor (P ≥ 1024) is TPM-bound — the
//     binding constraint is input-token headroom, not request count.
//
//  2. How large a warm set does that sustain over one TTL window?
//     sustainable warm-set = warms/min × T(min). At 5m that is a few hundred 4k
//     anchors; the 1h TTL buys 12× the set for the same headroom.
//
// The load-bearing rule (acceptance 2, last clause): when headroom is short the
// scheduler DEGRADES BY WARMING FEWER ANCHORS (lower hit rate), NEVER by 429-ing
// real traffic. Schedule therefore truncates the ranked candidate list to the
// sustainable set and drops the rest, rather than ever issuing a warm that would
// push real traffic over the rate limit. Budget at the uncached price (Law A3):
// the sustainable set is a best-effort rebate, never a pre-credited SLO.

// RateLimit is the tier's request/token budget and the share real traffic already
// consumes. All four fields are per-minute. R/X are the tier limits; RealRPM/RealTPM
// are the observed real-traffic usage; the difference is the headroom warms draw
// from. These are the M1 calibration outputs (the rate-limit headroom the issue's
// "Depends on" line names).
type RateLimit struct {
	TierRPM float64 // R  — tier requests/min ceiling
	TierTPM float64 // X  — tier input-tokens/min ceiling
	RealRPM float64 // R_real — real-traffic requests/min
	RealTPM float64 // X_real — real-traffic input-tokens/min
}

// WarmBudget is the scheduler's answer for one (tier, anchor-size, TTL) triple.
type WarmBudget struct {
	// WarmsPerMin is the sustainable warm rate inside headroom:
	// min(R − R_real, (X − X_real)/P). Zero when there is no headroom.
	WarmsPerMin float64
	// SustainableSet is how many anchors of size P that rate keeps warm over one
	// TTL window: WarmsPerMin × T(min), truncated to an integer count. Schedule
	// never warms more candidates than this in a TTL window.
	SustainableSet int
	// CrossoverTokens is P* = (X − X_real)/(R − R_real): the anchor size below which
	// the tier is RPM-starved and above which it is TPM-bound. ~100 tok on a
	// half-used 4000/400k tier, so every realistic anchor (P ≥ 1024) clears it.
	CrossoverTokens float64
	// TPMBound reports whether an anchor of size P is TPM-bound (P > P*), the
	// binding regime for every realistic anchor. False only for sub-P* anchors,
	// which the caller should pack behind one shared prefix instead of warming
	// individually.
	TPMBound bool
}

// PlanWarmBudget computes the sustainable warm budget for warming anchors of
// anchorTokens input tokens under rate limit r over a TTL window of ttlMillis.
// It is pure and allocation-free.
func PlanWarmBudget(r RateLimit, anchorTokens float64, ttlMillis int64) WarmBudget {
	availRPM := r.TierRPM - r.RealRPM
	availTPM := r.TierTPM - r.RealTPM

	crossover := 0.0
	if availRPM > 0 {
		crossover = availTPM / availRPM
	}

	warms := 0.0
	if availRPM > 0 && availTPM > 0 && anchorTokens > 0 {
		byTPM := availTPM / anchorTokens
		warms = math.Min(availRPM, byTPM)
	}

	ttlMin := float64(ttlMillis) / 60000.0
	sustainable := int(math.Floor(warms * ttlMin))
	if sustainable < 0 {
		sustainable = 0
	}

	return WarmBudget{
		WarmsPerMin:     warms,
		SustainableSet:  sustainable,
		CrossoverTokens: crossover,
		TPMBound:        crossover > 0 && anchorTokens > crossover,
	}
}

// WarmCandidate is one prefix under consideration for warming. The scheduler ranks
// candidates by the §5.2 cache-working-set signal — frequency × size × reuse-density
// — so the head of the ranked list is the few anchors that cover most of the volume
// on a steep (s≈1.74) workload. Secret/regulated candidates are filtered out before
// ranking (Law D4): a non-cacheable prefix can never be admitted to the warm set
// regardless of how hot it is.
type WarmCandidate struct {
	Key          string
	Frequency    float64 // arrival volume weight (e.g. requests/min for this prefix)
	Size         float64 // anchor input tokens P
	ReuseDensity float64 // reuse intensity — reuse cachemeta.Lifecycle.AccessRatePerSec
	Secret       SecretClassification
}

// RankScore is the §5.2 working-set weight: frequency × size × reuse-density. It is
// the product of three monotone-non-negative axes, so a prefix that is hot, large,
// and densely reused dominates one that is merely any one of those. The caller
// feeds cachemeta.Lifecycle.AccessRatePerSec as ReuseDensity so the ranking reuses
// the existing arrival-intensity signal rather than inventing a parallel one.
func (c WarmCandidate) RankScore() float64 {
	if !Warmable(c.Secret) {
		return 0
	}
	if c.Frequency < 0 {
		return 0
	}
	return c.Frequency * c.Size * c.ReuseDensity
}

// Rank returns the candidates ordered best-first by RankScore, with every
// non-cacheable (secret/regulated) candidate removed. Ties break by Key for
// determinism. The output is a new slice; the input is untouched.
func Rank(candidates []WarmCandidate) []WarmCandidate {
	out := make([]WarmCandidate, 0, len(candidates))
	for _, c := range candidates {
		if Warmable(c.Secret) {
			out = append(out, c)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		ri, rj := out[i].RankScore(), out[j].RankScore()
		if ri != rj {
			return ri > rj
		}
		return out[i].Key < out[j].Key
	})
	return out
}

// Schedule is the admission step the Warmer calls once per TTL window: rank the
// cacheable candidates by working-set weight and truncate to the budget's
// sustainable set. This is where "degrade by warming fewer anchors, never by
// 429-ing" is enforced — when headroom is short, SustainableSet shrinks and the
// tail of the ranked list is simply not warmed this window (lower hit rate),
// rather than issuing a warm that would push real traffic over the rate limit.
// It never returns more candidates than the budget allows or than were supplied.
func Schedule(candidates []WarmCandidate, budget WarmBudget) []WarmCandidate {
	ranked := Rank(candidates)
	n := budget.SustainableSet
	if n < 0 {
		n = 0
	}
	if n > len(ranked) {
		n = len(ranked)
	}
	return ranked[:n]
}

// ProjectCandidate builds a WarmCandidate from a cachemeta.Lifecycle plus the
// canonicalizer's frequency/size/secret signals — the bridge that lets the
// scheduler rank the SAME entries the warm set tracks, reusing
// Lifecycle.AccessRatePerSec as the reuse-density axis.
func ProjectCandidate(key string, lc cachemeta.Lifecycle, frequency, size float64, secret SecretClassification, nowMillis int64) WarmCandidate {
	return WarmCandidate{
		Key:          key,
		Frequency:    frequency,
		Size:         size,
		ReuseDensity: lc.AccessRatePerSec(nowMillis),
		Secret:       secret,
	}
}
