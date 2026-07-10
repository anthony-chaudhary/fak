package vcachecal

import (
	"math"
	"sort"
)

// allocation.go turns the scope-3 concentration MEASUREMENT into an ALLOCATION input —
// issue #4016 (AdaKV borrow; home epic #3661, parent #3983). When several buckets
// compete for one shared budget (memq render across N sessions; a fleet context-token
// slice), today's split is even or a flat scalar — the measured concentration is
// reporting-only. AdaKV's insight (pyramidkv normalize, pyramidkv_utils.py:1043 @
// 94255b6): a head whose attention mass is PEAKY deserves a larger slice of the layer
// budget than one whose mass is spread thin, because the same bytes retain more of its
// value. This file is that insight as a pure fold: per-bucket concentration = the
// top-K captured fraction of the bucket's cell values (read off FitConcentration's
// coverage curve), used as a multiplicative weight on the bucket's flat share and
// renormalized so the global budget is conserved exactly.
//
// Guardrails against the upstream tradeoff ("miscalibrated, it can starve
// diffuse-but-necessary heads"): a flat bucket of n cells still captures K/n > 0, so
// it is dampened, never zeroed; and a bucket with NO measured value (empty, or all
// non-positive weights) cannot be judged peaky or diffuse, so it keeps exactly its
// flat share instead of being starved by a measurement it never had. Everything is
// deterministic: no RNG, no map-order dependence, stable sort with an explicit
// input-index tie-break — same input, same output, to the byte.

// BudgetBucket is one competitor for a shared budget: a stable key plus the bucket's
// cell values as ranked vBlocks — the same rows FitConcentration consumes, sorted
// DESCENDING by Weight with non-negative weights (the FitConcentration contract;
// vcachescore.NormalizeRanked is the canonical normalizer). For the memq-render
// competition a caller maps each session's cells to rows (Frequency = access rate or
// relevance score, Size = bytes, ReuseDensity = 1); for a dispatch slice, each lane's
// ready-work values.
type BudgetBucket struct {
	Key    string         `json:"key"`
	Ranked []RankedVBlock `json:"ranked"`
}

// BucketShare is one bucket's allocation verdict: the measured concentration (top-K
// captured fraction; 0 = unmeasured or weighting off), the flat 1/N baseline, the
// renormalized concentration-weighted share, and the integer budget slice. Shares are
// conserved (they sum to 1) and budgets sum to the total EXACTLY via largest-remainder
// rounding — the AdaKV renormalization, so boosting one bucket can only come out of
// the others, never out of thin air.
type BucketShare struct {
	Key           string  `json:"key"`
	Concentration float64 `json:"concentration"`
	FlatShare     float64 `json:"flat_share"`
	Share         float64 `json:"share"`
	Budget        int64   `json:"budget"`
}

// AllocateByConcentration splits a shared budget across competing buckets,
// concentration-weighted. topK is both the OPT-IN gate and the K of the top-K
// captured fraction: topK <= 0 (the zero value) never measures anything and returns
// the flat even split — every Share == FlatShare regardless of bucket contents — so a
// caller whose knob is unset reproduces today's un-weighted division byte-for-byte.
// With topK >= 1, each measured bucket's flat share is multiplied by its concentration
// (sum of its top-K cell values / sum of all its cell values) and the measured mass is
// renormalized so the total is conserved; an unmeasured bucket keeps exactly its flat
// share. Output order mirrors input order. Pure and deterministic.
func AllocateByConcentration(buckets []BudgetBucket, total int64, topK int) []BucketShare {
	n := len(buckets)
	if n == 0 {
		return nil
	}
	flat := 1.0 / float64(n)
	shares := make([]BucketShare, n)
	for i, b := range buckets {
		shares[i] = BucketShare{Key: b.Key, FlatShare: flat, Share: flat}
		if topK > 0 {
			shares[i].Concentration = concentrationAt(FitConcentration(b.Ranked), topK)
		}
	}
	if topK > 0 {
		sumConc := 0.0
		unmeasured := 0
		for i := range shares {
			if shares[i].Concentration > 0 {
				sumConc += shares[i].Concentration
			} else {
				unmeasured++
			}
		}
		// The AdaKV weighting: after every unmeasured bucket keeps its flat share, the
		// measured buckets split the remaining mass in proportion to concentration.
		// sumConc == 0 (nothing measured) leaves every share on the flat baseline.
		if sumConc > 0 {
			remaining := 1.0 - flat*float64(unmeasured)
			for i := range shares {
				if shares[i].Concentration > 0 {
					shares[i].Share = remaining * shares[i].Concentration / sumConc
				}
			}
		}
	}
	distributeBudget(shares, total)
	return shares
}

// concentrationAt reads the top-K captured fraction off a fitted coverage curve —
// (sum of the top-K cell values) / (sum of all cell values), the #4016 weight. K is
// clamped to the curve's last rank (a bucket with fewer than K cells is fully
// captured: coverage 1); an unmeasured bucket (empty curve — zero total weight) reads
// 0, which the allocator treats as "keep the flat share", never as "starve".
func concentrationAt(c Concentration, k int) float64 {
	if len(c.TopNCoverage) == 0 {
		return 0
	}
	if k > len(c.TopNCoverage) {
		k = len(c.TopNCoverage)
	}
	return c.TopNCoverage[k]
}

// distributeBudget converts shares into integer budget slices that sum to total
// EXACTLY: floor each raw slice, then hand the leftover units one each to the largest
// fractional remainders, ties broken by input index ascending — explicit and
// deterministic, never map order. The reverse trim guards the pathological float case
// where the floors overshoot. total <= 0 leaves every budget 0 (shares still report).
func distributeBudget(shares []BucketShare, total int64) {
	if total <= 0 {
		return
	}
	type remainder struct {
		idx  int
		frac float64
	}
	rems := make([]remainder, len(shares))
	assigned := int64(0)
	for i := range shares {
		raw := shares[i].Share * float64(total)
		fl := int64(math.Floor(raw))
		shares[i].Budget = fl
		assigned += fl
		rems[i] = remainder{idx: i, frac: raw - float64(fl)}
	}
	sort.SliceStable(rems, func(a, b int) bool {
		if rems[a].frac != rems[b].frac {
			return rems[a].frac > rems[b].frac
		}
		return rems[a].idx < rems[b].idx
	})
	for i := 0; assigned < total && i < len(rems); i++ {
		shares[rems[i].idx].Budget++
		assigned++
	}
	for i := len(rems) - 1; assigned > total && i >= 0; i-- {
		if shares[rems[i].idx].Budget > 0 {
			shares[rems[i].idx].Budget--
			assigned--
		}
	}
}
