package fitplan

import (
	"fmt"
	"sort"
	"strings"
)

// enumerateRows prices every point of the tier x ctx x slots grid: one
// searchScored per enumerated operating point that still fits at least one
// stream. Rows come out in tier, ctx, ascending-slot order.
func enumerateRows(p *planner, in SearchInput, tiers []KVTier, ctxs, slotsList []int, isl, osl, target int) []searchScored {
	var rows []searchScored
	for _, tier := range tiers {
		c := Candidate{KVPrecision: tier.Precision, Context: tier.Context}
		if !kvSizable(c.Context) {
			continue
		}
		for _, ctx := range ctxs {
			c.ContextTokens = ctx
			lkv := min(isl, ctx)
			rate1 := p.stepRateAt(c, 1, lkv)
			for _, slots := range slotsList {
				q := p.atConcurrency(slots)
				pr := q.predict(c)
				bMem := pr.MaxConcurrency
				if bMem < 1 {
					continue // not even one stream fits at this ctx
				}
				bStar := min(bMem, slots)
				if in.SLO.TPOTMS > 0 && bStar > 0 {
					bStar = maxBatchAtRate(p, c, lkv, bStar, 1000.0/in.SLO.TPOTMS)
				}
				ttft := 0.0
				if bStar > 0 {
					ttft = p.chunkedTTFT(c, lkv, bStar, osl)
				}
				rows = append(rows, searchScored{
					row: SearchRow{
						KVPrecision:   tier.Precision,
						ContextTokens: ctx,
						Slots:         slots,
						BStar:         bStar,
						AggregateTokS: aggregateAt(p, c, lkv, bStar),
						TTFTSeconds:   ttft,
						Current:       in.CurrentKV != "" && strings.EqualFold(in.CurrentKV, tier.Precision) && ctx == in.ResolvedContext && slots == target,
					},
					ctxOK:   ctx >= isl,
					ttftOK:  in.SLO.TTFTMS <= 0 || ttft*1000 <= in.SLO.TTFTMS+1e-9,
					bMem:    bMem,
					tpot1MS: tpotMS(rate1),
				})
			}
		}
	}
	return rows
}

// dedupeRows collapses rows sharing (tier, ctx, B*): they are the same
// achievable configuration priced at different enumerated operating points, so
// keep the tightest one (rows are generated in ascending slot order, so the
// first occurrence wins).
func dedupeRows(rows []searchScored) []searchScored {
	deduped := rows[:0]
	seenKey := map[string]bool{}
	for _, s := range rows {
		key := fmt.Sprintf("%s|%d|%d", s.row.KVPrecision, s.row.ContextTokens, s.row.BStar)
		if seenKey[key] {
			continue
		}
		seenKey[key] = true
		deduped = append(deduped, s)
	}
	return deduped
}

// rankEligible keeps the rows that serve the stated workload — context covers
// the prompt, TTFT holds the SLO, B* reaches the target concurrency — ordered
// best-first: aggregate throughput desc, context desc, slots asc, then
// enumeration order (stable sort).
func rankEligible(rows []searchScored, target int) []searchScored {
	eligible := make([]searchScored, 0, len(rows))
	for _, s := range rows {
		if s.ctxOK && s.ttftOK && s.row.BStar >= target {
			eligible = append(eligible, s)
		}
	}
	sort.SliceStable(eligible, func(i, j int) bool {
		a, b := eligible[i].row, eligible[j].row
		if a.AggregateTokS != b.AggregateTokS {
			return a.AggregateTokS > b.AggregateTokS
		}
		if a.ContextTokens != b.ContextTokens {
			return a.ContextTokens > b.ContextTokens
		}
		return a.Slots < b.Slots
	})
	return eligible
}

// refuseNoEligible classifies a search with no eligible row into the typed
// refusal vocabulary: a pinned search judges the canonical pinned row, a pool
// shortfall names the bytes, and the rest fall through to context,
// concurrency, or SLO classification.
func refuseNoEligible(rep *SearchReport, p *planner, rows []searchScored, in SearchInput, tiers []KVTier, isl, target int) {
	if anyPinSet(in.Pins) {
		classifyPin(rep, rows, in, isl, target)
		return
	}
	if subSat(p.avail, p.weights) < 0 {
		if in.Model.NumExperts > 0 && in.Model.RoutedExpertBytes > 0 {
			refuseSearch(rep, ReasonNeedsRing, "", 0, 0, subSat(p.weights, p.avail), false)
		} else {
			refuseSearch(rep, ReasonWeightsExceedPool, "", 0, 0, subSat(p.weights, p.avail), false)
		}
		return
	}
	desired := in.ResolvedContext
	if desired < isl {
		desired = isl
	}
	maxOne, kvOne := p.maxSlotContextFor(tiers, 1)
	maxTarget, _ := p.maxSlotContextFor(tiers, target)
	switch {
	case maxOne == 0:
		refuseSearch(rep, ReasonContextExceedsPool, bestTierPrecision(tiers), 0, 0, p.shortfallOneStream(tiers, isl), false)
	case maxTarget < desired:
		if maxOne < desired {
			refuseSearch(rep, ReasonFitsOnlyAtCtxLT, kvOne, maxOne, 0, 0, false)
			rep.Refusal.RequestedContext = desired
		} else {
			refuseSearch(rep, ReasonConcurrencyExceedsPool, "", 0, maxConcAt(rows, desired), 0, false)
		}
	default:
		refuseSLO(rep, rows, in.SLO, target)
	}
}
