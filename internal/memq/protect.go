package memq

import "sort"

// protect.go — the protected-floor budget split (#4017, NACL/Scissorhands).
//
// applyBudget treats a durable cell identically to a relevance-matched ephemeral
// one: under pressure a low-relevance durable or recent cell tail-drops like
// anything else. The protected split ports the #2421 survival-class guarantee —
// which ctxplan already holds at the gateway/compaction boundary (the exact
// recent area in ctxplan/layout.go's DefaultLayout) — down into memq's
// byte-budget pass: the floor (every durable-class cell ∪ the top-N most-recent
// cells by Step) is charged against the cap FIRST, guaranteed to survive
// regardless of relevance rank, and only the leftover headroom is spent on the
// ephemeral remainder in working-set order.
//
// Upstream inspiration (technique only; no source vendored): NACL's
// protected-floor union (pins ∪ sink ∪ recent retained before any scored keep)
// and Scissorhands' budget accounting (spend protected first, then select only
// budget − |protected| by score). Their tradeoff carries over as the
// overflow-degrade rule: a floor that ALONE exceeds the cap keeps the
// most-recent protected cells that fit.
//
// The whole pass is deterministic: selection and every tie-break are total
// orders on (Step desc, then ID asc — the rankLess convention) with stable
// sorts, no RNG and no map iteration, so the same (working set, cap, recentN)
// yields byte-identical output — the audit contract #4021 states.

// applyProtectedBudget keeps the protected floor (durable ∪ top-recentN by
// Step) within cap first, then fills what is left with the remainder in
// working-set order — the same greedy skip-if-oversized rule applyBudget uses.
// The floor is spent most-recent-first (Step descending, ties by ascending ID),
// which is the overflow-degrade rule when the floor alone exceeds the cap.
// kept and dropped both preserve the working-set order, exactly as
// applyBudget's contract states, so the caller's MEMORY_INDEX_OVERFLOW verdict
// (#2430) is unchanged in shape. cap <= 0 stays unbounded, mirroring applyBudget.
func applyProtectedBudget(work []Cell, cap int64, recentN int) (kept, dropped []Cell) {
	if cap <= 0 {
		return work, nil
	}
	prot := protectedSet(work, recentN)

	// Spend the cap on the floor first, most-recent-first (the degrade order).
	order := make([]int, 0, len(work))
	for i := range work {
		if prot[i] {
			order = append(order, i)
		}
	}
	sortMostRecentFirst(order, work)
	keep := make([]bool, len(work))
	used := int64(0)
	for _, i := range order {
		if used+work[i].Bytes > cap {
			continue
		}
		used += work[i].Bytes
		keep[i] = true
	}

	// Then the ephemeral remainder, in the working-set order the caller ranked.
	for i, c := range work {
		if prot[i] {
			continue
		}
		if used+c.Bytes > cap {
			continue
		}
		used += c.Bytes
		keep[i] = true
	}

	// Emit kept/dropped in working-set order — the applyBudget contract, so the
	// overflow verdict's Dropped list stays a deterministic compaction work-list.
	kept = work[:0:0]
	for i, c := range work {
		if keep[i] {
			kept = append(kept, c)
		} else {
			dropped = append(dropped, c)
		}
	}
	return kept, dropped
}

// protectedSet marks, by working-set index, the protected floor: every
// durable-class cell, unioned with the top-recentN most-recent cells by Step
// (ties by ascending ID) regardless of their class — the NACL union shape
// (pins ∪ recent). recentN <= 0 protects the durable class alone.
func protectedSet(work []Cell, recentN int) []bool {
	prot := make([]bool, len(work))
	for i, c := range work {
		if NormDurability(c.Durability) == DurabilityDurable {
			prot[i] = true
		}
	}
	if recentN <= 0 {
		return prot
	}
	order := make([]int, len(work))
	for i := range order {
		order[i] = i
	}
	sortMostRecentFirst(order, work)
	if recentN > len(order) {
		recentN = len(order)
	}
	for _, i := range order[:recentN] {
		prot[i] = true
	}
	return prot
}

// sortMostRecentFirst orders working-set INDICES into the degrade order both floor passes rank
// by: most recent first (descending Step), ties broken on ascending ID. The tie-break is the
// rankLess one, which makes the order total and therefore deterministic — two runs over the same
// working set always spend the cap on the same cells. The sort is stable, so equal (Step, ID)
// cells — which cannot occur while IDs are unique — would keep their working-set order.
func sortMostRecentFirst(order []int, work []Cell) {
	sort.SliceStable(order, func(a, b int) bool {
		ca, cb := work[order[a]], work[order[b]]
		if ca.Step != cb.Step {
			return ca.Step > cb.Step
		}
		return ca.ID < cb.ID
	})
}
