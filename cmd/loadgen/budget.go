// Adapted from ai-dynamo/aiperf, src/aiperf/kubernetes/jobset.py:110 (split_weighted_total,
// _split_with_minimum) at ab610add4e8876fca223197596c728e17365f381. Apache-2.0; re-expressed as
// one exact-integer largest-remainder split with per-bucket minimums and typed validation errors.

package main

import (
	"errors"
	"fmt"
	"math/bits"
	"sort"
)

var (
	errBudgetNegative      = errors.New("budget: negative total")
	errBudgetNegativeInput = errors.New("budget: negative weight or minimum")
	errBudgetMinimums      = errors.New("budget: minimums exceed total")
	errBudgetZeroWeights   = errors.New("budget: weights sum to zero with budget left to distribute")
	errBudgetShape         = errors.New("budget: minimums length differs from weights")
	errBudgetWeightSum     = errors.New("budget: weight sum overflows uint64")
)

// SplitBudget splits total across len(weights) buckets. Each bucket first
// receives its minimum (minimums may be nil, meaning zero for every bucket);
// what remains is divided in proportion to weights by largest remainder, so
// the result always sums to exactly total. Remainder ties go to the lower
// index, matching upstream's stable descending sort.
func SplitBudget(total int, weights, minimums []int) ([]int, error) {
	if total < 0 {
		return nil, fmt.Errorf("%w: %d", errBudgetNegative, total)
	}
	if minimums != nil && len(minimums) != len(weights) {
		return nil, fmt.Errorf("%w: %d minimums, %d weights", errBudgetShape, len(minimums), len(weights))
	}
	if len(weights) == 0 {
		if total != 0 {
			return nil, fmt.Errorf("%w: %d with no buckets", errBudgetZeroWeights, total)
		}
		return []int{}, nil
	}

	shares := make([]int, len(weights))
	remaining := total
	var weightSum uint64
	for i, w := range weights {
		m := 0
		if minimums != nil {
			m = minimums[i]
		}
		if w < 0 || m < 0 {
			return nil, fmt.Errorf("%w: bucket %d weight=%d minimum=%d", errBudgetNegativeInput, i, w, m)
		}
		if m > remaining {
			return nil, fmt.Errorf("%w: total %d", errBudgetMinimums, total)
		}
		shares[i] = m
		remaining -= m
		var carry uint64
		weightSum, carry = bits.Add64(weightSum, uint64(w), 0)
		if carry != 0 {
			return nil, fmt.Errorf("%w: at bucket %d", errBudgetWeightSum, i)
		}
	}
	if remaining == 0 {
		return shares, nil
	}
	if weightSum == 0 {
		return nil, fmt.Errorf("%w: %d", errBudgetZeroWeights, remaining)
	}

	// Integer floor and remainder numerators (shared denominator weightSum)
	// keep the ranking exact where upstream's float shares could misorder.
	rems := make([]uint64, len(weights))
	distributed := 0
	for i, w := range weights {
		hi, lo := bits.Mul64(uint64(remaining), uint64(w))
		q, r := bits.Div64(hi, lo, weightSum)
		shares[i] += int(q)
		distributed += int(q)
		rems[i] = r
	}
	order := make([]int, len(weights))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return rems[order[a]] > rems[order[b]] })
	for _, idx := range order[:remaining-distributed] {
		shares[idx]++
	}
	return shares, nil
}
