// Adapted from ai-dynamo/aiperf, src/aiperf/kubernetes/jobset.py:110 (split_weighted_total,
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

// Inspired by ai-dynamo/aiperf, src/aiperf/kubernetes/jobset.py:162 at
// pinned requests are carved out first, the remainder is split by weight, and
// exhaustion names every term.

var errCPUSharesExhausted = errors.New("cpushares: pinned requests exceed total")

// CPUShareSpec is one consumer: a fixed Pinned request plus a Weight over the
// remainder (a zero-weight consumer receives only its pinned amount).
type CPUShareSpec struct {
	Name   string
	Pinned int
	Weight int
}

// CPUShares returns one share per spec, summing to exactly total when any
// weight exists, or to the pinned sum when none does and the budget is not
// exhausted.
func CPUShares(total int, specs []CPUShareSpec) ([]int, error) {
	pinned := 0
	for _, s := range specs {
		pinned += s.Pinned
	}
	if pinned > total {
		detail := ""
		for i, s := range specs {
			if i > 0 {
				detail += ", "
			}
			detail += fmt.Sprintf("%s=%d", s.Name, s.Pinned)
		}
		return nil, fmt.Errorf("%w: total %d, pinned %d (%s)", errCPUSharesExhausted, total, pinned, detail)
	}
	weights := make([]int, len(specs))
	mins := make([]int, len(specs))
	anyWeight := false
	for i, s := range specs {
		weights[i], mins[i] = s.Weight, s.Pinned
		anyWeight = anyWeight || s.Weight > 0
	}
	if !anyWeight {
		return mins, nil
	}
	return SplitBudget(total, weights, mins)
}
