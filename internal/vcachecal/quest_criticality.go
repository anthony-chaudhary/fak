// Ported from github.com/mit-han-lab/Quest (kernels/include/decode/decode_page.cuh and
// kernels/include/decode/decode_attn.cuh@01c1623bf9395009520874e989e29f683203b357), MIT.
// SPDX-License-Identifier: MIT

package vcachecal

import (
	"errors"
	"fmt"
	"sort"
)

// Quest query-aware page criticality: each KV page keeps only the channel-wise min and
// max of its keys, and a decode query scores the page by
//
//	score(page) = Σ_d max(q_d·kmax_d, q_d·kmin_d)
//
// which upper-bounds q·k for every key k in the page, because each term is the largest
// value q_d·k_d can take for k_d ∈ [kmin_d, kmax_d]. Pages are then ranked by that bound
// and only the top pages are attended. This file is an opt-in library primitive; it
// changes no serving default.

// ErrQuestDim reports query and page metadata whose channel counts disagree.
var ErrQuestDim = errors.New("vcachecal: quest dimension mismatch")

// QuestPageBounds folds a page's keys into its channel-wise (min, max) metadata, the
// same element-wise reduction Quest applies as keys are appended. ok is false for an
// empty page or ragged keys.
func QuestPageBounds(keys [][]float64) (kmin, kmax []float64, ok bool) {
	if len(keys) == 0 || len(keys[0]) == 0 {
		return nil, nil, false
	}
	dim := len(keys[0])
	kmin = append([]float64(nil), keys[0]...)
	kmax = append([]float64(nil), keys[0]...)
	for _, k := range keys[1:] {
		if len(k) != dim {
			return nil, nil, false
		}
		for d, v := range k {
			if v < kmin[d] {
				kmin[d] = v
			}
			if v > kmax[d] {
				kmax[d] = v
			}
		}
	}
	return kmin, kmax, true
}

// QuestCriticality returns the Quest upper bound on q·k over any key whose channels lie
// within [kmin, kmax]. The max over both products, rather than a sign test on q_d, is
// what keeps the bound valid without assuming kmin <= kmax.
func QuestCriticality(q, kmin, kmax []float64) (float64, error) {
	if len(kmin) != len(q) || len(kmax) != len(q) {
		return 0, fmt.Errorf("%w: q=%d kmin=%d kmax=%d", ErrQuestDim, len(q), len(kmin), len(kmax))
	}
	s := 0.0
	for d, qd := range q {
		a, b := qd*kmax[d], qd*kmin[d]
		if b > a {
			a = b
		}
		s += a
	}
	return s, nil
}

// QuestPageScore is one page's index and criticality bound.
type QuestPageScore struct {
	Page  int
	Score float64
}

// QuestRankPages scores every page and returns them ordered by descending bound, ties
// broken by ascending page index so the order is deterministic.
func QuestRankPages(q []float64, mins, maxs [][]float64) ([]QuestPageScore, error) {
	if len(mins) != len(maxs) {
		return nil, fmt.Errorf("%w: %d min rows vs %d max rows", ErrQuestDim, len(mins), len(maxs))
	}
	out := make([]QuestPageScore, len(mins))
	for i := range mins {
		s, err := QuestCriticality(q, mins[i], maxs[i])
		if err != nil {
			return nil, fmt.Errorf("page %d: %w", i, err)
		}
		out[i] = QuestPageScore{Page: i, Score: s}
	}
	sort.SliceStable(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return out[a].Page < out[b].Page
	})
	return out, nil
}

// QuestSelectPages picks the pages to attend under a page budget the way Quest does:
// the last (still-filling) page is always kept and excluded from the ranking, and the
// remaining budget-1 slots go to the highest-bound earlier pages. The result is sorted
// by page index. A budget at or above the page count selects every page; a budget
// below 1 selects none.
func QuestSelectPages(q []float64, mins, maxs [][]float64, budget int) ([]int, error) {
	n := len(mins)
	if len(maxs) != n {
		return nil, fmt.Errorf("%w: %d min rows vs %d max rows", ErrQuestDim, n, len(maxs))
	}
	if n == 0 || budget < 1 {
		return nil, nil
	}
	ranked, err := QuestRankPages(q, mins[:n-1], maxs[:n-1])
	if err != nil {
		return nil, err
	}
	if err := checkQuestDim(q, mins[n-1], maxs[n-1]); err != nil {
		return nil, fmt.Errorf("page %d: %w", n-1, err)
	}
	k := budget - 1
	if k > len(ranked) {
		k = len(ranked)
	}
	sel := make([]int, 0, k+1)
	for _, r := range ranked[:k] {
		sel = append(sel, r.Page)
	}
	sel = append(sel, n-1)
	sort.Ints(sel)
	return sel, nil
}

func checkQuestDim(q, kmin, kmax []float64) error {
	_, err := QuestCriticality(q, kmin, kmax)
	return err
}
