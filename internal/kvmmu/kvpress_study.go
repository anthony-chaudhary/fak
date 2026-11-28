// Conceptual INSPIRE-ONLY seam derived from NVIDIA/kvpress
// (94951a5b90a2594efbb674c69a9fb9cd28f300c0, Apache-2.0).
// No Python runtime, model weights, or served-token quality claim is made.
package kvmmu

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

var ErrInvalidKVPressPolicy = errors.New("kvmmu: invalid kvpress compression policy")

// KVPressScorerFamily identifies the scorer-based KV compression family.
type KVPressScorerFamily string

const (
	ScorerStreamingLLM KVPressScorerFamily = "streaming_llm"
	ScorerSnapKV       KVPressScorerFamily = "snapkv"
	ScorerPyramidKV    KVPressScorerFamily = "pyramidkv"
)

// KVPressPolicy parameterizes per-layer KV compression budget accounting.
type KVPressPolicy struct {
	Family       KVPressScorerFamily `json:"family"`
	KeepRatio    float64             `json:"keep_ratio"`
	SinkTokens   int                 `json:"sink_tokens"`
	RecentWindow int                 `json:"recent_window"`
}

// Validate checks that KeepRatio is in (0, 1] and sink/recent counts are non-negative.
func (p KVPressPolicy) Validate() error {
	switch p.Family {
	case ScorerStreamingLLM, ScorerSnapKV, ScorerPyramidKV:
	default:
		return fmt.Errorf("%w: unknown family %q", ErrInvalidKVPressPolicy, p.Family)
	}
	if math.IsNaN(p.KeepRatio) || math.IsInf(p.KeepRatio, 0) || p.KeepRatio <= 0 || p.KeepRatio > 1 {
		return fmt.Errorf("%w: keep_ratio=%v must be in (0, 1]", ErrInvalidKVPressPolicy, p.KeepRatio)
	}
	if p.SinkTokens < 0 || p.RecentWindow < 0 {
		return fmt.Errorf("%w: sink_tokens and recent_window must be >= 0", ErrInvalidKVPressPolicy)
	}
	return nil
}

// KVPressPlan records the kept token indices and budget accounting for one sequence.
type KVPressPlan struct {
	TotalTokens  int   `json:"total_tokens"`
	BudgetTokens int   `json:"budget_tokens"`
	KeptIndices  []int `json:"kept_indices"`
	EvictedCount int   `json:"evicted_count"`
}

// PlanKVCompression selects up to floor(KeepRatio * len(scores)) tokens while
// preserving SinkTokens at the prefix and RecentWindow at the tail, breaking ties
// deterministically by lower token index.
func PlanKVCompression(policy KVPressPolicy, scores []float64) (KVPressPlan, error) {
	if err := policy.Validate(); err != nil {
		return KVPressPlan{}, err
	}
	n := len(scores)
	if n == 0 {
		return KVPressPlan{}, fmt.Errorf("%w: empty token scores", ErrInvalidKVPressPolicy)
	}
	for i, s := range scores {
		if math.IsNaN(s) || math.IsInf(s, 0) {
			return KVPressPlan{}, fmt.Errorf("%w: non-finite score at %d", ErrInvalidKVPressPolicy, i)
		}
	}

	budget := int(math.Floor(policy.KeepRatio * float64(n)))
	if budget < 1 {
		budget = 1
	}
	if budget >= n {
		all := make([]int, n)
		for i := range all {
			all[i] = i
		}
		return KVPressPlan{TotalTokens: n, BudgetTokens: n, KeptIndices: all, EvictedCount: 0}, nil
	}

	kept := make([]bool, n)
	keptCount := 0
	// Pin attention sinks first, then recent window, bounded by budget.
	for i := 0; i < policy.SinkTokens && i < n && keptCount < budget; i++ {
		if !kept[i] {
			kept[i] = true
			keptCount++
		}
	}
	for i := max(0, n-policy.RecentWindow); i < n && keptCount < budget; i++ {
		if !kept[i] {
			kept[i] = true
			keptCount++
		}
	}

	if keptCount < budget {
		var middle []int
		for i := 0; i < n; i++ {
			if !kept[i] {
				middle = append(middle, i)
			}
		}
		sort.SliceStable(middle, func(a, b int) bool {
			if scores[middle[a]] != scores[middle[b]] {
				return scores[middle[a]] > scores[middle[b]]
			}
			return middle[a] < middle[b]
		})
		for _, idx := range middle {
			if keptCount >= budget {
				break
			}
			kept[idx] = true
			keptCount++
		}
	}

	out := make([]int, 0, keptCount)
	for i, k := range kept {
		if k {
			out = append(out, i)
		}
	}
	return KVPressPlan{
		TotalTokens:  n,
		BudgetTokens: budget,
		KeptIndices:  out,
		EvictedCount: n - len(out),
	}, nil
}
