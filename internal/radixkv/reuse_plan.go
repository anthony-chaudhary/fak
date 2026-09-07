package radixkv

import (
	"fmt"
	"math"
	"sort"
)

// ActionKind indicates what operation is applied to a span of tokens during execution.
type ActionKind string

const (
	// ActionDirectReuse specifies direct reuse of cached KV tokens without modification.
	ActionDirectReuse ActionKind = "direct_reuse"
	// ActionSelectiveRepair specifies selective repair of cached KV tokens.
	ActionSelectiveRepair ActionKind = "selective_repair"
	// ActionCompute specifies recomputing KV tokens via prefill.
	ActionCompute ActionKind = "compute"
	// ActionReject specifies rejecting the candidate span from reuse.
	ActionReject ActionKind = "reject"
)

// Standard tier constants.
const (
	// TierL1GPU specifies the L1 GPU device memory tier.
	TierL1GPU = "L1_GPU"
	// TierL2Host specifies the L2 host/system memory tier.
	TierL2Host = "L2_HOST"
	// TierL3Remote specifies the L3 remote/disaggregated storage tier.
	TierL3Remote = "L3_REMOTE"
)

// PlanSpan describes a single contiguous segment of tokens in the executed prompt.
type PlanSpan struct {
	// Start is the beginning token offset in the prompt (inclusive).
	Start int `json:"start"`
	// End is the ending token offset in the prompt (exclusive).
	End int `json:"end"`
	// Action is the execution operation applied to this span.
	Action ActionKind `json:"action"`
	// CandidateID is the identifier of the matched reuse candidate, if any.
	CandidateID string `json:"candidate_id,omitempty"`
	// SourceStart is the starting token offset in the source candidate.
	SourceStart int `json:"source_start,omitempty"`
	// SourceEnd is the ending token offset in the source candidate.
	SourceEnd int `json:"source_end,omitempty"`
	// Tier is the storage tier providing the cached KV state.
	Tier string `json:"tier,omitempty"`
	// Cost is the calculated execution cost for this span.
	Cost float64 `json:"cost"`
}

// ReusePlan represents the compiled execution and reuse layout for a prompt.
type ReusePlan struct {
	// PromptLen is the total number of tokens in the prompt.
	PromptLen int `json:"prompt_len"`
	// PrefixLen is the number of tokens locked as prefix reuse.
	PrefixLen int `json:"prefix_len"`
	// Spans contains the non-overlapping execution spans covering the prompt.
	Spans []PlanSpan `json:"spans"`
	// DirectTokens is the total token count scheduled for direct reuse.
	DirectTokens int `json:"direct_tokens"`
	// RepairTokens is the total token count scheduled for selective repair.
	RepairTokens int `json:"repair_tokens"`
	// ComputeTokens is the total token count scheduled for compute/prefill.
	ComputeTokens int `json:"compute_tokens"`
	// EstimatedCost is the total estimated cost of executing the plan.
	EstimatedCost float64 `json:"estimated_cost"`
	// SavingsRatio is the fraction of cost saved relative to full prefill.
	SavingsRatio float64 `json:"savings_ratio"`
}

// ReuseCandidate defines a matching span in cache candidate storage eligible for reuse.
type ReuseCandidate struct {
	// ID is the unique identifier for the reuse candidate.
	ID string `json:"id"`
	// DstStart is the start token offset in the destination prompt (inclusive).
	DstStart int `json:"dst_start"`
	// DstEnd is the end token offset in the destination prompt (exclusive).
	DstEnd int `json:"dst_end"`
	// SrcStart is the start token offset in the source cache.
	SrcStart int `json:"src_start"`
	// SrcEnd is the end token offset in the source cache.
	SrcEnd int `json:"src_end"`
	// Tier is the cache storage tier where candidate tokens reside.
	Tier string `json:"tier"`
	// Cost is the estimated cost of reusing this candidate span.
	Cost float64 `json:"cost"`
	// PrefillCost is the baseline cost if this span were computed via prefill.
	PrefillCost float64 `json:"prefill_cost"`
	// Compatible indicates whether candidate KV state is compatible with the model context.
	Compatible bool `json:"compatible"`
	// Action specifies the preferred action kind for this candidate.
	Action ActionKind `json:"action,omitempty"`
}

// CostModel calculates costs for prefill computation and candidate reuse.
type CostModel interface {
	PrefillCost(spanLen int) float64
	CandidateCost(c ReuseCandidate) float64
}

// DefaultCostModel provides linear token-cost based pricing.
type DefaultCostModel struct {
	// TokenCost is the cost multiplier per token for prefill computation.
	TokenCost float64
}

// PrefillCost computes the cost of full prefill generation for spanLen tokens.
func (d DefaultCostModel) PrefillCost(spanLen int) float64 {
	tc := d.TokenCost
	if tc <= 0 || math.IsNaN(tc) || math.IsInf(tc, 0) {
		tc = 1.0
	}
	return float64(spanLen) * tc
}

// CandidateCost returns the declared or evaluated cost for candidate reuse.
func (d DefaultCostModel) CandidateCost(c ReuseCandidate) float64 {
	return c.Cost
}

type candidateEval struct {
	candidate ReuseCandidate
	cost      float64
	prefill   float64
	savings   float64
}

// ComposeReusePlan constructs a non-overlapping token reuse plan covering [0, promptLen)
// with prefix [0, prefixLen) locked as ActionDirectReuse. It executes deterministic greedy
// interval selection to resolve candidate conflicts, maximizing net savings.
func ComposeReusePlan(promptLen int, prefixLen int, candidates []ReuseCandidate, cm CostModel) (*ReusePlan, error) {
	if promptLen < 0 || prefixLen < 0 || prefixLen > promptLen {
		return nil, fmt.Errorf("radixkv: invalid bounds promptLen=%d prefixLen=%d", promptLen, prefixLen)
	}

	if cm == nil {
		cm = DefaultCostModel{TokenCost: 1.0}
	}

	// Filter candidates strictly in [prefixLen, promptLen)
	var evals []candidateEval
	for _, c := range candidates {
		if c.DstStart < prefixLen || c.DstEnd > promptLen || c.DstStart >= c.DstEnd {
			continue
		}
		if !c.Compatible || c.Action == ActionReject {
			continue
		}

		cost := cm.CandidateCost(c)
		if math.IsNaN(cost) || math.IsInf(cost, 0) {
			continue
		}

		prefill := c.PrefillCost
		if math.IsNaN(prefill) || math.IsInf(prefill, 0) {
			continue
		}
		if prefill <= 0 {
			prefill = cm.PrefillCost(c.DstEnd - c.DstStart)
		}
		if math.IsNaN(prefill) || math.IsInf(prefill, 0) {
			continue
		}

		// Net-true value rule: reject if cost >= prefill
		if cost >= prefill {
			continue
		}

		evals = append(evals, candidateEval{
			candidate: c,
			cost:      cost,
			prefill:   prefill,
			savings:   prefill - cost,
		})
	}

	// Deterministic sort:
	// 1. Net savings (prefill - cost) descending
	// 2. Span length (c.DstEnd - c.DstStart) descending
	// 3. DstStart ascending
	// 4. DstEnd ascending
	// 5. ID ascending
	// 6. Tier ascending
	// 7. SrcStart ascending
	// 8. SrcEnd ascending
	// 9. cost ascending
	sort.Slice(evals, func(i, j int) bool {
		if evals[i].savings != evals[j].savings {
			return evals[i].savings > evals[j].savings
		}
		lenI := evals[i].candidate.DstEnd - evals[i].candidate.DstStart
		lenJ := evals[j].candidate.DstEnd - evals[j].candidate.DstStart
		if lenI != lenJ {
			return lenI > lenJ
		}
		if evals[i].candidate.DstStart != evals[j].candidate.DstStart {
			return evals[i].candidate.DstStart < evals[j].candidate.DstStart
		}
		if evals[i].candidate.DstEnd != evals[j].candidate.DstEnd {
			return evals[i].candidate.DstEnd < evals[j].candidate.DstEnd
		}
		if evals[i].candidate.ID != evals[j].candidate.ID {
			return evals[i].candidate.ID < evals[j].candidate.ID
		}
		if evals[i].candidate.Tier != evals[j].candidate.Tier {
			return evals[i].candidate.Tier < evals[j].candidate.Tier
		}
		if evals[i].candidate.SrcStart != evals[j].candidate.SrcStart {
			return evals[i].candidate.SrcStart < evals[j].candidate.SrcStart
		}
		if evals[i].candidate.SrcEnd != evals[j].candidate.SrcEnd {
			return evals[i].candidate.SrcEnd < evals[j].candidate.SrcEnd
		}
		return evals[i].cost < evals[j].cost
	})

	// Greedy interval selection: select non-overlapping candidates
	var selected []candidateEval
	for _, cand := range evals {
		overlap := false
		for _, sel := range selected {
			if cand.candidate.DstStart < sel.candidate.DstEnd && sel.candidate.DstStart < cand.candidate.DstEnd {
				overlap = true
				break
			}
		}
		if !overlap {
			selected = append(selected, cand)
		}
	}

	// Sort selected candidates in ascending DstStart order for span emission
	sort.Slice(selected, func(i, j int) bool {
		return selected[i].candidate.DstStart < selected[j].candidate.DstStart
	})

	spans := make([]PlanSpan, 0)
	cur := 0

	// Prefix span [0, prefixLen)
	if prefixLen > 0 {
		spans = append(spans, PlanSpan{
			Start:  0,
			End:    prefixLen,
			Action: ActionDirectReuse,
			Tier:   TierL1GPU,
			Cost:   0,
		})
		cur = prefixLen
	}

	for _, sel := range selected {
		if sel.candidate.DstStart > cur {
			gapLen := sel.candidate.DstStart - cur
			spans = append(spans, PlanSpan{
				Start:  cur,
				End:    sel.candidate.DstStart,
				Action: ActionCompute,
				Cost:   cm.PrefillCost(gapLen),
			})
		}

		action := sel.candidate.Action
		switch action {
		case ActionDirectReuse, ActionSelectiveRepair:
		default:
			action = ActionDirectReuse
		}

		spans = append(spans, PlanSpan{
			Start:       sel.candidate.DstStart,
			End:         sel.candidate.DstEnd,
			Action:      action,
			CandidateID: sel.candidate.ID,
			SourceStart: sel.candidate.SrcStart,
			SourceEnd:   sel.candidate.SrcEnd,
			Tier:        sel.candidate.Tier,
			Cost:        sel.cost,
		})
		cur = sel.candidate.DstEnd
	}

	if cur < promptLen {
		gapLen := promptLen - cur
		spans = append(spans, PlanSpan{
			Start:  cur,
			End:    promptLen,
			Action: ActionCompute,
			Cost:   cm.PrefillCost(gapLen),
		})
	}

	var directTokens, repairTokens, computeTokens int
	var estimatedCost float64

	for _, s := range spans {
		spanLen := s.End - s.Start
		switch s.Action {
		case ActionDirectReuse:
			directTokens += spanLen
		case ActionSelectiveRepair:
			repairTokens += spanLen
		case ActionCompute:
			computeTokens += spanLen
		default:
			directTokens += spanLen
		}
		estimatedCost += s.Cost
	}

	var savingsRatio float64
	fullPrefillCost := cm.PrefillCost(promptLen)
	if fullPrefillCost > 0 && !math.IsNaN(fullPrefillCost) && !math.IsInf(fullPrefillCost, 0) {
		ratio := (fullPrefillCost - estimatedCost) / fullPrefillCost
		if ratio < 0 || math.IsNaN(ratio) || math.IsInf(ratio, 0) {
			ratio = 0
		}
		savingsRatio = ratio
	}

	return &ReusePlan{
		PromptLen:     promptLen,
		PrefixLen:     prefixLen,
		Spans:         spans,
		DirectTokens:  directTokens,
		RepairTokens:  repairTokens,
		ComputeTokens: computeTokens,
		EstimatedCost: estimatedCost,
		SavingsRatio:  savingsRatio,
	}, nil
}
