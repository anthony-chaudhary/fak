package model

import (
	"fmt"
	"sort"
)

type KVPrefetchPolicy string

const (
	KVPrefetchDemandOnly KVPrefetchPolicy = "demand-only"
	KVPrefetchPredictive KVPrefetchPolicy = "predictive"
	KVPrefetchOracle     KVPrefetchPolicy = "oracle"
)

// KVPrefetchCandidate is one next-use block with the signals available before demand.
type KVPrefetchCandidate struct {
	BlockID          uint64  `json:"block_id"`
	Bytes            int64   `json:"bytes"`
	PositionDistance int     `json:"position_distance"`
	Recency          float64 `json:"recency"`
	RetrievalScore   float64 `json:"retrieval_score"`
	AttentionScore   float64 `json:"attention_score"`
	Needed           bool    `json:"needed"`
	ReadyByDemand    bool    `json:"ready_by_demand"`
}

type KVPrefetchReceipt struct {
	Schema                 string           `json:"schema"`
	Engine                 string           `json:"engine"`
	Policy                 KVPrefetchPolicy `json:"policy"`
	AcceptedTokens         int              `json:"accepted_tokens"`
	BudgetBytes            int64            `json:"budget_bytes"`
	FetchedBytes           int64            `json:"fetched_bytes"`
	UsefulBytes            int64            `json:"useful_bytes"`
	TimelyBytes            int64            `json:"timely_bytes"`
	PollutionBytes         int64            `json:"pollution_bytes"`
	FaultBytes             int64            `json:"fault_bytes"`
	WasteRatio             float64          `json:"waste_ratio"`
	UsefulBytesPerAccepted float64          `json:"useful_bytes_per_accepted_token"`
	SelectedBlocks         []uint64         `json:"selected_blocks"`
	QualityConstraint      string           `json:"quality_constraint"`
	StopRule               string           `json:"stop_rule"`
	Rollback               string           `json:"rollback"`
}

// EvaluateKVPrefetch applies one bounded policy to a frozen next-use trace. Prediction
// uses only position, recency, retrieval, and prior-attention signals; Needed is read
// only after selection to score the receipt.
func EvaluateKVPrefetch(policy KVPrefetchPolicy, candidates []KVPrefetchCandidate, budgetBytes int64, maxWasteRatio float64, acceptedTokens int) (KVPrefetchReceipt, error) {
	if budgetBytes < 0 || maxWasteRatio < 0 || maxWasteRatio > 1 || acceptedTokens < 0 {
		return KVPrefetchReceipt{}, fmt.Errorf("model: invalid KV prefetch envelope")
	}
	r := KVPrefetchReceipt{Schema: "fak-kv-prefetch-receipt/1", Engine: "fak-native", Policy: policy, AcceptedTokens: acceptedTokens, BudgetBytes: budgetBytes, QualityConstraint: "prefetch changes residency only; logits and accepted tokens remain identical", StopRule: "stop when measured waste exceeds max_waste_ratio", Rollback: "demand-only fetch"}
	order := append([]KVPrefetchCandidate(nil), candidates...)
	switch policy {
	case KVPrefetchDemandOnly:
		// No speculative fetch. Every needed byte is a demand fault.
	case KVPrefetchOracle:
		sort.SliceStable(order, func(i, j int) bool { return order[i].Needed && !order[j].Needed })
	case KVPrefetchPredictive:
		sort.SliceStable(order, func(i, j int) bool { return prefetchScore(order[i]) > prefetchScore(order[j]) })
	default:
		return KVPrefetchReceipt{}, fmt.Errorf("model: unknown KV prefetch policy %q", policy)
	}
	selected := map[uint64]bool{}
	if policy != KVPrefetchDemandOnly {
		for _, c := range order {
			if c.Bytes <= 0 {
				continue
			}
			if policy == KVPrefetchOracle && !c.Needed {
				continue
			}
			if r.FetchedBytes+c.Bytes > budgetBytes {
				continue
			}
			projectedPollution := r.PollutionBytes
			if !c.Needed {
				projectedPollution += c.Bytes
			}
			projectedFetched := r.FetchedBytes + c.Bytes
			if projectedFetched > 0 && float64(projectedPollution)/float64(projectedFetched) > maxWasteRatio {
				continue
			}
			selected[c.BlockID] = true
			r.SelectedBlocks = append(r.SelectedBlocks, c.BlockID)
			r.FetchedBytes = projectedFetched
			if c.Needed {
				r.UsefulBytes += c.Bytes
				if c.ReadyByDemand {
					r.TimelyBytes += c.Bytes
				}
			} else {
				r.PollutionBytes += c.Bytes
			}
		}
	}
	for _, c := range candidates {
		if c.Needed && (!selected[c.BlockID] || !c.ReadyByDemand) {
			r.FaultBytes += c.Bytes
		}
	}
	if r.FetchedBytes > 0 {
		r.WasteRatio = float64(r.PollutionBytes) / float64(r.FetchedBytes)
	}
	if acceptedTokens > 0 {
		r.UsefulBytesPerAccepted = float64(r.UsefulBytes) / float64(acceptedTokens)
	}
	return r, nil
}

func prefetchScore(c KVPrefetchCandidate) float64 {
	distance := float64(c.PositionDistance)
	if distance < 0 {
		distance = 0
	}
	return 2*c.AttentionScore + 1.5*c.RetrievalScore + c.Recency + 1/(1+distance)
}
