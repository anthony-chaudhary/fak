package compute

import (
	"fmt"
)

// ComponentizedMemoryBreakdown isolates the distinct physical memory demands.
type ComponentizedMemoryBreakdown struct {
	WeightsBytes       int64 `json:"weights_bytes"`
	ActivationsBytes   int64 `json:"activations_bytes"`
	SignalBuffersBytes int64 `json:"signal_buffers_bytes"`
	KVCacheBytes       int64 `json:"kv_cache_bytes"`
	ScratchpadBytes    int64 `json:"scratchpad_bytes"`
}

// TotalBytes returns the exact aggregate memory demand across all components.
func (b ComponentizedMemoryBreakdown) TotalBytes() int64 {
	return b.WeightsBytes + b.ActivationsBytes + b.SignalBuffersBytes + b.KVCacheBytes + b.ScratchpadBytes
}

// ComponentizedMemoryReceipt records whether the componentized demand fits and details component shares.
type ComponentizedMemoryReceipt struct {
	Allowed            bool                         `json:"allowed"`
	CapacityBytes      int64                        `json:"capacity_bytes"`
	EffectiveBudget    int64                        `json:"effective_budget"`
	DemandedBytes      int64                        `json:"demanded_bytes"`
	ExcessBytes        int64                        `json:"excess_bytes,omitempty"`
	RefusalReason      string                       `json:"refusal_reason,omitempty"`
	ViolatingComponent string                       `json:"violating_component,omitempty"`
	Breakdown          ComponentizedMemoryBreakdown `json:"breakdown"`
}

// EvaluateComponentizedMemoryFit checks memory demand with explicit breakdown against capacity.
// If total demand exceeds the effective budget (capacity after headroom), it refuses with
// the culpable component named.
func EvaluateComponentizedMemoryFit(
	breakdown ComponentizedMemoryBreakdown,
	capacityBytes int64,
	headroom float64,
) (ComponentizedMemoryReceipt, error) {
	if capacityBytes <= 0 {
		return ComponentizedMemoryReceipt{}, fmt.Errorf("capacityBytes must be positive, got %d", capacityBytes)
	}
	if headroom < 0 || headroom >= 1.0 {
		return ComponentizedMemoryReceipt{}, fmt.Errorf("headroom must be in [0, 1), got %v", headroom)
	}

	effectiveBudget := BudgetAfterHeadroom(capacityBytes, headroom)
	totalDemanded := breakdown.TotalBytes()

	receipt := ComponentizedMemoryReceipt{
		CapacityBytes:   capacityBytes,
		EffectiveBudget: effectiveBudget,
		DemandedBytes:   totalDemanded,
		Breakdown:       breakdown,
	}

	if totalDemanded <= effectiveBudget {
		receipt.Allowed = true
		return receipt, nil
	}

	receipt.Allowed = false
	receipt.ExcessBytes = totalDemanded - effectiveBudget

	// Identify primary contributor or single component pushing over budget
	// Check if any single component by itself exceeded budget, or which one tipped it
	culprit := "aggregate"
	switch {
	case breakdown.WeightsBytes > effectiveBudget:
		culprit = "weights"
	case breakdown.ActivationsBytes > effectiveBudget:
		culprit = "activations"
	case breakdown.SignalBuffersBytes > effectiveBudget:
		culprit = "signal_buffers"
	case breakdown.KVCacheBytes > effectiveBudget:
		culprit = "kv_cache"
	case breakdown.ScratchpadBytes > effectiveBudget:
		culprit = "scratchpad"
	default:
		// Tipped by the largest component
		largest := breakdown.WeightsBytes
		culprit = "weights"
		if breakdown.ActivationsBytes > largest {
			largest = breakdown.ActivationsBytes
			culprit = "activations"
		}
		if breakdown.SignalBuffersBytes > largest {
			largest = breakdown.SignalBuffersBytes
			culprit = "signal_buffers"
		}
		if breakdown.KVCacheBytes > largest {
			largest = breakdown.KVCacheBytes
			culprit = "kv_cache"
		}
		if breakdown.ScratchpadBytes > largest {
			culprit = "scratchpad"
		}
	}

	receipt.ViolatingComponent = culprit
	receipt.RefusalReason = fmt.Sprintf("memory demand (%s) exceeds effective budget (%s) by %s (culprit: %s)",
		memSize(totalDemanded), memSize(effectiveBudget), memSize(receipt.ExcessBytes), culprit)

	return receipt, nil
}
