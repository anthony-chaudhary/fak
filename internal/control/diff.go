package control

import (
	"fmt"
	"math"
)

// RiskLevel classifies the operational hazard of applying a dynamic configuration.
type RiskLevel string

const (
	RiskLow               RiskLevel = "LOW"
	RiskMedium            RiskLevel = "MEDIUM"
	RiskHighDrainRequired RiskLevel = "HIGH_DRAIN_REQUIRED"
)

// FieldDiff records the before and after values for a changed configuration parameter.
type FieldDiff struct {
	From any `json:"from"`
	To   any `json:"to"`
}

// ResourceImpact models the estimated memory, queue drain, and operational disruption
// of transitioning from one configuration to another.
type ResourceImpact struct {
	VRAMDeltaBytes   int64     `json:"vram_delta_bytes"`
	VRAMDeltaMB      float64   `json:"vram_delta_mb"`
	EstimatedDrainMS int64     `json:"estimated_drain_ms"`
	DrainRequired    bool      `json:"drain_required"`
	RiskLevel        RiskLevel `json:"risk_level"`
	Details          []string  `json:"details,omitempty"`
}

// ComputeDiff produces a structural diff between current and proposed configurations.
func ComputeDiff(current, proposed ServingConfig) map[string]FieldDiff {
	diff := make(map[string]FieldDiff)

	if current.CompletionDeadlineMs != proposed.CompletionDeadlineMs {
		diff["completion_deadline_ms"] = FieldDiff{From: current.CompletionDeadlineMs, To: proposed.CompletionDeadlineMs}
	}
	if current.StreamProgressTimeoutMs != proposed.StreamProgressTimeoutMs {
		diff["stream_progress_timeout_ms"] = FieldDiff{From: current.StreamProgressTimeoutMs, To: proposed.StreamProgressTimeoutMs}
	}
	if current.MaxWaitingSeqs != proposed.MaxWaitingSeqs {
		diff["max_waiting_seqs"] = FieldDiff{From: current.MaxWaitingSeqs, To: proposed.MaxWaitingSeqs}
	}
	if current.CompactHistoryBudget != proposed.CompactHistoryBudget {
		diff["compact_history_budget"] = FieldDiff{From: current.CompactHistoryBudget, To: proposed.CompactHistoryBudget}
	}
	if current.CompactAnchorHead != proposed.CompactAnchorHead {
		diff["compact_anchor_head"] = FieldDiff{From: current.CompactAnchorHead, To: proposed.CompactAnchorHead}
	}
	if current.LogLevel != proposed.LogLevel {
		diff["log_level"] = FieldDiff{From: current.LogLevel, To: proposed.LogLevel}
	}
	if current.SpeculativeDraftDepth != proposed.SpeculativeDraftDepth {
		diff["speculative_draft_depth"] = FieldDiff{From: current.SpeculativeDraftDepth, To: proposed.SpeculativeDraftDepth}
	}
	if current.SpeculativeAcceptanceThreshold != proposed.SpeculativeAcceptanceThreshold {
		diff["speculative_acceptance_threshold"] = FieldDiff{From: current.SpeculativeAcceptanceThreshold, To: proposed.SpeculativeAcceptanceThreshold}
	}

	if current.MaxBatchTokens != proposed.MaxBatchTokens {
		diff["max_batch_tokens"] = FieldDiff{From: current.MaxBatchTokens, To: proposed.MaxBatchTokens}
	}
	if current.MaxModelLen != proposed.MaxModelLen {
		diff["max_model_len"] = FieldDiff{From: current.MaxModelLen, To: proposed.MaxModelLen}
	}
	if current.MaxNumSeqs != proposed.MaxNumSeqs {
		diff["max_num_seqs"] = FieldDiff{From: current.MaxNumSeqs, To: proposed.MaxNumSeqs}
	}
	if current.PriorityStrategy != proposed.PriorityStrategy {
		diff["priority_strategy"] = FieldDiff{From: current.PriorityStrategy, To: proposed.PriorityStrategy}
	}
	if current.PreemptionStrategy != proposed.PreemptionStrategy {
		diff["preemption_strategy"] = FieldDiff{From: current.PreemptionStrategy, To: proposed.PreemptionStrategy}
	}

	if current.TargetKVBlocks != proposed.TargetKVBlocks {
		diff["target_kv_blocks"] = FieldDiff{From: current.TargetKVBlocks, To: proposed.TargetKVBlocks}
	}
	if current.BlockSizeBytes != proposed.BlockSizeBytes {
		diff["block_size_bytes"] = FieldDiff{From: current.BlockSizeBytes, To: proposed.BlockSizeBytes}
	}
	if current.MaxPreallocatedDraftLimit != proposed.MaxPreallocatedDraftLimit {
		diff["max_preallocated_draft_slots"] = FieldDiff{From: current.MaxPreallocatedDraftLimit, To: proposed.MaxPreallocatedDraftLimit}
	}
	if current.AvailableVRAMBytes != proposed.AvailableVRAMBytes {
		diff["available_vram_bytes"] = FieldDiff{From: current.AvailableVRAMBytes, To: proposed.AvailableVRAMBytes}
	}
	if current.ModelWeightsBytes != proposed.ModelWeightsBytes {
		diff["model_weights_bytes"] = FieldDiff{From: current.ModelWeightsBytes, To: proposed.ModelWeightsBytes}
	}
	if current.ActivationHeadroomBytes != proposed.ActivationHeadroomBytes {
		diff["activation_headroom_bytes"] = FieldDiff{From: current.ActivationHeadroomBytes, To: proposed.ActivationHeadroomBytes}
	}
	if current.DeclaredLatencySLAMS != proposed.DeclaredLatencySLAMS {
		diff["declared_latency_sla_ms"] = FieldDiff{From: current.DeclaredLatencySLAMS, To: proposed.DeclaredLatencySLAMS}
	}

	return diff
}

// ComputeImpact computes VRAM delta, drain duration, and transition risk level.
// activeSeqs represents the count of currently in-flight requests that would need draining
// if concurrency or memory boundaries contract.
func ComputeImpact(current, proposed ServingConfig, activeSeqs int) ResourceImpact {
	impact := ResourceImpact{
		RiskLevel: RiskLow,
	}

	// 1. VRAM Delta Calculation
	currentKVBytes := int64(current.TargetKVBlocks) * int64(current.BlockSizeBytes)
	proposedKVBytes := int64(proposed.TargetKVBlocks) * int64(proposed.BlockSizeBytes)
	vramDeltaBytes := proposedKVBytes - currentKVBytes
	impact.VRAMDeltaBytes = vramDeltaBytes
	impact.VRAMDeltaMB = math.Round((float64(vramDeltaBytes)/(1024*1024))*100) / 100

	if vramDeltaBytes != 0 {
		impact.Details = append(impact.Details, fmt.Sprintf("KV cache VRAM reservation changes by %+.2f MB", impact.VRAMDeltaMB))
	}

	// 2. Contraction & Drain Detection
	kvContraction := proposed.TargetKVBlocks < current.TargetKVBlocks || proposedKVBytes < currentKVBytes
	queueContraction := proposed.MaxWaitingSeqs < current.MaxWaitingSeqs
	concurrencyContraction := proposed.MaxNumSeqs < current.MaxNumSeqs
	batchContraction := proposed.MaxBatchTokens < current.MaxBatchTokens

	if kvContraction || queueContraction || concurrencyContraction || batchContraction {
		impact.DrainRequired = true
		impact.RiskLevel = RiskHighDrainRequired

		// Estimate drain time based on in-flight sequences and average token completion window
		var drainTime int64
		if activeSeqs > 0 {
			// Estimate 25ms per active sequence, capped by completion deadline
			drainTime = int64(activeSeqs) * 25
			if proposed.CompletionDeadlineMs > 0 && uint32(drainTime) > proposed.CompletionDeadlineMs {
				drainTime = int64(proposed.CompletionDeadlineMs)
			}
			if drainTime < 50 {
				drainTime = 50
			}
		} else {
			// Minimal quiescence barrier
			drainTime = 50
		}
		impact.EstimatedDrainMS = drainTime

		if kvContraction {
			impact.Details = append(impact.Details, "KV block allocation contracted: prefix-cache eviction and sequence drain required")
		}
		if queueContraction {
			impact.Details = append(impact.Details, "Wait queue ceiling reduced: queue drain required")
		}
		if concurrencyContraction {
			impact.Details = append(impact.Details, "Max concurrent sequences reduced: active sequence drain required")
		}
		if batchContraction {
			impact.Details = append(impact.Details, "Max batch token budget reduced: in-flight iterations must drain")
		}
		return impact
	}

	// 3. Medium Risk: Algorithmic changes or VRAM expansion
	algoChanged := current.PriorityStrategy != proposed.PriorityStrategy ||
		current.PreemptionStrategy != proposed.PreemptionStrategy ||
		current.MaxBatchTokens != proposed.MaxBatchTokens ||
		current.SpeculativeDraftDepth != proposed.SpeculativeDraftDepth

	if vramDeltaBytes > 0 || algoChanged {
		impact.RiskLevel = RiskMedium
		if vramDeltaBytes > 0 {
			impact.Details = append(impact.Details, "KV block allocation expanded: non-blocking memory growth")
		}
		if algoChanged {
			impact.Details = append(impact.Details, "Scheduler or speculative policy updated: takes effect at next iteration boundary")
		}
		return impact
	}

	// 4. Low Risk: Pure scalars
	impact.Details = append(impact.Details, "Scalar parameter update: lock-free atomic pointer swap with zero disruption")
	return impact
}
