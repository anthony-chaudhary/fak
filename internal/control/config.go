package control

import (
	"encoding/json"
	"strings"
)

// ServingConfig defines the multi-tier dynamic configuration parameters
// for AI inference serving, gateway operations, scheduler policies, and memory partitions.
type ServingConfig struct {
	// Tier 0: Pure Scalars & Soft Limits
	CompletionDeadlineMs           uint32  `json:"completion_deadline_ms"`
	StreamProgressTimeoutMs        uint32  `json:"stream_progress_timeout_ms"`
	MaxWaitingSeqs                 uint32  `json:"max_waiting_seqs"`
	CompactHistoryBudget           int     `json:"compact_history_budget"`
	CompactAnchorHead              int     `json:"compact_anchor_head"`
	LogLevel                       string  `json:"log_level"`
	SpeculativeDraftDepth          uint32  `json:"speculative_draft_depth"`
	SpeculativeAcceptanceThreshold float64 `json:"speculative_acceptance_threshold"`

	// Tier 1: Algorithmic & Scheduling Policies
	MaxBatchTokens     uint32 `json:"max_batch_tokens"`
	MaxModelLen        uint32 `json:"max_model_len"`
	MaxNumSeqs         uint32 `json:"max_num_seqs"`
	PriorityStrategy   string `json:"priority_strategy,omitempty"`
	PreemptionStrategy string `json:"preemption_strategy,omitempty"`

	// Tier 2: Memory & Resource Allocations
	TargetKVBlocks            uint32 `json:"target_kv_blocks"`
	BlockSizeBytes            uint32 `json:"block_size_bytes"`
	MaxPreallocatedDraftLimit uint32 `json:"max_preallocated_draft_slots"`
	AvailableVRAMBytes        uint64 `json:"available_vram_bytes"`
	ModelWeightsBytes         uint64 `json:"model_weights_bytes"`
	ActivationHeadroomBytes   uint64 `json:"activation_headroom_bytes"`

	// Operational SLA & Canary Bounds
	DeclaredLatencySLAMS float64 `json:"declared_latency_sla_ms,omitempty"`
}

// DefaultConfig returns a safe, initialized baseline configuration.
func DefaultConfig() ServingConfig {
	return ServingConfig{
		CompletionDeadlineMs:           30000,
		StreamProgressTimeoutMs:        0,
		MaxWaitingSeqs:                 1024,
		CompactHistoryBudget:           32000,
		CompactAnchorHead:              1,
		LogLevel:                       "info",
		SpeculativeDraftDepth:          3,
		SpeculativeAcceptanceThreshold: 0.80,

		MaxBatchTokens:     16384,
		MaxModelLen:        8192,
		MaxNumSeqs:         256,
		PriorityStrategy:   "fcfs",
		PreemptionStrategy: "recompute",

		TargetKVBlocks:            32768,
		BlockSizeBytes:            2048, // e.g. 16 tokens * 64 bytes per token
		MaxPreallocatedDraftLimit: 8,
		AvailableVRAMBytes:        24 * 1024 * 1024 * 1024, // 24 GB
		ModelWeightsBytes:         14 * 1024 * 1024 * 1024, // 14 GB
		ActivationHeadroomBytes:   2 * 1024 * 1024 * 1024,  // 2 GB

		DeclaredLatencySLAMS: 250.0,
	}
}

// VersionedConfig bundles a monotonic configuration epoch with an immutable snapshot.
type VersionedConfig struct {
	Epoch  uint64        `json:"epoch"`
	Config ServingConfig `json:"config"`
}

// ConfigPatch represents a partial update to ServingConfig.
// Nil pointers indicate unchanged fields.
type ConfigPatch struct {
	CompletionDeadlineMs           *uint32  `json:"completion_deadline_ms,omitempty"`
	StreamProgressTimeoutMs        *uint32  `json:"stream_progress_timeout_ms,omitempty"`
	MaxWaitingSeqs                 *uint32  `json:"max_waiting_seqs,omitempty"`
	CompactHistoryBudget           *int     `json:"compact_history_budget,omitempty"`
	CompactAnchorHead              *int     `json:"compact_anchor_head,omitempty"`
	LogLevel                       *string  `json:"log_level,omitempty"`
	SpeculativeDraftDepth          *uint32  `json:"speculative_draft_depth,omitempty"`
	SpeculativeAcceptanceThreshold *float64 `json:"speculative_acceptance_threshold,omitempty"`

	MaxBatchTokens     *uint32 `json:"max_batch_tokens,omitempty"`
	MaxModelLen        *uint32 `json:"max_model_len,omitempty"`
	MaxNumSeqs         *uint32 `json:"max_num_seqs,omitempty"`
	PriorityStrategy   *string `json:"priority_strategy,omitempty"`
	PreemptionStrategy *string `json:"preemption_strategy,omitempty"`

	TargetKVBlocks            *uint32 `json:"target_kv_blocks,omitempty"`
	BlockSizeBytes            *uint32 `json:"block_size_bytes,omitempty"`
	MaxPreallocatedDraftLimit *uint32 `json:"max_preallocated_draft_slots,omitempty"`
	AvailableVRAMBytes        *uint64 `json:"available_vram_bytes,omitempty"`
	ModelWeightsBytes         *uint64 `json:"model_weights_bytes,omitempty"`
	ActivationHeadroomBytes   *uint64 `json:"activation_headroom_bytes,omitempty"`

	DeclaredLatencySLAMS *float64 `json:"declared_latency_sla_ms,omitempty"`
}

// Apply applies patch fields onto the base ServingConfig and returns the new config.
func (c ServingConfig) Apply(p ConfigPatch) ServingConfig {
	next := c

	if p.CompletionDeadlineMs != nil {
		next.CompletionDeadlineMs = *p.CompletionDeadlineMs
	}
	if p.StreamProgressTimeoutMs != nil {
		next.StreamProgressTimeoutMs = *p.StreamProgressTimeoutMs
	}
	if p.MaxWaitingSeqs != nil {
		next.MaxWaitingSeqs = *p.MaxWaitingSeqs
	}
	if p.CompactHistoryBudget != nil {
		next.CompactHistoryBudget = *p.CompactHistoryBudget
	}
	if p.CompactAnchorHead != nil {
		next.CompactAnchorHead = *p.CompactAnchorHead
	}
	if p.LogLevel != nil {
		next.LogLevel = strings.ToLower(strings.TrimSpace(*p.LogLevel))
	}
	if p.SpeculativeDraftDepth != nil {
		next.SpeculativeDraftDepth = *p.SpeculativeDraftDepth
	}
	if p.SpeculativeAcceptanceThreshold != nil {
		next.SpeculativeAcceptanceThreshold = *p.SpeculativeAcceptanceThreshold
	}

	if p.MaxBatchTokens != nil {
		next.MaxBatchTokens = *p.MaxBatchTokens
	}
	if p.MaxModelLen != nil {
		next.MaxModelLen = *p.MaxModelLen
	}
	if p.MaxNumSeqs != nil {
		next.MaxNumSeqs = *p.MaxNumSeqs
	}
	if p.PriorityStrategy != nil {
		next.PriorityStrategy = strings.ToLower(strings.TrimSpace(*p.PriorityStrategy))
	}
	if p.PreemptionStrategy != nil {
		next.PreemptionStrategy = strings.ToLower(strings.TrimSpace(*p.PreemptionStrategy))
	}

	if p.TargetKVBlocks != nil {
		next.TargetKVBlocks = *p.TargetKVBlocks
	}
	if p.BlockSizeBytes != nil {
		next.BlockSizeBytes = *p.BlockSizeBytes
	}
	if p.MaxPreallocatedDraftLimit != nil {
		next.MaxPreallocatedDraftLimit = *p.MaxPreallocatedDraftLimit
	}
	if p.AvailableVRAMBytes != nil {
		next.AvailableVRAMBytes = *p.AvailableVRAMBytes
	}
	if p.ModelWeightsBytes != nil {
		next.ModelWeightsBytes = *p.ModelWeightsBytes
	}
	if p.ActivationHeadroomBytes != nil {
		next.ActivationHeadroomBytes = *p.ActivationHeadroomBytes
	}

	if p.DeclaredLatencySLAMS != nil {
		next.DeclaredLatencySLAMS = *p.DeclaredLatencySLAMS
	}

	return next
}

// Clone returns a deep copy of ServingConfig.
func (c ServingConfig) Clone() ServingConfig {
	return c
}

// MarshalJSON returns the JSON encoding of ServingConfig.
func (c ServingConfig) MarshalJSON() ([]byte, error) {
	type Alias ServingConfig
	return json.Marshal((Alias)(c))
}
