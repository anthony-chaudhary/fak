package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// SpeculativeBatchConfig defines the decoupled batching parameters for primary prompt chunking
// and speculative draft verification.
type SpeculativeBatchConfig struct {
	// UBatchSize is the primary chunk size for prompt evaluation (e.g. 1024).
	UBatchSize int `json:"ubatch_size"`

	// SpecDraftUBatchSize is the decoupled micro-batch size for speculative draft verification (e.g. 256 or 512).
	// Speculative verification runs are constrained to this fixed dimension to prevent
	// driver timeouts and CUDA graph capture invalidation.
	SpecDraftUBatchSize int `json:"spec_draft_ubatch_size"`

	// MaxContextLength is the maximum supported context window length (e.g. 262144 for >200K tokens).
	MaxContextLength int `json:"max_context_length"`

	// DeviceTag optionally tags the device architecture or stream identity for graph capture keys.
	DeviceTag string `json:"device_tag,omitempty"`
}

// DefaultSpeculativeBatchConfig returns safe production defaults with 1024-token primary
// prompt chunking, 512-token decoupled speculative micro-batching, and 256K max context length.
func DefaultSpeculativeBatchConfig() SpeculativeBatchConfig {
	return SpeculativeBatchConfig{
		UBatchSize:          1024,
		SpecDraftUBatchSize: 512,
		MaxContextLength:    262144, // 256K tokens (>200K)
	}
}

// Validate checks the consistency and bounds of the batch configuration.
func (cfg SpeculativeBatchConfig) Validate() error {
	if cfg.UBatchSize <= 0 {
		return fmt.Errorf("speculative batch: UBatchSize must be positive, got %d", cfg.UBatchSize)
	}
	if cfg.SpecDraftUBatchSize <= 0 {
		return fmt.Errorf("speculative batch: SpecDraftUBatchSize must be positive, got %d", cfg.SpecDraftUBatchSize)
	}
	if cfg.MaxContextLength <= 0 {
		return fmt.Errorf("speculative batch: MaxContextLength must be positive, got %d", cfg.MaxContextLength)
	}
	if cfg.MaxContextLength < cfg.UBatchSize {
		return fmt.Errorf("speculative batch: MaxContextLength (%d) cannot be smaller than UBatchSize (%d)",
			cfg.MaxContextLength, cfg.UBatchSize)
	}
	return nil
}

// GraphPlanner creates a compute.SpeculativeGraphPlanner corresponding to this configuration.
func (cfg SpeculativeBatchConfig) GraphPlanner() (*compute.SpeculativeGraphPlanner, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return compute.NewSpeculativeGraphPlanner(compute.SpeculativeGraphConfig{
		PrimaryUBatchSize:   cfg.UBatchSize,
		SpecDraftUBatchSize: cfg.SpecDraftUBatchSize,
		DeviceTag:           cfg.DeviceTag,
	})
}

// PromptChunkDispatch specifies one chunk of prompt evaluation during prefill.
type PromptChunkDispatch struct {
	Index      int                     `json:"index"`
	StartToken int                     `json:"start_token"`
	EndToken   int                     `json:"end_token"`
	NumTokens  int                     `json:"num_tokens"`
	IsLast     bool                    `json:"is_last"`
	CaptureKey compute.GraphCaptureKey `json:"capture_key"`
}

// SpeculativeDraftDispatch represents one micro-batch dispatch for verifying speculative draft tokens.
// The execution dimension is decoupled from prompt chunking and fixed to SpecDraftUBatchSize.
type SpeculativeDraftDispatch struct {
	Index          int                     `json:"index"`
	DraftOffset    int                     `json:"draft_offset"`
	NumTokens      int                     `json:"num_tokens"`      // actual draft tokens in this micro-batch
	PaddedTokens   int                     `json:"padded_tokens"`   // padding to reach SpecDraftUBatchSize
	FixedDimension int                     `json:"fixed_dimension"` // always SpecDraftUBatchSize
	BasePosition   int                     `json:"base_position"`   // sequence position in KV context
	CaptureKey     compute.GraphCaptureKey `json:"capture_key"`     // stable speculative capture key
}

// PlanPromptChunks plans prompt evaluation dispatches in bounded units of UBatchSize.
func (cfg SpeculativeBatchConfig) PlanPromptChunks(totalPromptTokens int, startPos int) ([]PromptChunkDispatch, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if totalPromptTokens < 0 {
		return nil, fmt.Errorf("totalPromptTokens must be non-negative, got %d", totalPromptTokens)
	}
	if startPos < 0 {
		return nil, fmt.Errorf("startPos must be non-negative, got %d", startPos)
	}
	if startPos+totalPromptTokens > cfg.MaxContextLength {
		return nil, fmt.Errorf("prompt range [%d, %d) exceeds MaxContextLength (%d)",
			startPos, startPos+totalPromptTokens, cfg.MaxContextLength)
	}
	if totalPromptTokens == 0 {
		return nil, nil
	}

	planner, err := cfg.GraphPlanner()
	if err != nil {
		return nil, err
	}

	var chunks []PromptChunkDispatch
	cursor := 0
	chunkIdx := 0

	for cursor < totalPromptTokens {
		remaining := totalPromptTokens - cursor
		chunkLen := remaining
		if chunkLen > cfg.UBatchSize {
			chunkLen = cfg.UBatchSize
		}

		isLast := (cursor + chunkLen) == totalPromptTokens
		start := startPos + cursor
		end := start + chunkLen

		chunks = append(chunks, PromptChunkDispatch{
			Index:      chunkIdx,
			StartToken: start,
			EndToken:   end,
			NumTokens:  chunkLen,
			IsLast:     isLast,
			CaptureKey: planner.PrimaryCaptureKey(chunkLen),
		})

		cursor += chunkLen
		chunkIdx++
	}

	return chunks, nil
}

// PlanSpeculativeVerification plans speculative draft verification dispatches using SpecDraftUBatchSize.
// Each dispatch is decoupled from prompt chunking and mapped to the stable speculative capture dimension.
func (cfg SpeculativeBatchConfig) PlanSpeculativeVerification(draftTokens int, basePos int) ([]SpeculativeDraftDispatch, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if draftTokens <= 0 {
		return nil, fmt.Errorf("draftTokens must be positive, got %d", draftTokens)
	}
	if basePos < 0 {
		return nil, fmt.Errorf("basePos must be non-negative, got %d", basePos)
	}
	if basePos+draftTokens > cfg.MaxContextLength {
		return nil, fmt.Errorf("speculative range [%d, %d) exceeds MaxContextLength (%d)",
			basePos, basePos+draftTokens, cfg.MaxContextLength)
	}

	planner, err := cfg.GraphPlanner()
	if err != nil {
		return nil, err
	}

	fixedDim := cfg.SpecDraftUBatchSize
	captureKey := planner.SpeculativeCaptureKey()

	var dispatches []SpeculativeDraftDispatch
	cursor := 0
	batchIdx := 0

	for cursor < draftTokens {
		remaining := draftTokens - cursor
		batchLen := remaining
		if batchLen > fixedDim {
			batchLen = fixedDim
		}

		padded := fixedDim - batchLen

		dispatches = append(dispatches, SpeculativeDraftDispatch{
			Index:          batchIdx,
			DraftOffset:    cursor,
			NumTokens:      batchLen,
			PaddedTokens:   padded,
			FixedDimension: fixedDim,
			BasePosition:   basePos + cursor,
			CaptureKey:     captureKey,
		})

		cursor += batchLen
		batchIdx++
	}

	return dispatches, nil
}

// DeepContextExecutionPlan summarizes the decomposed execution plan for deep context (>200K tokens).
type DeepContextExecutionPlan struct {
	PromptChunks       []PromptChunkDispatch      `json:"prompt_chunks"`
	SpeculativeBatches []SpeculativeDraftDispatch `json:"speculative_batches"`
	TotalPromptTokens  int                        `json:"total_prompt_tokens"`
	TotalDraftTokens   int                        `json:"total_draft_tokens"`
	ContextRemaining   int                        `json:"context_remaining"`
}

// PlanDeepContextExecution decomposes a deep-context request into primary prompt chunks
// (bounded by UBatchSize) and decoupled speculative verification micro-batches (bounded by SpecDraftUBatchSize).
func (cfg SpeculativeBatchConfig) PlanDeepContextExecution(promptTokens int, draftTokens int, startPos int) (*DeepContextExecutionPlan, error) {
	promptChunks, err := cfg.PlanPromptChunks(promptTokens, startPos)
	if err != nil {
		return nil, fmt.Errorf("deep context prompt planning: %w", err)
	}

	nextPos := startPos + promptTokens
	var specBatches []SpeculativeDraftDispatch
	if draftTokens > 0 {
		specBatches, err = cfg.PlanSpeculativeVerification(draftTokens, nextPos)
		if err != nil {
			return nil, fmt.Errorf("deep context speculative verification planning: %w", err)
		}
	}

	totalUsed := nextPos + draftTokens
	return &DeepContextExecutionPlan{
		PromptChunks:       promptChunks,
		SpeculativeBatches: specBatches,
		TotalPromptTokens:  promptTokens,
		TotalDraftTokens:   draftTokens,
		ContextRemaining:   cfg.MaxContextLength - totalUsed,
	}, nil
}
