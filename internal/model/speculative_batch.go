package model

import (
	"errors"
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

	// MaxActiveSeqs optionally limits the maximum active sequences concurrently permitted by the scheduler.
	MaxActiveSeqs int `json:"max_active_seqs,omitempty"`

	// MaxTokensPerReq optionally limits the maximum decode tokens (K + 1) per request.
	MaxTokensPerReq int `json:"max_tokens_per_req,omitempty"`
}

// DefaultSpeculativeBatchConfig returns safe production defaults with 1024-token primary
// prompt chunking, 512-token decoupled speculative micro-batching, and 256K max context length.
func DefaultSpeculativeBatchConfig() SpeculativeBatchConfig {
	return SpeculativeBatchConfig{
		UBatchSize:          1024,
		SpecDraftUBatchSize: 512,
		MaxContextLength:    262144, // 256K tokens (>200K)
		MaxActiveSeqs:       64,
		MaxTokensPerReq:     16,
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
	if cfg.MaxActiveSeqs < 0 {
		return fmt.Errorf("speculative batch: MaxActiveSeqs must be non-negative, got %d", cfg.MaxActiveSeqs)
	}
	if cfg.MaxTokensPerReq < 0 {
		return fmt.Errorf("speculative batch: MaxTokensPerReq must be non-negative, got %d", cfg.MaxTokensPerReq)
	}
	return nil
}

// SchedulerLimits extracts the physical runtime scheduler limits from the batch configuration.
func (cfg SpeculativeBatchConfig) SchedulerLimits() SchedulerLimits {
	maxSeqs := cfg.MaxActiveSeqs
	if maxSeqs <= 0 {
		maxSeqs = 64
	}
	maxTokensReq := cfg.MaxTokensPerReq
	if maxTokensReq <= 0 {
		maxTokensReq = 16
	}
	return SchedulerLimits{
		MaxActiveSeqs:       maxSeqs,
		SpecDraftUBatchSize: cfg.SpecDraftUBatchSize,
		MaxTokensPerReq:     maxTokensReq,
		UBatchSize:          cfg.UBatchSize,
		MaxContextLength:    cfg.MaxContextLength,
	}
}

// GraphPlanner creates a compute.SpeculativeGraphPlanner corresponding to this configuration.
func (cfg SpeculativeBatchConfig) GraphPlanner() (*compute.SpeculativeGraphPlanner, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	limits := cfg.SchedulerLimits()
	return compute.NewSpeculativeGraphPlanner(compute.SpeculativeGraphConfig{
		PrimaryUBatchSize:   cfg.UBatchSize,
		SpecDraftUBatchSize: cfg.SpecDraftUBatchSize,
		DeviceTag:           cfg.DeviceTag,
		MaxActiveSeqs:       limits.MaxActiveSeqs,
		MaxTokensPerReq:     limits.MaxTokensPerReq,
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

// SchedulerLimits defines the physical runtime scheduling constraints for speculative decoding
// and prefill execution.
type SchedulerLimits struct {
	// MaxActiveSeqs is the maximum number of active sequences allowed in a batch (N_seq <= MaxActiveSeqs).
	MaxActiveSeqs int `json:"max_active_seqs"`

	// SpecDraftUBatchSize is the ceiling on total speculative decode tokens in a micro-batch (sum(K_i + 1) <= SpecDraftUBatchSize).
	SpecDraftUBatchSize int `json:"spec_draft_ubatch_size"`

	// MaxTokensPerReq is the ceiling on decode tokens per request (K + 1 <= MaxTokensPerReq).
	MaxTokensPerReq int `json:"max_tokens_per_req"`

	// UBatchSize is the primary chunk size for prompt evaluation / prefill remainders.
	UBatchSize int `json:"ubatch_size"`

	// MaxContextLength is the maximum supported context window length.
	MaxContextLength int `json:"max_context_length"`
}

// DefaultSchedulerLimits returns safe production default scheduling limits.
func DefaultSchedulerLimits() SchedulerLimits {
	return SchedulerLimits{
		MaxActiveSeqs:       64,
		SpecDraftUBatchSize: 512,
		MaxTokensPerReq:     16,
		UBatchSize:          1024,
		MaxContextLength:    262144,
	}
}

// Validate checks that the scheduler limits are physically sound.
func (l SchedulerLimits) Validate() error {
	if l.MaxActiveSeqs <= 0 {
		return fmt.Errorf("scheduler limits: MaxActiveSeqs must be positive, got %d", l.MaxActiveSeqs)
	}
	if l.SpecDraftUBatchSize <= 0 {
		return fmt.Errorf("scheduler limits: SpecDraftUBatchSize must be positive, got %d", l.SpecDraftUBatchSize)
	}
	if l.MaxTokensPerReq <= 0 {
		return fmt.Errorf("scheduler limits: MaxTokensPerReq must be positive, got %d", l.MaxTokensPerReq)
	}
	if l.UBatchSize <= 0 {
		return fmt.Errorf("scheduler limits: UBatchSize must be positive, got %d", l.UBatchSize)
	}
	if l.MaxContextLength <= 0 {
		return fmt.Errorf("scheduler limits: MaxContextLength must be positive, got %d", l.MaxContextLength)
	}
	if l.MaxContextLength < l.UBatchSize {
		return fmt.Errorf("scheduler limits: MaxContextLength (%d) cannot be smaller than UBatchSize (%d)",
			l.MaxContextLength, l.UBatchSize)
	}
	return nil
}

// SpeculativeShape describes a concrete batch shape for speculative verification and mixed decode/prefill execution.
type SpeculativeShape struct {
	// NumSequences is the number of active sequences (N_seq).
	NumSequences int `json:"num_sequences"`

	// TokensPerSeq specifies the decode token count (K_i + 1) for each individual sequence.
	// When omitted or empty, MaxTokensPerSeq or uniform distribution is assumed.
	TokensPerSeq []int `json:"tokens_per_seq,omitempty"`

	// MaxTokensPerSeq is the maximum decode tokens (K + 1) across sequences in this shape.
	MaxTokensPerSeq int `json:"max_tokens_per_seq"`

	// TotalTokens is the total number of tokens (decode tokens + prefill tokens) in this step.
	TotalTokens int `json:"total_tokens"`

	// PrefillTokens is the number of prompt/prefill remainder tokens in a mixed step.
	PrefillTokens int `json:"prefill_tokens"`

	// ContextPosition is the sequence position in the KV context window.
	ContextPosition int `json:"context_position"`
}

// DecodeTokens returns the total speculative decode tokens represented by this shape.
func (s SpeculativeShape) DecodeTokens() int {
	if len(s.TokensPerSeq) > 0 {
		sum := 0
		for _, t := range s.TokensPerSeq {
			sum += t
		}
		return sum
	}
	if s.TotalTokens > s.PrefillTokens {
		return s.TotalTokens - s.PrefillTokens
	}
	if s.NumSequences > 0 && s.MaxTokensPerSeq > 0 {
		return s.NumSequences * s.MaxTokensPerSeq
	}
	return 0
}

// EffectiveTotalTokens returns the effective total tokens (decode + prefill) for this shape.
func (s SpeculativeShape) EffectiveTotalTokens() int {
	if s.TotalTokens > 0 {
		return s.TotalTokens
	}
	return s.DecodeTokens() + s.PrefillTokens
}

// ValidateSchedulerReachable checks whether shape can be physically produced by a real mixed decode/prefill
// step under the given scheduler limits, rejecting synthetic impossible shapes.
func ValidateSchedulerReachable(shape SpeculativeShape, limits SchedulerLimits) error {
	if err := limits.Validate(); err != nil {
		return fmt.Errorf("scheduler limits invalid: %w", err)
	}

	// A step must process at least one sequence or some prefill tokens.
	if shape.NumSequences < 0 {
		return fmt.Errorf("speculative shape: NumSequences cannot be negative, got %d", shape.NumSequences)
	}
	if shape.NumSequences == 0 && shape.PrefillTokens <= 0 {
		return errors.New("speculative shape: shape has no active sequences and no prefill tokens")
	}

	// 1. Max active sequences constraint (N_seq <= MaxActiveSeqs)
	if shape.NumSequences > limits.MaxActiveSeqs {
		return fmt.Errorf("speculative shape: NumSequences (%d) exceeds scheduler MaxActiveSeqs (%d)",
			shape.NumSequences, limits.MaxActiveSeqs)
	}

	// 2. Decode tokens per request (K + 1 <= MaxTokensPerReq)
	if len(shape.TokensPerSeq) > 0 {
		if shape.NumSequences > 0 && len(shape.TokensPerSeq) != shape.NumSequences {
			return fmt.Errorf("speculative shape: TokensPerSeq length (%d) does not match NumSequences (%d)",
				len(shape.TokensPerSeq), shape.NumSequences)
		}
		for i, k := range shape.TokensPerSeq {
			if k <= 0 {
				return fmt.Errorf("speculative shape: sequence %d has non-positive token count %d (must be >= 1 for K+1)", i, k)
			}
			if k > limits.MaxTokensPerReq {
				return fmt.Errorf("speculative shape: sequence %d token count (%d) exceeds scheduler MaxTokensPerReq (%d)",
					i, k, limits.MaxTokensPerReq)
			}
		}
	} else if shape.NumSequences > 0 {
		if shape.MaxTokensPerSeq <= 0 && shape.TotalTokens <= shape.PrefillTokens {
			return errors.New("speculative shape: active sequences present but no decode tokens specified")
		}
		if shape.MaxTokensPerSeq > limits.MaxTokensPerReq {
			return fmt.Errorf("speculative shape: MaxTokensPerSeq (%d) exceeds scheduler MaxTokensPerReq (%d)",
				shape.MaxTokensPerSeq, limits.MaxTokensPerReq)
		}
	}

	// 3. Speculative micro-batch limits (sum(K_i + 1) <= SpecDraftUBatchSize)
	decodeTokens := shape.DecodeTokens()
	if shape.NumSequences > 0 {
		if decodeTokens < shape.NumSequences {
			return fmt.Errorf("speculative shape: decode tokens (%d) cannot be less than NumSequences (%d) as each sequence requires at least 1 token",
				decodeTokens, shape.NumSequences)
		}
		maxPossible := shape.NumSequences * limits.MaxTokensPerReq
		if decodeTokens > maxPossible {
			return fmt.Errorf("speculative shape: decode tokens (%d) exceeds physical maximum (%d * %d = %d)",
				decodeTokens, shape.NumSequences, limits.MaxTokensPerReq, maxPossible)
		}
	}
	if decodeTokens > limits.SpecDraftUBatchSize {
		return fmt.Errorf("speculative shape: decode tokens (%d) exceeds scheduler SpecDraftUBatchSize (%d)",
			decodeTokens, limits.SpecDraftUBatchSize)
	}

	// 4. Context/prefill remainder constraints
	if shape.PrefillTokens < 0 {
		return fmt.Errorf("speculative shape: PrefillTokens cannot be negative, got %d", shape.PrefillTokens)
	}
	if shape.PrefillTokens > limits.UBatchSize {
		return fmt.Errorf("speculative shape: PrefillTokens (%d) exceeds scheduler UBatchSize (%d)",
			shape.PrefillTokens, limits.UBatchSize)
	}
	if shape.ContextPosition < 0 {
		return fmt.Errorf("speculative shape: ContextPosition cannot be negative, got %d", shape.ContextPosition)
	}

	effectiveTotal := shape.EffectiveTotalTokens()
	if shape.TotalTokens > 0 && shape.TotalTokens != (decodeTokens+shape.PrefillTokens) {
		return fmt.Errorf("speculative shape: TotalTokens (%d) inconsistent with decode tokens (%d) + prefill tokens (%d)",
			shape.TotalTokens, decodeTokens, shape.PrefillTokens)
	}
	if shape.ContextPosition+effectiveTotal > limits.MaxContextLength {
		return fmt.Errorf("speculative shape: context range [%d, %d) exceeds scheduler MaxContextLength (%d)",
			shape.ContextPosition, shape.ContextPosition+effectiveTotal, limits.MaxContextLength)
	}

	return nil
}

// DeriveProfileShapes generates candidate execution shapes derived strictly from scheduler constraints,
// guaranteeing every produced shape is scheduler-reachable.
func DeriveProfileShapes(limits SchedulerLimits) []SpeculativeShape {
	if err := limits.Validate(); err != nil {
		return nil
	}

	seqCounts := []int{1, 2, 4, 8, 16, 32, 64}
	var filteredSeqs []int
	hasMax := false
	for _, s := range seqCounts {
		if s <= limits.MaxActiveSeqs {
			filteredSeqs = append(filteredSeqs, s)
			if s == limits.MaxActiveSeqs {
				hasMax = true
			}
		}
	}
	if !hasMax && limits.MaxActiveSeqs > 0 {
		filteredSeqs = append(filteredSeqs, limits.MaxActiveSeqs)
	}

	tokensPerReqCandidates := []int{1, 2, 3, 4, 5, 8, 16}
	var filteredTokens []int
	hasMaxTok := false
	for _, t := range tokensPerReqCandidates {
		if t <= limits.MaxTokensPerReq {
			filteredTokens = append(filteredTokens, t)
			if t == limits.MaxTokensPerReq {
				hasMaxTok = true
			}
		}
	}
	if !hasMaxTok && limits.MaxTokensPerReq > 0 {
		filteredTokens = append(filteredTokens, limits.MaxTokensPerReq)
	}

	var shapes []SpeculativeShape

	// 1. Pure speculative decode shapes
	for _, nSeq := range filteredSeqs {
		for _, k := range filteredTokens {
			totalDecode := nSeq * k
			if totalDecode > limits.SpecDraftUBatchSize {
				continue
			}
			toks := make([]int, nSeq)
			for i := range toks {
				toks[i] = k
			}
			shape := SpeculativeShape{
				NumSequences:    nSeq,
				TokensPerSeq:    toks,
				MaxTokensPerSeq: k,
				TotalTokens:     totalDecode,
				PrefillTokens:   0,
				ContextPosition: 0,
			}
			if ValidateSchedulerReachable(shape, limits) == nil {
				shapes = append(shapes, shape)
			}
		}
	}

	// 2. Mixed decode/prefill remainder shapes for representative sequence counts
	prefillRemainders := []int{64, 128, 256, 512, 1024}
	for _, nSeq := range []int{1, 4} {
		if nSeq > limits.MaxActiveSeqs {
			continue
		}
		for _, pref := range prefillRemainders {
			if pref > limits.UBatchSize {
				continue
			}
			k := 4
			if k > limits.MaxTokensPerReq {
				k = limits.MaxTokensPerReq
			}
			totalDecode := nSeq * k
			if totalDecode > limits.SpecDraftUBatchSize {
				continue
			}
			toks := make([]int, nSeq)
			for i := range toks {
				toks[i] = k
			}
			shape := SpeculativeShape{
				NumSequences:    nSeq,
				TokensPerSeq:    toks,
				MaxTokensPerSeq: k,
				TotalTokens:     totalDecode + pref,
				PrefillTokens:   pref,
				ContextPosition: 0,
			}
			if ValidateSchedulerReachable(shape, limits) == nil {
				shapes = append(shapes, shape)
			}
		}
	}

	return shapes
}

// GenerateProfileShapes generates candidate execution shapes for this configuration,
// ensuring every produced shape is guaranteed to be scheduler-reachable.
func (cfg SpeculativeBatchConfig) GenerateProfileShapes() []SpeculativeShape {
	return DeriveProfileShapes(cfg.SchedulerLimits())
}

// DeepContextExecutionPlan summarizes the decomposed execution plan for deep context (>200K tokens).
type DeepContextExecutionPlan struct {
	PromptChunks       []PromptChunkDispatch      `json:"prompt_chunks"`
	SpeculativeBatches []SpeculativeDraftDispatch `json:"speculative_batches"`
	TotalPromptTokens  int                        `json:"total_prompt_tokens"`
	TotalDraftTokens   int                        `json:"total_draft_tokens"`
	ContextRemaining   int                        `json:"context_remaining"`
	Shape              SpeculativeShape           `json:"shape,omitempty"`
	SchedulerReachable bool                       `json:"scheduler_reachable"`
}

// SpeculativeBatchPlan describes an adjudicated speculative batch execution plan stamped with
// its selected profile shape and verified reachability.
type SpeculativeBatchPlan struct {
	Shape              SpeculativeShape           `json:"shape"`
	SchedulerReachable bool                       `json:"scheduler_reachable"`
	PromptChunks       []PromptChunkDispatch      `json:"prompt_chunks,omitempty"`
	SpeculativeBatches []SpeculativeDraftDispatch `json:"speculative_batches,omitempty"`
	TotalPromptTokens  int                        `json:"total_prompt_tokens"`
	TotalDraftTokens   int                        `json:"total_draft_tokens"`
	ContextRemaining   int                        `json:"context_remaining"`
}

// SpeculativeBatchReceipt captures the execution receipt of a verified speculative batch dispatch.
type SpeculativeBatchReceipt struct {
	Shape              SpeculativeShape        `json:"shape"`
	SchedulerReachable bool                    `json:"scheduler_reachable"`
	DraftTokens        int                     `json:"draft_tokens"`
	PaddedTokens       int                     `json:"padded_tokens"`
	CaptureKey         compute.GraphCaptureKey `json:"capture_key"`
}

// PlanSpeculativeBatch plans a speculative batch for a given shape, verifying reachability
// and stamping SchedulerReachable: true.
func (cfg SpeculativeBatchConfig) PlanSpeculativeBatch(shape SpeculativeShape) (*SpeculativeBatchPlan, error) {
	limits := cfg.SchedulerLimits()
	if err := ValidateSchedulerReachable(shape, limits); err != nil {
		return nil, fmt.Errorf("speculative batch planning rejected unreachable shape: %w", err)
	}

	var promptChunks []PromptChunkDispatch
	var err error
	if shape.PrefillTokens > 0 {
		promptChunks, err = cfg.PlanPromptChunks(shape.PrefillTokens, shape.ContextPosition)
		if err != nil {
			return nil, fmt.Errorf("planning prompt chunks for batch: %w", err)
		}
	}

	decodeTokens := shape.DecodeTokens()
	nextPos := shape.ContextPosition + shape.PrefillTokens
	var specBatches []SpeculativeDraftDispatch
	if decodeTokens > 0 {
		specBatches, err = cfg.PlanSpeculativeVerification(decodeTokens, nextPos)
		if err != nil {
			return nil, fmt.Errorf("planning speculative verification for batch: %w", err)
		}
	}

	totalUsed := nextPos + decodeTokens
	return &SpeculativeBatchPlan{
		Shape:              shape,
		SchedulerReachable: true,
		PromptChunks:       promptChunks,
		SpeculativeBatches: specBatches,
		TotalPromptTokens:  shape.PrefillTokens,
		TotalDraftTokens:   decodeTokens,
		ContextRemaining:   cfg.MaxContextLength - totalUsed,
	}, nil
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
	shape := SpeculativeShape{
		NumSequences:    1,
		MaxTokensPerSeq: draftTokens,
		TotalTokens:     draftTokens,
		PrefillTokens:   0,
		ContextPosition: nextPos,
	}
	if draftTokens > 0 {
		shape.TokensPerSeq = []int{draftTokens}
	}

	return &DeepContextExecutionPlan{
		PromptChunks:       promptChunks,
		SpeculativeBatches: specBatches,
		TotalPromptTokens:  promptTokens,
		TotalDraftTokens:   draftTokens,
		ContextRemaining:   cfg.MaxContextLength - totalUsed,
		Shape:              shape,
		SchedulerReachable: true,
	}, nil
}
