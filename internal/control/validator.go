package control

import (
	"fmt"
	"strings"
)

// Standard structured error codes for control validation failures.
const (
	ErrRelationalInvariantBatchTokens    = "ERR_RELATIONAL_INVARIANT_BATCH_TOKENS"
	ErrRelationalInvariantDraftDepth     = "ERR_RELATIONAL_INVARIANT_DRAFT_DEPTH"
	ErrRelationalInvariantVRAMOvercommit = "ERR_RELATIONAL_INVARIANT_VRAM_OVERCOMMIT"
	ErrSyntacticInvalidRange             = "ERR_SYNTACTIC_INVALID_RANGE"
	ErrSyntacticInvalidEnum              = "ERR_SYNTACTIC_INVALID_ENUM"
	ErrSyntacticRequiredField            = "ERR_SYNTACTIC_REQUIRED_FIELD"
)

// ValidationError describes a specific rule violation in a candidate configuration.
type ValidationError struct {
	Code    string `json:"code"`
	Field   string `json:"field,omitempty"`
	Message string `json:"message"`
}

func (e ValidationError) Error() string {
	if e.Field != "" {
		return fmt.Sprintf("[%s] %s: %s", e.Code, e.Field, e.Message)
	}
	return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

// ValidationErrors is an aggregate slice of ValidationError.
type ValidationErrors []ValidationError

func (ve ValidationErrors) Error() string {
	if len(ve) == 0 {
		return ""
	}
	msgs := make([]string, len(ve))
	for i, e := range ve {
		msgs[i] = e.Error()
	}
	return strings.Join(msgs, "; ")
}

// HasErrors reports whether any validation errors are present.
func (ve ValidationErrors) HasErrors() bool {
	return len(ve) > 0
}

// Validate performs both syntactic type bounds checking and relational invariant verification.
func Validate(cfg ServingConfig) ValidationErrors {
	var errs ValidationErrors

	// 1. Syntactic / Type Bounds Checking
	if cfg.CompletionDeadlineMs > 3600000 {
		errs = append(errs, ValidationError{
			Code:    ErrSyntacticInvalidRange,
			Field:   "completion_deadline_ms",
			Message: "completion_deadline_ms exceeds ceiling of 3600000ms (1h)",
		})
	}
	if cfg.StreamProgressTimeoutMs != 0 && (cfg.StreamProgressTimeoutMs < 5000 || cfg.StreamProgressTimeoutMs > 600000) {
		errs = append(errs, ValidationError{
			Code:    ErrSyntacticInvalidRange,
			Field:   "stream_progress_timeout_ms",
			Message: "stream_progress_timeout_ms must be 0 or within [5000, 600000]ms (5s-600s)",
		})
	}
	if cfg.MaxWaitingSeqs > 1000000 {
		errs = append(errs, ValidationError{
			Code:    ErrSyntacticInvalidRange,
			Field:   "max_waiting_seqs",
			Message: "max_waiting_seqs exceeds ceiling of 1000000",
		})
	}
	if cfg.CompactHistoryBudget < 0 || cfg.CompactHistoryBudget > 10000000 {
		errs = append(errs, ValidationError{
			Code:    ErrSyntacticInvalidRange,
			Field:   "compact_history_budget",
			Message: "compact_history_budget must be between 0 and 10000000",
		})
	}
	if cfg.CompactAnchorHead < 0 || cfg.CompactAnchorHead > 1 {
		errs = append(errs, ValidationError{
			Code:    ErrSyntacticInvalidRange,
			Field:   "compact_anchor_head",
			Message: "compact_anchor_head must be 0 or 1",
		})
	}
	if cfg.LogLevel != "" {
		lvl := strings.ToLower(strings.TrimSpace(cfg.LogLevel))
		switch lvl {
		case "debug", "info", "warn", "warning", "error":
		default:
			errs = append(errs, ValidationError{
				Code:    ErrSyntacticInvalidEnum,
				Field:   "log_level",
				Message: fmt.Sprintf("invalid log_level %q: must be debug, info, warn, or error", cfg.LogLevel),
			})
		}
	}
	if cfg.SpeculativeAcceptanceThreshold < 0.0 || cfg.SpeculativeAcceptanceThreshold > 1.0 {
		errs = append(errs, ValidationError{
			Code:    ErrSyntacticInvalidRange,
			Field:   "speculative_acceptance_threshold",
			Message: "speculative_acceptance_threshold must be in [0.0, 1.0]",
		})
	}

	if cfg.PriorityStrategy != "" {
		ps := strings.ToLower(strings.TrimSpace(cfg.PriorityStrategy))
		switch ps {
		case "fcfs", "deadline_first", "cost_fairness":
		default:
			errs = append(errs, ValidationError{
				Code:    ErrSyntacticInvalidEnum,
				Field:   "priority_strategy",
				Message: fmt.Sprintf("invalid priority_strategy %q: must be fcfs, deadline_first, or cost_fairness", cfg.PriorityStrategy),
			})
		}
	}
	if cfg.PreemptionStrategy != "" {
		pm := strings.ToLower(strings.TrimSpace(cfg.PreemptionStrategy))
		switch pm {
		case "recompute", "swap":
		default:
			errs = append(errs, ValidationError{
				Code:    ErrSyntacticInvalidEnum,
				Field:   "preemption_strategy",
				Message: fmt.Sprintf("invalid preemption_strategy %q: must be recompute or swap", cfg.PreemptionStrategy),
			})
		}
	}

	// 2. Relational Invariant Verification

	// Invariant 1: max_batch_tokens >= max_model_len
	// (prevents scheduler deadlock where a single sequence cannot fit in an iteration)
	if cfg.MaxBatchTokens > 0 && cfg.MaxModelLen > 0 {
		if cfg.MaxBatchTokens < cfg.MaxModelLen {
			errs = append(errs, ValidationError{
				Code:    ErrRelationalInvariantBatchTokens,
				Field:   "max_batch_tokens",
				Message: fmt.Sprintf("max_batch_tokens (%d) must be >= max_model_len (%d) to prevent single-sequence iteration deadlock", cfg.MaxBatchTokens, cfg.MaxModelLen),
			})
		}
	}

	// Invariant 2: speculative_draft_depth <= max_preallocated_draft_slots
	// (prevents GPU out-of-bounds pointer writes during speculative draft verification)
	if cfg.SpeculativeDraftDepth > cfg.MaxPreallocatedDraftLimit {
		errs = append(errs, ValidationError{
			Code:    ErrRelationalInvariantDraftDepth,
			Field:   "speculative_draft_depth",
			Message: fmt.Sprintf("speculative_draft_depth (%d) must be <= max_preallocated_draft_slots (%d) to prevent kernel out-of-bounds access", cfg.SpeculativeDraftDepth, cfg.MaxPreallocatedDraftLimit),
		})
	}

	// Invariant 3: target_kv_blocks * block_size_bytes <= available_vram - model_weights_bytes - activation_headroom
	// (prevents out-of-memory crashes due to KV cache overcommitment)
	if cfg.AvailableVRAMBytes > 0 && (cfg.ModelWeightsBytes > 0 || cfg.ActivationHeadroomBytes > 0) {
		fixedOverhead := cfg.ModelWeightsBytes + cfg.ActivationHeadroomBytes
		if fixedOverhead > cfg.AvailableVRAMBytes {
			errs = append(errs, ValidationError{
				Code:    ErrRelationalInvariantVRAMOvercommit,
				Field:   "available_vram_bytes",
				Message: fmt.Sprintf("fixed overhead (model weights %d bytes + headroom %d bytes = %d bytes) exceeds available VRAM (%d bytes)", cfg.ModelWeightsBytes, cfg.ActivationHeadroomBytes, fixedOverhead, cfg.AvailableVRAMBytes),
			})
		} else {
			availableKVHeadroom := cfg.AvailableVRAMBytes - fixedOverhead
			requestedKVBytes := uint64(cfg.TargetKVBlocks) * uint64(cfg.BlockSizeBytes)
			if requestedKVBytes > availableKVHeadroom {
				errs = append(errs, ValidationError{
					Code:    ErrRelationalInvariantVRAMOvercommit,
					Field:   "target_kv_blocks",
					Message: fmt.Sprintf("target_kv_blocks * block_size_bytes (%d bytes) exceeds available VRAM headroom (%d bytes = %d VRAM - %d weights - %d headroom)", requestedKVBytes, availableKVHeadroom, cfg.AvailableVRAMBytes, cfg.ModelWeightsBytes, cfg.ActivationHeadroomBytes),
				})
			}
		}
	}

	return errs
}
