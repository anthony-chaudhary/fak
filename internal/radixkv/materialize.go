package radixkv

import (
	"errors"
	"fmt"
)

// MaterializeMode specifies how a cached KV span is materialized into the
// execution context.
type MaterializeMode string

const (
	// MatDirectAttach attaches contiguous cached KV tokens directly without modification
	// when SourceStart == Start, SourceEnd == End, and the compatibility fence matches bit-identically.
	MatDirectAttach MaterializeMode = "direct_attach"
	// MatSelectiveRepair repairs moved spans where SourceStart != Start by re-rotating RoPE
	// rotary positions or performing selective recomputation.
	MatSelectiveRepair MaterializeMode = "selective_repair"
	// MatFullRecompute recomputes the entire span via prefill because the span fails the
	// compatibility fence or repair cost exceeds prefill.
	MatFullRecompute MaterializeMode = "full_recompute"
	// MatRejectUnsupported rejects reuse for unsupported decode regimes.
	MatRejectUnsupported MaterializeMode = "reject_unsupported"
)

// MaterializationMode is an alias for MaterializeMode.
type MaterializationMode = MaterializeMode

// MaterializeAction describes the materialization operation to apply to a single execution span.
type MaterializeAction struct {
	// Span is the underlying execution span from the reuse plan.
	Span PlanSpan `json:"span"`
	// Mode is the materialization mode determined for this span.
	Mode MaterializeMode `json:"mode"`
	// ShiftTokens is the token position offset (DstStart - SrcStart, or Start - SourceStart).
	ShiftTokens int `json:"shift_tokens"`
	// DeltaPosition is an alias for ShiftTokens.
	DeltaPosition int `json:"delta_position,omitempty"`
	// RequiresReRoPE indicates whether rotary position embeddings must be recomputed or adjusted.
	RequiresReRoPE bool `json:"requires_re_rope"`
	// Reason provides diagnostic context for the chosen materialization mode or downgrade.
	Reason string `json:"reason,omitempty"`
	// EstimatedTokens is the token count covered by this span (End - Start).
	EstimatedTokens int `json:"estimated_tokens"`
}

// MaterializeResult represents a fully adjudicated and fenced execution plan ready for execution.
type MaterializeResult struct {
	// Plan is the underlying compiled reuse plan.
	Plan *ReusePlan `json:"plan"`
	// Actions contains the materialization action for each span in Plan.Spans.
	Actions []MaterializeAction `json:"actions"`
	// DirectAttachedTokens is the total number of tokens materialized via MatDirectAttach.
	DirectAttachedTokens int `json:"direct_attached_tokens"`
	// RepairedTokens is the total number of tokens materialized via MatSelectiveRepair.
	RepairedTokens int `json:"repaired_tokens"`
	// RecomputedTokens is the total number of tokens materialized via MatFullRecompute or MatRejectUnsupported.
	RecomputedTokens int `json:"recomputed_tokens"`
	// FailsafeRecompute indicates whether any candidate span had to fall back to recompute or was rejected.
	FailsafeRecompute bool `json:"failsafe_recompute"`
}

// MaterializedPlan is an alias for MaterializeResult for backward compatibility.
type MaterializedPlan = MaterializeResult

// MaterializePlan fences and materializes an existing ReusePlan against candidate compatibility fences
// and the target execution fence. It validates compatibility, resolves moved KV spans with selective
// repair, downgrades incompatible spans to full recompute, and computes summary token distributions.
func MaterializePlan(
	plan *ReusePlan,
	targetFence RACCompatibilityFence,
	candidateFences map[string]RACCompatibilityFence,
) (*MaterializeResult, error) {
	if plan == nil {
		return nil, errors.New("radixkv: reuse plan is nil")
	}
	if targetFence.ModelID == "" {
		return nil, errors.New("radixkv: target fence missing ModelID")
	}
	if targetFence.MaxContextLen < 0 {
		return nil, errors.New("radixkv: target fence MaxContextLen cannot be negative")
	}
	for id, fence := range candidateFences {
		if fence.MaxContextLen < 0 {
			return nil, fmt.Errorf("radixkv: candidate %q MaxContextLen cannot be negative", id)
		}
	}

	actions := make([]MaterializeAction, 0, len(plan.Spans))
	failsafeRecompute := false
	var directAttachedTokens, repairedTokens, recomputedTokens int

	for _, span := range plan.Spans {
		spanLen := span.End - span.Start
		action := MaterializeAction{
			Span:            span,
			EstimatedTokens: spanLen,
		}

		switch span.Action {
		case ActionDirectReuse:
			if span.CandidateID == "" {
				// Prefix match attaches directly with zero shift
				action.Mode = MatDirectAttach
				action.ShiftTokens = 0
				action.DeltaPosition = 0
				action.RequiresReRoPE = false
				action.Reason = "prefix direct attach"
			} else {
				candFence, exists := candidateFences[span.CandidateID]
				if !exists {
					action.Mode = MatFullRecompute
					action.ShiftTokens = 0
					action.DeltaPosition = 0
					action.RequiresReRoPE = false
					action.Reason = fmt.Sprintf("candidate %q missing compatibility fence", span.CandidateID)
					failsafeRecompute = true
				} else if ok, reason := candFence.Match(targetFence); !ok {
					action.Mode = MatFullRecompute
					action.ShiftTokens = 0
					action.DeltaPosition = 0
					action.RequiresReRoPE = false
					action.Reason = fmt.Sprintf("incompatible fence: %s", reason)
					failsafeRecompute = true
				} else if span.Start == span.SourceStart {
					action.Mode = MatDirectAttach
					action.ShiftTokens = 0
					action.DeltaPosition = 0
					action.RequiresReRoPE = false
					action.Reason = "aligned direct attach"
				} else {
					shift := span.Start - span.SourceStart
					if targetFence.CompatibleWithShift(candFence) {
						action.Mode = MatSelectiveRepair
						action.ShiftTokens = shift
						action.DeltaPosition = shift
						action.RequiresReRoPE = true
						action.Reason = fmt.Sprintf("moved span selective repair (shift=%d)", shift)
					} else {
						action.Mode = MatFullRecompute
						action.ShiftTokens = 0
						action.DeltaPosition = 0
						action.RequiresReRoPE = false
						action.Reason = "regime does not support positional shift"
						failsafeRecompute = true
					}
				}
			}

		case ActionSelectiveRepair:
			if span.CandidateID != "" {
				candFence, exists := candidateFences[span.CandidateID]
				if !exists {
					action.Mode = MatFullRecompute
					action.ShiftTokens = 0
					action.DeltaPosition = 0
					action.RequiresReRoPE = false
					action.Reason = fmt.Sprintf("candidate %q missing compatibility fence", span.CandidateID)
					failsafeRecompute = true
				} else if ok, reason := candFence.Match(targetFence); !ok {
					action.Mode = MatFullRecompute
					action.ShiftTokens = 0
					action.DeltaPosition = 0
					action.RequiresReRoPE = false
					action.Reason = fmt.Sprintf("incompatible fence: %s", reason)
					failsafeRecompute = true
				} else {
					shift := span.Start - span.SourceStart
					if span.Start != span.SourceStart && !targetFence.CompatibleWithShift(candFence) {
						action.Mode = MatFullRecompute
						action.ShiftTokens = 0
						action.DeltaPosition = 0
						action.RequiresReRoPE = false
						action.Reason = "regime does not support positional shift"
						failsafeRecompute = true
					} else {
						action.Mode = MatSelectiveRepair
						action.ShiftTokens = shift
						action.DeltaPosition = shift
						action.RequiresReRoPE = (span.Start != span.SourceStart)
						action.Reason = fmt.Sprintf("selective repair (shift=%d)", shift)
					}
				}
			} else {
				shift := span.Start - span.SourceStart
				action.Mode = MatSelectiveRepair
				action.ShiftTokens = shift
				action.DeltaPosition = shift
				action.RequiresReRoPE = (span.Start != span.SourceStart)
				action.Reason = "selective repair"
			}

		case ActionCompute:
			action.Mode = MatFullRecompute
			action.ShiftTokens = 0
			action.DeltaPosition = 0
			action.RequiresReRoPE = false
			action.Reason = "prefill compute"

		case ActionReject:
			action.Mode = MatFullRecompute
			action.ShiftTokens = 0
			action.DeltaPosition = 0
			action.RequiresReRoPE = false
			action.Reason = "rejected span"
			failsafeRecompute = true

		default:
			action.Mode = MatFullRecompute
			action.ShiftTokens = 0
			action.DeltaPosition = 0
			action.RequiresReRoPE = false
			action.Reason = fmt.Sprintf("unknown action %q", span.Action)
			failsafeRecompute = true
		}

		switch action.Mode {
		case MatDirectAttach:
			directAttachedTokens += spanLen
		case MatSelectiveRepair:
			repairedTokens += spanLen
		case MatFullRecompute, MatRejectUnsupported:
			recomputedTokens += spanLen
		}

		actions = append(actions, action)
	}

	return &MaterializeResult{
		Plan:                 plan,
		Actions:              actions,
		DirectAttachedTokens: directAttachedTokens,
		RepairedTokens:       repairedTokens,
		RecomputedTokens:     recomputedTokens,
		FailsafeRecompute:    failsafeRecompute,
	}, nil
}
