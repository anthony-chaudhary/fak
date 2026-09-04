// Package qwensemanticstop validates server-side compute-reclamation receipts.
package qwensemanticstop

import (
	"errors"
	"fmt"
	"time"
)

const (
	Schema            = "fak.qwen38-semantic-stop-receipt/1"
	ClientLatencyOnly = "client_latency_only"
	ComputeReclaimed  = "compute_reclaimed"
	HoldUnsupported   = "HOLD_CANCELLATION_UNSUPPORTED"
)

var exactModel = ModelIdentity{
	Name: "Qwen/Qwen3.8-27B", Revision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
	DType: "bfloat16", TensorParallel: 2,
}

type ModelIdentity struct {
	Name           string `json:"name"`
	Revision       string `json:"revision"`
	DType          string `json:"dtype"`
	TensorParallel int    `json:"tensor_parallel_size"`
}

type Controls struct {
	PromptSHA256 string  `json:"prompt_sha256"`
	Seed         int64   `json:"seed"`
	Temperature  float64 `json:"temperature"`
	MaxTokens    int     `json:"max_tokens"`
}

type Arm struct {
	RequestID             string  `json:"request_id"`
	GeneratedTokens       int     `json:"generated_tokens"`
	ServerBusyMS          float64 `json:"server_busy_ms"`
	GPUUtilizationPercent float64 `json:"gpu_utilization_percent"`
	NextRequestLatencyMS  float64 `json:"next_request_latency_ms"`
	FinishState           string  `json:"finish_state"`
	DisconnectObserved    bool    `json:"disconnect_observed,omitempty"`
	CancelObserved        bool    `json:"cancel_observed,omitempty"`
	CancellationLatencyMS float64 `json:"cancellation_latency_ms,omitempty"`
	TokensAfterCancel     int     `json:"tokens_after_cancel,omitempty"`
	SchedulerReleased     bool    `json:"scheduler_released,omitempty"`
}

type Pair struct {
	Controls   Controls `json:"controls"`
	Control    Arm      `json:"control"`
	Comparison Arm      `json:"comparison"`
}

type Receipt struct {
	Schema                 string        `json:"schema"`
	Backend                string        `json:"backend"`
	BackendVersion         string        `json:"backend_version"`
	Model                  ModelIdentity `json:"model"`
	CancellationSupported  bool          `json:"cancellation_supported"`
	CancellationBoundMS    float64       `json:"cancellation_bound_ms"`
	NextLatencyTolerance   float64       `json:"next_latency_tolerance_fraction"`
	Pairs                  []Pair        `json:"pairs"`
	Semantics              string        `json:"semantics"`
	HoldReason             string        `json:"hold_reason,omitempty"`
	PromotionEvidence      string        `json:"promotion_evidence,omitempty"`
	DemotionEvidence       string        `json:"demotion_evidence,omitempty"`
	InvalidatingAssumption string        `json:"invalidating_assumption,omitempty"`
	EvaluatedAt            time.Time     `json:"evaluated_at,omitempty"`
}

// Invariant: semantic stop evaluation is fail-closed and deterministic.
// Guard: Evaluate rejects missing receipts, unsupported cancellation, model mismatches,
// insufficient evaluation pairs, invalid control parameters, or late cancellations.
// Evaluate fails closed unless a receipt proves that ten interleaved exact-model
// disconnects reached the scheduler and stopped generation within the declared bound.
func Evaluate(r *Receipt) error {
	if r == nil {
		return errors.New("RECEIPT_MISSING")
	}
	r.Schema = Schema
	r.Semantics = ClientLatencyOnly
	if !r.CancellationSupported {
		r.HoldReason = HoldUnsupported
		return errors.New(HoldUnsupported)
	}
	if r.Model != exactModel {
		return errors.New("EXACT_MODEL_MISMATCH")
	}
	if r.Backend == "" || r.BackendVersion == "" {
		return errors.New("BACKEND_SCOPE_MISSING")
	}
	if len(r.Pairs) < 10 {
		return fmt.Errorf("PAIR_COUNT_INSUFFICIENT: got %d, want at least 10", len(r.Pairs))
	}
	if r.CancellationBoundMS <= 0 {
		return errors.New("CANCELLATION_BOUND_MISSING")
	}
	for i, p := range r.Pairs {
		if p.Controls.MaxTokens != 256 || p.Controls.PromptSHA256 == "" || p.Controls.Temperature != 0 {
			return fmt.Errorf("PAIR_%d_CONTROLS_INVALID", i)
		}
		if p.Control.RequestID == "" || p.Comparison.RequestID == "" || p.Control.RequestID == p.Comparison.RequestID {
			return fmt.Errorf("PAIR_%d_REQUEST_ID_INVALID", i)
		}
		if p.Control.GeneratedTokens != 256 || p.Control.FinishState != "length" {
			return fmt.Errorf("PAIR_%d_CONTROL_NOT_DRAINED", i)
		}
		c := p.Comparison
		if !c.DisconnectObserved || !c.CancelObserved || !c.SchedulerReleased || c.FinishState != "cancelled" {
			return fmt.Errorf("PAIR_%d_CANCELLATION_UNPROVEN", i)
		}
		if c.CancellationLatencyMS < 0 || c.CancellationLatencyMS > r.CancellationBoundMS {
			return fmt.Errorf("PAIR_%d_CANCELLATION_LATE", i)
		}
		if c.GeneratedTokens >= 256 || c.TokensAfterCancel != 0 {
			return fmt.Errorf("PAIR_%d_COMPUTE_NOT_RECLAIMED", i)
		}
		if p.Control.ServerBusyMS <= 0 || c.ServerBusyMS <= 0 || p.Control.GPUUtilizationPercent < 0 || c.GPUUtilizationPercent < 0 {
			return fmt.Errorf("PAIR_%d_ACCOUNTING_MISSING", i)
		}
		limit := p.Control.NextRequestLatencyMS * (1 + r.NextLatencyTolerance)
		if p.Control.NextRequestLatencyMS <= 0 || c.NextRequestLatencyMS <= 0 || c.NextRequestLatencyMS > limit {
			return fmt.Errorf("PAIR_%d_NEXT_REQUEST_CONTAMINATED", i)
		}
	}
	if r.PromotionEvidence == "" || r.DemotionEvidence == "" || r.InvalidatingAssumption == "" {
		return errors.New("LIFECYCLE_EVIDENCE_MISSING")
	}
	r.Semantics = ComputeReclaimed
	r.HoldReason = ""
	r.EvaluatedAt = time.Now().UTC()
	return nil
}
