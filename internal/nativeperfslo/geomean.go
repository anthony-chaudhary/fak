// Structural adaptation of SLO-normalized geometric-mean speedup scoring from
// humanityslastsystem/hls (Apache-2.0, src/hls/scoring.py, baseline_credit.py @ 39cd872cd1d3).
package nativeperfslo

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// MinSLORatio floors per-request max(TTFT/budget, TPOT/budget) to prevent ln(1/0).
const MinSLORatio = 1e-9

var (
	// ErrInvalidSLOGeomean indicates malformed budgets, empty requests, or invalid baseline credit.
	ErrInvalidSLOGeomean = errors.New("nativeperfslo: invalid SLO geomean input")
	// ErrBaselineKeyMismatch indicates candidate and baseline fingerprint/load/GPU differ.
	ErrBaselineKeyMismatch = errors.New("nativeperfslo: baseline key mismatch")
)

// BaselineKey uniquely identifies a workload, arrival load, and GPU kind for baseline credit lookup.
type BaselineKey struct {
	WorkloadFingerprint string  `json:"workload_fingerprint"`
	Load                float64 `json:"load"`
	GPUKind             string  `json:"gpu_kind"`
}

// Validate ensures all BaselineKey fields are present and finite.
func (k BaselineKey) Validate() error {
	if strings.TrimSpace(k.WorkloadFingerprint) == "" {
		return fmt.Errorf("%w: workload_fingerprint is required", ErrInvalidSLOGeomean)
	}
	if math.IsNaN(k.Load) || math.IsInf(k.Load, 0) || k.Load <= 0 {
		return fmt.Errorf("%w: load=%v must be finite and > 0", ErrInvalidSLOGeomean, k.Load)
	}
	if strings.TrimSpace(k.GPUKind) == "" {
		return fmt.Errorf("%w: gpu_kind is required", ErrInvalidSLOGeomean)
	}
	return nil
}

// SLOBudget defines the per-request TTFT and TPOT latency targets in seconds.
type SLOBudget struct {
	TTFTSeconds float64 `json:"ttft_seconds"`
	TPOTSeconds float64 `json:"tpot_seconds"`
}

// Validate ensures both SLO budgets are finite and strictly positive.
func (b SLOBudget) Validate() error {
	if math.IsNaN(b.TTFTSeconds) || math.IsInf(b.TTFTSeconds, 0) || b.TTFTSeconds <= 0 {
		return fmt.Errorf("%w: ttft_seconds=%v must be finite and > 0", ErrInvalidSLOGeomean, b.TTFTSeconds)
	}
	if math.IsNaN(b.TPOTSeconds) || math.IsInf(b.TPOTSeconds, 0) || b.TPOTSeconds <= 0 {
		return fmt.Errorf("%w: tpot_seconds=%v must be finite and > 0", ErrInvalidSLOGeomean, b.TPOTSeconds)
	}
	return nil
}

// RequestLatency records one open-loop request's observed latencies and failure state.
type RequestLatency struct {
	TTFTSeconds float64 `json:"ttft_seconds"`
	TPOTSeconds float64 `json:"tpot_seconds"`
	Failed      bool    `json:"failed,omitempty"`
}

// BaselineCredit holds the reference engine's mean log-credit B for a BaselineKey.
type BaselineCredit struct {
	Key    BaselineKey `json:"key"`
	Credit float64     `json:"credit"`
}

// GeomeanInput bundles candidate requests, SLO budgets, baseline credit, and gate status.
type GeomeanInput struct {
	Key                BaselineKey      `json:"key"`
	Budget             SLOBudget        `json:"budget"`
	Baseline           BaselineCredit   `json:"baseline"`
	Requests           []RequestLatency `json:"requests"`
	FaithfulnessPassed bool             `json:"faithfulness_passed"`
}

// GeomeanResult holds the SLO-normalized geometric-mean speedup score exp(L - B).
// When any request fails or FaithfulnessPassed is false, Score is 0 and Zeroed is true,
// while UpperBound preserves exp(L - B) for diagnosis.
type GeomeanResult struct {
	Key                BaselineKey `json:"key"`
	TotalRequests      int         `json:"total_requests"`
	FailedRequests     int         `json:"failed_requests"`
	FaithfulnessPassed bool        `json:"faithfulness_passed"`
	MeanLogCredit      float64     `json:"mean_log_credit"`
	BaselineCredit     float64     `json:"baseline_credit"`
	UpperBound         float64     `json:"upper_bound"`
	Score              float64     `json:"score"`
	Zeroed             bool        `json:"zeroed"`
	ZeroReason         string      `json:"zero_reason,omitempty"`
}

// ComputeMeanLogCredit computes L = (1/N) * sum(ln(1 / max(r_i, MinSLORatio)))
// across requests, returning L and the count of failed/non-finite requests.
func ComputeMeanLogCredit(budget SLOBudget, requests []RequestLatency) (meanLogCredit float64, failed int, err error) {
	if err := budget.Validate(); err != nil {
		return 0, 0, err
	}
	if len(requests) == 0 {
		return 0, 0, fmt.Errorf("%w: no requests", ErrInvalidSLOGeomean)
	}
	sum := 0.0
	validCount := 0
	for _, req := range requests {
		if req.Failed ||
			math.IsNaN(req.TTFTSeconds) || math.IsInf(req.TTFTSeconds, 0) || req.TTFTSeconds < 0 ||
			math.IsNaN(req.TPOTSeconds) || math.IsInf(req.TPOTSeconds, 0) || req.TPOTSeconds < 0 {
			failed++
			continue
		}
		r := math.Max(req.TTFTSeconds/budget.TTFTSeconds, req.TPOTSeconds/budget.TPOTSeconds)
		if r < MinSLORatio {
			r = MinSLORatio
		}
		sum += math.Log(1.0 / r)
		validCount++
	}
	if validCount == 0 {
		return math.Inf(-1), failed, nil
	}
	return sum / float64(validCount), failed, nil
}

// EvaluateGeomeanSpeedup scores a run as exp(L - B) against a fingerprint-, load-,
// and GPU-matched baseline credit, zeroing Score (while keeping UpperBound) if any
// request failed or FaithfulnessPassed is false.
func EvaluateGeomeanSpeedup(in GeomeanInput) (GeomeanResult, error) {
	if err := in.Key.Validate(); err != nil {
		return GeomeanResult{}, err
	}
	if err := in.Baseline.Key.Validate(); err != nil {
		return GeomeanResult{}, err
	}
	if in.Key != in.Baseline.Key {
		return GeomeanResult{}, fmt.Errorf("%w: candidate=%+v baseline=%+v",
			ErrBaselineKeyMismatch, in.Key, in.Baseline.Key)
	}
	if math.IsNaN(in.Baseline.Credit) || math.IsInf(in.Baseline.Credit, 0) {
		return GeomeanResult{}, fmt.Errorf("%w: baseline credit=%v must be finite", ErrInvalidSLOGeomean, in.Baseline.Credit)
	}

	meanL, failed, err := ComputeMeanLogCredit(in.Budget, in.Requests)
	if err != nil {
		return GeomeanResult{}, err
	}

	upper := 0.0
	if !math.IsInf(meanL, -1) {
		upper = math.Exp(meanL - in.Baseline.Credit)
	}

	res := GeomeanResult{
		Key:                in.Key,
		TotalRequests:      len(in.Requests),
		FailedRequests:     failed,
		FaithfulnessPassed: in.FaithfulnessPassed,
		MeanLogCredit:      meanL,
		BaselineCredit:     in.Baseline.Credit,
		UpperBound:         upper,
		Score:              upper,
	}

	if failed > 0 {
		res.Score = 0
		res.Zeroed = true
		res.ZeroReason = fmt.Sprintf("%d failed request(s)", failed)
	} else if !in.FaithfulnessPassed {
		res.Score = 0
		res.Zeroed = true
		res.ZeroReason = "faithfulness gate failed"
	}

	return res, nil
}
