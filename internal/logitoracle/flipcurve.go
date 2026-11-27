// Structural adaptation of the teacher-forced flip-gap severity curve and
// Poisson count ceiling gate from humanityslastsystem/hls (Apache-2.0,
// src/hls/verify/poisson.py, flips.py, verifiers.py @ 39cd872cd1d3).
package logitoracle

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/quality"
)

var (
	// ErrInvalidFlipGateConfig indicates invalid rungs, delta, multiplier, or token count.
	ErrInvalidFlipGateConfig = errors.New("logitoracle: invalid flip-gap gate configuration")
)

// DefaultFlipGapRungs are the standard half-nat severity rungs starting at 0.5 nats.
var DefaultFlipGapRungs = []float64{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}

// PositionFlip records a single teacher-forced token comparison at a served position.
// Gap is logp_ref(ref_argmax) - logp_ref(served_token) in nats.
// Exact ties (ServedToken == RefArgmaxToken, or LogPServed == LogPRefArgmax) are not flips.
type PositionFlip struct {
	Position       int     `json:"position"`
	ServedToken    int     `json:"served_token"`
	RefArgmaxToken int     `json:"ref_argmax_token"`
	LogPRefArgmax  float64 `json:"logp_ref_argmax"`
	LogPServed     float64 `json:"logp_served"`
}

// FlipGap returns the non-negative flip severity in nats, whether the position is a flip,
// and whether the inputs are finite and valid.
func (p PositionFlip) FlipGap() (gap float64, isFlip bool, finite bool) {
	if math.IsNaN(p.LogPRefArgmax) || math.IsInf(p.LogPRefArgmax, 0) ||
		math.IsNaN(p.LogPServed) || math.IsInf(p.LogPServed, 0) {
		return math.NaN(), false, false
	}
	if p.ServedToken == p.RefArgmaxToken {
		return 0, false, true
	}
	gap = p.LogPRefArgmax - p.LogPServed
	if math.IsNaN(gap) || math.IsInf(gap, 0) {
		return math.NaN(), false, false
	}
	if gap <= 0 {
		// Exact tie in log-probability (or served >= ref argmax): not a flip.
		return 0, false, true
	}
	return gap, true, true
}

// FlipGateConfig parameterizes the teacher-forced flip-gap Poisson ceiling gate.
type FlipGateConfig struct {
	// Rungs are the severity thresholds in nats (must be strictly increasing and > 0).
	// Defaults to DefaultFlipGapRungs when empty.
	Rungs []float64 `json:"rungs"`
	// LambdaUp holds the upper-bound per-token flip rate at each rung (len == len(Rungs)).
	LambdaUp []float64 `json:"lambda_up"`
	// QMax is the maximum allowed overall flip rate (before Multiplier).
	QMax float64 `json:"q_max"`
	// Multiplier scales QMax and each rung ceiling k(c, N) (must be > 0, typically >= 1).
	Multiplier float64 `json:"multiplier"`
	// Delta is the family-wise error budget split across rungs (delta_rung = Delta / len(Rungs)).
	Delta float64 `json:"delta"`
}

// Validate checks that FlipGateConfig has finite, well-ordered rungs and valid bounds.
func (c FlipGateConfig) Validate() ([]float64, error) {
	rungs := c.Rungs
	if len(rungs) == 0 {
		rungs = DefaultFlipGapRungs
	}
	for i, r := range rungs {
		if math.IsNaN(r) || math.IsInf(r, 0) || r <= 0 {
			return nil, fmt.Errorf("%w: rung[%d]=%v must be finite and > 0", ErrInvalidFlipGateConfig, i, r)
		}
		if i > 0 && r <= rungs[i-1] {
			return nil, fmt.Errorf("%w: rungs must be strictly increasing (%v <= %v)", ErrInvalidFlipGateConfig, r, rungs[i-1])
		}
	}
	if len(c.LambdaUp) != len(rungs) {
		return nil, fmt.Errorf("%w: lambda_up length %d != rungs length %d", ErrInvalidFlipGateConfig, len(c.LambdaUp), len(rungs))
	}
	for i, lam := range c.LambdaUp {
		if math.IsNaN(lam) || math.IsInf(lam, 0) || lam < 0 {
			return nil, fmt.Errorf("%w: lambda_up[%d]=%v must be finite and >= 0", ErrInvalidFlipGateConfig, i, lam)
		}
	}
	if math.IsNaN(c.QMax) || math.IsInf(c.QMax, 0) || c.QMax < 0 {
		return nil, fmt.Errorf("%w: q_max=%v must be finite and >= 0", ErrInvalidFlipGateConfig, c.QMax)
	}
	if math.IsNaN(c.Multiplier) || math.IsInf(c.Multiplier, 0) || c.Multiplier <= 0 {
		return nil, fmt.Errorf("%w: multiplier=%v must be finite and > 0", ErrInvalidFlipGateConfig, c.Multiplier)
	}
	if math.IsNaN(c.Delta) || math.IsInf(c.Delta, 0) || c.Delta <= 0 || c.Delta >= 1 {
		return nil, fmt.Errorf("%w: delta=%v must be in (0, 1)", ErrInvalidFlipGateConfig, c.Delta)
	}
	return rungs, nil
}

// RungResult summarizes observed nested flip counts and Poisson ceiling at one severity rung.
type RungResult struct {
	RungNats    float64 `json:"rung_nats"`
	LambdaUp    float64 `json:"lambda_up"`
	Observed    int     `json:"observed"`
	BaseCeiling int     `json:"base_ceiling"`
	ScaledLimit int     `json:"scaled_limit"`
	TailPValue  float64 `json:"tail_p_value"`
	Exceeded    bool    `json:"exceeded"`
}

// FlipCurveReport is the full verdict of EvaluateFlipCurve.
type FlipCurveReport struct {
	Tokens         int          `json:"tokens"`
	TotalFlips     int          `json:"total_flips"`
	NonFiniteCount int          `json:"non_finite_count"`
	FlipRate       float64      `json:"flip_rate"`
	MaxAllowedRate float64      `json:"max_allowed_rate"`
	DeltaRung      float64      `json:"delta_rung"`
	Rungs          []RungResult `json:"rungs"`
	Pass           bool         `json:"pass"`
	FailureReason  string       `json:"failure_reason,omitempty"`
}

// EvaluateFlipCurve grades a slice of teacher-forced position comparisons against
// the severity-rung Poisson ceiling gate and overall flip-rate bound.
func EvaluateFlipCurve(flips []PositionFlip, cfg FlipGateConfig) (FlipCurveReport, error) {
	rungs, err := cfg.Validate()
	if err != nil {
		return FlipCurveReport{}, err
	}
	n := len(flips)
	if n == 0 {
		return FlipCurveReport{}, fmt.Errorf("%w: zero tokens", ErrInvalidFlipGateConfig)
	}

	nestedCounts := make([]int, len(rungs))
	totalFlips := 0
	nonFinite := 0

	for _, p := range flips {
		gap, isFlip, finite := p.FlipGap()
		if !finite {
			nonFinite++
			continue
		}
		if !isFlip {
			continue
		}
		totalFlips++
		for i, c := range rungs {
			if gap >= c {
				nestedCounts[i]++
			}
		}
	}

	deltaRung := cfg.Delta / float64(len(rungs))
	flipRate := float64(totalFlips) / float64(n)
	maxAllowedRate := cfg.Multiplier * cfg.QMax

	rungResults := make([]RungResult, len(rungs))
	pass := true
	failureReason := ""

	if nonFinite > 0 {
		pass = false
		failureReason = fmt.Sprintf("non-finite gap at %d position(s)", nonFinite)
	} else if flipRate > maxAllowedRate {
		pass = false
		failureReason = fmt.Sprintf("flip rate %.6g exceeds %.6g (multiplier=%.4g * q_max=%.6g)",
			flipRate, maxAllowedRate, cfg.Multiplier, cfg.QMax)
	}

	for i, c := range rungs {
		mu := cfg.LambdaUp[i] * float64(n)
		kBase := PoissonCeiling(mu, deltaRung)
		scaledLimit := int(math.Floor(cfg.Multiplier * float64(kBase)))
		obs := nestedCounts[i]
		pVal := 1.0
		if obs > 0 {
			pVal = PoissonUpperTail(mu, obs-1)
		}
		exceeded := obs > scaledLimit
		if exceeded && pass {
			pass = false
			failureReason = fmt.Sprintf("rung %.2f nats observed %d > scaled ceiling %d (base %d)",
				c, obs, scaledLimit, kBase)
		}
		rungResults[i] = RungResult{
			RungNats:    c,
			LambdaUp:    cfg.LambdaUp[i],
			Observed:    obs,
			BaseCeiling: kBase,
			ScaledLimit: scaledLimit,
			TailPValue:  pVal,
			Exceeded:    exceeded,
		}
	}

	return FlipCurveReport{
		Tokens:         n,
		TotalFlips:     totalFlips,
		NonFiniteCount: nonFinite,
		FlipRate:       flipRate,
		MaxAllowedRate: maxAllowedRate,
		DeltaRung:      deltaRung,
		Rungs:          rungResults,
		Pass:           pass,
		FailureReason:  failureReason,
	}, nil
}

// ToMetricComparisons converts per-rung Poisson tail p-values into quality.MetricComparison
// entries so callers can adjudicate the severity curve through quality.ControlMultiplicity.
func (r FlipCurveReport) ToMetricComparisons(model, slice string, tier quality.Tier, costSeconds float64, ev quality.Evidence) []quality.MetricComparison {
	out := make([]quality.MetricComparison, len(r.Rungs))
	for i, rr := range r.Rungs {
		e := ev
		if e.CaseID == "" {
			e.CaseID = fmt.Sprintf("flip-rung-%.2f", rr.RungNats)
		}
		if e.State == "" {
			if rr.Exceeded {
				e.State = quality.StateFail
			} else {
				e.State = quality.StatePass
			}
		}
		out[i] = quality.MetricComparison{
			Model:       model,
			Slice:       slice,
			Metric:      fmt.Sprintf("flip_rung_%.2f", rr.RungNats),
			P:           rr.TailPValue,
			Tier:        tier,
			CostSeconds: costSeconds,
			Evidence:    e,
		}
	}
	return out
}

// PoissonCeiling returns min{k >= 0 : P(Poisson(mu) > k) <= delta} using a
// numerically stable log-space tail evaluation.
func PoissonCeiling(mu, delta float64) int {
	if mu <= 0 || math.IsNaN(mu) {
		return 0
	}
	if delta <= 0 {
		return math.MaxInt32
	}
	if delta >= 1 {
		return 0
	}
	// Check k = 0 first.
	if PoissonUpperTail(mu, 0) <= delta {
		return 0
	}
	lo := 0
	hi := max(1, int(math.Ceil(mu))+4)
	for PoissonUpperTail(mu, hi) > delta {
		lo = hi
		hi *= 2
		if hi > 1<<28 {
			return hi
		}
	}
	for lo+1 < hi {
		mid := lo + (hi-lo)/2
		if PoissonUpperTail(mu, mid) <= delta {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}

// PoissonUpperTail computes P(Poisson(mu) > k) in log-space without underflow or
// catastrophic cancellation.
func PoissonUpperTail(mu float64, k int) float64 {
	if k < 0 {
		return 1.0
	}
	if mu <= 0 {
		return 0.0
	}
	if float64(k) < mu {
		// Lower CDF is at most ~0.5, so 1 - cdf has no cancellation.
		logMu := math.Log(mu)
		var terms []float64
		for i := 0; i <= k; i++ {
			lg, _ := math.Lgamma(float64(i + 1))
			terms = append(terms, -mu+float64(i)*logMu-lg)
		}
		cdf := math.Exp(logSumExp(terms))
		if cdf >= 1 {
			return 0
		}
		return 1.0 - cdf
	}
	// k >= mu: sum upper tail j = k+1, k+2, ... directly in log-space.
	logMu := math.Log(mu)
	lg0, _ := math.Lgamma(float64(k + 2))
	logTerm := -mu + float64(k+1)*logMu - lg0
	if logTerm < -745 {
		return 0
	}
	logSum := logTerm
	for j := k + 2; j <= k+100000; j++ {
		logTerm += logMu - math.Log(float64(j))
		if logTerm < logSum-40 {
			break
		}
		logSum = logAddExp(logSum, logTerm)
	}
	if logSum >= 0 {
		return 1.0
	}
	return math.Exp(logSum)
}

func logAddExp(a, b float64) float64 {
	if a < b {
		a, b = b, a
	}
	if math.IsInf(b, -1) {
		return a
	}
	return a + math.Log1p(math.Exp(b-a))
}

func logSumExp(vals []float64) float64 {
	if len(vals) == 0 {
		return math.Inf(-1)
	}
	sorted := append([]float64(nil), vals...)
	sort.Float64s(sorted)
	maxV := sorted[len(sorted)-1]
	if math.IsInf(maxV, -1) {
		return maxV
	}
	var sum float64
	for _, v := range sorted {
		sum += math.Exp(v - maxV)
	}
	return maxV + math.Log(sum)
}
