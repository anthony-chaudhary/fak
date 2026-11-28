// Adapted from ai-dynamo/aiperf
// modified from the original): BenchmarkPoint uncertainty contract enforcing
// ci_low <= mean <= ci_high, 2D covariance cov_xy, and n_runs < 3 low-confidence
// flagging.
package perfledger

import (
	"errors"
	"fmt"
	"math"
)

// MinConfidentRuns is the minimum replicate run count before a point stops
// carrying the low-confidence warning flag.
const MinConfidentRuns = 3

// DefaultBenchmarkConfidence is the default two-sided confidence level.
const DefaultBenchmarkConfidence = 0.95

// BenchmarkPoint carries 2D summary statistics, confidence bounds, sample
// covariance, replicate count, and low-confidence flagging for multi-run plots.
type BenchmarkPoint struct {
	Label         string  `json:"label"`
	XMean         float64 `json:"x_mean"`
	YMean         float64 `json:"y_mean"`
	XCILow        float64 `json:"x_ci_low"`
	XCIHigh       float64 `json:"x_ci_high"`
	YCILow        float64 `json:"y_ci_low"`
	YCIHigh       float64 `json:"y_ci_high"`
	CovXY         float64 `json:"cov_xy"`
	VarX          float64 `json:"var_x,omitempty"`
	VarY          float64 `json:"var_y,omitempty"`
	NRuns         int     `json:"n_runs"`
	LowConfidence bool    `json:"low_confidence"`
}

// Validate enforces finiteness, NRuns >= 1, ci_low <= mean <= ci_high on both
// axes, non-negative variances, and the n_runs < 3 low-confidence flag.
func (p BenchmarkPoint) Validate() error {
	if p.NRuns < 1 {
		return fmt.Errorf("perfledger: BenchmarkPoint.NRuns must be >= 1, got %d", p.NRuns)
	}
	for _, f := range []struct {
		name string
		val  float64
	}{
		{"x_mean", p.XMean},
		{"y_mean", p.YMean},
		{"x_ci_low", p.XCILow},
		{"x_ci_high", p.XCIHigh},
		{"y_ci_low", p.YCILow},
		{"y_ci_high", p.YCIHigh},
		{"cov_xy", p.CovXY},
		{"var_x", p.VarX},
		{"var_y", p.VarY},
	} {
		if !isFiniteFloat(f.val) {
			return fmt.Errorf("perfledger: BenchmarkPoint.%s must be finite, got %v", f.name, f.val)
		}
	}
	if p.VarX < 0 || p.VarY < 0 {
		return fmt.Errorf("perfledger: BenchmarkPoint variance cannot be negative: var_x=%v var_y=%v", p.VarX, p.VarY)
	}
	if p.XCILow > p.XMean || p.XMean > p.XCIHigh {
		return fmt.Errorf("perfledger: BenchmarkPoint X CI invariant violated: [%v, %v] does not contain mean %v", p.XCILow, p.XCIHigh, p.XMean)
	}
	if p.YCILow > p.YMean || p.YMean > p.YCIHigh {
		return fmt.Errorf("perfledger: BenchmarkPoint Y CI invariant violated: [%v, %v] does not contain mean %v", p.YCILow, p.YCIHigh, p.YMean)
	}
	if p.NRuns < MinConfidentRuns && !p.LowConfidence {
		return fmt.Errorf("perfledger: BenchmarkPoint with n_runs=%d (< %d) must set LowConfidence=true", p.NRuns, MinConfidentRuns)
	}
	if p.NRuns >= MinConfidentRuns && p.LowConfidence {
		return fmt.Errorf("perfledger: BenchmarkPoint with n_runs=%d (>= %d) must not set LowConfidence=true", p.NRuns, MinConfidentRuns)
	}
	return nil
}

// NewBenchmarkPoint populates LowConfidence from NRuns (< MinConfidentRuns)
// and validates all uncertainty invariants.
func NewBenchmarkPoint(p BenchmarkPoint) (BenchmarkPoint, error) {
	p.LowConfidence = p.NRuns < MinConfidentRuns
	if err := p.Validate(); err != nil {
		return BenchmarkPoint{}, err
	}
	return p, nil
}

// ComputeBenchmarkPoint builds a validated BenchmarkPoint from paired per-run
// observations (xs, ys) with Student-t confidence intervals at confidence
// (defaulting to DefaultBenchmarkConfidence when confidence == 0).
func ComputeBenchmarkPoint(label string, xs, ys []float64, confidence float64) (BenchmarkPoint, error) {
	if len(xs) != len(ys) {
		return BenchmarkPoint{}, fmt.Errorf("perfledger: ComputeBenchmarkPoint: xs (%d) and ys (%d) length mismatch", len(xs), len(ys))
	}
	n := len(xs)
	if n == 0 {
		return BenchmarkPoint{}, errors.New("perfledger: ComputeBenchmarkPoint: empty observations")
	}
	if confidence == 0 {
		confidence = DefaultBenchmarkConfidence
	}
	if !isFiniteFloat(confidence) || confidence <= 0 || confidence >= 1 {
		return BenchmarkPoint{}, fmt.Errorf("perfledger: ComputeBenchmarkPoint: confidence must be in (0, 1), got %v", confidence)
	}

	var sumX, sumY float64
	for i := 0; i < n; i++ {
		if !isFiniteFloat(xs[i]) || !isFiniteFloat(ys[i]) {
			return BenchmarkPoint{}, fmt.Errorf("perfledger: ComputeBenchmarkPoint: non-finite observation at index %d", i)
		}
		sumX += xs[i]
		sumY += ys[i]
	}
	xMean := sumX / float64(n)
	yMean := sumY / float64(n)

	p := BenchmarkPoint{
		Label:         label,
		XMean:         xMean,
		YMean:         yMean,
		XCILow:        xMean,
		XCIHigh:       xMean,
		YCILow:        yMean,
		YCIHigh:       yMean,
		NRuns:         n,
		LowConfidence: n < MinConfidentRuns,
	}
	if n == 1 {
		return p, nil
	}

	var ssX, ssY, ssXY float64
	for i := 0; i < n; i++ {
		dx := xs[i] - xMean
		dy := ys[i] - yMean
		ssX += dx * dx
		ssY += dy * dy
		ssXY += dx * dy
	}
	df := float64(n - 1)
	p.VarX = ssX / df
	p.VarY = ssY / df
	p.CovXY = ssXY / df

	tCrit := studentTCritical(n-1, confidence)
	seX := math.Sqrt(p.VarX / float64(n))
	seY := math.Sqrt(p.VarY / float64(n))
	hwX := tCrit * seX
	hwY := tCrit * seY

	p.XCILow = xMean - hwX
	p.XCIHigh = xMean + hwX
	p.YCILow = yMean - hwY
	p.YCIHigh = yMean + hwY

	if err := p.Validate(); err != nil {
		return BenchmarkPoint{}, err
	}
	return p, nil
}

// studentTCritical returns the two-sided Student-t critical value t_{alpha/2, df}
// where P(|T| <= tCrit) = confidence via bisection on the exact Student-t CDF.
func studentTCritical(df int, confidence float64) float64 {
	targetOneTail := (1.0 - confidence) / 2.0
	lo := 0.0
	hi := 1.0
	for studentTTailProb(hi, df) > targetOneTail && hi < 1e9 {
		hi *= 2.0
	}
	for iter := 0; iter < 96; iter++ {
		mid := 0.5 * (lo + hi)
		if studentTTailProb(mid, df) > targetOneTail {
			lo = mid
		} else {
			hi = mid
		}
	}
	return 0.5 * (lo + hi)
}

// studentTTailProb computes P(T >= t) for t >= 0 using the regularized
// incomplete beta function: 0.5 * I_{df/(df+t^2)}(df/2, 1/2).
func studentTTailProb(t float64, df int) float64 {
	if t <= 0 {
		return 0.5
	}
	fdf := float64(df)
	x := fdf / (fdf + t*t)
	return 0.5 * regularizedIncompleteBeta(x, 0.5*fdf, 0.5)
}

// regularizedIncompleteBeta evaluates I_x(a, b) via Lentz continued fraction.
func regularizedIncompleteBeta(x, a, b float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	lgA, _ := math.Lgamma(a)
	lgB, _ := math.Lgamma(b)
	lgAB, _ := math.Lgamma(a + b)
	front := math.Exp(lgAB - lgA - lgB + a*math.Log(x) + b*math.Log(1-x))
	if x < (a+1)/(a+b+2) {
		return front * betaContinuedFraction(x, a, b) / a
	}
	return 1.0 - front*betaContinuedFraction(1-x, b, a)/b
}

func betaContinuedFraction(x, a, b float64) float64 {
	const (
		maxIter = 200
		eps     = 3e-14
		tiny    = 1e-300
	)
	qab := a + b
	qap := a + 1
	qam := a - 1
	c := 1.0
	d := 1.0 - qab*x/qap
	if math.Abs(d) < tiny {
		d = tiny
	}
	d = 1.0 / d
	h := d
	for m := 1; m <= maxIter; m++ {
		fm := float64(m)
		m2 := 2 * fm
		aa := fm * (b - fm) * x / ((qam + m2) * (a + m2))
		d = 1.0 + aa*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1.0 + aa/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1.0 / d
		h *= d * c

		aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2))
		d = 1.0 + aa*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1.0 + aa/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1.0 / d
		delta := d * c
		h *= delta
		if math.Abs(delta-1.0) < eps {
			break
		}
	}
	return h
}
