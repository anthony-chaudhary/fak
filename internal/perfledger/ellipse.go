// Structural INSPIRE-ONLY adaptation of ai-dynamo/aiperf confidence-ellipse geometry
// (src/aiperf/plot/geometry/ellipse.py:16 @ ab610add4e88, Apache-2.0); clean-room Go.
package perfledger

import (
	"errors"
	"math"
)

var ErrEllipseInput = errors.New("perfledger: invalid confidence-ellipse input")

// ConfidenceEllipse describes a 2-D confidence ellipse: center, semi-axes
// (major >= minor) and the rotation of the major axis in radians.
type ConfidenceEllipse struct {
	CenterX   float64 `json:"center_x"`
	CenterY   float64 `json:"center_y"`
	SemiMajor float64 `json:"semi_major"`
	SemiMinor float64 `json:"semi_minor"`
	AngleRad  float64 `json:"angle_rad"`
}

// chi2Quantile2DOF is the closed-form chi-square quantile for 2 degrees of
// freedom: -2 ln(1-p).
func chi2Quantile2DOF(confidence float64) float64 {
	return -2 * math.Log(1-confidence)
}

// ComputeConfidenceEllipse builds the confidence ellipse of paired samples at the
// given confidence in (0, 1) from the 2x2 sample covariance eigendecomposition.
// Semi-axes are sqrt(chi2 * eigenvalue); negative eigenvalues from rounding are
// clamped to zero.
func ComputeConfidenceEllipse(xs, ys []float64, confidence float64) (ConfidenceEllipse, error) {
	n := len(xs)
	if n < 2 || n != len(ys) || !(confidence > 0 && confidence < 1) {
		return ConfidenceEllipse{}, ErrEllipseInput
	}
	var mx, my float64
	for i := 0; i < n; i++ {
		if math.IsNaN(xs[i]) || math.IsInf(xs[i], 0) || math.IsNaN(ys[i]) || math.IsInf(ys[i], 0) {
			return ConfidenceEllipse{}, ErrEllipseInput
		}
		mx += xs[i]
		my += ys[i]
	}
	mx /= float64(n)
	my /= float64(n)
	var sxx, syy, sxy float64
	for i := 0; i < n; i++ {
		dx, dy := xs[i]-mx, ys[i]-my
		sxx += dx * dx
		syy += dy * dy
		sxy += dx * dy
	}
	d := float64(n - 1)
	sxx, syy, sxy = sxx/d, syy/d, sxy/d

	// Eigenvalues of [[sxx, sxy], [sxy, syy]].
	tr := sxx + syy
	disc := math.Sqrt(math.Max(0, (sxx-syy)*(sxx-syy)/4+sxy*sxy))
	l1 := math.Max(0, tr/2+disc)
	l2 := math.Max(0, tr/2-disc)
	angle := 0.5 * math.Atan2(2*sxy, sxx-syy)
	chi2 := chi2Quantile2DOF(confidence)
	return ConfidenceEllipse{
		CenterX:   mx,
		CenterY:   my,
		SemiMajor: math.Sqrt(chi2 * l1),
		SemiMinor: math.Sqrt(chi2 * l2),
		AngleRad:  angle,
	}, nil
}
