package compute

import (
	"math"
	"sort"
	"testing"
)

// approx_distribution_test.go — issue #4267 (epic #3236, borrowed from zml's equivalence oracle).
//
// fak grades its Approx compute paths by a single SCALAR cosine floor per dtype (cudaFP16CosineMin
// et al. in cuda_accuracy_gates.go). A scalar cosine is a whole-vector reduction: a small fraction
// of catastrophically-wrong elements barely moves a cosine sitting near 1.0, so a LOCALIZED numeric
// fault can slip past a 0.997 floor. zml's oracle (compareSlices, zml/testing.zig:164) instead
// grades two float buffers as an error DISTRIBUTION — a close-fraction counted against a per-element
// (absolute_tol + relative_tol*scale) threshold with a pass-FRACTION floor (minimum_close_fraction
// = 0.999), plus tail quantiles. This file mirrors that oracle on fak's CPU-reference path (no GPU
// needed) so an Approx gate can floor on the distribution BESIDE cosine; the -tags cuda fp16 gate
// (cuda_fp16_test.go) consumes it as a real device gate.

// Approx distribution gate floors (RECORDED, mirroring zml's minimum_close_fraction = 0.999). They
// are scale-free (close-fraction and a RELATIVE-error tail), so a healthy Approx op — whose realized
// per-element relative error is far under approxDefaultRelTol — passes regardless of output
// magnitude, while a handful of blown elements collapses the close-fraction and spikes the tail.
const (
	approxDefaultAbsTol    = 1e-3 // absolute slack: |a-b| below this is close regardless of scale
	approxDefaultRelTol    = 1e-2 // per-element relative slack: |a-b| <= relTol*max(|a|,|b|) is close
	approxMinCloseFraction = 0.999 // >= this fraction of elements must be close (zml's floor)
	approxMaxRelP999       = 0.05  // the 99.9th-pct RELATIVE error must be <= this (scale-free tail bound)

	// approxDemoCosineFloor is the fp16 lane's recorded scalar cosine floor (cudaFP16CosineMin =
	// 0.997), duplicated here because that constant lives under //go:build cuda and is absent from
	// this default (non-cuda) build. Used only to demonstrate that the SCALAR gate passes a fault
	// the DISTRIBUTION gate catches.
	approxDemoCosineFloor = 0.997
)

// approxDistReport is the element-wise error distribution graded BESIDE cosine, faithful to zml's
// compareSlices report: a close-fraction, RMSE, and RELATIVE-error tail quantiles.
type approxDistReport struct {
	CloseFraction       float64 // fraction of elements within (absTol + relTol*scale)
	RMSE                float64 // sqrt(mean((a-b)^2))
	P50, P90, P99, P999 float64 // quantiles of the per-element RELATIVE error |a-b|/max(|a|,|b|)
}

// passes floors the report the way #4267 specifies: close-fraction >= floor AND the relative-error
// 99.9th-pct tail <= bound. A high cosine can no longer hide a handful of blown elements, because
// they collapse the close-fraction and spike the p999 tail.
func (r approxDistReport) passes(minCloseFraction, maxRelP999 float64) bool {
	return r.CloseFraction >= minCloseFraction && r.P999 <= maxRelP999
}

// compareDistribution grades an approx/device buffer b against a reference a as an error
// distribution (#4267 first checkable step). No GPU required — it is pure f32 arithmetic over two
// equal-length slices, so it runs on the CPU-reference path and is reusable by the -tags cuda Approx
// gates. An element is "close" when |a-b| <= absTol + relTol*max(|a|,|b|); the reported quantiles
// are of the per-element relative error so the tail bound is dimensionless.
func compareDistribution(a, b []float32, absTol, relTol float64) approxDistReport {
	n := len(a)
	if n == 0 || n != len(b) {
		return approxDistReport{}
	}
	rel := make([]float64, n)
	closeN := 0
	var sq float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		scale := math.Max(math.Abs(float64(a[i])), math.Abs(float64(b[i])))
		if d <= absTol+relTol*scale {
			closeN++
		}
		sq += d * d
		if scale > 0 {
			rel[i] = d / scale
		}
	}
	sort.Float64s(rel)
	q := func(p float64) float64 {
		idx := int(math.Round(p * float64(n-1)))
		if idx < 0 {
			idx = 0
		} else if idx > n-1 {
			idx = n - 1
		}
		return rel[idx]
	}
	return approxDistReport{
		CloseFraction: float64(closeN) / float64(n),
		RMSE:          math.Sqrt(sq / float64(n)),
		P50:           q(0.50),
		P90:           q(0.90),
		P99:           q(0.99),
		P999:          q(0.999),
	}
}

// injectLocalizedFault returns a copy of ref with k elements (spread evenly through the buffer) each
// knocked frac off — a "small fraction of catastrophically-wrong elements" that a whole-vector
// cosine barely notices, but a per-element close-fraction / tail quantile does.
func injectLocalizedFault(ref []float32, k int, frac float32) []float32 {
	dev := make([]float32, len(ref))
	copy(dev, ref)
	if k <= 0 || len(ref) == 0 {
		return dev
	}
	step := len(ref) / k
	if step == 0 {
		step = 1
	}
	for j := 0; j < k; j++ {
		i := j * step
		if i >= len(ref) {
			break
		}
		dev[i] = ref[i] * (1 + frac)
	}
	return dev
}

// TestApproxErrorDistributionCatchesLocalizedFaultCosinePasses is the #4267 discrimination witness:
// a localized fault that clears the scalar cosine floor yet fails the distribution floor — the
// gap the whole-vector cosine cannot see. Pure CPU-reference arithmetic, no GPU.
func TestApproxErrorDistributionCatchesLocalizedFaultCosinePasses(t *testing.T) {
	const n = 4096
	// A reference vector with element magnitudes ~1 so the whole-vector norm (~sqrt(n)) dwarfs a
	// handful of blown elements — exactly the regime where a scalar cosine hides a localized fault.
	var s lcg = 4267
	ref := make([]float32, n)
	for i := range ref {
		ref[i] = 1 + 0.25*s.f() // ~[0.875, 1.125)
	}

	// Localized fault: 24 of 4096 elements (~0.59%) each knocked 15% off. Catastrophic per-element,
	// negligible for the whole-vector cosine.
	dev := injectLocalizedFault(ref, 24, 0.15)

	// (1) The SCALAR cosine gate PASSES — the fault hides behind a cosine sitting at ~1.0.
	c := cosine(ref, dev)
	if c < approxDemoCosineFloor {
		t.Fatalf("setup: cosine %.6f should still clear the fp16 floor %.4f so the fault is HIDDEN from the scalar gate", c, approxDemoCosineFloor)
	}

	// (2) The DISTRIBUTION gate CATCHES it — the close-fraction collapses below the floor and the
	// p999 tail spikes, precisely the localized-fault discrimination the scalar cosine lacks.
	rep := compareDistribution(ref, dev, approxDefaultAbsTol, approxDefaultRelTol)
	if rep.passes(approxMinCloseFraction, approxMaxRelP999) {
		t.Fatalf("distribution gate FAILED to catch the localized fault: %+v (cosine=%.6f passed)", rep, c)
	}
	if rep.CloseFraction >= approxMinCloseFraction {
		t.Fatalf("close-fraction %.6f should be < floor %.4f (a localized fault must collapse it)", rep.CloseFraction, approxMinCloseFraction)
	}
	if rep.P999 <= approxMaxRelP999 {
		t.Fatalf("relative p999 %.6f should exceed bound %.4f (the blown tail must show)", rep.P999, approxMaxRelP999)
	}
	// Note the p99 (top 1%) still looks clean — only the p999 tail + close-fraction expose the
	// fault, which is the whole point of grading the tail, not a scalar reduction.
	t.Logf("#4267 discrimination: cosine=%.6f PASS(>=%.4f) but closeFraction=%.6f<%.4f p99=%.4f p999=%.4f>%.4f RMSE=%.4f => distribution CATCHES it",
		c, approxDemoCosineFloor, rep.CloseFraction, approxMinCloseFraction, rep.P99, rep.P999, approxMaxRelP999, rep.RMSE)

	// (3) Negative control: legitimate Approx drift (tiny uniform rounding noise, no localized
	// blow-up) must pass BOTH gates — the distribution gate does not false-positive on honest drift.
	clean := make([]float32, n)
	for i := range ref {
		clean[i] = ref[i] * (1 + 0.002*s.f()) // <=0.1% element noise, well under relTol
	}
	cc := cosine(ref, clean)
	cr := compareDistribution(ref, clean, approxDefaultAbsTol, approxDefaultRelTol)
	if cc < approxDemoCosineFloor || !cr.passes(approxMinCloseFraction, approxMaxRelP999) {
		t.Fatalf("negative control: legitimate drift should pass both gates: cosine=%.6f report=%+v", cc, cr)
	}
}
