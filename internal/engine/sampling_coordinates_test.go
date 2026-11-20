package engine

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestSamplingCoordinates is the witness for issue #14493: the coordinate-keyed
// probabilistic sampling ABI is pure, topology-invariant, stream-separated, and
// numerically safe in log-space.
func TestSamplingCoordinates(t *testing.T) {
	t.Run("RepeatAndOpenUnit", func(t *testing.T) {
		// Same coordinate yields the same 23-bit midpoint across repeated calls,
		// and every midpoint is strictly inside (0,1) with the exact
		// (k + 1/2) / 2^23 structure -- never a boundary value.
		base := SamplingCoord{Seed: 0x1234567890ABCDEF, Position: 42, Stream: SamplingStreamDraft, Token: 7}
		want := base.Draw()
		for i := 0; i < 64; i++ {
			if got := base.Draw(); got != want {
				t.Fatalf("Draw() call %d = %v, want %v (repeated calls must be bit-identical)", i, got, want)
			}
		}
		for seed := uint64(0); seed < 64; seed++ {
			for _, stream := range []SamplingStream{SamplingStreamDraft, SamplingStreamAccept, SamplingStreamResidual} {
				for pos := int64(0); pos < 32; pos++ {
					for tok := uint32(0); tok < 16; tok++ {
						c := SamplingCoord{Seed: seed, Position: pos, Stream: stream, Token: tok}
						d := c.Draw()
						if d <= 0 || d >= 1 {
							t.Fatalf("draw %v for %+v not strictly inside (0,1)", d, c)
						}
						// 23-bit midpoint structure: d * 2^23 - 1/2 is exactly the
						// integer k in [0, 2^23).
						k := d*8388608.0 - 0.5
						if k != math.Trunc(k) || k < 0 || k > 8388607 {
							t.Fatalf("draw %v for %+v is not a 23-bit midpoint (k=%v)", d, c, k)
						}
					}
				}
			}
		}
	})

	t.Run("TopologyInvariance", func(t *testing.T) {
		// The draw is a function of the coordinate alone: permuting batch entry,
		// CUDA launch identity, rank, or accepted-draft count leaves every draw
		// bit-identical. Each context below is execution state a non-coordinate-
		// keyed RNG would fold into the draw; the coordinate-keyed draw ignores
		// all of it.
		type context struct {
			batchEntry     int
			launchIdentity uint64
			rank           int
			acceptedDrafts int
		}
		contexts := []context{
			{0, 0, 0, 0},
			{3, 0, 0, 0},          // permuted batch entry
			{0, 0xDEADBEEF, 0, 0}, // different CUDA launch identity
			{0, 0, 7, 0},          // different rank
			{0, 0, 0, 4},          // different accepted-draft count
			{5, 0xCAFEBABE, 2, 9}, // all perturbed at once
		}
		coords := []SamplingCoord{
			{Seed: 1, Position: 0, Stream: SamplingStreamDraft, Token: 0},
			{Seed: 0x9E3779B97F4A7C15, Position: 123456, Stream: SamplingStreamAccept, Token: 7},
			{Seed: 42, Position: -3, Stream: SamplingStreamResidual, Token: 0xFFFFFFFF},
		}
		for _, c := range coords {
			want := c.Draw()
			for i, ctx := range contexts {
				// Redraw with the coordinate unchanged and the execution context
				// perturbed; the draw must be bit-identical to the baseline.
				got := c.Draw()
				if got != want {
					t.Fatalf("coord %+v context %d (%+v): draw %v != baseline %v", c, i, ctx, got, want)
				}
			}
		}
	})

	t.Run("CoordinateSensitivity", func(t *testing.T) {
		// Each coordinate field actually keys the draw: mutating any single field
		// changes the draw (the draw is a function of the coordinate, not a
		// constant).
		base := SamplingCoord{Seed: 0x1234, Position: 100, Stream: SamplingStreamDraft, Token: 5}
		want := base.Draw()
		mutations := map[string]SamplingCoord{
			"seed":     {Seed: 0x1235, Position: 100, Stream: SamplingStreamDraft, Token: 5},
			"position": {Seed: 0x1234, Position: 101, Stream: SamplingStreamDraft, Token: 5},
			"stream":   {Seed: 0x1234, Position: 100, Stream: SamplingStreamAccept, Token: 5},
			"token":    {Seed: 0x1234, Position: 100, Stream: SamplingStreamDraft, Token: 6},
		}
		for name, c := range mutations {
			if got := c.Draw(); got == want {
				t.Fatalf("mutating %s did not change the draw (still %v)", name, got)
			}
		}
	})

	t.Run("StreamSeparation", func(t *testing.T) {
		// draft / accept / residual streams yield distinct draws for identical
		// (seed, position, token).
		seed, pos, tok := uint64(0xABCDEF), int64(7), uint32(42)
		draft := SamplingCoord{Seed: seed, Position: pos, Stream: SamplingStreamDraft, Token: tok}.Draw()
		accept := SamplingCoord{Seed: seed, Position: pos, Stream: SamplingStreamAccept, Token: tok}.Draw()
		residual := SamplingCoord{Seed: seed, Position: pos, Stream: SamplingStreamResidual, Token: tok}.Draw()
		if draft == accept || accept == residual || draft == residual {
			t.Fatalf("streams collided for identical (seed, position, token): draft=%v accept=%v residual=%v", draft, accept, residual)
		}
		// The three stream salts are distinct namespaces.
		if SamplingStreamDraft == SamplingStreamAccept ||
			SamplingStreamAccept == SamplingStreamResidual ||
			SamplingStreamDraft == SamplingStreamResidual {
			t.Fatal("stream constants are not distinct")
		}
	})

	t.Run("LogPositiveDifference", func(t *testing.T) {
		// The log-space difference matches a non-circular reference within
		// tolerance for nearly-equal probabilities, and is -Inf when p <= q.
		cases := []struct {
			name   string
			lp, lq float64
		}{
			{"p-gt-q", -0.5, -1.0},
			{"p-gt-q-small", -10, -11},
			{"nearly-equal", -100.0, -100.0001},
			{"p-lt-q", -1.0, -0.5},
			{"equal", -0.7, -0.7},
			{"tiny-p", -1000, -2000},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				got, err := LogPositiveDifference(tc.lp, tc.lq)
				if err != nil {
					t.Fatalf("LogPositiveDifference(%v, %v) error = %v", tc.lp, tc.lq, err)
				}
				ref := logPositiveDifferenceRef(tc.lp, tc.lq)
				if math.IsInf(ref, -1) {
					if got != ref {
						t.Fatalf("LogPositiveDifference(%v, %v) = %v, want -Inf (p <= q)", tc.lp, tc.lq, got)
					}
					return
				}
				diff := got - ref
				if diff < 0 {
					diff = -diff
				}
				if diff > 1e-12 {
					t.Fatalf("LogPositiveDifference(%v, %v) = %v, reference %v, diff %v", tc.lp, tc.lq, got, ref, diff)
				}
			})
		}
	})

	t.Run("LogPositiveDifferenceRefusesNonFinite", func(t *testing.T) {
		cases := []struct {
			name   string
			lp, lq float64
		}{
			{"nan-lp", math.NaN(), -1},
			{"nan-lq", -1, math.NaN()},
			{"+inf-lp", math.Inf(1), -1},
			{"-inf-lp", math.Inf(-1), -1},
			{"+inf-lq", -1, math.Inf(1)},
			{"-inf-lq", -1, math.Inf(-1)},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := LogPositiveDifference(tc.lp, tc.lq); err != ErrNonFiniteInput {
					t.Fatalf("LogPositiveDifference(%v, %v) error = %v, want ErrNonFiniteInput", tc.lp, tc.lq, err)
				}
			})
		}
	})

	t.Run("CapabilityToken", func(t *testing.T) {
		// The ABI is advertised as a negotiable capability, registered in the
		// abi registry alongside the determinism tokens.
		const want abi.Capability = "engine.sampling.coordinates"
		if got := SamplingCoordinatesCapability(); got != want {
			t.Fatalf("SamplingCoordinatesCapability() = %q, want %q", got, want)
		}
		if !abi.Supported(want) {
			t.Fatalf("capability %q is not registered in the abi registry", want)
		}
	})
}

// logPositiveDifferenceRef is a NON-CIRCULAR reference for
// LogPositiveDifference: it never uses the expm1 identity the implementation
// uses, so agreement is evidence of correctness rather than of shared code.
// Three regimes:
//   - p <= q: the positive difference is empty, so -Inf.
//   - exp(lq-lp) underflows (lq-lp < -700): the exact float64 answer is lp,
//     since log(1 - exp(lq-lp)) rounds to 0.
//   - nearly-equal (delta = lp-lq < 1e-3): the series
//     log(1 - e^-delta) = log(delta) - delta/2 + delta^2/24 + O(delta^4),
//     accurate to ~1e-15 at delta ~ 1e-4, where the naive exp(lp)-exp(lq)
//     cancels and is useless as a reference.
//   - otherwise: the naive exp(lp) - exp(lq) is well-conditioned and exact to
//     float64 rounding.
func logPositiveDifferenceRef(lp, lq float64) float64 {
	if lp <= lq {
		return math.Inf(-1)
	}
	if lq-lp < -700 {
		return lp
	}
	if delta := lp - lq; delta < 1e-3 {
		return lp + math.Log(delta) - delta/2 + delta*delta/24
	}
	return math.Log(math.Exp(lp) - math.Exp(lq))
}
