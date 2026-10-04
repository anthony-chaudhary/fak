package model

import (
	"math"
	"testing"
)

// legacyExecuteSwiGLU is the PRE-#13452 body, transcribed verbatim from the commit
// before the migration. It is the independent original-arithmetic oracle: the
// migrated executeSwiGLU is compared against it with math.Float32bits over the same
// weights and inputs. In particular it pins the projection reduction order, the
// default SiLU formula and the down-projection loop order.
func legacyExecuteSwiGLU(x, WAct, WUp, WDown []float32, inDim, interDim int) []float32 {
	inter := make([]float32, interDim)
	for i := 0; i < interDim; i++ {
		var gSum, uSum float32
		rowOff := i * inDim
		for j := 0; j < inDim; j++ {
			xj := x[j]
			gSum += WAct[rowOff+j] * xj
			uSum += WUp[rowOff+j] * xj
		}
		inter[i] = silu(gSum) * uSum
	}

	out := make([]float32, inDim)
	for i := 0; i < inDim; i++ {
		var sum float32
		rowOff := i * interDim
		for j := 0; j < interDim; j++ {
			sum += WDown[rowOff+j] * inter[j]
		}
		out[i] = sum
	}
	return out
}

// TestGLM5NextExecuteSwiGLUFFNGatedParity proves the migrated executeSwiGLU is
// bit-identical to the pre-refactor loop over the model's own dense and sparse
// consumers. It exercises the real consumer path (ExecuteGLM5NextDenseMLP and
// ExecuteGLM5NextSparseMoE) rather than the bare helper, and compares with
// math.Float32bits, not a tolerance.
func TestGLM5NextExecuteSwiGLUFFNGatedParity(t *testing.T) {
	t.Run("dense consumer", func(t *testing.T) {
		const inDim, interDim = 5, 7
		x := []float32{0.75, -0.5, 0.25, -1.25, 1.5}
		p := GLM5NextDenseMLPParams{
			InDim:    inDim,
			InterDim: interDim,
			WAct:     glm5NextValues(interDim*inDim, 0.07),
			WUp:      glm5NextValues(interDim*inDim, -0.05),
			WDown:    glm5NextValues(inDim*interDim, 0.03),
		}
		want := legacyExecuteSwiGLU(x, p.WAct, p.WUp, p.WDown, inDim, interDim)
		got := ExecuteGLM5NextDenseMLP(x, p)
		assertFloat32BitsEqual(t, "dense", want, got)
	})

	t.Run("sparse consumer", func(t *testing.T) {
		const inDim, interDim, numExperts = 4, 6, 8
		x := []float32{1.25, -0.75, 0.5, 2.0}

		initWeight := func(scale float32) GLM5NextExpertWeight {
			return GLM5NextExpertWeight{
				WAct:  glm5NextValues(interDim*inDim, scale),
				WUp:   glm5NextValues(interDim*inDim, -scale*0.5),
				WDown: glm5NextValues(inDim*interDim, scale*0.25),
			}
		}

		routed := make([]GLM5NextExpertWeight, numExperts)
		for e := range routed {
			routed[e] = initWeight(float32(e+1) * 0.013)
		}
		params := GLM5NextMoEMLPParams{
			InDim:         inDim,
			MoEInterDim:   interDim,
			SharedExpert:  initWeight(0.05),
			RoutedExperts: routed,
		}
		route := GLM5NextMoERouteResult{
			ExpertIndices: []int{0, 3, 6},
			Weights:       []float32{0.6, 0.3, 0.1},
		}

		// Independent oracle: shared + weighted routed experts, each through the
		// pre-refactor executeSwiGLU body.
		want := make([]float32, inDim)
		shared := legacyExecuteSwiGLU(x, params.SharedExpert.WAct, params.SharedExpert.WUp, params.SharedExpert.WDown, inDim, interDim)
		for i := range want {
			want[i] += shared[i]
		}
		for k, expIdx := range route.ExpertIndices {
			exp := params.RoutedExperts[expIdx]
			expOut := legacyExecuteSwiGLU(x, exp.WAct, exp.WUp, exp.WDown, inDim, interDim)
			for i := range want {
				want[i] += route.Weights[k] * expOut[i]
			}
		}

		got := ExecuteGLM5NextSparseMoE(x, route, params)
		assertFloat32BitsEqual(t, "sparse", want, got)
	})

	t.Run("special values preserved", func(t *testing.T) {
		const inDim, interDim = 1, 3
		// A gate value large enough that silu saturates toward 1, plus a zero row,
		// exercises the activation endpoint and signed-zero payload through the
		// shared helper without any tolerance substitution.
		x := []float32{1}
		p := GLM5NextDenseMLPParams{
			InDim:    inDim,
			InterDim: interDim,
			WAct:     []float32{40, -40, 0},
			WUp:      []float32{1, 1, 1},
			WDown:    []float32{1, 1, 1},
		}
		want := legacyExecuteSwiGLU(x, p.WAct, p.WUp, p.WDown, inDim, interDim)
		got := ExecuteGLM5NextDenseMLP(x, p)
		assertFloat32BitsEqual(t, "special-values", want, got)
	})
}

// glm5NextValues builds a deterministic weight slice from a per-index formula so
// the parity oracle and the production path consume identical bits.
func glm5NextValues(n int, scale float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.Sin(float64(i)*0.37+0.1)) * scale
	}
	return v
}

func TestExecuteGLM5NextDenseMLP(t *testing.T) {
	const inDim = 4
	const interDim = 8

	x := []float32{1.0, 0.5, -0.5, 2.0}

	WAct := make([]float32, interDim*inDim)
	WUp := make([]float32, interDim*inDim)
	WDown := make([]float32, inDim*interDim)

	for i := range WAct {
		WAct[i] = 0.1
		WUp[i] = 0.2
	}
	for i := range WDown {
		WDown[i] = 0.05
	}

	params := GLM5NextDenseMLPParams{
		InDim:    inDim,
		InterDim: interDim,
		WAct:     WAct,
		WUp:      WUp,
		WDown:    WDown,
	}

	out := ExecuteGLM5NextDenseMLP(x, params)
	if len(out) != inDim {
		t.Fatalf("len(out) = %d, want %d", len(out), inDim)
	}

	// Verify non-zero output
	for i, v := range out {
		if v == 0 {
			t.Fatalf("dense MLP output[%d] = 0", i)
		}
	}
}

func TestExecuteGLM5NextSparseMoE(t *testing.T) {
	const inDim = 4
	const interDim = 8
	const numExperts = 16

	x := []float32{1.0, 1.0, 1.0, 1.0}

	initWeight := func(scale float32) GLM5NextExpertWeight {
		g := make([]float32, interDim*inDim)
		u := make([]float32, interDim*inDim)
		d := make([]float32, inDim*interDim)
		for i := range g {
			g[i] = scale
			u[i] = scale
		}
		for i := range d {
			d[i] = scale
		}
		return GLM5NextExpertWeight{WAct: g, WUp: u, WDown: d}
	}

	routed := make([]GLM5NextExpertWeight, numExperts)
	for e := 0; e < numExperts; e++ {
		routed[e] = initWeight(float32(e+1) * 0.01)
	}

	params := GLM5NextMoEMLPParams{
		InDim:         inDim,
		MoEInterDim:   interDim,
		SharedExpert:  initWeight(0.05),
		RoutedExperts: routed,
	}

	// Route to expert 0 and expert 1 with weights 0.6 and 0.4
	route := GLM5NextMoERouteResult{
		ExpertIndices: []int{0, 1},
		Weights:       []float32{0.6, 0.4},
	}

	out := ExecuteGLM5NextSparseMoE(x, route, params)
	if len(out) != inDim {
		t.Fatalf("len(out) = %d, want %d", len(out), inDim)
	}

	for i, v := range out {
		if v <= 0 || math.IsNaN(float64(v)) {
			t.Fatalf("sparse MoE output[%d] = %g, expected positive finite", i, v)
		}
	}
}
