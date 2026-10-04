package model

import (
	"math"
	"testing"
)

// glm_shared_ffn_composition_test.go pins the migration of glmSharedExperts onto the
// shared ffn.Gated ordered activation+down projection (fak#13447). The shared-expert
// tail used to carry its own act(g)*u loop; it now delegates that ordered activation
// sequence and the single down projection to the ffn package, while the production
// caller keeps ownership of the gate/up projection kernel and the prepared down input.
// The migration must be numerically invisible: bit-for-bit identical output for the
// real adapter over the real GLM checkpoint geometry.

// legacyGLMSharedExperts is the PRE-#13447 body, transcribed verbatim from the commit
// before the migration. It is the independent original-arithmetic oracle: the migrated
// production glmSharedExperts is compared against it with math.Float32bits over the same
// model, weights, and inputs.
func legacyGLMSharedExperts(m *Model, layer int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	I := cfg.MoEIntermediateSize * cfg.NSharedExperts
	if I == 0 {
		I = cfg.IntermediateSize
	}
	prefix := layerName(layer, "mlp.shared_experts.")
	g := mat.mul(prefix+"gate_proj.weight", xn, I, H)
	u := mat.mul(prefix+"up_proj.weight", xn, I, H)
	for i := 0; i < I; i++ {
		g[i] = act(g[i], cfg) * u[i]
	}
	return mat.mul(prefix+"down_proj.weight", mat.prep(g), H, I)
}

// TestGLMSharedExpertsFFNGatedParity proves the migrated glmSharedExperts is
// bit-identical to the pre-refactor loop over a real GLM DSA model's shared-expert
// weights, on the real matKernel the adapter uses (residentKernel). It exercises the
// real consumer path (glmSharedExperts) rather than a bare helper, and compares the
// signed-zero and NaN-carrying behavior with math.Float32bits, not a tolerance.
func TestGLMSharedExpertsFFNGatedParity(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if cfg.NSharedExperts <= 0 {
		t.Fatalf("fixture NSharedExperts=%d, want a shared expert", cfg.NSharedExperts)
	}
	layer := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if m.hasWeight(layerName(l, "mlp.shared_experts.gate_proj.weight")) {
			layer = l
			break
		}
	}
	if layer < 0 {
		t.Fatalf("no shared-expert layer in the glm fixture")
	}
	mat := residentKernel{m}

	// Multiple inputs catch loop-order differences across more than one activation.
	for _, tc := range []struct {
		name string
		vals []float32
	}{
		{name: "ramp", vals: rampHidden(cfg.HiddenSize, func(i int) float32 { return float32(i-5) * 0.25 })},
		{name: "sign-mixed", vals: rampHidden(cfg.HiddenSize, func(i int) float32 {
			return float32(math.Sin(float64(i)*0.11+0.3)) + float32((i*3)%7) - 3
		})},
		{name: "signed-zero", vals: rampHidden(cfg.HiddenSize, func(i int) float32 {
			if i%2 == 0 {
				return float32(0)
			}
			return float32(math.Copysign(0, -1))
		})},
		{name: "large", vals: rampHidden(cfg.HiddenSize, func(i int) float32 { return float32(i*i) * 1e3 })},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := glmSharedExperts(m, layer, append([]float32(nil), tc.vals...), mat)
			want := legacyGLMSharedExperts(m, layer, append([]float32(nil), tc.vals...), mat)
			if len(got) != len(want) {
				t.Fatalf("length=%d, want %d", len(got), len(want))
			}
			for i := range want {
				if gb, wb := math.Float32bits(got[i]), math.Float32bits(want[i]); gb != wb {
					t.Fatalf("shared expert[%d] = %.9g (0x%08x), want %.9g (0x%08x)",
						i, got[i], gb, want[i], wb)
				}
			}
		})
	}
}

// rampHidden builds a HiddenSize vector from a per-index formula.
func rampHidden(hidden int, f func(int) float32) []float32 {
	x := make([]float32, hidden)
	for i := range x {
		x[i] = f(i)
	}
	return x
}
