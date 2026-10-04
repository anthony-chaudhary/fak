package model

import (
	"math"
	"testing"
)

// qwen_shared_ffn_composition_test.go pins the migration of the generic tail of
// qwen35SharedExpert onto the shared ffn.Gated ordered activation+down projection
// (fak#13448). Before the migration the generic tail carried its own `act(g[i])*u[i]`
// loop followed by one down projection; it now delegates that ordered activation
// sequence and the single projection to the ffn package, while the production caller
// keeps ownership of the gate/up projection kernel, the I/SharedIntermediateSize width
// fallback, and the post-down scalar sigmoid output gate. The fused Q4_K early return is
// untouched.
//
// The migration must be numerically invisible: bit-for-bit identical output for the
// real adapter over the real Qwen3.5-MoE shared-expert geometry, across every
// activation the generic tail can take.

// legacyQwen35SharedExpert is the PRE-#13448 body with its own ordered act(g)*u loop.
// It is the independent original-arithmetic oracle: the migrated production
// qwen35SharedExpert is compared against it with math.Float32bits over the same model,
// weights, and inputs. It is deliberately kept outside the migrated seam.
func legacyQwen35SharedExpert(m *Model, layer int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	I := cfg.SharedIntermediateSize
	if I == 0 {
		I = cfg.expertIntermediate()
	}
	gn := qwen35SharedExpertName(layer, "gate_proj.weight")
	un := qwen35SharedExpertName(layer, "up_proj.weight")
	dn := qwen35SharedExpertName(layer, "down_proj.weight")
	g := mat.mul(gn, xn, I, H)
	u := mat.mul(un, xn, I, H)
	for i := 0; i < I; i++ {
		g[i] = act(g[i], cfg) * u[i]
	}
	out := mat.mul(dn, mat.prep(g), H, I)
	gate := sigmoid(mat.mul(qwen35SharedExpertName(layer, "gate.weight"), xn, 1, H)[0])
	for i := 0; i < H; i++ {
		out[i] *= gate
	}
	return out
}

// buildQwen35SharedExpertModel assembles a one-layer qwen3_5_moe model carrying the
// singular shared-expert tensors at width SI, so qwen35SharedExpert's generic tail is
// the code under test. The routed experts are present so the model is a real MoE with
// the production geometry; only the shared-expert branch is exercised.
func buildQwen35SharedExpertModel(t *testing.T, sharedIntermediate int, mutate func(*Config)) *Model {
	t.Helper()
	const H, I, E = 2, 2, 4
	cfg := parsedQwen35MoETestConfig(t)
	cfg.SharedIntermediateSize = sharedIntermediate
	if mutate != nil {
		mutate(&cfg)
	}
	router := []float32{2, 0, 0, 3, 1, 1, -1, -1}
	tensors := qwen35MoEBaseTensors(H, I, E, router)
	// Shared expert (SwiGLU at width I=2) + scalar hidden->1 sigmoid gate, as in the
	// existing TestQwen35MoESharedExpertAdded fixture. Deliberately non-symmetric values
	// so the gated tail scales a non-trivial vector.
	tensors = append(tensors,
		NamedTensorF32{Name: qwen35SharedExpertName(0, "gate_proj.weight"), Shape: []int{I, H}, Data: []float32{0.5, 0, 0, 0.5}},
		NamedTensorF32{Name: qwen35SharedExpertName(0, "up_proj.weight"), Shape: []int{I, H}, Data: []float32{0.4, 0, 0, 0.4}},
		NamedTensorF32{Name: qwen35SharedExpertName(0, "down_proj.weight"), Shape: []int{H, I}, Data: []float32{0.6, 0, 0, 0.6}},
		NamedTensorF32{Name: qwen35SharedExpertName(0, "gate.weight"), Shape: []int{1, H}, Data: []float32{0.25, 0.25}},
	)
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("build qwen35 shared-expert model: %v", err)
	}
	return m
}

// TestQwen35SharedExpertFFNGatedParity proves the migrated generic tail of
// qwen35SharedExpert is bit-identical to the pre-refactor loop over the real Qwen3.5-MoE
// shared-expert geometry and the real adapter (qwen35SharedExpert), on the real matKernel
// the adapter uses (f32Kernel). It sweeps every activation the generic tail can take
// (SiLU, tanh-GELU, erf-GELU), the SharedIntermediateSize=0 width fallback, and inputs
// that expose signed-zero and large-magnitude behavior, comparing math.Float32bits, not a
// tolerance.
func TestQwen35SharedExpertFFNGatedParity(t *testing.T) {
	type actCase struct {
		name   string
		mutate func(*Config)
	}
	activations := []actCase{
		{name: "silu", mutate: nil},
		{name: "gelu-tanh", mutate: func(c *Config) { c.ActGeluTanh = true }},
		{name: "gelu-erf", mutate: func(c *Config) { c.ActGeluErf = true }},
	}
	widths := []struct {
		name string
		si   int
	}{
		{name: "shared-width", si: 2},
		{name: "width-fallback", si: 0}, // I==0 -> expertIntermediate() == MoEIntermediateSize
	}
	inputs := []struct {
		name string
		vals []float32
	}{
		{name: "ramp", vals: rampHidden(2, func(i int) float32 { return float32(i-1) * 0.75 })},
		{name: "sign-mixed", vals: []float32{float32(math.Sin(0.4)), float32(math.Cos(0.9))}},
		{name: "signed-zero", vals: []float32{float32(0), float32(math.Copysign(0, -1))}},
		{name: "large", vals: []float32{1234.5, -987.25}},
	}

	for _, act := range activations {
		for _, w := range widths {
			m := buildQwen35SharedExpertModel(t, w.si, act.mutate)
			mat := f32Kernel{m}
			if !m.has(qwen35SharedExpertName(0, "gate.weight")) {
				t.Fatalf("%s/%s: fixture lacks the singular shared-expert gate", act.name, w.name)
			}
			for _, in := range inputs {
				t.Run(act.name+"/"+w.name+"/"+in.name, func(t *testing.T) {
					got := qwen35SharedExpert(m, 0, append([]float32(nil), in.vals...), mat)
					want := legacyQwen35SharedExpert(m, 0, append([]float32(nil), in.vals...), mat)
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
	}
}
