package model

import (
	"encoding/binary"
	"math"
	"testing"
)

// addF32Bias appends a real f32 bias tensor to the model's raw buffer and registers it in
// the manifest, so addBiasIfPresent reads the same values the oracle applies (a manifest
// entry with no backing bytes would make addBias index an empty slice).
func addF32Bias(m *Model, name string, vals []float32) {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	off := len(m.raw)
	m.raw = append(m.raw, buf...)
	m.manifest[name] = tensorMeta{Dtype: "f32", Shape: []int{len(vals)}, Offset: off, Nbytes: len(buf)}
}

// expert_ffn_composition_test.go pins the migration of the generic routed-expert host
// tail of expertSwiGLU onto the shared ffn.Gated ordered activation+down projection
// (fak#13449). Before the migration the generic (non-residentInput) tail carried its own
// `act(g[i])*u[i]` loop followed by one down projection; it now delegates that ordered
// activation sequence and the single projection to the ffn package, exactly as
// denseSwiGLU (fak#13435) and qwen35SharedExpert (fak#13448) already do. The production
// caller keeps ownership of the gate/up projection kernel, the HAL/split/Metal/residency
// branches above, the optional bias adds and their order, and the residentInput arm.
//
// The migration must be numerically invisible: bit-for-bit identical output for the real
// adapter (expertSwiGLU) over the real routed-expert geometry, across every activation the
// host tail can take and with the expert bias tensors present and absent.

// legacyExpertSwiGLU is the PRE-#13449 host tail with its own ordered act(g)*u loop. It
// is the independent original-arithmetic oracle: the migrated production expertSwiGLU is
// compared against it with math.Float32bits over the same model, weights, and inputs. It
// is deliberately kept outside the migrated seam.
func legacyExpertSwiGLU(m *Model, layer, expert int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.expertIntermediate()
	gn := expertName(layer, expert, "gate_proj.weight")
	un := expertName(layer, expert, "up_proj.weight")
	dn := expertName(layer, expert, "down_proj.weight")
	gu := mulGroup(mat, []string{gn, un}, xn, []int{I, I}, H)
	g, u := gu[0], gu[1]
	m.addBiasIfPresent(g, expertName(layer, expert, "gate_proj.bias"))
	m.addBiasIfPresent(u, expertName(layer, expert, "up_proj.bias"))
	for i := 0; i < I; i++ {
		g[i] = act(g[i], cfg) * u[i]
	}
	out := mat.mul(dn, mat.prep(g), H, I)
	m.addBiasIfPresent(out, expertName(layer, expert, "down_proj.bias"))
	return out
}

// buildRoutedExpertModel assembles a one-layer MoE model whose single routed expert is the
// code under test. Hidden and intermediate widths are deliberately non-square so a
// transposed projection cannot pass vacuously. With biased=true the expert carries
// gate/up/down bias tensors so the bias-order contract is exercised too.
func buildRoutedExpertModel(t *testing.T, I int, mutate func(*Config), biased bool) *Model {
	t.Helper()
	const H = 4
	cfg := Config{
		HiddenSize:          H,
		IntermediateSize:    I,
		MoEIntermediateSize: I,
		NumLayers:           1,
		NumHeads:            1,
		NumKVHeads:          1,
		HeadDim:             1,
		VocabSize:           4,
		RMSNormEps:          1e-5,
		RopeTheta:           10000,
		NumExperts:          1,
		NumExpertsPerTok:    1,
		NormTopKProb:        true,
		EOSTokenID:          -1,
	}
	if mutate != nil {
		mutate(&cfg)
	}
	m := NewSyntheticMoE(cfg)
	if biased {
		// Deliberately non-symmetric bias values so a wrong add order or a dropped add is
		// visible; the oracle applies the identical tensors.
		addF32Bias(m, expertName(0, 0, "gate_proj.bias"), []float32{0.5, -0.25, 0.125, -0.75, 1.5, -1.25})
		addF32Bias(m, expertName(0, 0, "up_proj.bias"), []float32{-0.5, 0.25, -0.125, 0.75, -1.5, 1.25})
		addF32Bias(m, expertName(0, 0, "down_proj.bias"), []float32{0.3, -0.7, 0.9, -0.1})
	}
	return m
}

// TestExpertSwiGLUHostFallbackFFNGatedParity proves the migrated generic host tail of
// expertSwiGLU is bit-identical to the pre-refactor loop over the real routed-expert
// geometry and the real adapter (expertSwiGLU), on the real host kernels it is called
// with (residentKernel and f32Kernel). It sweeps every activation the host tail can take
// (SiLU, tanh-GELU, erf-GELU), expert bias presence, and inputs that expose signed-zero
// and large-magnitude behavior, comparing math.Float32bits, not a tolerance.
func TestExpertSwiGLUHostFallbackFFNGatedParity(t *testing.T) {
	type actCase struct {
		name   string
		mutate func(*Config)
	}
	activations := []actCase{
		{name: "silu", mutate: nil},
		{name: "gelu-tanh", mutate: func(c *Config) { c.ActGeluTanh = true }},
		{name: "gelu-erf", mutate: func(c *Config) { c.ActGeluErf = true }},
	}
	inputs := []struct {
		name string
		vals []float32
	}{
		{name: "ramp", vals: rampHidden(4, func(i int) float32 { return float32(i-1) * 0.75 })},
		{name: "sign-mixed", vals: []float32{float32(math.Sin(0.4)), float32(math.Cos(0.9)), -1.5, 2.25}},
		{name: "signed-zero", vals: []float32{float32(0), float32(math.Copysign(0, -1)), 1, -1}},
		{name: "large", vals: []float32{1234.5, -987.25, 6.5, -0.125}},
	}

	for _, act := range activations {
		for _, biased := range []bool{false, true} {
			m := buildRoutedExpertModel(t, 6, act.mutate, biased)
			for kernelName, mat := range map[string]matKernel{
				"resident": residentKernel{m},
				"f32":      f32Kernel{m},
			} {
				for _, in := range inputs {
					name := act.name + "/biased=" + boolWord(biased) + "/" + kernelName + "/" + in.name
					t.Run(name, func(t *testing.T) {
						got := expertSwiGLU(m, 0, 0, append([]float32(nil), in.vals...), mat)
						want := legacyExpertSwiGLU(m, 0, 0, append([]float32(nil), in.vals...), mat)
						if len(got) != len(want) {
							t.Fatalf("length=%d, want %d", len(got), len(want))
						}
						for i := range want {
							if gb, wb := math.Float32bits(got[i]), math.Float32bits(want[i]); gb != wb {
								t.Fatalf("expert output[%d] = %.9g (0x%08x), want %.9g (0x%08x)",
									i, got[i], gb, want[i], wb)
							}
						}
					})
				}
			}
		}
	}
}

func boolWord(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}
