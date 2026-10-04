package model

import (
	"math"
	"testing"
)

// TestGatedMLPFusionEligibility pins the centralized gated-FFN fusion admission that
// denseSwiGLU, expertSwiGLU (HAL and Metal) and qwen35SharedExpert now share. The table
// walks the default SiLU path, each/both GELU flags, and each bias-presence position, and
// cross-checks the shared decision against an independent inline reference so a regression
// in the extracted predicate cannot pass by matching itself.
//
// Comparing the decision, not the arithmetic, is deliberate: the fused on-GPU fast path is
// inert on a non-Metal build (q4kFusedMLP declines), so the only platform-independent
// contract is the admission decision the four call sites now delegate to. The consumer
// subtests below then exercise a real adapter on both sides of that decision.
func TestGatedMLPFusionEligibility(t *testing.T) {
	const I = 4
	p := func(suffix string) string { return layerName(0, suffix) }
	gateBias, upBias, downBias := p("mlp.gate_proj.bias"), p("mlp.up_proj.bias"), p("mlp.down_proj.bias")

	// reference is the pre-extraction inline decision, spelled out independently.
	reference := func(cfg Config, biases map[string]bool) bool {
		if cfg.ActGeluTanh || cfg.ActGeluErf {
			return false
		}
		return !biases[gateBias] && !biases[upBias] && !biases[downBias]
	}

	cases := []struct {
		name   string
		cfg    Config
		biases []string
	}{
		{name: "default_silu_bias_free", cfg: Config{}},
		{name: "gelu_tanh_refuses", cfg: Config{ActGeluTanh: true}},
		{name: "gelu_erf_refuses", cfg: Config{ActGeluErf: true}},
		{name: "both_gelu_flags_refuse", cfg: Config{ActGeluTanh: true, ActGeluErf: true}},
		{name: "gate_bias_refuses", cfg: Config{}, biases: []string{gateBias}},
		{name: "up_bias_refuses", cfg: Config{}, biases: []string{upBias}},
		{name: "down_bias_refuses", cfg: Config{}, biases: []string{downBias}},
		{name: "all_biases_refuse", cfg: Config{}, biases: []string{gateBias, upBias, downBias}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &Model{Cfg: tc.cfg, manifest: map[string]tensorMeta{}}
			biases := map[string]bool{}
			for _, b := range tc.biases {
				m.manifest[b] = tensorMeta{Shape: []int{I}}
				biases[b] = true
			}
			want := reference(tc.cfg, biases)
			got := siluGatedMLP(tc.cfg) && m.biasFreeGatedMLP(gateBias, upBias, downBias)
			if got != want {
				t.Fatalf("shared admission = %v, want %v (cfg=%+v biases=%v)", got, want, tc.cfg, tc.biases)
			}
		})
	}
}

// TestGatedMLPFusionEligibilityConsumers proves the extracted decision is the one the real
// adapters consult: an eligible (default-SiLU, bias-free) projection triple and a refusing
// (GELU or biased) one both run their adapter to a correct, finite result, and the refusing
// configuration reproduces the non-fused arithmetic bit-for-bit. The dense, routed-expert
// and Qwen shared-expert consumers are each exercised through the shared predicate.
func TestGatedMLPFusionEligibilityConsumers(t *testing.T) {
	const H, I = 3, 5
	xn := []float32{0.75, -0.5, 0.25}

	t.Run("dense", func(t *testing.T) {
		for _, pol := range []struct {
			name  string
			cfg   Config
			bias  bool
			admit bool
		}{
			{name: "silu_bias_free_admits", cfg: Config{}, admit: true},
			{name: "gelu_refuses", cfg: Config{ActGeluTanh: true}},
			{name: "biased_refuses", cfg: Config{}, bias: true},
		} {
			t.Run(pol.name, func(t *testing.T) {
				cfg := pol.cfg
				cfg.HiddenSize, cfg.IntermediateSize = H, I
				p := func(s string) string { return layerName(0, s) }
				tensors := []NamedTensorF32{
					{Name: p("mlp.gate_proj.weight"), Shape: []int{I, H}, Data: ffnTestValues(I*H, 0.11)},
					{Name: p("mlp.up_proj.weight"), Shape: []int{I, H}, Data: ffnTestValues(I*H, -0.09)},
					{Name: p("mlp.down_proj.weight"), Shape: []int{H, I}, Data: ffnTestValues(H*I, 0.13)},
				}
				if pol.bias {
					tensors = append(tensors, NamedTensorF32{Name: p("mlp.gate_proj.bias"), Shape: []int{I}, Data: []float32{0.1, -0.2, 0.3, 0.05, -0.15}})
				}
				m, err := NewFromF32Tensors(cfg, tensors)
				if err != nil {
					t.Fatalf("NewFromF32Tensors: %v", err)
				}
				admit := siluGatedMLP(cfg) && m.biasFreeGatedMLP(p("mlp.gate_proj.bias"), p("mlp.up_proj.bias"), p("mlp.down_proj.bias"))
				if admit != pol.admit {
					t.Fatalf("admission = %v, want %v", admit, pol.admit)
				}

				// The consumer must produce the exact non-fused arithmetic in every case:
				// with the predicate refusing it is the only path, and with it admitting the
				// non-Metal fused fast path declines and falls through to the same arithmetic.
				g := matRows(m.tensor(p("mlp.gate_proj.weight")), xn, I, H)
				u := matRows(m.tensor(p("mlp.up_proj.weight")), xn, I, H)
				if pol.bias {
					m.addBiasIfPresent(g, p("mlp.gate_proj.bias"))
				}
				for i := range g {
					g[i] = act(g[i], cfg) * u[i]
				}
				want := matRows(m.tensor(p("mlp.down_proj.weight")), g, H, I)
				got := denseSwiGLU{}.apply(m, 0, xn, f32Kernel{m})
				assertFloat32BitsEqual(t, "dense adapter "+pol.name, want, got)
				assertFinite(t, got)
			})
		}
	})

	t.Run("routed_expert", func(t *testing.T) {
		base := Config{HiddenSize: H, IntermediateSize: I, MoEIntermediateSize: I}
		gn := expertName(0, 0, "gate_proj.weight")
		un := expertName(0, 0, "up_proj.weight")
		dn := expertName(0, 0, "down_proj.weight")
		for _, pol := range []struct {
			name  string
			cfg   Config
			bias  bool
			admit bool
		}{
			{name: "silu_bias_free_admits", cfg: base, admit: true},
			{name: "gelu_refuses", cfg: func() Config { c := base; c.ActGeluErf = true; return c }()},
			{name: "biased_refuses", cfg: base, bias: true},
		} {
			t.Run(pol.name, func(t *testing.T) {
				tensors := []NamedTensorF32{
					{Name: gn, Shape: []int{I, H}, Data: ffnTestValues(I*H, 0.11)},
					{Name: un, Shape: []int{I, H}, Data: ffnTestValues(I*H, -0.09)},
					{Name: dn, Shape: []int{H, I}, Data: ffnTestValues(H*I, 0.13)},
				}
				if pol.bias {
					tensors = append(tensors, NamedTensorF32{Name: expertName(0, 0, "up_proj.bias"), Shape: []int{I}, Data: []float32{0.05, -0.1, 0.15, 0.2, -0.25}})
				}
				m, err := NewFromF32Tensors(pol.cfg, tensors)
				if err != nil {
					t.Fatalf("NewFromF32Tensors: %v", err)
				}
				admit := siluGatedMLP(pol.cfg) && m.biasFreeGatedMLP(
					expertName(0, 0, "gate_proj.bias"), expertName(0, 0, "up_proj.bias"), expertName(0, 0, "down_proj.bias"))
				if admit != pol.admit {
					t.Fatalf("admission = %v, want %v", admit, pol.admit)
				}
				g := matRows(m.tensor(gn), xn, I, H)
				u := matRows(m.tensor(un), xn, I, H)
				if pol.bias {
					m.addBiasIfPresent(u, expertName(0, 0, "up_proj.bias"))
				}
				for i := range g {
					g[i] = act(g[i], pol.cfg) * u[i]
				}
				want := matRows(m.tensor(dn), g, H, I)
				got := expertSwiGLU(m, 0, 0, xn, residentKernel{m})
				assertFloat32BitsEqual(t, "expert adapter "+pol.name, want, got)
				assertFinite(t, got)
			})
		}
	})

	t.Run("qwen_shared_expert", func(t *testing.T) {
		for _, pol := range []struct {
			name  string
			cfg   Config
			bias  bool
			admit bool
		}{
			{name: "silu_bias_free_admits", cfg: Config{}, admit: true},
			{name: "gelu_refuses", cfg: Config{ActGeluTanh: true}},
			{name: "biased_refuses", cfg: Config{}, bias: true},
		} {
			t.Run(pol.name, func(t *testing.T) {
				cfg := pol.cfg
				cfg.HiddenSize, cfg.IntermediateSize, cfg.SharedIntermediateSize = H, I, I
				gn := qwen35SharedExpertName(0, "gate_proj.weight")
				un := qwen35SharedExpertName(0, "up_proj.weight")
				dn := qwen35SharedExpertName(0, "down_proj.weight")
				tensors := []NamedTensorF32{
					{Name: gn, Shape: []int{I, H}, Data: ffnTestValues(I*H, 0.11)},
					{Name: un, Shape: []int{I, H}, Data: ffnTestValues(I*H, -0.09)},
					{Name: dn, Shape: []int{H, I}, Data: ffnTestValues(H*I, 0.13)},
					{Name: qwen35SharedExpertName(0, "gate.weight"), Shape: []int{1, H}, Data: []float32{0.25, -0.1, 0.4}},
				}
				if pol.bias {
					tensors = append(tensors, NamedTensorF32{Name: qwen35SharedExpertName(0, "down_proj.bias"), Shape: []int{H}, Data: []float32{0.01, -0.02, 0.03}})
				}
				m, err := NewFromF32Tensors(cfg, tensors)
				if err != nil {
					t.Fatalf("NewFromF32Tensors: %v", err)
				}
				admit := siluGatedMLP(cfg) && m.biasFreeGatedMLP(
					qwen35SharedExpertName(0, "gate_proj.bias"), qwen35SharedExpertName(0, "up_proj.bias"), qwen35SharedExpertName(0, "down_proj.bias"))
				if admit != pol.admit {
					t.Fatalf("admission = %v, want %v", admit, pol.admit)
				}
				g := matRows(m.tensor(gn), xn, I, H)
				u := matRows(m.tensor(un), xn, I, H)
				for i := range g {
					g[i] = act(g[i], cfg) * u[i]
				}
				want := matRows(m.tensor(dn), g, H, I)
				gate := sigmoid(matRows(m.tensor(qwen35SharedExpertName(0, "gate.weight")), xn, 1, H)[0])
				for i := range want {
					want[i] *= gate
				}
				got := qwen35SharedExpert(m, 0, xn, residentKernel{m})
				assertFloat32BitsEqual(t, "qwen shared adapter "+pol.name, want, got)
				assertFinite(t, got)
			})
		}
	})
}

// assertFinite fails when any element of v is NaN or Inf.
func assertFinite(t *testing.T, v []float32) {
	t.Helper()
	for i, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Fatalf("value[%d] = %v, want finite", i, x)
		}
	}
}
