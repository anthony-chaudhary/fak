package model

import (
	"math"
	"testing"
)

func TestFFNGatedAdaptersBitExact(t *testing.T) {
	t.Run("v41 routed expert", func(t *testing.T) {
		const H, I = 3, 5
		cfg := Config{ActGeluTanh: true}
		xn := []float32{0.75, -0.5, 0.25}
		w1 := ffnTestValues(I*H, 0.07)
		w3 := ffnTestValues(I*H, -0.05)
		w2 := ffnTestValues(H*I, 0.03)
		gate := matRows(w1, xn, I, H)
		up := matRows(w3, xn, I, H)
		for i := range gate {
			gate[i] = act(gate[i], cfg) * up[i]
		}
		want := matRows(w2, gate, H, I)
		got := v41SwiGLU(w1, w3, w2, xn, I, H, cfg)
		assertFloat32BitsEqual(t, "v41 routed expert", want, got)
		legacyAllocs := testing.AllocsPerRun(100, func() {
			ffnAdapterTestSink = legacyV41SwiGLU(w1, w3, w2, xn, I, H, cfg)
		})
		adapterAllocs := testing.AllocsPerRun(100, func() {
			ffnAdapterTestSink = v41SwiGLU(w1, w3, w2, xn, I, H, cfg)
		})
		t.Logf("v41 routed allocations: adapter=%v legacy=%v", adapterAllocs, legacyAllocs)
		if adapterAllocs > legacyAllocs {
			t.Fatalf("v41 routed allocations adapter=%v legacy=%v, want no regression", adapterAllocs, legacyAllocs)
		}
	})

	t.Run("v41 shared expert", func(t *testing.T) {
		m := v41DecodeStateModel(t, 1)
		cfg := m.Cfg
		xn := ffnTestValues(cfg.HiddenSize, 0.015625)
		h1, err := m.v41ProjMatRows(0, "ffn.shared_experts.w1.weight", xn, cfg.MoEIntermediateSize, cfg.HiddenSize)
		if err != nil {
			t.Fatalf("reference gate projection: %v", err)
		}
		h3, err := m.v41ProjMatRows(0, "ffn.shared_experts.w3.weight", xn, cfg.MoEIntermediateSize, cfg.HiddenSize)
		if err != nil {
			t.Fatalf("reference up projection: %v", err)
		}
		for i := range h1 {
			h1[i] = act(h1[i], cfg) * h3[i]
		}
		want, err := m.v41ProjMatRows(0, "ffn.shared_experts.w2.weight", h1, cfg.HiddenSize, cfg.MoEIntermediateSize)
		if err != nil {
			t.Fatalf("reference down projection: %v", err)
		}
		got, err := m.v41SharedExpertSwiGLU(0, xn, cfg)
		if err != nil {
			t.Fatalf("shared expert adapter: %v", err)
		}
		assertFloat32BitsEqual(t, "v41 shared expert", want, got)
		legacyAllocs := testing.AllocsPerRun(100, func() {
			g, callErr := m.v41ProjMatRows(0, "ffn.shared_experts.w1.weight", xn, cfg.MoEIntermediateSize, cfg.HiddenSize)
			if callErr != nil {
				panic(callErr)
			}
			u, callErr := m.v41ProjMatRows(0, "ffn.shared_experts.w3.weight", xn, cfg.MoEIntermediateSize, cfg.HiddenSize)
			if callErr != nil {
				panic(callErr)
			}
			h := make([]float32, len(g))
			for i := range h {
				h[i] = act(g[i], cfg) * u[i]
			}
			ffnAdapterTestSink, callErr = m.v41ProjMatRows(0, "ffn.shared_experts.w2.weight", h, cfg.HiddenSize, cfg.MoEIntermediateSize)
			if callErr != nil {
				panic(callErr)
			}
		})
		adapterAllocs := testing.AllocsPerRun(100, func() {
			var callErr error
			ffnAdapterTestSink, callErr = m.v41SharedExpertSwiGLU(0, xn, cfg)
			if callErr != nil {
				panic(callErr)
			}
		})
		t.Logf("v41 shared allocations: adapter=%v legacy=%v", adapterAllocs, legacyAllocs)
		if adapterAllocs > legacyAllocs {
			t.Fatalf("v41 shared allocations adapter=%v legacy=%v, want no regression", adapterAllocs, legacyAllocs)
		}
	})

	t.Run("dense non-fused with biases", func(t *testing.T) {
		cfg := Config{HiddenSize: 2, IntermediateSize: 3, ActGeluErf: true}
		p := func(s string) string { return layerName(0, s) }
		m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
			{Name: p("mlp.gate_proj.weight"), Shape: []int{3, 2}, Data: ffnTestValues(6, 0.11)},
			{Name: p("mlp.gate_proj.bias"), Shape: []int{3}, Data: []float32{0.1, -0.2, 0.3}},
			{Name: p("mlp.up_proj.weight"), Shape: []int{3, 2}, Data: ffnTestValues(6, -0.09)},
			{Name: p("mlp.up_proj.bias"), Shape: []int{3}, Data: []float32{-0.05, 0.15, -0.25}},
			{Name: p("mlp.down_proj.weight"), Shape: []int{2, 3}, Data: ffnTestValues(6, 0.13)},
			{Name: p("mlp.down_proj.bias"), Shape: []int{2}, Data: []float32{0.01, -0.02}},
		})
		if err != nil {
			t.Fatalf("NewFromF32Tensors: %v", err)
		}
		xn := []float32{0.75, -0.5}
		gate := matRows(m.tensor(p("mlp.gate_proj.weight")), xn, cfg.IntermediateSize, cfg.HiddenSize)
		up := matRows(m.tensor(p("mlp.up_proj.weight")), xn, cfg.IntermediateSize, cfg.HiddenSize)
		m.addBiasIfPresent(gate, p("mlp.gate_proj.bias"))
		m.addBiasIfPresent(up, p("mlp.up_proj.bias"))
		for i := range gate {
			gate[i] = act(gate[i], cfg) * up[i]
		}
		want := matRows(m.tensor(p("mlp.down_proj.weight")), gate, cfg.HiddenSize, cfg.IntermediateSize)
		m.addBiasIfPresent(want, p("mlp.down_proj.bias"))
		got := denseSwiGLU{}.apply(m, 0, xn, f32Kernel{m})
		assertFloat32BitsEqual(t, "dense adapter", want, got)
		legacyAllocs := testing.AllocsPerRun(100, func() {
			ffnAdapterTestSink = legacyDenseSwiGLUApply(m, 0, xn, f32Kernel{m})
		})
		adapterAllocs := testing.AllocsPerRun(100, func() {
			ffnAdapterTestSink = denseSwiGLU{}.apply(m, 0, xn, f32Kernel{m})
		})
		t.Logf("dense allocations: adapter=%v legacy=%v", adapterAllocs, legacyAllocs)
		if adapterAllocs > legacyAllocs {
			t.Fatalf("dense allocations adapter=%v legacy=%v, want no regression", adapterAllocs, legacyAllocs)
		}
	})
}

var ffnAdapterTestSink []float32

func legacyV41SwiGLU(w1, w3, w2, xn []float32, I, H int, cfg Config) []float32 {
	h1 := matRows(w1, xn, I, H)
	h3 := matRows(w3, xn, I, H)
	h := make([]float32, I)
	for i := 0; i < I; i++ {
		h[i] = act(h1[i], cfg) * h3[i]
	}
	return matRows(w2, h, H, I)
}

func legacyDenseSwiGLUApply(m *Model, layer int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	p := func(s string) string { return layerName(layer, s) }
	if !cfg.ActGeluTanh && !cfg.ActGeluErf &&
		!m.has(p("mlp.gate_proj.bias")) && !m.has(p("mlp.up_proj.bias")) && !m.has(p("mlp.down_proj.bias")) {
		if sk, ok := mat.(sessionQ4KKernel); ok {
			if xf, ok2 := xn.([]float32); ok2 {
				if out := sk.s.q4kFusedMLP(p("mlp.gate_proj.weight"), p("mlp.up_proj.weight"), p("mlp.down_proj.weight"), xf); out != nil {
					return out
				}
			}
		}
	}
	gu := mulGroup(mat, []string{p("mlp.gate_proj.weight"), p("mlp.up_proj.weight")}, xn, []int{I, I}, H)
	g, u := gu[0], gu[1]
	m.addBiasIfPresent(g, p("mlp.gate_proj.bias"))
	m.addBiasIfPresent(u, p("mlp.up_proj.bias"))
	for i := 0; i < I; i++ {
		g[i] = act(g[i], cfg) * u[i]
	}
	out := mat.mul(p("mlp.down_proj.weight"), mat.prep(g), H, I)
	m.addBiasIfPresent(out, p("mlp.down_proj.bias"))
	return out
}

func BenchmarkFFNGatedAdapters(b *testing.B) {
	b.Run("V41Routed", func(b *testing.B) {
		const H, I = 64, 128
		xn := ffnTestValues(H, 0.015625)
		w1 := ffnTestValues(I*H, 0.007)
		w3 := ffnTestValues(I*H, -0.005)
		w2 := ffnTestValues(H*I, 0.003)
		for _, activation := range []struct {
			name string
			cfg  Config
		}{
			{name: "SiLU", cfg: Config{}},
			{name: "GELUTanh", cfg: Config{ActGeluTanh: true}},
		} {
			b.Run(activation.name, func(b *testing.B) {
				for _, bench := range []struct {
					name string
					run  func() []float32
				}{
					{name: "Adapter", run: func() []float32 { return v41SwiGLU(w1, w3, w2, xn, I, H, activation.cfg) }},
					{name: "Legacy", run: func() []float32 { return legacyV41SwiGLU(w1, w3, w2, xn, I, H, activation.cfg) }},
				} {
					b.Run(bench.name, func(b *testing.B) {
						b.ReportAllocs()
						for i := 0; i < b.N; i++ {
							ffnAdapterTestSink = bench.run()
						}
					})
				}
			})
		}
	})

	b.Run("DenseNonFused", func(b *testing.B) {
		for _, activation := range []struct {
			name string
			gelu bool
		}{
			{name: "SiLU"},
			{name: "GELUErf", gelu: true},
		} {
			b.Run(activation.name, func(b *testing.B) {
				cfg := denseCfgForMoETest()
				cfg.ActGeluErf = activation.gelu
				m := NewSynthetic(cfg)
				xn := ffnTestValues(cfg.HiddenSize, 0.015625)
				kernel := f32Kernel{m}
				for _, bench := range []struct {
					name string
					run  func() []float32
				}{
					{name: "Adapter", run: func() []float32 { return denseSwiGLU{}.apply(m, 0, xn, kernel) }},
					{name: "Legacy", run: func() []float32 { return legacyDenseSwiGLUApply(m, 0, xn, kernel) }},
				} {
					b.Run(bench.name, func(b *testing.B) {
						b.ReportAllocs()
						for i := 0; i < b.N; i++ {
							ffnAdapterTestSink = bench.run()
						}
					})
				}
			})
		}
	})
}

func ffnTestValues(n int, scale float32) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = float32((i%7)-3) * scale
		if math.Float32bits(values[i]) == 0 {
			values[i] = scale / 2
		}
	}
	return values
}
