package model

import (
	"math"
	"testing"
)

// Pin the configured V4.1 expert equation through both projection adapters.
// Rectangular matrices expose every clamp branch without a square-only shortcut.
// fak-test:runtime fast est=10ms lane=default
func TestV41SwigluLimitRoutedAndStreaming(t *testing.T) {
	t.Parallel()
	const H, I = 3, 5
	x := []float32{1, -0.5, 0.25}
	w1 := []float32{
		-5, 0, 0,
		5, 0, 0,
		1, 0, 0,
		-1, 0, 0,
		0.5, 0, 0,
	}
	w3 := []float32{
		5, 0, 0,
		-5, 0, 0,
		4, 0, 0,
		-4, 0, 0,
		0.75, 0, 0,
	}
	w2 := []float32{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0.5, 0.25,
	}
	for _, tc := range []struct {
		name  string
		limit float64
	}{
		{name: "zero preserves plain SwiGLU"},
		{name: "upper gate and symmetric up clamp", limit: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{HiddenSize: H, MoEIntermediateSize: I, SwigluLimit: tc.limit}
			want := v41SwigluLimitReference(w1, w3, w2, x, I, H, float32(tc.limit))
			m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
				{Name: layerName(0, "ffn.shared_experts.w1.weight"), Shape: []int{I, H}, Data: w1},
				{Name: layerName(0, "ffn.shared_experts.w3.weight"), Shape: []int{I, H}, Data: w3},
				{Name: layerName(0, "ffn.shared_experts.w2.weight"), Shape: []int{H, I}, Data: w2},
			})
			if err != nil {
				t.Fatalf("shared f32 store fixture: %v", err)
			}
			for _, adapter := range []struct {
				name string
				run  func() ([]float32, error)
			}{
				{name: "routed", run: func() ([]float32, error) { return v41SwiGLU(w1, w3, w2, x, I, H, cfg), nil }},
				{name: "streaming shared", run: func() ([]float32, error) { return m.v41SharedExpertSwiGLU(0, x, cfg) }},
			} {
				t.Run(adapter.name, func(t *testing.T) {
					got, err := adapter.run()
					if err != nil {
						t.Fatalf("expert projection: %v", err)
					}
					if len(got) != H {
						t.Fatalf("output width=%d, want %d", len(got), H)
					}
					for i := range want {
						if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > 2e-6*math.Max(1, math.Abs(float64(want[i]))) {
							t.Errorf("limit=%g output[%d]=%g, want independent f32 oracle %g", tc.limit, i, got[i], want[i])
						}
					}
					if tc.limit == 0 {
						plain := v41SwiGLU(w1, w3, w2, x, I, H, Config{})
						assertFloat32BitsEqual(t, "zero limit preserves plain activation", plain, got)
					}
				})
			}
		})
	}
	// The negative gate is -5 while the limit is 2. Its output remains the
	// negative-gate SiLU result; a lower gate clamp changes this isolated channel.
	t.Log("SW-VERIFIED routed and f32-store shared projections; no physical GPU qualification")
}

// This oracle does not call matRows, act, silu, or the shared FFN component.
func v41SwigluLimitReference(w1, w3, w2, x []float32, intermediate, hidden int, limit float32) []float32 {
	project := func(weights, input []float32, rows, cols int) []float32 {
		out := make([]float32, rows)
		for row := range out {
			for col, value := range input[:cols] {
				out[row] += weights[row*cols+col] * value
			}
		}
		return out
	}
	gate := project(w1, x, intermediate, hidden)
	up := project(w3, x, intermediate, hidden)
	for i, g := range gate {
		u := up[i]
		if limit > 0 {
			if g > limit {
				g = limit
			}
			if u < -limit {
				u = -limit
			}
			if u > limit {
				u = limit
			}
		}
		gate[i] = (g / (1 + float32(math.Exp(float64(-g))))) * u
	}
	return project(w2, gate, hidden, intermediate)
}
