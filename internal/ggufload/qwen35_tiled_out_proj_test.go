package ggufload

import (
	"bytes"
	"math"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Witnesses for the tiled native GDN out_proj route: ssm_out keeps its GGUF bytes and tiled
// input-column order, and the forward tiles the core, instead of requantizing to Q8_0.

// fak-test:runtime fast est=20ms
func TestQwen35TiledOutProjColumnContractMatchesNormalizer(t *testing.T) {
	// The loader's float normalizer is the column authority; the activation tiling must be its
	// exact inverse: canonical column h*d+i is GGUF column c*d+i exactly when src[c] == h.
	for _, g := range []struct{ nK, nV, d int }{{2, 4, 64}, {16, 48, 128}, {4, 12, 8}} {
		cfg := nativeRowsTestCfg(true)
		cfg.HiddenSize = 1
		cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads = g.nK, g.nV
		cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim = g.d, g.d
		valDim := g.nV * g.d
		probe := make([]float32, valDim)
		for j := range probe {
			probe[j] = float32(j)
		}
		canonical, err := normalizeCanonicalTensorData("model.layers.0.linear_attn.out_proj.weight", probe, cfg)
		if err != nil {
			t.Fatal(err)
		}
		src, err := compute.Qwen35TiledValueHeadSource(g.nK, g.nV)
		if err != nil {
			t.Fatal(err)
		}
		for c, h := range src {
			for i := 0; i < g.d; i++ {
				if got, want := canonical[h*g.d+i], float32(c*g.d+i); got != want {
					t.Fatalf("nK=%d nV=%d: canonical column %d holds GGUF column %v, tiling expects %v", g.nK, g.nV, h*g.d+i, got, want)
				}
			}
		}
	}
}

// fak-test:runtime fast est=20ms
func TestQwen35TiledOutProjAdmission(t *testing.T) {
	cfg := nativeRowsTestCfg(true)
	out := "model.layers.0.linear_attn.out_proj.weight"
	shape := []int{256, 256}
	iq := q4kLoadOptions{residentDenseIQ: map[TensorType]bool{TensorIQ3_S: true}, residentDenseQ2K: true}
	cases := []struct {
		name  string
		canon string
		typ   TensorType
		shape []int
		opts  q4kLoadOptions
		want  bool
	}{
		{"retained i-quant", out, TensorIQ3_S, shape, iq, true},
		{"Q4_K", out, TensorQ4_K, shape, iq, true},
		{"Q2_K retained", out, TensorQ2_K, shape, iq, true},
		{"i-quant the backend does not retain", out, TensorIQ2_S, shape, iq, false},
		{"Q3_K not retained stays Q8", out, TensorQ3_K, shape, iq, false},
		{"no raw i-quant backend", out, TensorQ4_K, shape, q4kLoadOptions{residentDenseQ2K: true}, false},
		{"option off", out, TensorQ4_K, shape, q4kLoadOptions{residentDenseIQ: iq.residentDenseIQ, nativeSSMOutOff: true}, false},
		{"prism grouped columns", out, TensorQ4_K, shape, q4kLoadOptions{residentDenseIQ: iq.residentDenseIQ, prismGDNVGrouped: true}, false},
		{"wrong reduction width", out, TensorQ4_K, []int{256, 512}, iq, false},
		{"in_proj_z is a row transform", "model.layers.0.linear_attn.in_proj_z.weight", TensorIQ3_S, shape, iq, false},
		{"mtp out_proj", "mtp.layers.0.linear_attn.out_proj.weight", TensorQ4_K, shape, iq, false},
	}
	for _, tc := range cases {
		if got := qwen35TiledOutProjResident(cfg, tc.canon, tc.typ, tc.shape, tc.opts); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
	if qwen35NativeRowResident(cfg, out, TensorQ4_K, shape, iq) {
		t.Error("out_proj leaked into the row-permutation route")
	}
	t.Setenv("FAK_QWEN35_NATIVE_SSM_OUT", "0")
	if qwen35TiledOutProjResident(cfg, out, TensorQ4_K, shape, iq) {
		t.Error("FAK_QWEN35_NATIVE_SSM_OUT=0 did not restore the Q8 route")
	}
}

// fak-test:runtime fast est=2s
func TestQwen35TiledOutProjLoaderEstimatorAndForward(t *testing.T) {
	load := func(opts ...Q4KLoadOption) (*model.Model, int64) {
		ws := qwen35NativeRowsFixture(t)
		plan, err := ws.EstimateQ4KLoadMemoryPlan(opts...)
		if err != nil {
			t.Fatalf("estimate: %v", err)
		}
		m, err := ws.QuantModelQ4KProfileOptions(nil, opts...)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		t.Cleanup(func() { _ = m.CloseWeights() })
		if got := m.ResidentReport().TotalResidentBytes; got != plan.Total() {
			t.Fatalf("estimate %d != loaded resident %d", plan.Total(), got)
		}
		return m, plan.Total()
	}
	iq := WithDenseIQResident(TensorIQ3_S)
	tiled, tiledBytes := load(iq, WithQwen35NativeSSMOut(true))
	q8, q8Bytes := load(iq, WithQwen35NativeSSMOut(false))
	if tiledBytes >= q8Bytes {
		t.Fatalf("tiled native out_proj resident %d not below the Q8 route %d", tiledBytes, q8Bytes)
	}

	ws := qwen35NativeRowsFixture(t)
	for l := 0; l < 3; l++ {
		name := "model.layers." + strconv.Itoa(l) + ".linear_attn.out_proj.weight"
		if !tiled.HasQ4K(name) || !tiled.Qwen35OutProjTiled(name) {
			t.Fatalf("%s: tiled route store q4k=%v tiled=%v", name, tiled.HasQ4K(name), tiled.Qwen35OutProjTiled(name))
		}
		if !q8.HasQ8(name) || q8.Qwen35OutProjTiled(name) {
			t.Fatalf("%s: Q8 route store q8=%v tiled=%v", name, q8.HasQ8(name), q8.Qwen35OutProjTiled(name))
		}
		raw, _, err := ws.TensorBytes("blk." + strconv.Itoa(l) + ".ssm_out.weight")
		if err != nil {
			t.Fatal(err)
		}
		stored, ok := tiled.Q4KRaw(name)
		if !ok || !bytes.Equal(stored[:len(raw)], raw) {
			t.Fatalf("%s: stored bytes are not the GGUF ssm_out bytes", name)
		}
	}
	if !tiled.HasQwen35TiledOutProj() || q8.HasQwen35TiledOutProj() {
		t.Fatal("model-level tiled flag disagrees with the route")
	}

	// Forward parity: native tiled out_proj vs the same weights deinterleaved into Q8.
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 1, 2, 4, 6, 8, 10, 12}
	follow := []int{9, 13, 14, 15}
	run := func(m *model.Model) [][]float32 {
		s := m.NewSession()
		s.Q4K = true
		out := [][]float32{append([]float32(nil), s.Prefill(prompt)...)}
		for _, id := range follow {
			out = append(out, append([]float32(nil), s.Step(id)...))
		}
		return out
	}
	got, want := run(tiled), run(q8)
	for i := range want {
		var dot, ng, nw float64
		for j := range want[i] {
			g, w := float64(got[i][j]), float64(want[i][j])
			if math.IsNaN(g) || math.IsInf(g, 0) {
				t.Fatalf("step %d tiled logit %d not finite: %v", i, j, g)
			}
			dot, ng, nw = dot+g*w, ng+g*g, nw+w*w
		}
		cos := dot / math.Sqrt(ng*nw)
		t.Logf("step %d tiled-vs-Q8 logits cosine %.7f argmax %d/%d", i, cos, argmaxF32(got[i]), argmaxF32(want[i]))
		if cos < 0.9999 {
			t.Errorf("step %d tiled-vs-Q8 logits cosine %.6f < 0.9999", i, cos)
		}
		if a, b := argmaxF32(got[i]), argmaxF32(want[i]); a != b {
			t.Errorf("step %d argmax tiled=%d Q8=%d", i, a, b)
		}
	}
}
