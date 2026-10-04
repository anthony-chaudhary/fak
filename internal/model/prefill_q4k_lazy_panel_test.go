package model

import (
	"reflect"
	"testing"
)

// TestPrefillQ4KLazyQ8Panel pins the lazy activation-panel contract of prefillBatchedQ4K
// (#13694): the Q8_0 activation panel is read ONLY on the q8 branch of proj, so it must be
// built only when some projection consuming that activation actually takes the q8 branch,
// and at most once per activation. The eager version quantized four panels per layer (q/k/v
// input, o input, gate/up input, down input) even when every projection was Q4_K-resident —
// pure CPU waste that dominated the short-prompt TTFT floor on Metal.
//
// Bit-identity of the lazily built panel against the frozen eager oracle is held separately
// by TestPrefillQ4KNormReuseExactAndAllocation (Q/K on the Q8 path, exact hidden/logits/KV).
func TestPrefillQ4KLazyQ8Panel(t *testing.T) {
	// The CPU int8 q4kGemm quantizes its OWN private activation panel inside the GEMM; pin the
	// f32 q4kGemm (what the Metal q4_k GEMM consumes) so the hook counts only the prefill's
	// shared Q8 panels — the surface under test.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := Config{
		HiddenSize: 256, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		IntermediateSize: 512, VocabSize: 64, RMSNormEps: 1e-6, AttnSoftcap: 50.0, RopeTheta: 10000.0,
	}
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	all := []string{"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
		"self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
	outOf := map[string]int{
		"self_attn.q_proj.weight": cfg.NumHeads * cfg.HeadDim,
		"self_attn.k_proj.weight": cfg.NumKVHeads * cfg.HeadDim,
		"self_attn.v_proj.weight": cfg.NumKVHeads * cfg.HeadDim,
		"self_attn.o_proj.weight": H, "mlp.gate_proj.weight": I, "mlp.up_proj.weight": I,
		"mlp.down_proj.weight": H,
	}
	cases := []struct {
		name string
		// q8 reports whether layer l's projection stays OUT of q4kw (so it takes the q8
		// branch via m.q8's on-demand quantization of the f32 manifest tensor).
		q8 func(l int, proj string) bool
		// want is the exact sequence of panel widths the prefill must quantize.
		want []int
	}{
		{"all-q4k-resident", func(int, string) bool { return false }, nil},
		{"qk-q8-minority", func(_ int, p string) bool {
			return p == "self_attn.q_proj.weight" || p == "self_attn.k_proj.weight"
		}, []int{H, H}}, // one shared q/k panel per layer, NOT one per projection
		{"qk-plus-layer0-down-q8", func(l int, p string) bool {
			return p == "self_attn.q_proj.weight" || p == "self_attn.k_proj.weight" ||
				(l == 0 && p == "mlp.down_proj.weight")
		}, []int{H, I, H}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := NewSynthetic(cfg)
			var projs [][2]any
			for l := 0; l < cfg.NumLayers; l++ {
				for _, p := range all {
					if !tc.q8(l, p) {
						projs = append(projs, [2]any{layerPrefix(l) + p, outOf[p]})
					}
				}
			}
			fillQ4KW(t, m, projs, 13694)
			var got []int
			q8PanelBuildHook = func(P, width int) { got = append(got, width) }
			defer func() { q8PanelBuildHook = nil }()
			s := m.NewSession()
			defer s.Close()
			s.Q4K = true
			if s.MetalQ4K {
				t.Fatal("requires native CPU dispatch")
			}
			ids := []int{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33}
			s.prefillBatchedQ4K(ids)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("panel widths built = %v (%d builds), want %v", got, len(got), tc.want)
			}
		})
	}
}
