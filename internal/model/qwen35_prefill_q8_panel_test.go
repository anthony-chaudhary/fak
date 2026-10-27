package model

import "testing"

// fak-test:runtime fast est=20ms
func TestQwen35UnusedQ8PanelRequiresCompleteConsumerProof(t *testing.T) {
	oldMode, oldProf, oldHook := qgemmMode, qprofOn, q8PanelBuildHook
	t.Cleanup(func() { qgemmMode, qprofOn, q8PanelBuildHook = oldMode, oldProf, oldHook })
	qgemmMode, qprofOn, q8PanelBuildHook = qgemmModeTile, false, nil
	const width = 256
	manifest, raw := synthBuildRaw([]synthTensor{{"rotated.weight", []int{3, width}}, {"plain.weight", []int{2, width}}}, func(_ string, next func() float32) float32 { return next() })
	m := &Model{manifest: manifest, raw: raw, q4kw: map[string]*q4kTensor{"q4": {in: width}}, kqw: map[string]*kQuantTensor{"k": {in: width}}, q2w: map[string]*q2Tensor{"q2": {in: width}}, q8w: map[string]*q8Tensor{"q4": {in: width}, "plain.weight": {in: width}}}
	t.Cleanup(func() { _ = m.CloseWeights() })
	signs := make([]int, width)
	for i := range signs {
		signs[i] = 1
	}
	if err := m.SetPrismHadamard(PrismHadamardSpec{BlockSize: 64, SignWidths: []int{width}, SignValues: signs, WeightNames: []string{"rotated.weight"}}); err != nil {
		t.Fatal(err)
	}
	s := &Session{M: m}
	x := make([]float32, 3*width)
	for _, names := range [][]string{{"q4"}, {"q4", "k", "q2"}, {"rotated.weight"}, {"q4", "rotated.weight"}} {
		if !s.qwen35PrefillQ8PanelUnused(x, 3, width, names) {
			t.Fatalf("proven unused %v declined", names)
		}
	}
	for _, names := range [][]string{nil, {"q4", "plain.weight"}, {"missing.weight"}, {"rotated.weight", "plain.weight"}} {
		if s.qwen35PrefillQ8PanelUnused(x, 3, width, names) {
			t.Fatalf("needed or unknown %v skipped", names)
		}
	}
	names := []string{"q4"}
	refuse := func(label string) {
		t.Helper()
		if s.qwen35PrefillQ8PanelUnused(x, 3, width, names) {
			t.Fatal(label + " skipped historical eager panel")
		}
	}
	m.q4kw["q4"].in = width * 2
	refuse("resident width")
	m.q4kw["q4"].in = width
	s.MetalQ4K = true
	refuse("Metal")
	s.MetalQ4K = false
	qprofOn = true
	refuse("profile")
	qprofOn = false
	qgemmMode = qgemmModeLegacy
	refuse("legacy")
	qgemmMode = qgemmModeTile
	calls := 0
	q8PanelBuildHook = func(int, int) { calls++ }
	refuse("custom observer")
	q8PanelBuildHook = nil
	if calls != 0 {
		t.Fatal("admission invoked observer")
	}
	if s.qwen35PrefillQ8PanelUnused(x[:width], 1, width, names) || s.qwen35PrefillQ8PanelUnused(x[:len(x)-1], 3, width, names) || s.qwen35PrefillQ8PanelUnused(x, 3, 0, names) {
		t.Fatal("singleton/malformed shape skipped")
	}
}

// fak-test:runtime slow est=2s
func TestQwen35UnusedQ8PanelsKeepOrdinaryHybridState(t *testing.T) {
	oldMode, oldProf, oldHook, oldForce := qgemmMode, qprofOn, q8PanelBuildHook, q4kSDOTForce
	t.Cleanup(func() { qgemmMode, qprofOn, q8PanelBuildHook, q4kSDOTForce = oldMode, oldProf, oldHook, oldForce })
	qgemmMode, qprofOn, q4kSDOTForce = qgemmModeTile, false, -1
	t.Setenv("FAK_QPROFILE", "")
	t.Setenv("FAK_QWEN35_PREFILL_TOKEN_LOOP", "")
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	t.Cleanup(func() { _ = m.CloseWeights() })
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73}
	baselinePanels := 0
	q8PanelBuildHook = func(int, int) { baselinePanels++ } // conservatively retains old eager route
	ref := m.NewSession()
	ref.Q4K = true
	want, ok := ref.tryPrefillQwen35HybridQ4K(prompt, true)
	if !ok {
		t.Fatal("ordinary reference declined")
	}
	if baselinePanels != 4*cfg.NumLayers {
		t.Fatalf("eager fixture built%d panels want%d", baselinePanels, 4*cfg.NumLayers)
	}
	q8PanelBuildHook = nil
	got := m.NewSession()
	got.Q4K = true
	actual, ok := got.tryPrefillQwen35HybridQ4K(prompt, true)
	if !ok {
		t.Fatal("ordinary candidate declined")
	}
	// This real producer gives Q4_K gate/up/down and a Q8 GDN minority: the former
	// has no unrotated-Q8 consumer, while the latter must keep its shared panel.
	p := layerPrefix(0)
	x := make([]float32, len(prompt)*cfg.HiddenSize)
	if !got.qwen35PrefillQ8PanelUnused(x, len(prompt), cfg.HiddenSize, []string{p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}) || got.qwen35PrefillQ8PanelUnused(x, len(prompt), cfg.HiddenSize, []string{p + "linear_attn.in_proj_qkv.weight", p + "linear_attn.in_proj_z.weight"}) {
		t.Fatal("real resident precedence classification changed")
	}
	assertFloat32BitsEqual(t, "hybrid logits", want, actual)
	if got.Cache.Len() != ref.Cache.Len() {
		t.Fatal("cache length changed")
	}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		assertFloat32BitsEqual(t, "K", ref.Cache.K[layer], got.Cache.K[layer])
		assertFloat32BitsEqual(t, "Kraw", ref.Cache.Kraw[layer], got.Cache.Kraw[layer])
		assertFloat32BitsEqual(t, "V", ref.Cache.V[layer], got.Cache.V[layer])
		a, b := ref.Cache.linear.layers[layer], got.Cache.linear.layers[layer]
		if len(a.recurrent) != len(b.recurrent) || len(a.conv) != len(b.conv) {
			t.Fatal("linear state geometry changed")
		}
		for i := range a.recurrent {
			assertFloat32BitsEqual(t, "recurrent", a.recurrent[i], b.recurrent[i])
		}
		for i := range a.conv {
			assertFloat32BitsEqual(t, "conv", a.conv[i], b.conv[i])
		}
	}
}
