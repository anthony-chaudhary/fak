package model

import (
	"reflect"
	"testing"
)

func TestQwen35VerifyPanelResumesNativeState(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	// Exercise the physical override selected by ordinary mlpNorms.
	m.manifest[layerName(0, "pre_feedforward_layernorm.weight")] = m.manifest[layerName(0, "input_layernorm.weight")]
	m.tensor(layerName(0, "pre_feedforward_layernorm.weight"))[0] += 0.5
	panel, serial := m.NewSession(), m.NewSession()
	defer panel.Close()
	defer serial.Close()
	panel.captureTargetHidden, serial.captureTargetHidden = true, true
	prefix, draft := []int{3, 7, 11, 5, 17, 19}, []int{23, 2, 29}
	panel.Prefill(prefix)
	serial.Prefill(prefix)
	before := panel.Cache.Clone()
	panel.PhaseProfiler = NewPhaseProfiler()
	layers := 0
	got, err := panel.qwen35VerifyPanel(draft, func(layer, rows int) {
		if layer != layers || rows != len(draft) {
			t.Fatalf("layer input %d/%d, want layer %d and only draft rows", layer, rows, layers)
		}
		layers++
	})
	if err != nil {
		t.Fatal(err)
	}
	if layers != m.Cfg.NumLayers || len(got) != len(draft) {
		t.Fatal("incomplete layer panel or logits")
	}
	for i, id := range draft {
		assertFloat32BitsEqual(t, "draft logits", serial.Step(id), got[i])
	}
	assertQwen35MTPTargetStateEqual(t, panel, serial)
	for layer := range before.K {
		assertFloat32BitsEqual(t, "prefix K", before.K[layer], panel.Cache.K[layer][:len(before.K[layer])])
		assertFloat32BitsEqual(t, "prefix V", before.V[layer], panel.Cache.V[layer][:len(before.V[layer])])
		assertFloat32BitsEqual(t, "prefix Kraw", before.Kraw[layer], panel.Cache.Kraw[layer][:len(before.Kraw[layer])])
	}
	if panel.PhaseProfiler.stat["mlp_decode"] != nil || panel.PhaseProfiler.stat["qwen35_linear_step_in_proj"] != nil {
		t.Fatal("panel executed ordinary token steps")
	}
	assertFloat32BitsEqual(t, "continuation", serial.Step(31), panel.Step(31))
	assertQwen35MTPTargetStateEqual(t, panel, serial)
	checkpoint, err := panel.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer checkpoint.Close()
	if _, err := panel.qwen35VerifyPanel([]int{m.Cfg.VocabSize}, nil); err == nil {
		t.Fatal("invalid draft accepted")
	}
	if !reflect.DeepEqual(checkpoint.Cache, panel.Cache.Clone()) {
		t.Fatal("admission failure mutated cache")
	}
	weight := layerName(m.Cfg.NumLayers-1, "mlp.down_proj.weight")
	saved := m.manifest[weight]
	delete(m.manifest, weight)
	if _, err := panel.qwen35VerifyPanel(draft, nil); err == nil {
		t.Fatal("missing F32 projection admitted")
	}
	badShape := saved
	badShape.Shape = []int{1, 1}
	m.manifest[weight] = badShape
	if _, err := panel.qwen35VerifyPanel(draft, nil); err == nil {
		t.Fatal("malformed F32 projection admitted")
	}
	m.manifest[weight] = saved
	if !reflect.DeepEqual(checkpoint.Cache, panel.Cache.Clone()) {
		t.Fatal("projection admission failure mutated cache")
	}
}
