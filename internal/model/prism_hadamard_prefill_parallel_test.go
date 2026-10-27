package model

import (
	"bytes"
	"math"
	"testing"
)

// fak-test:runtime fast est=250ms
func TestPrismPrefillPanelRowParallelKeepsScalarTransform(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
	})
	const name, width = "declared.weight", 256
	for _, block := range []int{1, 4, 64, 256} {
		manifest, raw := synthBuildRaw([]synthTensor{{name, []int{3, width}}}, func(_ string, next func() float32) float32 { return next() })
		m := &Model{manifest: manifest, raw: raw}
		t.Cleanup(func() { _ = m.CloseWeights() })
		signs := make([]int, width)
		for i := range signs {
			signs[i] = 1
			if i%5 == 1 {
				signs[i] = -1
			}
		}
		if err := m.SetPrismHadamard(PrismHadamardSpec{BlockSize: block, SignWidths: []int{width}, SignValues: signs, WeightNames: []string{name}}); err != nil {
			t.Fatal(err)
		}
		rawBefore := bytes.Clone(raw)
		for _, rows := range []int{0, 1, 3, 257} {
			x := make([]float32, rows*width)
			for i := range x {
				x[i] = float32(i%31-15) / 29
			}
			if len(x) > 0 {
				x[0] = float32(math.Copysign(0, -1))
			}
			before := append([]float32(nil), x...)
			want := make([]float32, len(x))
			for row := 0; row < rows; row++ {
				copy(want[row*width:(row+1)*width], m.prismProjectInput(name, x[row*width:(row+1)*width]))
			}
			for _, workers := range []int{1, 4} {
				if err := SetWorkers(workers); err != nil {
					t.Fatal(err)
				}
				got := m.prismProjectPanel(name, x, rows)
				assertFloat32BitsEqual(t, "parallel signed panel", want, got)
				assertFloat32BitsEqual(t, "borrowed input", before, x)
				if !bytes.Equal(raw, rawBefore) {
					t.Fatal("resident weights mutated")
				}
				if len(got) > 0 {
					got[0]++
					assertFloat32BitsEqual(t, "result must not alias input", before, x)
				}
				if undeclared := m.prismProjectPanel("undeclared.weight", x, rows); len(x) > 0 && &undeclared[0] != &x[0] {
					t.Fatal("undeclared weight stopped returning the original panel")
				}
			}
			if rows == 257 {
				// A real batched resident projection reaches the transformed
				// panel; this shape also exceeds the existing worker threshold.
				if len(x) < parThreshold {
					t.Fatal("fixture does not reach parallel threshold")
				}
				if err := SetWorkers(1); err != nil {
					t.Fatal(err)
				}
				reference := m.residentMatMulBatch(name, x, 3, width, rows)
				if err := SetWorkers(4); err != nil {
					t.Fatal(err)
				}
				actual := m.residentMatMulBatch(name, x, 3, width, rows)
				assertFloat32BitsEqual(t, "ordinary resident projection", reference, actual)
			}
		}
		// Shape validation still happens before dispatch; preserve the old panic.
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("malformed panel stopped panicking")
				}
			}()
			m.prismProjectPanel(name, make([]float32, width-1), 1)
		}()
	}
}

// fak-test:runtime fast est=20ms
func TestPrismPrefillGroupRequiresProducerIdentity(t *testing.T) {
	oldGroup, oldMode, oldProf, oldHook := qgemmGroup, qgemmMode, qprofOn, q8PanelBuildHook
	t.Cleanup(func() { qgemmGroup, qgemmMode, qprofOn, q8PanelBuildHook = oldGroup, oldMode, oldProf, oldHook })
	qgemmGroup, qgemmMode, qprofOn, q8PanelBuildHook = true, qgemmModeTile, false, nil
	const width = 64
	names := []string{"a.weight", "b.weight"}
	manifest, raw := synthBuildRaw([]synthTensor{{names[0], []int{3, width}}, {names[1], []int{5, width}}, {"other.weight", []int{2, 32}}}, func(_ string, next func() float32) float32 { return next() })
	m := &Model{manifest: manifest, raw: raw}
	t.Cleanup(func() { _ = m.CloseWeights() })
	signs := make([]int, width+32)
	for i := range signs {
		signs[i] = 1
		if i%3 == 1 {
			signs[i] = -1
		}
	}
	if err := m.SetPrismHadamard(PrismHadamardSpec{BlockSize: 4, SignWidths: []int{width, 32}, SignValues: signs, WeightNames: append(append([]string(nil), names...), "other.weight")}); err != nil {
		t.Fatal(err)
	}
	// Real loader-facing producer, not a crafted state: same-width names resolve
	// to the exact same state-owned sign slice and block geometry.
	state := m.prism
	if &state.signs[state.weightWidth[names[0]]][0] != &state.signs[state.weightWidth[names[1]]][0] {
		t.Fatal("producer did not share sign identity")
	}
	session := &Session{M: m}
	x := make([]float32, 3*width)
	for i := range x {
		x[i] = float32(i%37-18) / 19
	}
	before := append([]float32(nil), x...)
	got := session.prismPrefillGroupPanel(names, x, 3)
	if got == nil {
		t.Fatal("validated producer identity declined")
	}
	for _, name := range names {
		assertFloat32BitsEqual(t, name, m.prismProjectPanel(name, x, 3), got)
	}
	got[0]++
	assertFloat32BitsEqual(t, "input remains borrowed", before, x)
	if m.prism != state {
		t.Fatal("group changed model state")
	}
	decline := func(label string, ns []string, panel []float32, rows int) {
		t.Helper()
		if session.prismPrefillGroupPanel(ns, panel, rows) != nil {
			t.Fatal(label + " must decline")
		}
	}
	decline("undeclared", []string{names[0], "missing.weight"}, x, 3)
	decline("different width", []string{names[0], "other.weight"}, x, 3)
	decline("singleton", names, x[:width], 1)
	decline("shape", names, x[:len(x)-1], 3)
	m.qwen35TiledOutProj = map[string]struct{}{names[1]: {}}
	decline("tiled input", names, x, 3)
	m.qwen35TiledOutProj = nil
	qprofOn = true
	decline("profile", names, x, 3)
	qprofOn = false
	qgemmGroup = false
	decline("group disabled", names, x, 3)
	qgemmGroup = true
	qgemmMode = qgemmModeLegacy
	decline("legacy", names, x, 3)
	qgemmMode = qgemmModeTile
	calls := 0
	q8PanelBuildHook = func(int, int) { calls++ }
	decline("custom panel observer", names, x, 3)
	if calls != 0 {
		t.Fatal("decline invoked custom observer")
	}
	assertFloat32BitsEqual(t, "declines preserve input", before, x)
}

// fak-test:runtime slow est=2s
func TestPrismQwen35HybridPrefillGroupMatchesIndependentTransforms(t *testing.T) {
	oldGroup, oldMode, oldProf, oldHook, oldForce := qgemmGroup, qgemmMode, qprofOn, q8PanelBuildHook, q4kSDOTForce
	t.Cleanup(func() {
		qgemmGroup, qgemmMode, qprofOn, q8PanelBuildHook, q4kSDOTForce = oldGroup, oldMode, oldProf, oldHook, oldForce
	})
	qgemmMode, qprofOn, q8PanelBuildHook, q4kSDOTForce = qgemmModeTile, false, nil, -1
	t.Setenv("FAK_QPROFILE", "")
	t.Setenv("FAK_QWEN35_PREFILL_TOKEN_LOOP", "")
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	t.Cleanup(func() { _ = m.CloseWeights() })
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	names := []string{}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		p := layerPrefix(layer)
		names = append(names, p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight")
		if cfg.isLinearAttnLayer(layer) {
			names = append(names, p+"linear_attn.in_proj_qkv.weight", p+"linear_attn.in_proj_z.weight", p+"linear_attn.in_proj_b.weight", p+"linear_attn.in_proj_a.weight")
		} else {
			names = append(names, p+"self_attn.q_proj.weight", p+"self_attn.k_proj.weight", p+"self_attn.v_proj.weight")
		}
	}
	signs := make([]int, cfg.HiddenSize)
	for i := range signs {
		signs[i] = 1
		if i%5 == 2 {
			signs[i] = -1
		}
	}
	if err := m.SetPrismHadamard(PrismHadamardSpec{BlockSize: 64, SignWidths: []int{cfg.HiddenSize}, SignValues: signs, WeightNames: names}); err != nil {
		t.Fatal(err)
	}
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73}
	if !q4kQwen35HybridPrefillOK(cfg, len(prompt)) {
		t.Fatal("fixture misses ordinary hybrid route")
	}
	run := func(group bool) (*Session, []float32) {
		qgemmGroup = group
		session := m.NewSession()
		session.Q4K = true
		logits, ok := session.tryPrefillQwen35HybridQ4K(prompt, true)
		if !ok {
			t.Fatal("ordinary hybrid selection declined")
		}
		return session, logits
	}
	ref, want := run(false)
	got, actual := run(true)
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
			t.Fatal("linear state shape changed")
		}
		for i := range a.recurrent {
			assertFloat32BitsEqual(t, "recurrent", a.recurrent[i], b.recurrent[i])
		}
		for i := range a.conv {
			assertFloat32BitsEqual(t, "conv", a.conv[i], b.conv[i])
		}
	}
}
