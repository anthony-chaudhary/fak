package model

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=100ms
func TestPrismBatchedResidentProjection(t *testing.T) {
	t.Run("signed-panel", func(t *testing.T) {
		const declared, ordinary = "declared.weight", "ordinary.weight"
		weight := []float32{0.5, 0, -0.5, 0}
		i := 0
		manifest, raw := synthBuildRaw([]synthTensor{
			{declared, []int{1, len(weight)}},
			{ordinary, []int{1, len(weight)}},
		}, func(string, func() float32) float32 {
			v := weight[i%len(weight)]
			i++
			return v
		})
		m := &Model{manifest: manifest, raw: raw}
		t.Cleanup(func() { _ = m.CloseWeights() })
		x := []float32{1, 2, 3, 4, -2, 1, 0, 3}
		before := append([]float32(nil), x...)
		weightsBefore := append([]byte(nil), raw...)
		rows := len(x) / len(weight)
		unrotated := []float32{-1, -1}
		assertFloat32BitsEqual(t, "unconfigured panel", unrotated,
			m.residentMatMulBatch(declared, x, 1, len(weight), rows))
		if err := m.SetPrismHadamard(PrismHadamardSpec{
			BlockSize: 4, SignWidths: []int{len(weight)},
			SignValues: []int{1, -1, 1, 1}, WeightNames: []string{declared},
		}); err != nil {
			t.Fatal(err)
		}
		// Literal signed-H4 outputs are [3,1,-4,2] and [0,-2,-3,1].
		// These expectations do not call the panel transform under test.
		want := []float32{3.5, 1.5}
		for row := range want {
			assertFloat32BitsEqual(t, "literal scalar oracle", want[row:row+1],
				m.residentMatRows(declared, x[row*len(weight):(row+1)*len(weight)], 1, len(weight)))
		}
		assertFloat32BitsEqual(t, "undeclared panel", unrotated,
			m.residentMatMulBatch(ordinary, x, 1, len(weight), rows))
		got := m.residentMatMulBatch(declared, x, 1, len(weight), rows)
		if !reflect.DeepEqual(x, before) || !reflect.DeepEqual(raw, weightsBefore) {
			t.Fatal("batch projection mutated borrowed activations or resident weights")
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("Prism batch omitted declared transform: got=%v want=%v scalar_verified=true input_unchanged=true", got, want)
		}
		set := NewLoRASet()
		if err := set.Add(&LoRAAdapter{Adapter: "delta", Target: declared, In: len(weight), Out: 1,
			Rank: 1, Alpha: 1, A: []float32{1, 0, 0, 0}, B: []float32{2}}); err != nil {
			t.Fatal(err)
		}
		m.SetLoRA(set)
		assertFloat32BitsEqual(t, "LoRA uses original input", []float32{5.5, -2.5},
			m.residentMatMulBatch(declared, x, 1, len(weight), rows))
		m.SetLoRA(nil)
		if got := m.residentMatMulBatch(declared, nil, 1, len(weight), 0); len(got) != 0 {
			t.Fatalf("empty panel produced rows: %v", got)
		}
		if !reflect.DeepEqual(x, before) || !reflect.DeepEqual(raw, weightsBefore) {
			t.Fatal("fallback projection mutated borrowed activations or resident weights")
		}
	})
	t.Run("stateful-gdn", func(t *testing.T) {
		cfg := qwen35HybridTestCfg()
		m := NewSynthetic(cfg)
		t.Cleanup(func() { _ = m.CloseWeights() })
		const layer = 0
		_, _, _, _, _, valueWidth, _ := cfg.linearAttnDims()
		if !cfg.isLinearAttnLayer(layer) || valueWidth != cfg.HiddenSize {
			t.Fatal("fixture requires a linear layer with equal input/output sign widths")
		}
		names := []string{
			layerName(layer, "linear_attn.in_proj_qkv.weight"),
			layerName(layer, "linear_attn.in_proj_z.weight"),
			layerName(layer, "linear_attn.in_proj_b.weight"),
			layerName(layer, "linear_attn.in_proj_a.weight"),
			layerName(layer, "linear_attn.out_proj.weight"),
		}
		signs := make([]int, cfg.HiddenSize)
		for i := range signs {
			signs[i] = 1
			if i%4 == 1 {
				signs[i] = -1
			}
		}
		if err := m.SetPrismHadamard(PrismHadamardSpec{BlockSize: 4,
			SignWidths: []int{len(signs)}, SignValues: signs, WeightNames: names}); err != nil {
			t.Fatal(err)
		}
		panel := make([][]float32, cfg.LinearConvKernelDim+4)
		before := make([][]float32, len(panel))
		for row := range panel {
			panel[row] = make([]float32, cfg.HiddenSize)
			for i := range panel[row] {
				panel[row][i] = float32(math.Sin(float64((row+1)*(i+3)) * 0.017))
			}
			before[row] = append([]float32(nil), panel[row]...)
		}
		want := m.linearAttnSeq(layer, panel)
		got := m.linearAttnSeqBatched(layer, panel)
		if len(got) != len(want) || len(got) != len(panel) {
			t.Fatalf("GDN output rows=%d scalar=%d request=%d", len(got), len(want), len(panel))
		}
		for row := range want {
			assertFloat32BitsEqual(t, "Prism cacheless GDN", want[row], got[row])
		}
		s := m.NewSession()
		t.Cleanup(s.Close)
		prefix := cfg.LinearConvKernelDim + 1
		for _, row := range panel[:prefix] {
			s.linearAttnStep(layer, row, residentKernel{m})
		}
		state := s.Cache.linear.layers[layer].clone()
		want = make([][]float32, len(panel)-prefix)
		for i, row := range panel[prefix:] {
			want[i] = s.linearAttnStep(layer, row, residentKernel{m})
		}
		got, err := m.linearAttnSeqBatchedStateful(layer, panel[prefix:], &state)
		if err != nil || len(got) != len(want) {
			t.Fatalf("resumed GDN rows=%d want=%d err=%v", len(got), len(want), err)
		}
		for row := range want {
			assertFloat32BitsEqual(t, "Prism resumed GDN", want[row], got[row])
		}
		if !reflect.DeepEqual(state, s.Cache.linear.layers[layer]) {
			t.Fatal("Prism batched recurrence or convolution differs from native token steps")
		}
		if !reflect.DeepEqual(panel, before) {
			t.Fatal("GDN consumer mutated borrowed activation rows")
		}
	})
}
