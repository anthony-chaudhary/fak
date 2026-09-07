package model

import (
	"math"
	"reflect"
	"testing"
)

func TestQwen35LinearAttnBatchedResumesState(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	s := m.NewSession()
	defer s.Close()
	layer := 0
	for !cfg.isLinearAttnLayer(layer) {
		layer++
	}
	panel := make([][]float32, cfg.LinearConvKernelDim+5)
	for row := range panel {
		panel[row] = make([]float32, cfg.HiddenSize)
		for i := range panel[row] {
			panel[row][i] = float32(math.Sin(float64((row+1)*(i+3)) * 0.017))
		}
	}
	prefix := len(panel) - 3
	for _, row := range panel[:prefix] {
		s.linearAttnStep(layer, row, residentKernel{m})
	}
	state := s.Cache.linear.layers[layer].clone()
	want := make([][]float32, 3)
	for i, row := range panel[prefix:] {
		want[i] = s.linearAttnStep(layer, row, residentKernel{m})
	}
	got, err := m.linearAttnSeqBatchedState(layer, panel[prefix:], &state)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		assertFloat32BitsEqual(t, "resumed row", want[i], got[i])
	}
	if !reflect.DeepEqual(state, s.Cache.linear.layers[layer]) {
		t.Fatal("resumed convolution or recurrent state differs from native token steps")
	}
	cacheless := m.linearAttnSeqBatched(layer, panel)
	for i, row := range m.linearAttnSeq(layer, panel) {
		assertFloat32BitsEqual(t, "cacheless row", row, cacheless[i])
	}
	before := state.clone()
	if _, err := m.linearAttnSeqBatchedState(layer, [][]float32{{1}}, &state); err == nil {
		t.Fatal("malformed input accepted")
	}
	if !reflect.DeepEqual(state, before) {
		t.Fatal("malformed input mutated borrowed state")
	}
	state.recurrent[0] = state.recurrent[0][:1]
	before = state.clone()
	if _, err := m.linearAttnSeqBatchedState(layer, panel[prefix:], &state); err == nil {
		t.Fatal("malformed recurrent state accepted")
	}
	if !reflect.DeepEqual(state, before) {
		t.Fatal("malformed geometry mutated borrowed state")
	}
}

// TestQwen35LinearAttnBatchedMatchesScalar is the issue #443 box-1 witness: the batched-projection
// Gated-DeltaNet prefill path (linearAttnSeqBatched) is bit-for-bit identical to the scalar
// reference (linearAttnSeq) on the f32 path, layer-by-layer over an arbitrary input panel AND
// end-to-end through Forward with the FAK_GDN_BATCHED opt-in engaged. Float32bits equality (not a
// tolerance) is the strongest possible f32 parity gate — every matMulBatch row reduces in the same
// fdot i-order as the per-token GEMV it replaces, so nothing rounds differently.
func TestQwen35LinearAttnBatchedMatchesScalar(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	H := cfg.HiddenSize

	// A deterministic, non-trivial normalized-input panel per linear-attention layer.
	mkPanel := func(seq, seed int) [][]float32 {
		xn := make([][]float32, seq)
		for t := 0; t < seq; t++ {
			row := make([]float32, H)
			for i := 0; i < H; i++ {
				row[i] = float32(math.Sin(float64((t+1)*(i+3)*(seed+1)) * 0.017))
			}
			xn[t] = row
		}
		return xn
	}

	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		for _, seq := range []int{1, 2, 5, 9} {
			xn := mkPanel(seq, l*7+seq)
			want := m.linearAttnSeq(l, xn)
			got := m.linearAttnSeqBatched(l, xn)
			if len(got) != len(want) {
				t.Fatalf("layer %d seq %d: len(got)=%d want %d", l, seq, len(got), len(want))
			}
			for tk := range want {
				assertFloat32BitsEqual(t, "layer "+itoa(l)+" seq "+itoa(seq)+" tok "+itoa(tk), want[tk], got[tk])
			}
		}
	}

	// End-to-end: the batched opt-in must reproduce the scalar Forward logits bit-for-bit.
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 2, 29}
	scalarLogits := m.Forward(prompt).Logits

	old := gdnBatchedPrefill
	gdnBatchedPrefill = true
	defer func() { gdnBatchedPrefill = old }()
	batchedLogits := m.Forward(prompt).Logits

	if len(batchedLogits) != len(scalarLogits) {
		t.Fatalf("Forward len mismatch: batched=%d scalar=%d", len(batchedLogits), len(scalarLogits))
	}
	for t0 := range scalarLogits {
		assertFloat32BitsEqual(t, "Forward batched logits pos "+itoa(t0), scalarLogits[t0], batchedLogits[t0])
	}
}
