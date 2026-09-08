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

// TestQwen35ChunkedBatchedMatchesScalar aliases TestQwen35LinearAttnBatchedMatchesScalar
// under the TestQwen35Chunked test witness namespace.
func TestQwen35ChunkedBatchedMatchesScalar(t *testing.T) {
	TestQwen35LinearAttnBatchedMatchesScalar(t)
}

// TestQwen35ChunkedLinearAttnBatchedResumedStateMatchesStep is the issue #11987 witness:
// resumes batched GDN from a committed prefix of length > K (e.g. prefix 5, K=4), verifies
// that processing three new rows via linearAttnSeqBatchedStateful produces bit-identical
// output, conv state, and recurrent state compared to ordinary native linearAttnStep.
// Also proves cacheless wrapper equality and malformed-state refusal before mutation.
func TestQwen35ChunkedLinearAttnBatchedResumedStateMatchesStep(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	H := cfg.HiddenSize
	nV := cfg.LinearNumValueHeads
	K := cfg.LinearConvKernelDim

	mkRow := func(seed int) []float32 {
		row := make([]float32, H)
		for i := 0; i < H; i++ {
			row[i] = float32(math.Sin(float64((i+3)*(seed+1)) * 0.017))
		}
		return row
	}

	// Find the first linear attention layer.
	var linearLayer int = -1
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			linearLayer = l
			break
		}
	}
	if linearLayer < 0 {
		t.Fatal("no linear attention layer in test cfg")
	}

	// 1. Cacheless wrapper equality: linearAttnSeqBatched == linearAttnSeqBatchedStateful(nil).
	seqSample := [][]float32{mkRow(1), mkRow(2), mkRow(3)}
	wrapperOut := m.linearAttnSeqBatched(linearLayer, seqSample)
	directOut, err := m.linearAttnSeqBatchedStateful(linearLayer, seqSample, nil)
	if err != nil {
		t.Fatalf("cacheless stateful returned error: %v", err)
	}
	if len(wrapperOut) != len(directOut) {
		t.Fatalf("wrapper len mismatch: %d vs %d", len(wrapperOut), len(directOut))
	}
	for i := range wrapperOut {
		assertFloat32BitsEqual(t, "cacheless wrapper equality row "+itoa(i), wrapperOut[i], directOut[i])
	}

	// 2. Malformed-state refusal before mutation.
	badState := linearAttnLayerState{
		recurrent: make([][]float32, nV-1), // malformed: wrong head count
	}
	preStateRecurrent := len(badState.recurrent)
	_, badErr := m.linearAttnSeqBatchedStateful(linearLayer, seqSample, &badState)
	if badErr == nil {
		t.Fatal("expected error on malformed recurrent head count, got nil")
	}
	if len(badState.recurrent) != preStateRecurrent {
		t.Fatal("malformed state was mutated despite error refusal")
	}

	// 3. Resumed-prefix > K (prefix len 5 > K=4) plus three new rows.
	prefixLen := K + 1 // 5
	prefixRows := make([][]float32, prefixLen)
	for i := 0; i < prefixLen; i++ {
		prefixRows[i] = mkRow(10 + i)
	}
	newRows := [][]float32{mkRow(101), mkRow(102), mkRow(103)} // 3 new rows

	// Advance a session using linearAttnStep through the prefix to build committed native state.
	sStep := m.NewSession()
	mat := residentKernel{m}
	for i := 0; i < prefixLen; i++ {
		sStep.linearAttnStep(linearLayer, prefixRows[i], mat)
	}
	stepCacheLayer := sStep.Cache.linear.layer(cfg, linearLayer)

	// Clone the committed prefix state for the batched resume path.
	batchedState := stepCacheLayer.clone()

	// Step the remaining 3 new rows sequentially using linearAttnStep.
	wantStepOutputs := make([][]float32, len(newRows))
	for i := 0; i < len(newRows); i++ {
		wantStepOutputs[i] = sStep.linearAttnStep(linearLayer, newRows[i], mat)
	}

	// Process the 3 new rows in a single batch using linearAttnSeqBatchedStateful, resuming from committed state.
	gotBatchedOutputs, err := m.linearAttnSeqBatchedStateful(linearLayer, newRows, &batchedState)
	if err != nil {
		t.Fatalf("linearAttnSeqBatchedStateful resume failed: %v", err)
	}

	// Check outputs: all 3 rows must be bit-identical.
	if len(gotBatchedOutputs) != len(wantStepOutputs) {
		t.Fatalf("output rows count mismatch: got %d, want %d", len(gotBatchedOutputs), len(wantStepOutputs))
	}
	for i := range wantStepOutputs {
		assertFloat32BitsEqual(t, "resumed GDN output row "+itoa(i), wantStepOutputs[i], gotBatchedOutputs[i])
	}

	// Check convolution state: must match exactly.
	if len(batchedState.conv) != len(stepCacheLayer.conv) {
		t.Fatalf("conv window len mismatch: got %d, want %d", len(batchedState.conv), len(stepCacheLayer.conv))
	}
	for i := range stepCacheLayer.conv {
		assertFloat32BitsEqual(t, "resumed GDN conv row "+itoa(i), stepCacheLayer.conv[i], batchedState.conv[i])
	}

	// Check recurrent state: all nV heads must match bit-for-bit.
	if len(batchedState.recurrent) != nV {
		t.Fatalf("recurrent head count mismatch: got %d, want %d", len(batchedState.recurrent), nV)
	}
	for h := 0; h < nV; h++ {
		assertFloat32BitsEqual(t, "resumed GDN recurrent head "+itoa(h), stepCacheLayer.recurrent[h], batchedState.recurrent[h])
	}
}

// TestQwen35LinearAttnBatchedResumedStateMatchesStep aliases
// TestQwen35ChunkedLinearAttnBatchedResumedStateMatchesStep for compatibility.
func TestQwen35LinearAttnBatchedResumedStateMatchesStep(t *testing.T) {
	TestQwen35ChunkedLinearAttnBatchedResumedStateMatchesStep(t)
}
