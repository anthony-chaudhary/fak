package model

// dspine_bridge_test.go — reference-equivalence pins for the DSpine
// adjacent-causal-injection drafter lineage (anthony-chaudhary/fak#14441).
// Synthetic fp32 weights only: no checkpoint download, no quantization, no
// file I/O. The oracle helpers below share no code with the drafter; they
// re-implement the documented math by hand so a wiring drift turns red here
// (the v41SpecDecodeOracleFromLogits pattern).

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"
)

const (
	dspineTestHidden = 8
	dspineTestRank   = 4
	dspineTestVocab  = 32
	dspineTestLayers = 3
	dspineTestBlock  = 16
	dspineTestMask   = 31
	dspineTestTopK   = 4
)

// dspineTestWeights builds deterministic synthetic fp32 weights.
func dspineTestWeights(t *testing.T) *DSpineWeights {
	t.Helper()
	rng := rand.New(rand.NewSource(14441))
	fill := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	w := &DSpineWeights{
		Hidden: dspineTestHidden,
		Rank:   dspineTestRank,
		Vocab:  dspineTestVocab,
		Dense:  make([][]float32, dspineTestLayers),
	}
	for l := range w.Dense {
		w.Dense[l] = fill(dspineTestHidden * dspineTestHidden)
	}
	w.Transfer = fill(dspineTestRank * dspineTestHidden)
	w.Gate = fill(dspineTestRank * dspineTestHidden)
	w.Lift = fill(dspineTestHidden * dspineTestRank)
	w.Embed = fill(dspineTestVocab * dspineTestHidden)
	w.Head = fill(dspineTestVocab * dspineTestHidden)
	return w
}

// dspineTestContract is the supported checkpoint contract: version stamp 1,
// terminal-disabled with an all-zero terminal projection, independent writes,
// serial decode, separated selector, final write on the last layer, agreeing
// refinement widths, top-k in [2, 64].
func dspineTestContract() DSpineCheckpointContract {
	return DSpineCheckpointContract{
		Version:            1,
		WritesIndependent:  true,
		SerialDecode:       true,
		SelectorSeparated:  true,
		FinalLayerWrite:    dspineTestLayers - 1,
		RefinementWidths:   []int{dspineTestTopK, dspineTestTopK, dspineTestTopK, dspineTestTopK},
		TopK:               dspineTestTopK,
		TerminalProjection: []float32{},
	}
}

// dspineTestConfig wraps layers in an enabled config over the supported
// contract.
func dspineTestConfig(layers []DSpineLayerSpec) DSpineBridgeConfig {
	return DSpineBridgeConfig{
		Enabled:    true,
		Checkpoint: dspineTestContract(),
		Layers:     layers,
		BlockSize:  dspineTestBlock,
		TopK:       dspineTestTopK,
		Rank:       dspineTestRank,
		MaskToken:  dspineTestMask,
		Eps:        1e-6,
	}
}

// Oracle primitives: hand-written recomputations of the documented math.

func dspineOracleRMS(v []float32, eps float32) []float32 {
	sum := float32(0)
	for idx := range v {
		sum += v[idx] * v[idx]
	}
	scale := float32(1.0 / math.Sqrt(float64(sum/float32(len(v)))+float64(eps)))
	out := make([]float32, len(v))
	for idx, val := range v {
		out[idx] = val * scale
	}
	return out
}

func dspineOracleMatVec(m, v []float32, outDim, inDim int) []float32 {
	out := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		acc := float32(0)
		for i := 0; i < inDim; i++ {
			acc += m[o*inDim+i] * v[i]
		}
		out[o] = acc
	}
	return out
}

func dspineOracleSigmoid(g float32) float32 {
	return g / (1 + float32(math.Exp(float64(-g))))
}

// dspineOracleForward recomputes the whole forward pass from scratch and
// returns per-layer norms/outputs, final features, and logits.
func dspineOracleForward(w *DSpineWeights, layers []DSpineLayerSpec, anchor, mask, rows int, eps float32) ([][]float32, [][]float32, []float32, [][]float32) {
	h, r := dspineTestHidden, dspineTestRank
	x := make([]float32, rows*h)
	copy(x[:h], w.Embed[anchor*h:(anchor+1)*h])
	for i := 1; i < rows; i++ {
		copy(x[i*h:], w.Embed[mask*h:(mask+1)*h])
	}
	anchorEmb := w.Embed[anchor*h : (anchor+1)*h]
	norms := make([][]float32, dspineTestLayers)
	outs := make([][]float32, dspineTestLayers)
	prev := []float32(nil)
	for l := 0; l < dspineTestLayers; l++ {
		n := make([]float32, rows*h)
		for i := 0; i < rows; i++ {
			norm := dspineOracleRMS(x[i*h:(i+1)*h], eps)
			if l > 0 && i >= 1 {
				var src []float32
				if i == 1 {
					src = anchorEmb
				} else {
					src = prev[(i-1)*h : i*h]
				}
				pred := dspineOracleMatVec(w.Transfer, src, r, h)
				gamma := dspineOracleMatVec(w.Gate, prev[i*h:(i+1)*h], r, h)
				gated := make([]float32, r)
				for k := 0; k < r; k++ {
					gated[k] = dspineOracleSigmoid(gamma[k]) * pred[k]
				}
				msg := dspineOracleMatVec(w.Lift, gated, h, r)
				for k := 0; k < h; k++ {
					norm[k] += msg[k]
				}
			}
			copy(n[i*h:], norm)
		}
		out := make([]float32, rows*h)
		for i := 0; i < rows; i++ {
			proj := dspineOracleMatVec(w.Dense[l], n[i*h:(i+1)*h], h, h)
			start := 0
			if layers[l].Direction == DSpineSlidingWindow {
				start = i - layers[l].Window + 1
				if start < 0 {
					start = 0
				}
			}
			mix := make([]float32, h)
			for j := start; j <= i; j++ {
				for k := 0; k < h; k++ {
					mix[k] += n[j*h+k]
				}
			}
			for k := 0; k < h; k++ {
				out[i*h+k] = x[i*h+k] + proj[k] + mix[k]
			}
		}
		norms[l] = n
		outs[l] = out
		prev = out
	}
	feat := make([]float32, rows*h)
	for i := 0; i < rows; i++ {
		copy(feat[i*h:], dspineOracleRMS(prev[i*h:(i+1)*h], eps))
	}
	logits := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		logits[i] = dspineOracleMatVec(w.Head, feat[i*h:(i+1)*h], dspineTestVocab, h)
	}
	return norms, outs, feat, logits
}

func TestDSpineBridgeWriteMatchesReference(t *testing.T) {
	ctx := context.Background()
	w := dspineTestWeights(t)
	layers := []DSpineLayerSpec{
		{Direction: DSpineCausal},
		{Direction: DSpineSlidingWindow, Window: 4},
		{Direction: DSpineCausal},
	}
	d, err := NewDSpineBridge(dspineTestConfig(layers), w)
	if err != nil {
		t.Fatalf("admission refused a valid config: %v", err)
	}
	const anchor, rows = 7, 6
	trace, err := d.Forward(ctx, anchor, rows)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	norms, outs, feat, logits := dspineOracleForward(w, layers, anchor, dspineTestMask, rows, float32(1e-6))
	h := dspineTestHidden
	for l := 0; l < dspineTestLayers; l++ {
		if len(trace.Layers[l].Norm) != len(norms[l]) || len(trace.Layers[l].Out) != len(outs[l]) {
			t.Fatalf("layer %d: trace shape mismatch (%d/%d vs %d/%d)", l, len(trace.Layers[l].Norm), len(trace.Layers[l].Out), len(norms[l]), len(outs[l]))
		}
		for i := range norms[l] {
			if trace.Layers[l].Norm[i] != norms[l][i] {
				t.Fatalf("layer %d norm[%d]: wired %v != reference %v", l, i, trace.Layers[l].Norm[i], norms[l][i])
			}
			if trace.Layers[l].Out[i] != outs[l][i] {
				t.Fatalf("layer %d out[%d]: wired %v != reference %v", l, i, trace.Layers[l].Out[i], outs[l][i])
			}
		}
	}
	for i := range feat {
		if trace.Features[i] != feat[i] {
			t.Fatalf("feature[%d]: wired %v != reference %v", i, trace.Features[i], feat[i])
		}
	}
	for i := range logits {
		for c := range logits[i] {
			if trace.Logits[i][c] != logits[i][c] {
				t.Fatalf("logits[%d][%d]: wired %v != reference %v", i, c, trace.Logits[i][c], logits[i][c])
			}
		}
	}
	// The anchor row is never written: at every layer >= 1 its input norm stays
	// the plain RMS of the anchor embedding.
	anchorEmb := w.Embed[anchor*h : (anchor+1)*h]
	for l := 1; l < dspineTestLayers; l++ {
		want := dspineOracleRMS(anchorEmb, float32(1e-6))
		for k := 0; k < h; k++ {
			if trace.Layers[l].Norm[k] != want[k] {
				t.Fatalf("layer %d: anchor row was written (norm[%d] %v != plain rms %v)", l, k, trace.Layers[l].Norm[k], want[k])
			}
		}
	}
	// Fused next-input-norm: every non-anchor row received a nonzero write at
	// every layer >= 1 (synthetic random weights make a zero message
	// measure-zero).
	for l := 1; l < dspineTestLayers; l++ {
		for i := 1; i < rows; i++ {
			plain := dspineOracleRMS(w.Embed[dspineTestMask*h:(dspineTestMask+1)*h], float32(1e-6))
			written := false
			for k := 0; k < h; k++ {
				if trace.Layers[l].Norm[i*h+k] != plain[k] {
					written = true
				}
			}
			if !written {
				t.Fatalf("layer %d row %d: bridge write did not land", l, i)
			}
		}
	}
}

func TestCausalBlockTruncationIsExact(t *testing.T) {
	ctx := context.Background()
	w := dspineTestWeights(t)
	layers := []DSpineLayerSpec{
		{Direction: DSpineCausal},
		{Direction: DSpineCausal},
		{Direction: DSpineSlidingWindow, Window: 5},
	}
	d, err := NewDSpineBridge(dspineTestConfig(layers), w)
	if err != nil {
		t.Fatalf("admission refused a valid config: %v", err)
	}
	full, err := d.Forward(ctx, 7, dspineTestBlock)
	if err != nil {
		t.Fatalf("full block forward: %v", err)
	}
	h := dspineTestHidden
	for _, cap := range []int{3, 9, 15} {
		trunc, err := d.Forward(ctx, 7, cap+1)
		if err != nil {
			t.Fatalf("cap %d: truncated forward refused: %v", cap, err)
		}
		for i := 0; i <= cap; i++ {
			for k := 0; k < h; k++ {
				if trunc.Features[i*h+k] != full.Features[i*h+k] {
					t.Fatalf("cap %d row %d feature %d: truncated %v != full %v", cap, i, k, trunc.Features[i*h+k], full.Features[i*h+k])
				}
			}
			for c := 0; c < dspineTestVocab; c++ {
				if trunc.Logits[i][c] != full.Logits[i][c] {
					t.Fatalf("cap %d row %d logit %d: truncated %v != full %v", cap, i, c, trunc.Logits[i][c], full.Logits[i][c])
				}
			}
		}
		tokens, err := d.Draft(ctx, 7, cap)
		if err != nil {
			t.Fatalf("cap %d: draft: %v", cap, err)
		}
		if len(tokens) != cap {
			t.Fatalf("cap %d: drafted %d tokens, want %d", cap, len(tokens), cap)
		}
	}
}

func TestBidirectionalBlockRefusesTruncation(t *testing.T) {
	ctx := context.Background()
	w := dspineTestWeights(t)
	// Shipped-head shape: both directions declared across the layers.
	layers := []DSpineLayerSpec{
		{Direction: DSpineCausal},
		{Direction: DSpineBidirectional},
		{Direction: DSpineCausal},
	}
	d, err := NewDSpineBridge(dspineTestConfig(layers), w)
	if err != nil {
		t.Fatalf("admission refused a valid config: %v", err)
	}
	if _, err := d.Forward(ctx, 7, dspineTestBlock); err != nil {
		t.Fatalf("full block forward must succeed: %v", err)
	}
	// Caps 3 and 14 draft 4 and 15 rows: both truncate the trained 16-row block.
	for _, cap := range []int{3, 14} {
		if _, err := d.Forward(ctx, 7, cap+1); !errors.Is(err, ErrDSpineTruncationRefused) {
			t.Fatalf("cap %d: Forward wants a structured truncation refusal, got %v", cap, err)
		}
		if _, err := d.Draft(ctx, 7, cap); !errors.Is(err, ErrDSpineTruncationRefused) {
			t.Fatalf("cap %d: Draft wants a structured truncation refusal, got %v", cap, err)
		}
	}
}

func TestDSpineContractRefusals(t *testing.T) {
	w := dspineTestWeights(t)
	// Three layers: the supported contract stamps its final write on the last
	// layer (index 2), so the out-of-range case names index 3.
	base := dspineTestConfig([]DSpineLayerSpec{{Direction: DSpineCausal}, {Direction: DSpineCausal}, {Direction: DSpineCausal}})
	cases := []struct {
		name   string
		mutate func(*DSpineCheckpointContract)
		want   error
	}{
		{"version zero", func(c *DSpineCheckpointContract) { c.Version = 0 }, ErrDSpineContractVersion},
		{"version two", func(c *DSpineCheckpointContract) { c.Version = 2 }, ErrDSpineContractVersion},
		{"terminal enabled", func(c *DSpineCheckpointContract) { c.TerminalEnabled = true }, ErrDSpineContractTerminalEnabled},
		{"dependent writes", func(c *DSpineCheckpointContract) { c.WritesIndependent = false }, ErrDSpineContractWritesDependent},
		{"non-serial decode", func(c *DSpineCheckpointContract) { c.SerialDecode = false }, ErrDSpineContractDecodeNonSerial},
		{"co-located selector", func(c *DSpineCheckpointContract) { c.SelectorSeparated = false }, ErrDSpineContractSelectorCoLocated},
		{"missing final write", func(c *DSpineCheckpointContract) { c.FinalLayerWrite = -1 }, ErrDSpineContractFinalWrite},
		{"out-of-range final write", func(c *DSpineCheckpointContract) { c.FinalLayerWrite = 3 }, ErrDSpineContractFinalWrite},
		{"disagreeing refinement widths", func(c *DSpineCheckpointContract) { c.RefinementWidths = []int{4, 8} }, ErrDSpineContractRefinementWidths},
		{"no refinement widths", func(c *DSpineCheckpointContract) { c.RefinementWidths = nil }, ErrDSpineContractRefinementWidths},
		{"top-k below range", func(c *DSpineCheckpointContract) { c.TopK = 1 }, ErrDSpineContractTopK},
		{"top-k above range", func(c *DSpineCheckpointContract) { c.TopK = 65 }, ErrDSpineContractTopK},
		{"nonzero terminal projection despite disabled terminal", func(c *DSpineCheckpointContract) {
			c.TerminalProjection = []float32{0, 0, 1}
		}, ErrDSpineContractTerminalProjection},
	}
	for _, tc := range cases {
		cfg := base
		tc.mutate(&cfg.Checkpoint)
		if _, err := NewDSpineBridge(cfg, w); !errors.Is(err, tc.want) {
			t.Fatalf("%s: want %v, got %v", tc.name, tc.want, err)
		}
	}
	// Control: the supported contract admits.
	if _, err := NewDSpineBridge(base, w); err != nil {
		t.Fatalf("supported contract must admit, got %v", err)
	}
	// A runtime top-k that disagrees with the checkpoint width refuses too.
	cfg := base
	cfg.TopK = 8
	if _, err := NewDSpineBridge(cfg, w); !errors.Is(err, ErrDSpineContractRefinementWidths) {
		t.Fatalf("runtime/checkpoint width disagreement must refuse, got %v", err)
	}
}

// TestDSpineRefinementLatticeFeedsWalk pins acceptance criterion 3: the
// refinement table has the shared [slots, K predecessor, K candidate] shape and
// the greedy lattice walk consumes it unchanged.
func TestDSpineRefinementLatticeFeedsWalk(t *testing.T) {
	ctx := context.Background()
	w := dspineTestWeights(t)
	layers := []DSpineLayerSpec{
		{Direction: DSpineCausal},
		{Direction: DSpineSlidingWindow, Window: 4},
		{Direction: DSpineCausal},
	}
	d, err := NewDSpineBridge(dspineTestConfig(layers), w)
	if err != nil {
		t.Fatalf("admission refused a valid config: %v", err)
	}
	const anchor, cap = 7, 5
	trace, err := d.Forward(ctx, anchor, cap+1)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	tab, err := d.Refine(trace)
	if err != nil {
		t.Fatalf("refine: %v", err)
	}
	slots := cap
	k := dspineTestTopK
	if tab.K != k {
		t.Fatalf("lattice K = %d, want %d", tab.K, k)
	}
	if len(tab.Cells) != slots || len(tab.Senders) != slots || len(tab.Candidates) != slots {
		t.Fatalf("lattice holds %d/%d/%d slot slices, want %d each", len(tab.Cells), len(tab.Senders), len(tab.Candidates), slots)
	}
	for s := 0; s < slots; s++ {
		if len(tab.Senders[s]) != k || len(tab.Candidates[s]) != k || len(tab.Cells[s]) != k {
			t.Fatalf("slot %d: sender/candidate/cell widths %d/%d/%d, want %d", s, len(tab.Senders[s]), len(tab.Candidates[s]), len(tab.Cells[s]), k)
		}
		for si := 0; si < k; si++ {
			if len(tab.Cells[s][si]) != k {
				t.Fatalf("slot %d sender %d: %d cells, want %d", s, si, len(tab.Cells[s][si]), k)
			}
		}
	}
	// Slot 1 sends the anchor's exact code: its backbone row already used that
	// same code, so the anchor-slice scores equal the backbone logits exactly.
	if tab.Senders[0][0] != anchor {
		t.Fatalf("slot 1 sender[0] = %d, want anchor %d", tab.Senders[0][0], anchor)
	}
	for ci, c := range tab.Candidates[0] {
		if tab.Cells[0][0][ci] != trace.Logits[1][c] {
			t.Fatalf("slot 1 candidate %d: refined %v != backbone logit %v", c, tab.Cells[0][0][ci], trace.Logits[1][c])
		}
	}
	// Sentinel senders score -Inf.
	for si := 1; si < k; si++ {
		if tab.Senders[0][si] != -1 {
			t.Fatalf("slot 1 sender[%d] = %d, want sentinel -1", si, tab.Senders[0][si])
		}
		for ci := 0; ci < k; ci++ {
			if !math.IsInf(float64(tab.Cells[0][si][ci]), -1) {
				t.Fatalf("slot 1 sentinel sender %d cell %d = %v, want -Inf", si, ci, tab.Cells[0][si][ci])
			}
		}
	}
	// The greedy walk over the lattice matches the wired Draft path.
	path, err := tab.Walk(anchor)
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	drafted, err := d.Draft(ctx, anchor, cap)
	if err != nil {
		t.Fatalf("draft: %v", err)
	}
	if len(path) != cap || len(drafted) != cap {
		t.Fatalf("walk/draft lengths %d/%d, want %d", len(path), len(drafted), cap)
	}
	for i := 0; i < cap; i++ {
		if path[i] != drafted[i] {
			t.Fatalf("slot %d: lattice walk %d != draft %d", i, path[i], drafted[i])
		}
	}
}

// TestDSpineDisabledBridgeInert pins the quarantine fallback: the zero-value
// config never engages, so an absent bridge keeps plain DFlash behavior.
func TestDSpineDisabledBridgeInert(t *testing.T) {
	var zero DSpineBridgeConfig
	if err := zero.Validate(); !errors.Is(err, ErrDSpineDisabled) {
		t.Fatalf("zero config must report %v, got %v", ErrDSpineDisabled, err)
	}
	if _, err := NewDSpineBridge(zero, dspineTestWeights(t)); !errors.Is(err, ErrDSpineDisabled) {
		t.Fatalf("zero config must refuse admission, got %v", err)
	}
}
