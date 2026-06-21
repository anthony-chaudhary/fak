package model

import (
	"math"
	"sort"
)

// logitlens.go — the "logit lens": project the residual stream after every layer
// through the model's own final-norm + LM head, so you can read off what token the
// model would predict if decoding stopped at that depth. This is the data source for
// the visual next-token debugger (cmd/lensviz): it turns one Forward pass into a
// per-layer × position view of how the prediction crystallizes through depth.
//
// The projection is exactly the one Forward uses for the real (final-layer) logits —
// finalNorm + headName + the resident kernel + logitScaleInPlace — applied to each
// intermediate Hidden[l] instead of only the last. For tied-embedding models that is
// the textbook logit lens; for untied heads it is the same head matrix, which is the
// standard and only sensible choice.

// LayerLogits projects the residual stream captured in act at the given position
// through the final norm + LM head for every layer, returning the per-layer logits.
//
// out[l] is the vocab-sized logit vector obtained by decoding act.Hidden[l] (the
// residual stream entering layer l, with l==0 the embedding output and the last index
// the post-final-block stream) as if it were the final hidden state. len(out) ==
// len(act.Hidden) == NumLayers+1. out[len-1] equals the model's real logits at pos
// (act.Logits[pos]) up to floating-point reassociation, since it runs the identical
// projection on the identical input.
func (m *Model) LayerLogits(act *Activations, pos int) [][]float32 {
	if act == nil || pos < 0 || pos >= act.Seq {
		return nil
	}
	cfg := m.Cfg
	H := cfg.HiddenSize
	mat := residentKernel{m}
	head := m.headName()
	out := make([][]float32, len(act.Hidden))
	for l, hidden := range act.Hidden {
		// hidden is flattened [seq*hidden]; slice out this position's vector.
		hv := hidden[pos*H : (pos+1)*H]
		xf := m.finalNorm(hv)
		logits := mat.mul(head, mat.prep(xf), cfg.VocabSize, H)
		logitScaleInPlace(logits, cfg)
		out[l] = logits
	}
	return out
}

// TokenProb is one ranked next-token candidate: its vocab id, raw logit, and softmax
// probability over the full vocabulary at that layer/position.
type TokenProb struct {
	ID    int
	Logit float32
	Prob  float32
}

// TopK returns the k highest-probability tokens for one logit vector, sorted by
// descending probability. Probabilities are the full-vocabulary softmax of logits, so
// they sum to 1 across the whole vocab even though only k are returned. k is clamped to
// [1, vocab]; a nil/empty logits returns nil.
func TopK(logits []float32, k int) []TokenProb {
	n := len(logits)
	if n == 0 || k <= 0 {
		return nil
	}
	if k > n {
		k = n
	}
	// Softmax over the full vocab (numerically stable).
	var max float32 = logits[0]
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	// Exact exp here (not the fast approximation): this is an off-hot-path debugger and
	// the displayed probabilities must be trustworthy, not within forward-pass tolerance.
	probs := make([]float32, n)
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - max))
		probs[i] = float32(e)
		sum += e
	}
	if sum > 0 {
		inv := float32(1 / sum)
		for i := range probs {
			probs[i] *= inv
		}
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	// Partial sort is enough, but vocab-sized full sort is simple and fast enough for
	// an interactive, off-hot-path debugger.
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	out := make([]TokenProb, k)
	for i := 0; i < k; i++ {
		id := idx[i]
		out[i] = TokenProb{ID: id, Logit: logits[id], Prob: probs[id]}
	}
	return out
}
