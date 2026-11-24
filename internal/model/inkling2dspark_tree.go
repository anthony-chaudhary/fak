package model

import (
	"fmt"
	"math"
	"sort"
)

// Markov beam-tree drafting, adapted from Niko1221/Strata (MIT) at
// c6f6427d9de9: the backbone hidden rows depend only on the anchor and mask
// block, so one backbone pass serves every beam; only the recurrent Markov
// head runs per beam.

type inkling2DSparkBeam struct {
	state []float32
	prev  int
	logp  float64
	node  int // tree index of the beam's last token, -1 at the anchor
}

// DraftTree proposes a token tree after anchor (the token at pos): at each of
// the Block-1 depths every live beam offers its width best next tokens, and
// the width best by cumulative draft log-probability become tree nodes and
// the next beams. Nodes are ordered by depth, then by descending score, so a
// node's parent always precedes it. maxNodes bounds the tree (the verify
// cost); width 1 reproduces Draft's greedy chain.
func (d *Inkling2DSparkDraft) DraftTree(anchor, pos, width, maxNodes int) ([]Inkling2TreeNode, error) {
	hp := d.HP
	if width < 1 || maxNodes < 1 {
		return nil, fmt.Errorf("inkling2 dspark: tree width %d and node budget %d must be positive", width, maxNodes)
	}
	if len(d.positions) == 0 || d.positions[len(d.positions)-1] >= pos {
		return nil, fmt.Errorf("inkling2 dspark: anchor position %d needs committed context before it", pos)
	}
	hidden := d.backbone(anchor, pos)
	fRNN := d.fRNN[len(d.fRNN)-1]
	beams := []inkling2DSparkBeam{{state: make([]float32, hp.MarkovRank), prev: anchor, node: -1}}
	var nodes []Inkling2TreeNode
	type candidate struct {
		beam, rank, token int
		logp              float64
	}
	for i := 1; i < hp.Block && len(nodes) < maxNodes; i++ {
		var cands []candidate
		next := make([][]float32, len(beams))
		for b := range beams {
			state, logits := d.markovStep(beams[b].state, beams[b].prev, hidden[i], fRNN)
			next[b] = state
			norm := inkling2LogSumExp(logits)
			for rank, tok := range inkling2TopK(logits, min(width, len(logits))) {
				cands = append(cands, candidate{b, rank, tok, beams[b].logp + float64(logits[tok]) - norm})
			}
		}
		sort.SliceStable(cands, func(a, b int) bool { return cands[a].logp > cands[b].logp })
		keep := min(width, len(cands), maxNodes-len(nodes))
		live := make([]inkling2DSparkBeam, 0, keep)
		for _, c := range cands[:keep] {
			nodes = append(nodes, Inkling2TreeNode{Token: c.token, Parent: beams[c.beam].node})
			live = append(live, inkling2DSparkBeam{state: next[c.beam], prev: c.token, logp: c.logp, node: len(nodes) - 1})
		}
		beams = live
	}
	return nodes, nil
}

// markovStep is one step of Draft's recurrent head: it advances a copy of
// state from prev and returns it with the unpadded draft logits.
func (d *Inkling2DSparkDraft) markovStep(state []float32, prev int, h, fRNN []float32) ([]float32, []float32) {
	hp := d.HP
	r := hp.MarkovRank
	emb := make([]float32, r)
	d.W.MarkovW1.Row(prev, emb)
	in := make([]float32, 0, d.W.RnnProj.Cols())
	in = append(append(append(append(in, state...), emb...), h...), fRNN...)
	raw := inkling2MatVec(d.W.RnnProj, in)
	out := append([]float32(nil), state...)
	latent := make([]float32, r)
	for c := 0; c < r; c++ {
		raw[c] += d.W.RnnBias[c]
		raw[r+c] += d.W.RnnBias[r+c]
		raw[2*r+c] += d.W.RnnBias[2*r+c]
		gate := sigmoid32(raw[c])
		cand := float32(math.Tanh(float64(raw[r+c])))
		out[c] = gate*out[c] + (1-gate)*cand
		latent[c] = float32(math.Tanh(float64(raw[2*r+c])))
	}
	y := inkling2MatVec(d.W.HiddenOut, latent)
	for j := range y {
		y[j] += h[j] + d.W.HiddenOutBias[j]
	}
	logits := make([]float32, d.W.Output.Rows())
	d.W.Output.MatVec(logits, y)
	return out, logits[:hp.VocabUnpadded]
}

func inkling2LogSumExp(v []float32) float64 {
	m := math.Inf(-1)
	for _, x := range v {
		m = math.Max(m, float64(x))
	}
	var s float64
	for _, x := range v {
		s += math.Exp(float64(x) - m)
	}
	return m + math.Log(s)
}
