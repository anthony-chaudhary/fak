package model

import (
	"math"
	"sort"
)

// minimax_m3.go — MiniMax-M3 "MiniMax Sparse Attention" (MSA) wired cacheless forward.
//
// This is the MSA analogue of glm_dsa.go (GLM-5.2 Dynamic Sparse Attention). It wires
// the real MiniMax-M3 text-decoder forward for the cacheless full-prefill Forward:
//
//   - full_attention layers run the standard dense causal GQA path (m.attnSeq) — the
//     SAME per-head qk-norm + partial-RoPE attention the existing axes already prove,
//     so they need no new code.
//   - minimax_m3_sparse layers run the lightning-indexer block-sparse path: a separate
//     index branch (self_attn.indexer.{q_proj,k_proj,q_norm,k_norm}) scores every key,
//     max-pools the per-key scores into blocks of IndexBlockSize keys, then PER INDEX
//     HEAD (= per GQA group) forces the always-on IndexLocalBlocks local window to +inf
//     and keeps the top-IndexTopKBlocks blocks (a fixed count: a forced local block
//     displaces the lowest-scored block — HF's scatter-to-inf-then-topk, NOT a union),
//     broadcasts the per-block choice back onto keys, and runs GQA softmax over only the
//     admitted keys. Block max-pool + key broadcast reuse the host-witnessed msa_index.go
//     helpers (msaBlockScores / msaSelectedKeyPositions); the HF-exact fixed-count
//     selection is minimaxSelectBlocks here, and the MSA-specific numeric code is the
//     learned indexer projection and the per-group masked GQA.
//   - the FFN is the SwiGLU-OAI gated MoE (minimaxMoeFFN): sigmoid routing + score
//     correction bias + top-k + renorm + routed-scaling (reusing glmRoute with no
//     expert grouping, which matches MiniMax's router exactly) over OAI-gated experts,
//     plus an always-on shared expert.
//
// FIDELITY / WITNESS STATUS. The tensor names, the indexer scoring formula
// (scores = idx_q · idx_kᵀ, k-side single head, RoPE on both, no extra scale), the
// per-index-head=per-GQA-group selection, the partial RoPE (first rotary_dim dims),
// the per-head qk-norm, and the SwiGLU-OAI gate
// (glu = clamp(gate,≤limit)·σ(clamp(gate,≤limit)·alpha); out = (clamp(up,±limit)+1)·glu)
// are taken VERBATIM from the HuggingFace `minimax_m3_vl` modeling reference. The
// structural correctness of the wiring is witnessed host-side (synthetic finite
// forward + the budget-covers-all-blocks ⇒ dense-GQA superset reduction + the OAI gate
// unit value). The NUMERIC parity against a real checkpoint is gated on a tiny
// `minimax_m3` HF oracle (.cache/oracle-minimax), exactly as every other family's
// production oracle is — see TestOptionalMiniMaxM3Oracle* in oracle_test.go. Some
// details that a real export will confirm and which are inferred from the public
// reference (not a local checkpoint) are flagged in comments below.

// layerMiniMax applies one MiniMax-M3 decoder block to x in place under the PreNorm
// topology MiniMax uses: attention (dense GQA on full_attention layers, lightning-
// indexer block-sparse on minimax_m3_sparse layers) + SwiGLU-OAI MoE FFN, each with a
// pre-norm and a residual add. It mirrors layerGLMDsa.
func (m *Model) layerMiniMax(l int, x [][]float32, rp rope) {
	cfg := m.Cfg
	eps := float32(cfg.RMSNormEps)
	attnNorm := m.attentionNorms(l)
	mlpNorm := m.mlpNorms(l)
	attnSub := func(xn [][]float32) [][]float32 { return m.minimaxAttnSeq(l, xn, rp) }
	mlpSub := func(xn [][]float32) [][]float32 { return m.mlpSeq(l, xn) }
	composeSeqSublayer(cfg.BlockTopology, x, attnNorm, eps, cfg, attnSub)
	composeSeqSublayer(cfg.BlockTopology, x, mlpNorm, eps, cfg, mlpSub)
}

// minimaxAttnSeq routes one layer's whole-sequence attention: a minimax_m3_sparse
// layer takes the lightning-indexer block-sparse path; a full_attention layer reuses
// the dense causal GQA path (m.attnSeq), which already applies per-head qk-norm and
// partial RoPE from the existing config axes.
func (m *Model) minimaxAttnSeq(l int, xn [][]float32, rp rope) [][]float32 {
	if m.Cfg.isMSALayer(l) {
		return m.minimaxMSAAttnSeq(l, xn, rp)
	}
	return m.attnSeq(l, xn, rp)
}

// minimaxMSAAttnSeq is the MiniMax-M3 MSA attention sublayer over a whole sequence of
// already-normalized inputs. It projects the main-branch q/k/v (per-head qk-norm +
// partial RoPE, identical to the dense path), computes the lightning-indexer per-GQA-
// group block selection, and runs GQA softmax over only the admitted causal keys.
func (m *Model) minimaxMSAAttnSeq(l int, xn [][]float32, rp rope) [][]float32 {
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	seq := len(xn)
	p := func(s string) string { return layerName(l, s) }

	// main branch q/k/v — same per-head qk-norm (post-projection, pre-RoPE) + partial
	// RoPE the dense GQA path uses, so a full vs sparse layer differ ONLY in which keys
	// the softmax sees.
	q := make([][]float32, seq)
	k := make([][]float32, seq)
	v := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		q[t] = m.residentMatRows(p("self_attn.q_proj.weight"), xn[t], nH*hd, H)
		k[t] = m.residentMatRows(p("self_attn.k_proj.weight"), xn[t], nKV*hd, H)
		v[t] = m.residentMatRows(p("self_attn.v_proj.weight"), xn[t], nKV*hd, H)
		m.applyProjBias(l, q[t], k[t], v[t])
		m.applyLayerQKNorm(l, q[t], k[t])
		ropeRowQKInto(q[t], k[t], rp.cos[t], rp.sin[t], hd, nH, nKV)
	}

	// lightning-indexer block selection, one admitted-key set per (GQA group, query).
	sel := m.minimaxIndexerSelect(l, xn, rp)

	scale := cfg.attnScale()
	attnOut := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		attnOut[t] = make([]float32, nH*hd)
		for h := 0; h < nH; h++ {
			// index_n_heads == num_key_value_heads, so each GQA group has exactly one
			// index head, and every query head in the group shares its key selection.
			kvh := h / grp
			admitted := sel[kvh][t]
			qh := q[t][h*hd : (h+1)*hd]
			scores := make([]float32, len(admitted))
			for i, kp := range admitted {
				scores[i] = dot(qh, k[kp][kvh*hd:(kvh+1)*hd]) * scale
			}
			softmaxInPlace(scores)
			o := attnOut[t][h*hd : (h+1)*hd]
			for i, kp := range admitted {
				vh := v[kp][kvh*hd : (kvh+1)*hd]
				w := scores[i]
				for d := 0; d < hd; d++ {
					o[d] += w * vh[d]
				}
			}
		}
		attnOut[t] = m.residentMatRows(p("self_attn.o_proj.weight"), attnOut[t], H, nH*hd)
		m.addBiasIfPresent(attnOut[t], p("self_attn.o_proj.bias"))
	}
	return attnOut
}

// minimaxIndexerSelect runs the MiniMax-M3 lightning indexer over a whole sequence and
// returns, per index head (== GQA group) and per query, the ascending set of admitted
// causal key positions. The index branch projects a low-dim per-head query
// (self_attn.indexer.q_proj, IndexNHeads heads of IndexHeadDim) and a single shared key
// (self_attn.indexer.k_proj, one head of IndexHeadDim), RMS-norms both (q_norm/k_norm),
// applies partial RoPE (first rotary_dim dims, the same rope row as the main branch),
// scores every causal key by idx_q·idx_k, max-pools to blocks of IndexBlockSize, and
// selects the top-IndexTopKBlocks blocks plus the always-on IndexLocalBlocks local
// window (block-level causality) — the selection helpers in msa_index.go.
func (m *Model) minimaxIndexerSelect(l int, xn [][]float32, rp rope) [][][]int {
	blocks := m.minimaxIndexerSelectBlocks(l, xn, rp)
	seq := len(xn)
	blockSize := m.Cfg.IndexBlockSize
	keyPos := make([]int, seq)
	for i := range keyPos {
		keyPos[i] = i
	}
	out := make([][][]int, len(blocks))
	for g := range blocks {
		out[g] = make([][]int, seq)
		for qpos := 0; qpos < seq; qpos++ {
			keys, ok := msaSelectedKeyPositions(keyPos, qpos, blockSize, blocks[g][qpos])
			if !ok {
				panic("model: minimax_m3 indexer key broadcast failed")
			}
			out[g][qpos] = keys
		}
	}
	return out
}

// minimaxIndexerSelectBlocks runs the indexer and returns, per index head (== GQA group)
// and per query, the ASCENDING set of selected causal key-BLOCK indices (the HF-exact
// fixed-count top-k with the local window forced in; see minimaxSelectBlocks). It is the
// selection the HF MSA trace records, so the optional oracle test can reproduce it
// directly; minimaxIndexerSelect broadcasts these blocks back onto key positions for the
// actual attention.
func (m *Model) minimaxIndexerSelectBlocks(l int, xn [][]float32, rp rope) [][][]int {
	cfg := m.Cfg
	nIdx := cfg.IndexNHeads
	seq := len(xn)
	blockSize := cfg.IndexBlockSize
	topK := cfg.IndexTopKBlocks
	local := cfg.IndexLocalBlocks
	idxQ, idxK := m.minimaxIndexerProject(l, xn, rp)

	keyPos := make([]int, seq)
	for i := range keyPos {
		keyPos[i] = i
	}
	out := make([][][]int, nIdx)
	for g := 0; g < nIdx; g++ {
		out[g] = make([][]int, seq)
		for qpos := 0; qpos < seq; qpos++ {
			scores := make([]float64, seq) // entries > qpos are unused (causal skip in msaBlockScores)
			for kk := 0; kk <= qpos; kk++ {
				scores[kk] = float64(dot(idxQ[qpos][g], idxK[kk]))
			}
			bs, ok := msaBlockScores(scores, keyPos, qpos, blockSize)
			if !ok {
				panic("model: minimax_m3 indexer block-score failed")
			}
			out[g][qpos] = minimaxSelectBlocks(bs, qpos, blockSize, topK, local)
		}
	}
	return out
}

// minimaxSelectBlocks reproduces HF's MiniMax-M3 lightning-indexer block selection
// EXACTLY: force the always-on local-window blocks (the query's own block and the
// localBlocks-1 immediately before it) to +inf, then keep the top-min(topKBlocks,
// #causal blocks) by score (ties broken on the smaller block index). This differs from
// the scaffold's generic msaSelectBlocks union: HF scatters the local blocks to +inf and
// then takes a FIXED-count topk, so a forced local block DISPLACES the lowest-scored
// block rather than adding to the set. Returns the selected block indices ascending.
func minimaxSelectBlocks(blockScores map[int]float64, qpos, blockSize, topKBlocks, localBlocks int) []int {
	qb := qpos / blockSize
	type cand struct {
		b int
		s float64
	}
	cands := make([]cand, 0, len(blockScores))
	for b, s := range blockScores {
		if localBlocks > 0 && b <= qb && b >= qb-localBlocks+1 {
			s = math.Inf(1) // always-on local window forced visible
		}
		cands = append(cands, cand{b, s})
	}
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].s == cands[j].s {
			return cands[i].b < cands[j].b
		}
		return cands[i].s > cands[j].s
	})
	n := topKBlocks
	if n > len(cands) {
		n = len(cands)
	}
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, cands[i].b)
	}
	sort.Ints(out)
	return out
}

// minimaxIndexerProject projects the lightning-indexer query (IndexNHeads heads of
// IndexHeadDim) and the single shared key head from each normalized input row, applies
// q_norm/k_norm and partial RoPE (first rotary_dim dims), and returns idxQ[t][head] and
// idxK[t]. The score idx_q·idx_k carries no extra scale (verbatim HF: matmul only).
func (m *Model) minimaxIndexerProject(l int, xn [][]float32, rp rope) ([][][]float32, [][]float32) {
	cfg := m.Cfg
	H := cfg.HiddenSize
	nIdx := cfg.IndexNHeads
	idxDim := cfg.IndexHeadDim
	rot := cfg.rotaryDim()
	seq := len(xn)
	eps := float32(cfg.RMSNormEps)
	ip := func(s string) string { return layerName(l, "self_attn.indexer."+s) }
	qNorm := m.tensor(ip("q_norm.weight"))
	kNorm := m.tensor(ip("k_norm.weight"))

	idxQ := make([][][]float32, seq)
	idxK := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		qf := m.residentMatRows(ip("q_proj.weight"), xn[t], nIdx*idxDim, H)
		kf := m.residentMatRows(ip("k_proj.weight"), xn[t], idxDim, H)
		idxQ[t] = make([][]float32, nIdx)
		for h := 0; h < nIdx; h++ {
			head := qf[h*idxDim : (h+1)*idxDim]
			applyRMSNormInPlaceCfg(head, qNorm, eps, cfg)
			applyRopeRow(head[:rot], rp.cos[t], rp.sin[t])
			idxQ[t][h] = head
		}
		applyRMSNormInPlaceCfg(kf, kNorm, eps, cfg)
		applyRopeRow(kf[:rot], rp.cos[t], rp.sin[t])
		idxK[t] = kf
	}
	return idxQ, idxK
}

// minimaxIndexerNormalizedBlocks is the optional-oracle hook: from a layer's RAW input
// hidden (pre-input_layernorm), normalize it the way the forward does and return the
// per-(index head, query) selected key-block indices, so the MSA-trace oracle test can
// compare the Go selection against the HF lightning-indexer trace.
func (m *Model) minimaxIndexerNormalizedBlocks(l int, hidden []float32, seq int) [][][]int {
	cfg := m.Cfg
	H := cfg.HiddenSize
	w := m.tensor(layerName(l, "input_layernorm.weight"))
	xn := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xn[t] = normCfg(hidden[t*H:(t+1)*H], w, m.tensorOptional(layerName(l, "input_layernorm.bias")), float32(cfg.RMSNormEps), cfg)
	}
	return m.minimaxIndexerSelectBlocks(l, xn, newRopeForLayer(cfg, l, seq))
}

// minimaxMoeFFN is the MiniMax-M3 SwiGLU-OAI MoE FFN: sigmoid router + score-
// correction bias + top-k + renorm + routed-scaling (glmRoute, which reduces to exactly
// MiniMax's router when no expert grouping is configured) over OAI-gated experts, plus
// an always-on shared expert. The routed_scaling_factor multiplies only the routed
// experts (folded into glmRoute's pick weights), never the shared expert — matching HF.
type minimaxMoeFFN struct{}

func (minimaxMoeFFN) apply(m *Model, layer int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	delta := make([]float32, H)
	for _, pk := range glmRoute(m, layer, xn, mat) {
		out := expertMiniMaxOAI(m, layer, pk.expert, xn, mat)
		for i := 0; i < H; i++ {
			delta[i] += pk.weight * out[i]
		}
	}
	if cfg.NSharedExperts > 0 && m.has(layerName(layer, "mlp.shared_experts.gate_proj.weight")) {
		shared := minimaxSharedExpertOAI(m, layer, xn, mat)
		for i := 0; i < H; i++ {
			delta[i] += shared[i]
		}
	}
	return delta
}

// expertMiniMaxOAI runs one routed expert's SwiGLU-OAI gate over xn. The OAI ("OpenAI")
// gated SwiGLU clamps the gate to swiglu_limit and the up branch to ±swiglu_limit, then
// out = (up+1) * (gate * sigmoid(gate*swiglu_alpha)). It generalizes the gpt-oss clamped
// SwiGLU (which fixes limit=7, alpha=1.702) to MiniMax's configured swiglu_limit/alpha.
func expertMiniMaxOAI(m *Model, layer, expert int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	g := mat.mul(expertName(layer, expert, "gate_proj.weight"), xn, I, H)
	u := mat.mul(expertName(layer, expert, "up_proj.weight"), xn, I, H)
	m.addBiasIfPresent(g, expertName(layer, expert, "gate_proj.bias"))
	m.addBiasIfPresent(u, expertName(layer, expert, "up_proj.bias"))
	swigluOAIInPlace(g, u, cfg)
	out := mat.mul(expertName(layer, expert, "down_proj.weight"), mat.prep(g), H, I)
	m.addBiasIfPresent(out, expertName(layer, expert, "down_proj.bias"))
	return out
}

// minimaxSharedExpertOAI runs the always-on shared expert (a dense SwiGLU-OAI MLP of
// width SharedIntermediateSize) over xn. It is added to the routed-expert sum unscaled.
func minimaxSharedExpertOAI(m *Model, layer int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	I := cfg.SharedIntermediateSize
	if I == 0 {
		I = cfg.IntermediateSize
	}
	prefix := layerName(layer, "mlp.shared_experts.")
	g := mat.mul(prefix+"gate_proj.weight", xn, I, H)
	u := mat.mul(prefix+"up_proj.weight", xn, I, H)
	swigluOAIInPlace(g, u, cfg)
	return mat.mul(prefix+"down_proj.weight", mat.prep(g), H, I)
}

// swigluOAIInPlace applies MiniMax-M3's SwiGLU-OAI gate to g (gate) and u (up) in place,
// writing the gated activation into g: clamp the gate to swiglu_limit and the up branch
// to ±swiglu_limit, then g = (up+1) * (gate * sigmoid(gate*alpha)). A zero swiglu_limit
// means "no clamp"; a zero swiglu_alpha falls back to alpha=1.702 (the gpt-oss/OAI
// default) so a partially-specified export is still well-defined.
func swigluOAIInPlace(g, u []float32, cfg Config) {
	limit := float32(cfg.SwigluLimit)
	alpha := float32(cfg.SwigluAlpha)
	if alpha == 0 {
		alpha = 1.702
	}
	for i := range g {
		gate := g[i]
		up := u[i]
		if limit > 0 {
			if gate > limit {
				gate = limit
			}
			if up > limit {
				up = limit
			} else if up < -limit {
				up = -limit
			}
		}
		glu := gate * sigmoid(gate*alpha)
		g[i] = (up + 1) * glu
	}
}
