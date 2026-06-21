package model

import "fmt"

// tensor_parallel_forward.go — WIRING the tensor-parallel decomposition into the LIVE
// forward path. tensor_parallel.go / tensor_parallel_attn.go proved the Megatron FFN and
// attention blocks as STANDALONE primitives over explicit weights, deliberately OMITTING
// RoPE, qk-norm, attention softcap, ALiBi, the attention output gate, q/k/v/o biases, and
// the sliding window. This file closes the step GLM-5.2-NATIVE-ENGINE-GAP names as the
// remaining work on the TP lever: run the SHARDED decomposition over the REAL model layer —
// every feature the live attnSeq / dense SwiGLU FFN apply RE-ENTERS — and prove it equals
// the live (HF-oracle-gated) path up to the single AllReduce's documented round-off.
//
// WHY THIS TRANSITIVELY INHERITS THE HF ORACLE. The gate chain (tensor_parallel_forward_test.go):
//
//	ForwardTP(ranks=1)  ==(bit-exact, max|Δ|=0)        Forward          [this file's tests]
//	Forward             ==(HF tolerance)               HuggingFace      [oracle_test.go]
//	ForwardTP(ranks=N)  ==(AllReduce reassociation)    ForwardTP(ranks=1)
//
// so ForwardTP(ranks=N) reproduces HF up to the documented reassociation round-off. The
// ranks=1 leg is BIT-EXACT because a 1-shard plan makes every AllGather a no-op concat and
// every AllReduce a single matRows over the full width — the row-parallel reassociation,
// the only non-bit-exact step, vanishes. And the per-rank projection slices the weight's
// OUTPUT ROWS, so it is the SAME fdot-per-row as the monolith; RoPE / qk-norm / the gate /
// softcap are all head-local (or per-element), so each rank's pre-o_proj band stays
// BIT-EXACT vs the monolith's matching head slice — the column-parallel disjointness witness
// (TestTPForwardColumnParallelBandDisjoint), now WITH the omitted features re-entered.
//
// HEAD-PARALLEL ATTENTION (Megatron): shard the nKV kv-head GROUPS across ranks; query heads
// follow under GQA (head h -> kv head h/grp). Each rank projects+attends only its heads
// (column-parallel Q/K/V, NO collective — heads are independent), then the output projection
// is ROW-parallel over the concatenated head dim with exactly ONE AllReduceSum; the o_proj
// bias is added once, AFTER the reduce. FFN: shard the intermediate dim I; gate/up are
// column-parallel (the activation band never gathered), down is row-parallel with one
// AllReduceSum; the down bias is added once after the reduce.
//
// SCOPE — honest, fail-closed. This wires the STANDARD GQA attention (attnSeq) + DENSE
// SwiGLU FFN (the path the landed primitives model) for the composeSeqSublayer topologies
// (PreNorm / PostNorm / SandwichNorm). It does NOT yet shard:
//   - GLM-5.2's MLA + Dynamic-Sparse attention — its latent KV is shared across heads, so
//     head-parallel TP is a DIFFERENT decomposition (a later sub-lever);
//   - MoE FFN — expert-parallelism is a different decomposition (a later sub-lever);
//   - the ParallelResidual graph (NeoX/Cohere), the DenseMLP activation-MLP, and
//     linear-attention (GDN) layers.
// ForwardTP fails closed on each rather than silently mis-serving it. It also requires
// f32-resident projection weights (it slices raw output rows); a quant-resident model
// returns an error — quant-aware sharding is a later lever. The real NCCL/RDMA Collective
// behind LocalCollective is the OTHER remaining lever and needs the DGX hardware.

// TPConfig parameters the wired tensor-parallel forward. AttnRanks shards the attention over
// the kv-head groups (nKV); FFNRanks shards the FFN over the intermediate dim (I). They may
// differ — a real plan often shards attention by head and the FFN by intermediate width
// independently. Coll is the AllGather/AllReduce seam (nil -> LocalCollective, the single-box
// bit-exact default; a real NCCL/RDMA collective is a two-method swap).
type TPConfig struct {
	AttnRanks int
	FFNRanks  int
	Coll      Collective
}

// clampRanks bounds a requested rank count to [1, dim]: a dimension cannot be sharded across
// more ranks than it has indices (NewTPPlan would reject it), and 0/negative means "1".
func clampRanks(r, dim int) int {
	if r < 1 {
		return 1
	}
	if r > dim {
		return dim
	}
	return r
}

// forwardTPSupported reports (as an error) the first reason this model's architecture is not
// yet covered by the wired TP path, so ForwardTP fails closed instead of mis-serving an arch
// whose TP decomposition differs. Each rejected case names the later sub-lever it belongs to.
func (m *Model) forwardTPSupported() error {
	cfg := m.Cfg
	switch {
	case cfg.isGLMMoeDsa():
		return fmt.Errorf("model: ForwardTP does not yet shard glm_moe_dsa (MLA+DSA shares a latent KV across heads — head-parallel TP is a separate sub-lever)")
	case cfg.IsMoE():
		return fmt.Errorf("model: ForwardTP does not yet shard MoE FFN (expert-parallel is a separate sub-lever)")
	case cfg.DenseMLP:
		return fmt.Errorf("model: ForwardTP shards dense SwiGLU FFN only (the DenseMLP activation-MLP is a later lever)")
	case cfg.BlockTopology == ParallelResidual:
		return fmt.Errorf("model: ForwardTP does not yet support the ParallelResidual topology (shared-norm two-branch residual is a later lever)")
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			return fmt.Errorf("model: ForwardTP does not shard linear-attention layer %d (the GDN recurrence is a later lever)", l)
		}
	}
	return nil
}

// tpAttnBandPreProj computes the pre-output-projection attention output for the query-head
// band [band.QLo,band.QHi) (mapped onto kv-head band [band.KVLo,band.KVHi)) over the whole
// sequence of already-normalized inputs, applying EXACTLY the live attnSeq feature set —
// projection (sliced raw f32 output rows), q/k/v bias, qk-norm, RoPE (or ALiBi), the sliding
// window, attention softcap, the attention-sink softmax, and the attention output gate — but
// only for this band's heads. Each returned row has width (band.QHi-band.QLo)*HeadDim.
//
// It is BIT-EXACT vs the monolith attnSeq's matching head slice: matRows over the band's
// output rows is the identical fdot-per-row as the monolith over those same rows, and every
// feature is head-local (qk-norm/softcap/gate are per-head; RoPE depends only on position and
// within-head dim, never the head index). It requires f32-resident q/k/v projection weights.
func (m *Model) tpAttnBandPreProj(l int, xn [][]float32, rp rope, band attnHeadBand) ([][]float32, error) {
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	grp := cfg.GroupSize()
	seq := len(xn)
	attnCap := float32(cfg.AttnSoftcap)
	p := func(s string) string { return layerName(l, s) }

	for _, name := range []string{"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"} {
		if !m.has(p(name)) {
			return nil, fmt.Errorf("model: ForwardTP requires f32-resident %s (quant-aware TP sharding is a later lever)", p(name))
		}
	}
	qW := m.tensor(p("self_attn.q_proj.weight"))
	kW := m.tensor(p("self_attn.k_proj.weight"))
	vW := m.tensor(p("self_attn.v_proj.weight"))

	gated := cfg.AttnOutputGate
	bandNQ := band.QHi - band.QLo
	bandNKV := band.KVHi - band.KVLo

	// per-position q,k,v for the band, after projection (+ bias, + qk-norm, + RoPE).
	q := make([][]float32, seq) // [seq][bandNQ*hd]
	k := make([][]float32, seq) // [seq][bandNKV*hd]
	v := make([][]float32, seq)
	var gates [][]float32
	if gated {
		gates = make([][]float32, seq)
	}
	for t := 0; t < seq; t++ {
		x := xn[t]
		if gated {
			// gated q_proj packs [query|gate] per head, so the band's rows are the contiguous
			// span [QLo*2*hd, QHi*2*hd). Split per head into the query and the sigmoid gate,
			// exactly as attnSeq does (just over this band's heads).
			qf := matRows(shardWeightRows(qW, H, band.QLo*2*hd, band.QHi*2*hd), x, bandNQ*2*hd, H)
			qv := make([]float32, bandNQ*hd)
			gv := make([]float32, bandNQ*hd)
			for h := 0; h < bandNQ; h++ {
				copy(qv[h*hd:(h+1)*hd], qf[h*2*hd:h*2*hd+hd])
				copy(gv[h*hd:(h+1)*hd], qf[h*2*hd+hd:h*2*hd+2*hd])
			}
			q[t], gates[t] = qv, gv
		} else {
			q[t] = matRows(shardWeightRows(qW, H, band.QLo*hd, band.QHi*hd), x, bandNQ*hd, H)
		}
		k[t] = matRows(shardWeightRows(kW, H, band.KVLo*hd, band.KVHi*hd), x, bandNKV*hd, H)
		v[t] = matRows(shardWeightRows(vW, H, band.KVLo*hd, band.KVHi*hd), x, bandNKV*hd, H)
		m.tpApplyProjBiasBand(l, q[t], k[t], v[t], band, hd)
		m.tpApplyQKNormBand(l, q[t], k[t], bandNQ, bandNKV)
		if !cfg.Alibi {
			ropeRowQKInto(q[t], k[t], rp.cos[t], rp.sin[t], hd, bandNQ, bandNKV)
		}
	}

	// scaled-dot-product attention, causal, GQA, optionally windowed — identical math to
	// attnSeq, restricted to this band's query heads (GLOBAL head index drives ALiBi and the
	// attention sink, so the per-head behavior is unchanged by sharding).
	W := cfg.windowForLayer(l)
	scale := cfg.attnScale()
	attnOut := make([][]float32, seq) // [seq][bandNQ*hd]
	for t := 0; t < seq; t++ {
		attnOut[t] = make([]float32, bandNQ*hd)
		lo := 0
		if W >= 0 {
			if lo = t - W + 1; lo < 0 {
				lo = 0
			}
		}
		for hl := 0; hl < bandNQ; hl++ {
			gh := band.QLo + hl           // global query head
			localKV := gh/grp - band.KVLo // local kv-head index within this band
			qh := q[t][hl*hd : (hl+1)*hd]
			scores := make([]float32, t+1-lo)
			for j := lo; j <= t; j++ {
				kh := k[j][localKV*hd : (localKV+1)*hd]
				scores[j-lo] = dot(qh, kh)*scale + cfg.alibiScoreBias(gh, j, seq)
			}
			softcapInPlace(scores, attnCap)
			m.softmaxAttentionScores(l, gh, scores)
			o := attnOut[t][hl*hd : (hl+1)*hd]
			for j := lo; j <= t; j++ {
				vh := v[j][localKV*hd : (localKV+1)*hd]
				w := scores[j-lo]
				for d := 0; d < hd; d++ {
					o[d] += w * vh[d]
				}
			}
		}
		if gated {
			gt := gates[t]
			for i := 0; i < bandNQ*hd; i++ {
				attnOut[t][i] *= sigmoidf(gt[i])
			}
		}
	}
	return attnOut, nil
}

// tpApplyProjBiasBand adds the q/k/v projection biases for this band's heads (the matching
// slice of each full bias), mirroring applyProjBias but band-local. Only physically-present
// biases are added, so Llama (no bias) is a no-op and Qwen2 (all three) adds each band slice.
func (m *Model) tpApplyProjBiasBand(l int, q, k, v []float32, band attnHeadBand, hd int) {
	p := func(s string) string { return layerName(l, s) }
	addBand := func(y []float32, name string, headLo int) {
		if !m.has(name) {
			return
		}
		b := m.tensor(name)
		off := headLo * hd
		for i := range y {
			y[i] += b[off+i]
		}
	}
	addBand(q, p("self_attn.q_proj.bias"), band.QLo)
	addBand(k, p("self_attn.k_proj.bias"), band.KVLo)
	addBand(v, p("self_attn.v_proj.bias"), band.KVLo)
}

// tpApplyQKNormBand runs per-head qk-norm over this band's q/k heads using the layer's
// q_norm/k_norm weights (length HeadDim, applied per head), mirroring applyLayerQKNorm with
// the band's head counts. No-op when QKNorm is off. Per-head normalization is head-local, so
// the band result equals the monolith's matching head slice bit-for-bit.
func (m *Model) tpApplyQKNormBand(l int, q, k []float32, bandNQ, bandNKV int) {
	cfg := m.Cfg
	if !cfg.QKNorm {
		return
	}
	p := func(s string) string { return layerName(l, s) }
	eps := cfg.qkNormEps()
	applyQKNormCfg(q, m.tensor(p("self_attn.q_norm.weight")), bandNQ, cfg.HeadDim, eps, cfg)
	applyQKNormCfg(k, m.tensor(p("self_attn.k_norm.weight")), bandNKV, cfg.HeadDim, eps, cfg)
}

// tpAttnLayerPartials computes each rank's [seq][hidden] o_proj partial for the head-sharded
// attention of layer l: per rank, tpAttnBandPreProj over its head band, then the row-parallel
// output projection over that band's head columns (a [hidden] partial per position). The
// per-position AllReduceSum of these partials (tpAttnLayer / tpAttnLayerReference) is the one
// collective. plan.Dim must == nKV. Requires an f32-resident o_proj.
func (m *Model) tpAttnLayerPartials(l int, xn [][]float32, rp rope, plan TPPlan) ([][][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	if plan.Dim != nKV {
		return nil, fmt.Errorf("model: tpAttnLayer plan.Dim = %d, want nKV = %d (shard whole KV-head groups)", plan.Dim, nKV)
	}
	oName := layerName(l, "self_attn.o_proj.weight")
	if !m.has(oName) {
		return nil, fmt.Errorf("model: ForwardTP requires f32-resident %s (quant-aware TP sharding is a later lever)", oName)
	}
	oW := m.tensor(oName) // [hidden, nH*hd]
	qHeadDim := nH * hd
	seq := len(xn)
	rankPartials := make([][][]float32, len(plan.Shards))
	for r, s := range plan.Shards {
		band := attnBandForShard(s, grp)
		attnOut, err := m.tpAttnBandPreProj(l, xn, rp, band)
		if err != nil {
			return nil, err
		}
		oCols := shardWeightColumns(oW, H, qHeadDim, band.QLo*hd, band.QHi*hd)
		bandW := (band.QHi - band.QLo) * hd
		part := make([][]float32, seq)
		for t := 0; t < seq; t++ {
			part[t] = matRows(oCols, attnOut[t], H, bandW)
		}
		rankPartials[r] = part
	}
	return rankPartials, nil
}

// tpAttnLayer is the head-sharded live attention for layer l: it reduces the per-rank o_proj
// partials through the Collective (one AllReduceSum per position) and adds the o_proj bias
// once. Result [seq][hidden], matching attnSeq within the AllReduce round-off (bit-exact at
// ranks=1). A nil collective defaults to LocalCollective.
func (m *Model) tpAttnLayer(l int, xn [][]float32, rp rope, plan TPPlan, coll Collective) ([][]float32, error) {
	parts, err := m.tpAttnLayerPartials(l, xn, rp, plan)
	if err != nil {
		return nil, err
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	return m.tpReduceWithBias(parts, coll, layerName(l, "self_attn.o_proj.bias"))
}

// tpAttnLayerReference is the bit-exact rank-order oracle for tpAttnLayer: the identical
// per-rank partials, summed per position in rank order directly (not through the Collective),
// then the o_proj bias added once. Pinning tpAttnLayer == this at max|Δ|=0 proves the
// collective reduces the o_proj partials in rank order — the row-parallel contract the loose
// vs-monolith round-off bound cannot see.
func (m *Model) tpAttnLayerReference(l int, xn [][]float32, rp rope, plan TPPlan) ([][]float32, error) {
	parts, err := m.tpAttnLayerPartials(l, xn, rp, plan)
	if err != nil {
		return nil, err
	}
	return m.tpReduceRefWithBias(parts, layerName(l, "self_attn.o_proj.bias")), nil
}

// tpFFNLayerPartials computes each rank's [seq][hidden] down-projection partial for the
// intermediate-sharded dense SwiGLU FFN of layer l: per rank, gate/up are column-parallel
// over its I-band (the REAL activation act(g)*u with any gate/up bias, the intermediate never
// gathered), and down is row-parallel over that same I-band (a [hidden] partial). The
// per-position AllReduceSum is the one collective. plan.Dim must == IntermediateSize. Requires
// f32-resident gate/up/down weights.
func (m *Model) tpFFNLayerPartials(l int, xn [][]float32, plan TPPlan) ([][][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	if plan.Dim != I {
		return nil, fmt.Errorf("model: tpFFNLayer plan.Dim = %d, want intermediate = %d", plan.Dim, I)
	}
	p := func(s string) string { return layerName(l, s) }
	for _, name := range []string{"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"} {
		if !m.has(p(name)) {
			return nil, fmt.Errorf("model: ForwardTP requires f32-resident %s (dense SwiGLU TP only; quant/MoE TP are later levers)", p(name))
		}
	}
	gateW := m.tensor(p("mlp.gate_proj.weight")) // [I,H]
	upW := m.tensor(p("mlp.up_proj.weight"))     // [I,H]
	downW := m.tensor(p("mlp.down_proj.weight")) // [H,I]
	gBias := m.tensorOptional(p("mlp.gate_proj.bias"))
	uBias := m.tensorOptional(p("mlp.up_proj.bias"))
	seq := len(xn)
	rankPartials := make([][][]float32, len(plan.Shards))
	for r, s := range plan.Shards {
		lo, w := s.Lo, s.Width()
		gSlice := shardWeightRows(gateW, H, s.Lo, s.Hi)
		uSlice := shardWeightRows(upW, H, s.Lo, s.Hi)
		downSlice := shardWeightColumns(downW, H, I, s.Lo, s.Hi)
		part := make([][]float32, seq)
		for t := 0; t < seq; t++ {
			g := matRows(gSlice, xn[t], w, H)
			u := matRows(uSlice, xn[t], w, H)
			if gBias != nil {
				for i := 0; i < w; i++ {
					g[i] += gBias[lo+i]
				}
			}
			if uBias != nil {
				for i := 0; i < w; i++ {
					u[i] += uBias[lo+i]
				}
			}
			for i := 0; i < w; i++ {
				g[i] = act(g[i], cfg) * u[i]
			}
			part[t] = matRows(downSlice, g, H, w)
		}
		rankPartials[r] = part
	}
	return rankPartials, nil
}

// tpFFNLayer is the intermediate-sharded live dense SwiGLU FFN for layer l: it reduces the
// per-rank down partials through the Collective (one AllReduceSum per position) and adds the
// down bias once. Result [seq][hidden], matching mlpSeq within the AllReduce round-off
// (bit-exact at ranks=1). A nil collective defaults to LocalCollective.
func (m *Model) tpFFNLayer(l int, xn [][]float32, plan TPPlan, coll Collective) ([][]float32, error) {
	parts, err := m.tpFFNLayerPartials(l, xn, plan)
	if err != nil {
		return nil, err
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	return m.tpReduceWithBias(parts, coll, layerName(l, "mlp.down_proj.bias"))
}

// tpFFNLayerReference is the bit-exact rank-order oracle for tpFFNLayer (see
// tpAttnLayerReference): identical partials, rank-order sum, down bias once.
func (m *Model) tpFFNLayerReference(l int, xn [][]float32, plan TPPlan) ([][]float32, error) {
	parts, err := m.tpFFNLayerPartials(l, xn, plan)
	if err != nil {
		return nil, err
	}
	return m.tpReduceRefWithBias(parts, layerName(l, "mlp.down_proj.bias")), nil
}

// tpReduceWithBias reduces per-rank [seq][hidden] partials position-by-position through the
// Collective's AllReduceSum and adds the named bias once per position after the reduce (the
// row-parallel bias is applied to the reduced sum, NOT per rank). Used by both the attention
// (o_proj) and FFN (down) row-parallel blocks.
func (m *Model) tpReduceWithBias(parts [][][]float32, coll Collective, biasName string) ([][]float32, error) {
	if len(parts) == 0 {
		return nil, fmt.Errorf("model: tp reduce got no rank partials")
	}
	seq := len(parts[0])
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		col := make([][]float32, len(parts))
		for r := range parts {
			col[r] = parts[r][t]
		}
		reduced, err := coll.AllReduceSum(col)
		if err != nil {
			return nil, fmt.Errorf("model: tp AllReduce at position %d: %w", t, err)
		}
		m.addBiasIfPresent(reduced, biasName)
		out[t] = reduced
	}
	return out, nil
}

// tpReduceRefWithBias is tpReduceWithBias's bit-exact reference: rank-order summation
// (sumPartialsRankOrder) instead of the Collective, same once-after-reduce bias.
func (m *Model) tpReduceRefWithBias(parts [][][]float32, biasName string) [][]float32 {
	seq := len(parts[0])
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		col := make([][]float32, len(parts))
		for r := range parts {
			col[r] = parts[r][t]
		}
		reduced := sumPartialsRankOrder(col)
		m.addBiasIfPresent(reduced, biasName)
		out[t] = reduced
	}
	return out
}

// zeroRows returns a [seq][width] slab of zeros — a shape-safe stand-in returned by the
// ForwardTP sub-layer closures when a sharded sub-layer errors, so composeSeqSublayer's
// residual add never indexes a nil row before ForwardTP surfaces the captured error.
func zeroRows(seq, width int) [][]float32 {
	out := make([][]float32, seq)
	for t := range out {
		out[t] = make([]float32, width)
	}
	return out
}

// ForwardTP is the tensor-parallel twin of Forward: a full cacheless prefill whose attention
// and dense-SwiGLU FFN sub-layers are SHARDED across ranks (tpAttnLayer / tpFFNLayer) and
// recombined through the Collective, with every live feature applied. It returns the same
// Activations as Forward and, at ranks=1, is bit-identical to it (the transitive HF-oracle
// gate); across ranks it matches within the single AllReduce's documented round-off.
//
// It fails closed (forwardTPSupported) on any architecture whose TP decomposition it does not
// yet implement — GLM MLA+DSA, MoE, DenseMLP, ParallelResidual, linear-attention — and on a
// quant-resident model (the sharded matmuls slice raw f32 output rows). The norm/residual
// composition reuses the live composeSeqSublayer, so PreNorm / PostNorm / SandwichNorm all
// stay bit-exact at ranks=1.
func (m *Model) ForwardTP(ids []int, tp TPConfig) (*Activations, error) {
	if err := m.forwardTPSupported(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)
	coll := tp.Coll
	if coll == nil {
		coll = LocalCollective{}
	}
	attnPlan, err := NewTPPlan(cfg.NumKVHeads, clampRanks(tp.AttnRanks, cfg.NumKVHeads))
	if err != nil {
		return nil, fmt.Errorf("model: ForwardTP attention plan: %w", err)
	}
	ffnPlan, err := NewTPPlan(cfg.IntermediateSize, clampRanks(tp.FFNRanks, cfg.IntermediateSize))
	if err != nil {
		return nil, fmt.Errorf("model: ForwardTP ffn plan: %w", err)
	}

	embed := m.embedRows()
	x := make([][]float32, seq)
	for t, id := range ids {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg)
	}
	act := &Activations{Seq: seq, Hidden: [][]float32{flatten(x)}}

	var subErr error
	for l := 0; l < cfg.NumLayers; l++ {
		rp := newRopeForLayer(cfg, l, seq)
		attnSub := func(xn [][]float32) [][]float32 {
			if subErr != nil {
				return zeroRows(seq, H)
			}
			out, err := m.tpAttnLayer(l, xn, rp, attnPlan, coll)
			if err != nil {
				subErr = err
				return zeroRows(seq, H)
			}
			return out
		}
		mlpSub := func(xn [][]float32) [][]float32 {
			if subErr != nil {
				return zeroRows(seq, H)
			}
			out, err := m.tpFFNLayer(l, xn, ffnPlan, coll)
			if err != nil {
				subErr = err
				return zeroRows(seq, H)
			}
			return out
		}
		composeSeqSublayer(cfg.BlockTopology, x, m.attentionNorms(l), eps, cfg, attnSub)
		composeSeqSublayer(cfg.BlockTopology, x, m.mlpNorms(l), eps, cfg, mlpSub)
		if subErr != nil {
			return nil, subErr
		}
		act.Hidden = append(act.Hidden, flatten(x))
	}

	mat := residentKernel{m}
	act.Logits = make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xf := m.finalNorm(x[t])
		logits := mat.mul(m.headName(), mat.prep(xf), cfg.VocabSize, H)
		logitScaleInPlace(logits, cfg)
		act.Logits[t] = logits
	}
	return act, nil
}
