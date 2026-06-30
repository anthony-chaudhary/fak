package model

import "fmt"

// expert_parallel_forward.go — WIRING the expert-parallel (EP) MoE decomposition into the
// LIVE glm_moe_dsa forward. expert_parallel.go proved the EP MoE FFN as a STANDALONE
// per-layer primitive (ExpertParallelDelta / expertParallelGLMMoEDelta, bit-exact vs the
// glmMoeFFN monolith at ranks=1, within the single AllReduceSum's reassociation round-off
// across ranks). This file closes the gap GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md names
// as the next rung — "EP wired into the live glmMoeFFN forward" — by running a full
// glm_moe_dsa prefill whose ROUTED MoE FFN is sharded across expert-parallel ranks and
// reduced through the Collective, exactly as ForwardTP (tensor_parallel_forward.go) wired the
// dense row-parallel decomposition into the live Forward.
//
// THE GATE CHAIN — the EP twin of ForwardTP's, transitively inheriting the HF oracle:
//
//	ForwardEP(ranks=1)  ==(bit-exact, max|Δ|=0)        Forward          [this file's tests]
//	Forward             ==(HF tolerance)               HuggingFace      [glm_test.go / oracle_test.go]
//	ForwardEP(ranks=N)  ==(AllReduce reassociation)    ForwardEP(ranks=1)
//
// so ForwardEP(ranks=N) reproduces the live glm_moe_dsa forward up to the documented
// reassociation round-off. The ranks=1 leg is BIT-EXACT because a 1-rank expert plan owns
// every expert in one band and sums the gate-weighted experts in the SAME expert-ascending
// order glmMoeFFN uses (glmRoute returns picks sorted by expert index), and AllReduceSum of
// one partial is the identity — so the only non-bit-exact step (regrouping the routed sum
// across bands) vanishes (expert_parallel_test.go's TestExpertParallelGLMSharedExpert pins
// this per-layer; the tests here pin it through the whole forward).
//
// WHAT IS SHARDED AND WHAT IS REPLICATED — honest, fail-closed scope. EP is the GLM-5.2
// MoE multi-GPU sub-lever: a 753B glm_moe_dsa model's parameters are dominated by its routed
// experts, so partitioning the experts across ranks (each holds ≈ the expert bulk / R) is the
// decomposition that escapes the cpu-offload wall. ForwardEP shards ONLY that routed MoE FFN.
// The MLA + Dynamic-Sparse attention runs REPLICATED (the monolith glmDsaAttnSeqShared on
// every rank): its latent KV is shared across heads, so head-parallel TP is a DIFFERENT
// decomposition (a separate sub-lever, the same one ForwardTP fails closed on for glm_moe_dsa).
// The dense first-k FFN layers (no router) also run replicated (the monolith mlpSeq) — dense
// FFN intermediate-parallel TP is ForwardTP's job, not EP's. So ForwardEP composes the live
// attention + dense layers UNCHANGED with an expert-parallel MoE FFN; on real multi-GPU the
// replicated halves cost redundant flops but no extra collective, and the expert GEMMs — the
// dominant work — move off one device onto the rank fleet.
//
// THE DEVICE LINE — what this does NOT yet claim. The Collective ForwardEP reduces through
// is LocalCollective (the single-box, bit-exact default) or BackendCollective on cpu-ref. A
// real cross-DEVICE reduction needs the NCCL/RCCL compute.CollectiveBackend rung
// (hardware-gated, native-753b-track-staged-plan.md P3) — only with that, plus a multi-GPU
// binary on the box, may a multi-GPU EP SERVE be claimed. This file makes the EP path
// live-selectable and proves it bit-exact on one box, so the residual to a live DGX tok/s
// number is the device collective + the on-box binary, not the wiring.

// EPConfig parameters the wired expert-parallel forward. Ranks shards the routed experts
// across [1,NumExperts] near-even contiguous bands (ExpertParallelPlan); over-large or
// non-positive counts are clamped (clampRanks), mirroring TPConfig. Coll is the single
// AllReduceSum seam (nil -> LocalCollective, the single-box bit-exact default; a real
// NCCL/RDMA collective is a two-method swap behind compute.CollectiveBackend).
type EPConfig struct {
	Ranks int
	Coll  Collective
}

// forwardEPSupported reports (as an error) the first reason this model is not covered by the
// wired EP path, so ForwardEP fails closed instead of mis-serving an architecture whose MoE
// decomposition it does not implement. EP is the glm_moe_dsa MoE sub-lever: it requires a
// glm_moe_dsa MoE config. The per-layer expertParallelGLMMoEDelta additionally fails closed
// on gptoss experts (the expertGPTOSS form) and the qwen3.5 singular gated shared expert —
// surfaced (captured) by ForwardEP's sub-layer closure rather than silently mis-computed.
func (m *Model) forwardEPSupported() error {
	cfg := m.Cfg
	switch {
	case !cfg.isGLMMoeDsa():
		return fmt.Errorf("model: ForwardEP is the glm_moe_dsa expert-parallel sub-lever (use ForwardTP for dense/standard-attention models)")
	case !cfg.IsMoE():
		return fmt.Errorf("model: ForwardEP requires an MoE config (NumExperts>0); this glm_moe_dsa config is dense")
	}
	return nil
}

// epMoeLayer computes the FFN sub-layer output [seq][hidden] for layer l with the routed MoE
// experts sharded expert-parallel across `plan` and reduced through `coll`. It dispatches per
// layer with the SAME gate ffnForLayer uses for glm_moe_dsa: a routed MoE layer (l past the
// dense prefix AND carrying a router) goes through the EP delta; a dense first-k layer (no
// router) runs the monolith dense SwiGLU (mlpSeq) bit-identically to Forward. Keying on
// hasWeight(router) — not m.has — so a quantized-resident model (router in q8w/q4kw) dispatches
// the same way Forward does, never falling a dense layer through to the router path.
func (m *Model) epMoeLayer(l int, xn [][]float32, mat matKernel, plan TPPlan, coll Collective) ([][]float32, error) {
	// A dense first-k GLM layer carries no router -> run the monolith dense FFN unchanged
	// (the exact path mlpSeq takes for it, so it stays bit-identical to Forward).
	if !(l >= m.Cfg.FirstKDenseReplace && m.hasWeight(routerName(l))) {
		return m.mlpSeq(l, xn), nil
	}
	out := make([][]float32, len(xn))
	for t := range xn {
		// Prep the normed input exactly as glmMoeFFN does (mlpSeq passes mat.prep(xn[t]) to
		// ffn.apply), then route + reduce the routed experts across the expert plan. The shared
		// expert (if any) is added once inside expertParallelGLMMoEDelta, after the reduce —
		// matching glmMoeFFN's routed-then-shared order.
		xp := mat.prep(xn[t])
		picks := glmRoute(m, l, xp, mat)
		delta, err := m.expertParallelGLMMoEDelta(l, xp, mat, picks, plan, coll)
		if err != nil {
			return nil, err
		}
		out[t] = delta
	}
	return out, nil
}

// ForwardEP is the expert-parallel twin of Forward for glm_moe_dsa: a full cacheless prefill
// whose ROUTED MoE FFN is sharded across `ep.Ranks` expert-parallel ranks (epMoeLayer) and
// recombined through the Collective (one AllReduceSum per position per MoE layer), with the
// MLA+DSA attention and dense first-k FFN run replicated (the live monolith). It returns the
// same Activations as Forward and, at ranks=1, is bit-identical to it (the transitive HF-oracle
// gate); across ranks it matches within the single AllReduce's documented reassociation round-off.
//
// It fails closed (forwardEPSupported) on any non-glm_moe_dsa or dense config, and — through
// the captured sub-layer error — on the gptoss / qwen3.5-shared-expert forms the EP delta
// refuses. The attention, norms, residual composition, and dense layers reuse the live
// glmDsaAttnSeqShared / composeSeqSublayer / mlpSeq, so every PreNorm/PostNorm/SandwichNorm
// (and the ParallelResidual two-branch) topology stays bit-exact at ranks=1.
func (m *Model) ForwardEP(ids []int, ep EPConfig) (*Activations, error) {
	if err := m.forwardEPSupported(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)
	coll := ep.Coll
	if coll == nil {
		coll = LocalCollective{}
	}
	plan, err := ExpertParallelPlan(cfg.NumExperts, clampRanks(ep.Ranks, cfg.NumExperts))
	if err != nil {
		return nil, fmt.Errorf("model: ForwardEP expert plan: %w", err)
	}

	embed := m.embedRows()
	x := make([][]float32, seq)
	for t, id := range ids {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg)
	}
	act := &Activations{Seq: seq, Hidden: [][]float32{flatten(x)}}

	mat := residentKernel{m}
	topo := cfg.BlockTopology
	var glmDsaSharedTopK [][]int // threaded across layers: a shared-indexer layer reuses the prior full layer's top-k
	var subErr error
	for l := 0; l < cfg.NumLayers; l++ {
		attnNorm := m.attentionNorms(l)
		// Attention is REPLICATED (not sharded): the monolith MLA+DSA path, identical to
		// layerGLMDsa. Head-parallel TP over MLA's shared latent KV is a separate sub-lever.
		attnSub := func(xn [][]float32) [][]float32 {
			return m.glmDsaAttnSeqShared(l, xn, &glmDsaSharedTopK)
		}
		// FFN is EXPERT-PARALLEL on routed MoE layers, monolith on dense first-k layers.
		mlpSub := func(xn [][]float32) [][]float32 {
			if subErr != nil {
				return zeroRows(seq, H)
			}
			out, err := m.epMoeLayer(l, xn, mat, plan, coll)
			if err != nil {
				subErr = err
				return zeroRows(seq, H)
			}
			return out
		}

		if topo == ParallelResidual {
			// Both branches read the original residual (mirrors layerGLMDsa's ParallelResidual leg).
			mlpNorm := m.parallelMLPNorms(l, attnNorm)
			o := attnSub(normSeq(x, attnNorm, eps, cfg))
			d := mlpSub(normSeq(x, mlpNorm, eps, cfg))
			for t := 0; t < seq; t++ {
				for i := 0; i < H; i++ {
					x[t][i] += o[t][i] + d[t][i]
				}
			}
		} else {
			composeSeqSublayer(topo, x, attnNorm, eps, cfg, attnSub)
			composeSeqSublayer(topo, x, m.mlpNorms(l), eps, cfg, mlpSub)
		}
		if subErr != nil {
			return nil, subErr
		}
		act.Hidden = append(act.Hidden, flatten(x))
	}

	act.Logits = make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xf := m.finalNorm(x[t])
		logits := mat.mul(m.headName(), mat.prep(xf), cfg.VocabSize, H)
		logitScaleInPlace(logits, cfg)
		act.Logits[t] = logits
	}
	return act, nil
}
