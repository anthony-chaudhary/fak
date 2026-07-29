package model

// family_minimax_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU
// numeric oracle for the MiniMax-M3 "MiniMax Sparse Attention" (MSA) family.
// Companion to family_cpu_oracle_test.go (which states the doctrine and owns the
// shared scalar primitives), to family_cohere_cpu_oracle_test.go (the closest
// structural template), and to family_gemma_cpu_oracle_test.go (the AxesAreLive
// anti-vacuity pattern this file reuses).
//
// WHY THIS FAMILY IS DISTINCT (and therefore worth an oracle). MiniMax does NOT
// resolve through the resolveSpecFor token switch the way gptneox/falcon/cohere do —
// its covmatrix ResolverToken is empty. It resolves through CONFIG PREDICATES:
// Config.isMiniMax() (a substring of archFamilyKey), Config.isMiniMaxSparseAttn(),
// and the per-layer Config.isMSALayer(l). Those predicates fan out to four distinct
// dispatch sites — forward.go's cacheless layerMiniMax, kv.go's cached
// tokenHiddenMiniMax (Prefill/Step), moe.go's minimaxMoeFFN/minimaxDenseFFN, and
// arch_support.go's ForwardMiniMax classification — and the MSA attention itself is
// its own numeric code (minimaxMSAAttnSeq + minimaxIndexerProject +
// minimaxIndexerSelectBlocks + minimaxSelectBlocks, with a cached twin in
// minimax_m3_session.go and its own KV extension minimaxKVCache). It is NOT a silent
// dense-Llama fall-through, so an oracle here tests real family-specific production
// code rather than re-testing attnSeq under a new name.
//
// INDEPENDENCE DISCIPLINE, AND ITS HONEST LIMIT. The reference below is a plain
// scalar transcription. It reuses NONE of the production machinery: tensors come out
// of the fixture's own pristine HF-order bytes through this file's own offset table
// (not m.manifest, which the loader mutates when it splits the batched/fused expert
// rows), every matmul / norm / softmax / sigmoid / rotary is a naive in-order scalar
// loop here, and the block dataflow is hardcoded to a minimaxOracleSpec whose fields
// are derived by hand from the published HF rules — never read out of Config. Each
// semantic claim below was read off an upstream HuggingFace modeling file, cited:
//
//   - ROUTER (transformers/models/minimax_m2/modeling_minimax_m2.py,
//     MiniMaxM2TopKRouter.forward, corroborated by
//     transformers/models/deepseek_v3/modeling_deepseek_v3.py,
//     DeepseekV3MoE.route_tokens_to_experts): sigmoid on the router logits, then
//     scores_for_choice = routing_weights + e_score_correction_bias, then topk on
//     THAT, then `top_k_weights = routing_weights.gather(...)` — the weights are the
//     RAW sigmoid, not the corrected score — then `/= sum`, then
//     `* routed_scaling_factor` (the DeepSeek-V3 line; M2 has no such factor). Two
//     independent upstream files agree, and this is exactly what glmRoute does with
//     n_group == 1.
//   - SHARED EXPERT (DeepseekV3MoE.forward): `hidden_states = hidden_states +
//     self.shared_experts(residuals)` — ADDED to the routed weighted sum, over the
//     SAME normalized input, and NOT multiplied by routed_scaling_factor (the factor
//     is folded into topk_weights one line earlier, inside route_tokens_to_experts).
//     This is the risk the lane brief names, and production is correct on it:
//     minimaxMoeFFN.apply accumulates `delta[i] += shared[i]` unscaled.
//   - SwiGLU-OAI (transformers/models/gpt_oss/modeling_gpt_oss.py,
//     GptOssExperts._apply_gate): `gate.clamp(max=limit)`, `up.clamp(-limit, +limit)`,
//     `glu = gate * sigmoid(gate * alpha)`, `(up + 1) * glu`. Upper clamp on the gate
//     ONLY; symmetric clamp on up. MiniMax generalizes gpt-oss's fixed limit=7.0 /
//     alpha=1.702 to configured swiglu_limit / swiglu_alpha.
//   - PARTIAL ROTARY (MiniMaxM2RotaryEmbedding.compute_default_rope_parameters +
//     apply_rotary_pos_emb + rotate_half): `dim = int(head_dim *
//     partial_rotary_factor)` and inv_freq = 1/base**(arange(0,dim,2)/dim) — the
//     DENOMINATOR IS THE ROTARY WIDTH, not head_dim (fak encodes this as
//     invFreqDenom() == rotaryDim() for minimax). apply_rotary_pos_emb splits
//     `q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]`, rotates only q_rot
//     with the NON-interleaved rotate_half (element j pairs with j+rotary_dim/2), and
//     concatenates q_pass back untouched.
//   - GEMMA-STYLE NORM GAIN (transformers/models/gemma3/modeling_gemma3.py
//     Gemma3RMSNorm.forward: `output * (1.0 + self.weight.float())`). config.go turns
//     this on for the whole minimax family (use_gemma_norm), so the reference applies
//     (1+w) to the block norms, the final norm, the per-head qk-norm AND the
//     lightning-indexer q/k norms — every norm, and no other scaling.
//   - ATTENTION (MiniMaxM2Attention): causal GQA at head_dim**-0.5, softmax in fp32,
//     qk-norm applied POST-projection and PRE-rotary, no ALiBi/sink/softcap.
//
// The one semantic this file CANNOT source from an upstream file is the MSA block
// selector itself: transformers 5.10.2 (the newest available in this environment)
// ships minimax and minimax_m2 but no minimax_m3_vl, and its nearest relatives
// (glm_moe_dsa / deepseek_v4 lightning indexers) select individual KEYS, not blocks.
// So the selector reference here is an independent re-IMPLEMENTATION of the
// documented algorithm (per-index-head dot scores, max-pool into blocks of
// index_block_size, force the always-on local window to +inf, fixed-count topk,
// broadcast back onto keys) written from the algorithm rather than transcribed from
// upstream code. It can catch an implementation/transcription defect in fak; it
// cannot catch a spec misreading that fak and this file would share. That residual is
// stated plainly rather than papered over — see the report note on the
// TestMiniMaxCPUNumericOracleSparsityIsNonVacuous / ...AxesAreLive pair, which at
// least pins the selector's OBSERVABLE effect (sparsity is real, and every selector
// knob moves the answer).
//
// The fixture is built on MiniMax's REAL checkpoint layout — BATCHED experts
// (mlp.experts.gate_up_proj.weight [E, 2I, H] and mlp.experts.down_proj.weight
// [E, H, I], the MiniMaxM2Experts 3D parameters) and a FUSED shared expert
// (mlp.shared_experts.gate_up_proj.weight [2*shared_I, H]) — and loaded through
// newHFCheckpointModel, so production also exercises splitBatchedMoEExperts and
// materializeMiniMaxSharedExperts on the way in. Geometry is deliberately
// non-degenerate: nH*hd (32) != HiddenSize (24) so a projection/hidden conflation
// cannot cancel; nKV (2) != nH (4) so a GQA grouping bug cannot cancel; rotary_dim (4)
// < head_dim (8) so the partial-rotary pass-through is observable; index_head_dim (6)
// differs from head_dim (8) and from rotary_dim so an indexer/main-branch conflation
// cannot cancel; and the routed (10), dense (14) and shared (6) FFN widths are three
// different numbers so a width mix-up cannot cancel. swiglu_limit is 0.25 rather than
// the checkpoint's 7.0 because at this fixture's activation scale a limit of 7 never
// binds and the clamp axis would be numerically inert (the same fixture-scaling note
// the Gemma oracle makes about its soft-caps); swiglu_alpha is 1.9 rather than 1.702
// so the axis is distinguishable from swigluOAIInPlace's zero-value fallback.

import (
	"encoding/binary"
	"math"
	"sort"
	"testing"
)

// ---- independent scalar primitives -------------------------------------------------

// minimaxOracleSigmoid is torch.sigmoid in f32.
func minimaxOracleSigmoid(z float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-z))))
}

// minimaxOracleRMSNorm is MiniMaxM3VLRMSNorm: x*rsqrt(mean(x^2)+eps) scaled by the
// learned gain. gain1p selects the Gemma form (1+w) the family's use_gemma_norm turns
// on; false is the plain-w Llama form, kept so the AxesAreLive gate can show the (1+w)
// choice is not inert.
func minimaxOracleRMSNorm(x, w []float32, eps float32, gain1p bool) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		g := w[i]
		if gain1p {
			g = 1 + w[i]
		}
		out[i] = v * inv * g
	}
	return out
}

// minimaxOracleRope rotates ONE vector in place at position pos with HF's PARTIAL
// non-interleaved rotary: only the first rotaryDim components rotate (rotate_half pairs
// j with j+rotaryDim/2), the tail is passed through, and the inverse frequency
// denominator is rotaryDim — NOT len(hv). Deliberately not applyRopeRow.
func minimaxOracleRope(hv []float32, pos, rotaryDim int, theta float64) {
	if rotaryDim > len(hv) {
		rotaryDim = len(hv)
	}
	half := rotaryDim / 2
	for j := 0; j < half; j++ {
		angle := float64(pos) / math.Pow(theta, float64(2*j)/float64(rotaryDim))
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		a, b := hv[j], hv[j+half]
		hv[j] = a*c - b*s
		hv[j+half] = b*c + a*s
	}
}

// minimaxOracleSwigluOAI is the OAI clamped gate: gate clamped ABOVE at limit, up
// clamped symmetrically at ±limit, glu = gate*sigmoid(gate*alpha), out = (up+1)*glu.
// limit <= 0 means "no clamp".
func minimaxOracleSwigluOAI(g, u []float32, alpha, limit float32) []float32 {
	out := make([]float32, len(g))
	for i := range g {
		gate, up := g[i], u[i]
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
		glu := gate * minimaxOracleSigmoid(gate*alpha)
		out[i] = (up + 1) * glu
	}
	return out
}

// ---- family semantics, hardcoded ---------------------------------------------------

// minimaxOracleSpec is the MiniMax-M3 semantics the reference hardcodes. Every field is
// derived HERE from the published HF rule, never read out of Config — that is what lets
// the numeric comparison witness config.go's derivation as well as the kernels.
type minimaxOracleSpec struct {
	// sparse[l] is true when layer l is a minimax_m3_sparse (MSA) layer.
	sparse []bool
	// moe[l] is true when layer l carries a router (the first-k layers are dense).
	moe []bool

	normGain1p bool // Gemma-style (1+w) on EVERY norm
	qkNorm     bool // per-head RMSNorm on q/k, post-projection, pre-rotary
	rotaryDim  int  // int(head_dim * partial_rotary_factor)
	theta      float64

	blockSize    int
	topKBlocks   int
	localBlocks  int
	blockPoolMax bool // block score = MAX over member keys (vs mean)

	swigluAlpha float32
	swigluLimit float32
	denseOAI    bool // the first-k dense MLP uses the OAI gate, not plain SiLU

	topK                int
	normTopKProb        bool
	routedScaling       float32
	selectOnCorrected   bool // topk ranks on sigmoid + e_score_correction_bias
	weightFromCorrected bool // WRONG variant: pick weight = the corrected score
	sharedExpert        bool
	sharedScaled        bool // WRONG variant: shared expert scaled by routed_scaling
}

// minimaxOracleSpecFor is the published MiniMax-M3 spec at this fixture's geometry.
func minimaxOracleSpecFor() minimaxOracleSpec {
	return minimaxOracleSpec{
		// layer_types = [full_attention, minimax_m3_sparse, minimax_m3_sparse];
		// layer 0 is also the first-k DENSE FFN layer (no router), as in the real
		// 60-layer checkpoint whose moe_layer_freq leaves the first layers dense.
		sparse:              []bool{false, true, true},
		moe:                 []bool{false, true, true},
		normGain1p:          true,
		qkNorm:              true,
		rotaryDim:           4, // int(head_dim 8 * partial_rotary_factor 0.5)
		theta:               10000,
		blockSize:           2,
		topKBlocks:          2,
		localBlocks:         1,
		blockPoolMax:        true,
		swigluAlpha:         1.9,
		swigluLimit:         0.25,
		denseOAI:            true,
		topK:                2,
		normTopKProb:        true,
		routedScaling:       1.7,
		selectOnCorrected:   true,
		weightFromCorrected: false,
		sharedExpert:        true,
		sharedScaled:        false,
	}
}

// ---- fixture -----------------------------------------------------------------------

// minimaxOracleCfg is the tiny MiniMax-M3 fixture config. QKNorm, NormGain1p,
// NormTopKProb and NSharedExperts are left ZERO on purpose so the test proves
// deriveConfigAxes lands them for the family rather than the fixture asserting them by
// hand. tie_word_embeddings is false (as the real M3 config is), so the fixture carries
// a separate lm_head and the untied-head role is under test.
func minimaxOracleCfg() Config {
	return Config{
		HiddenSize:             24,
		NumLayers:              3,
		NumHeads:               4,
		NumKVHeads:             2,
		HeadDim:                8,
		IntermediateSize:       10,
		DenseIntermediateSize:  14,
		SharedIntermediateSize: 6,
		VocabSize:              53,
		ModelType:              "minimax_m3",
		Architectures:          []string{"MiniMaxM3VLForCausalLM"},
		LayerTypes:             []string{"full_attention", "minimax_m3_sparse", "minimax_m3_sparse"},
		RMSNormEps:             1e-5,
		RopeTheta:              10000,
		PartialRotaryFactor:    0.5,
		NumExperts:             4,
		NumExpertsPerTok:       2,
		RoutedScalingFactor:    1.7,
		SwigluAlpha:            1.9,
		SwigluLimit:            0.25,
		IndexNHeads:            2,
		IndexHeadDim:           6,
		IndexBlockSize:         2,
		IndexTopKBlocks:        2,
		IndexLocalBlocks:       1,
		EOSTokenID:             -1,
	}
}

// minimaxOracleTensors is MiniMax-M3's REAL tensor roster at the fixture geometry: a
// dense first layer (no router, its own mlp.{gate,up,down}_proj at
// dense_intermediate_size), MSA layers carrying the lightning indexer, BATCHED routed
// experts, a FUSED shared expert, and an untied lm_head.
func minimaxOracleTensors(cfg Config, sp minimaxOracleSpec) []synthTensor {
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V, E := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.NumExperts
	denseI, sharedI := cfg.DenseIntermediateSize, cfg.SharedIntermediateSize
	nIdx, idxDim := cfg.IndexNHeads, cfg.IndexHeadDim
	type ts = synthTensor
	out := []ts{{"model.embed_tokens.weight", []int{V, H}}}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		ap := p + "self_attn."
		out = append(out,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{ap + "q_proj.weight", []int{nH * hd, H}},
			ts{ap + "k_proj.weight", []int{nKV * hd, H}},
			ts{ap + "v_proj.weight", []int{nKV * hd, H}},
			ts{ap + "o_proj.weight", []int{H, nH * hd}},
			ts{ap + "q_norm.weight", []int{hd}},
			ts{ap + "k_norm.weight", []int{hd}},
		)
		if sp.sparse[l] {
			out = append(out,
				ts{ap + "indexer.q_proj.weight", []int{nIdx * idxDim, H}},
				ts{ap + "indexer.k_proj.weight", []int{idxDim, H}},
				ts{ap + "indexer.q_norm.weight", []int{idxDim}},
				ts{ap + "indexer.k_norm.weight", []int{idxDim}},
			)
		}
		out = append(out, ts{p + "post_attention_layernorm.weight", []int{H}})
		if sp.moe[l] {
			out = append(out,
				ts{p + "mlp.gate.weight", []int{E, H}},
				ts{p + "mlp.gate.e_score_correction_bias", []int{E}},
				ts{p + "mlp.experts.gate_up_proj.weight", []int{E, 2 * I, H}},
				ts{p + "mlp.experts.down_proj.weight", []int{E, H, I}},
				ts{p + "mlp.shared_experts.gate_up_proj.weight", []int{2 * sharedI, H}},
				ts{p + "mlp.shared_experts.down_proj.weight", []int{H, sharedI}},
			)
		} else {
			out = append(out,
				ts{p + "mlp.gate_proj.weight", []int{denseI, H}},
				ts{p + "mlp.up_proj.weight", []int{denseI, H}},
				ts{p + "mlp.down_proj.weight", []int{H, denseI}},
			)
		}
	}
	return append(out,
		ts{"model.norm.weight", []int{H}},
		ts{"lm_head.weight", []int{V, H}},
	)
}

// minimaxOracleFixture holds the loaded production Model plus a PRISTINE copy of the
// checkpoint bytes in HF order and this file's OWN name->offset table. The private table
// matters: newHFCheckpointModel DELETES the batched/fused source rows from the manifest
// when it splits them, so a reference that read m.manifest could not see the checkpoint
// as HuggingFace wrote it.
type minimaxOracleFixture struct {
	m   *Model
	hf  []byte
	off map[string]int // element offset into hf (f32 units)
	n   map[string]int // element count
	sp  minimaxOracleSpec
}

func (f *minimaxOracleFixture) tensor(t *testing.T, name string) []float32 {
	t.Helper()
	o, ok := f.off[name]
	if !ok {
		t.Fatalf("fixture tensor %q missing", name)
	}
	out := make([]float32, f.n[name])
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(f.hf[(o+i)*4:]))
	}
	return out
}

func newMinimaxOracleFixture(t *testing.T) *minimaxOracleFixture {
	t.Helper()
	cfg := minimaxOracleCfg()
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	sp := minimaxOracleSpecFor()
	list := minimaxOracleTensors(cfg, sp)
	man, raw := synthBuildRaw(list, func(name string, next func() float32) float32 {
		if isCPUOracleNormWeight(name) {
			return 1 + 0.25*next() // distinct NON-UNIT gains, well-conditioned
		}
		return synthMatmulFill(name, next)
	})
	hf := append([]byte(nil), raw...)
	off := make(map[string]int, len(list))
	cnt := make(map[string]int, len(list))
	elems := 0
	for _, ts := range list {
		k := 1
		for _, d := range ts.shape {
			k *= d
		}
		off[ts.name] = elems
		cnt[ts.name] = k
		elems += k
	}
	m, err := newHFCheckpointModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newHFCheckpointModel(minimax fixture): %v", err)
	}
	return &minimaxOracleFixture{m: m, hf: hf, off: off, n: cnt, sp: sp}
}

// ---- the independent reference -----------------------------------------------------

// minimaxOracleSelect is the independent lightning-indexer block selection: per query,
// the ascending admitted causal key positions. Written from the documented MSA algorithm
// (see the header's honesty note about the missing upstream file), NOT by calling
// minimaxIndexerSelect / minimaxSelectBlocks / msaBlockScores / msaSelectedKeyPositions.
func minimaxOracleSelect(t *testing.T, f *minimaxOracleFixture, l int, xn [][]float32, sp minimaxOracleSpec) [][]int {
	t.Helper()
	cfg := f.m.Cfg
	H, nIdx, idxDim := cfg.HiddenSize, cfg.IndexNHeads, cfg.IndexHeadDim
	eps := float32(cfg.RMSNormEps)
	ip := layerPrefix(l) + "self_attn.indexer."
	wq := f.tensor(t, ip+"q_proj.weight")
	wk := f.tensor(t, ip+"k_proj.weight")
	qn := f.tensor(t, ip+"q_norm.weight")
	kn := f.tensor(t, ip+"k_norm.weight")
	seq := len(xn)

	idxQ := make([][][]float32, seq)
	idxK := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		qf := cpuOracleMatVec(wq, xn[tt], nIdx*idxDim, H)
		idxQ[tt] = make([][]float32, nIdx)
		for h := 0; h < nIdx; h++ {
			head := minimaxOracleRMSNorm(qf[h*idxDim:(h+1)*idxDim], qn, eps, sp.normGain1p)
			minimaxOracleRope(head, tt, sp.rotaryDim, sp.theta)
			idxQ[tt][h] = head
		}
		kf := minimaxOracleRMSNorm(cpuOracleMatVec(wk, xn[tt], idxDim, H), kn, eps, sp.normGain1p)
		minimaxOracleRope(kf, tt, sp.rotaryDim, sp.theta)
		idxK[tt] = kf
	}

	out := make([][]int, seq)
	for qp := 0; qp < seq; qp++ {
		qb := qp / sp.blockSize
		score := map[int]float64{}
		count := map[int]int{}
		for h := 0; h < nIdx; h++ {
			for kp := 0; kp <= qp; kp++ {
				var s float32
				for d := 0; d < idxDim; d++ {
					s += idxQ[qp][h][d] * idxK[kp][d]
				}
				b := kp / sp.blockSize
				if sp.blockPoolMax {
					if cur, seen := score[b]; !seen || float64(s) > cur {
						score[b] = float64(s)
					}
					continue
				}
				score[b] += float64(s)
				count[b]++
			}
		}
		if !sp.blockPoolMax {
			for b := range score {
				score[b] /= float64(count[b])
			}
		}
		type cand struct {
			b int
			s float64
		}
		cands := make([]cand, 0, len(score))
		for b, s := range score {
			// Always-on local window: scattered to +inf, then a FIXED-count topk — so a
			// forced local block DISPLACES the lowest-scored block rather than adding to
			// the set.
			if sp.localBlocks > 0 && b <= qb && b >= qb-sp.localBlocks+1 {
				s = math.Inf(1)
			}
			cands = append(cands, cand{b, s})
		}
		sort.Slice(cands, func(i, j int) bool {
			if cands[i].s == cands[j].s {
				return cands[i].b < cands[j].b
			}
			return cands[i].s > cands[j].s
		})
		keepN := sp.topKBlocks
		if keepN > len(cands) {
			keepN = len(cands)
		}
		keep := make(map[int]bool, keepN)
		for i := 0; i < keepN; i++ {
			keep[cands[i].b] = true
		}
		keys := make([]int, 0, qp+1)
		for kp := 0; kp <= qp; kp++ {
			if keep[kp/sp.blockSize] {
				keys = append(keys, kp)
			}
		}
		out[qp] = keys
	}
	return out
}

// minimaxOracleMoE is the independent SwiGLU-OAI MoE FFN for one position: the HF
// sigmoid router (+ e_score_correction_bias for SELECTION only), top-k, renorm, routed
// scaling, OAI-gated batched experts, plus the always-on shared expert ADDED unscaled.
func minimaxOracleMoE(xn []float32, router, corr, gateUp, down, sGateUp, sDown []float32, cfg Config, sp minimaxOracleSpec) []float32 {
	H, I, E := cfg.HiddenSize, cfg.IntermediateSize, cfg.NumExperts
	sharedI := cfg.SharedIntermediateSize

	logits := cpuOracleMatVec(router, xn, E, H)
	raw := make([]float32, E)
	choice := make([]float32, E)
	for e := range logits {
		raw[e] = minimaxOracleSigmoid(logits[e])
		choice[e] = raw[e] + corr[e]
	}
	rank := choice
	if !sp.selectOnCorrected {
		rank = raw
	}
	order := make([]int, E)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return rank[order[i]] > rank[order[j]] })
	k := sp.topK
	if k > E {
		k = E
	}
	picks := order[:k]
	wts := make([]float32, k)
	var sum float32
	for i, e := range picks {
		wts[i] = raw[e]
		if sp.weightFromCorrected {
			wts[i] = choice[e]
		}
		sum += wts[i]
	}
	if sp.normTopKProb && sum != 0 {
		for i := range wts {
			wts[i] /= sum
		}
	}
	for i := range wts {
		wts[i] *= sp.routedScaling
	}

	out := make([]float32, H)
	for i, e := range picks {
		base := e * 2 * I * H
		g := cpuOracleMatVec(gateUp[base:base+I*H], xn, I, H)
		u := cpuOracleMatVec(gateUp[base+I*H:base+2*I*H], xn, I, H)
		act := minimaxOracleSwigluOAI(g, u, sp.swigluAlpha, sp.swigluLimit)
		d := cpuOracleMatVec(down[e*H*I:(e+1)*H*I], act, H, I)
		for j := 0; j < H; j++ {
			out[j] += wts[i] * d[j]
		}
	}
	if sp.sharedExpert {
		g := cpuOracleMatVec(sGateUp[:sharedI*H], xn, sharedI, H)
		u := cpuOracleMatVec(sGateUp[sharedI*H:2*sharedI*H], xn, sharedI, H)
		act := minimaxOracleSwigluOAI(g, u, sp.swigluAlpha, sp.swigluLimit)
		d := cpuOracleMatVec(sDown, act, H, sharedI)
		s := float32(1)
		if sp.sharedScaled {
			s = sp.routedScaling
		}
		for j := 0; j < H; j++ {
			out[j] += s * d[j]
		}
	}
	return out
}

// minimaxReference runs the independent MiniMax-M3 forward: per-position logits for ids.
// Every step is the HF MiniMaxM3VL decoder dataflow, hardcoded — NOT routed through
// cfg.BlockTopology, normCfg, composeSeqSublayer, applyQKNormCfg, applyRopeRow,
// glmRoute, swigluOAIInPlace, ffnForLayer or the msa_index.go helpers.
func minimaxReference(t *testing.T, f *minimaxOracleFixture, ids []int, sp minimaxOracleSpec) [][]float32 {
	t.Helper()
	cfg := f.m.Cfg
	H, V := cfg.HiddenSize, cfg.VocabSize
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	grp := nH / nKV
	denseI := cfg.DenseIntermediateSize
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)

	tensor := func(name string) []float32 { return f.tensor(t, name) }

	embed := tensor("model.embed_tokens.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		ap := p + "self_attn."

		// --- attention sublayer (PreNorm: x += attn(norm(x))) ---
		inNorm := tensor(p + "input_layernorm.weight")
		xn := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn[tt] = minimaxOracleRMSNorm(x[tt], inNorm, eps, sp.normGain1p)
		}
		wq := tensor(ap + "q_proj.weight")
		wk := tensor(ap + "k_proj.weight")
		wv := tensor(ap + "v_proj.weight")
		wo := tensor(ap + "o_proj.weight")
		qn := tensor(ap + "q_norm.weight")
		kn := tensor(ap + "k_norm.weight")

		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			q[tt] = cpuOracleMatVec(wq, xn[tt], nH*hd, H)
			k[tt] = cpuOracleMatVec(wk, xn[tt], nKV*hd, H)
			v[tt] = cpuOracleMatVec(wv, xn[tt], nKV*hd, H)
			if sp.qkNorm {
				for h := 0; h < nH; h++ {
					copy(q[tt][h*hd:(h+1)*hd], minimaxOracleRMSNorm(q[tt][h*hd:(h+1)*hd], qn, eps, sp.normGain1p))
				}
				for h := 0; h < nKV; h++ {
					copy(k[tt][h*hd:(h+1)*hd], minimaxOracleRMSNorm(k[tt][h*hd:(h+1)*hd], kn, eps, sp.normGain1p))
				}
			}
			for h := 0; h < nH; h++ {
				minimaxOracleRope(q[tt][h*hd:(h+1)*hd], tt, sp.rotaryDim, sp.theta)
			}
			for h := 0; h < nKV; h++ {
				minimaxOracleRope(k[tt][h*hd:(h+1)*hd], tt, sp.rotaryDim, sp.theta)
			}
		}

		admitted := make([][]int, seq)
		if sp.sparse[l] {
			admitted = minimaxOracleSelect(t, f, l, xn, sp)
		} else {
			for tt := 0; tt < seq; tt++ {
				keys := make([]int, tt+1)
				for j := range keys {
					keys[j] = j
				}
				admitted[tt] = keys
			}
		}

		scale := float32(1.0 / math.Sqrt(float64(hd))) // head_dim**-0.5
		for tt := 0; tt < seq; tt++ {
			concat := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				kvh := h / grp
				qh := q[tt][h*hd : (h+1)*hd]
				scores := make([]float32, len(admitted[tt]))
				for i, kp := range admitted[tt] {
					kh := k[kp][kvh*hd : (kvh+1)*hd]
					var s float32
					for d := 0; d < hd; d++ {
						s += qh[d] * kh[d]
					}
					scores[i] = s * scale
				}
				cpuOracleSoftmax(scores)
				o := concat[h*hd : (h+1)*hd]
				for i, kp := range admitted[tt] {
					vh := v[kp][kvh*hd : (kvh+1)*hd]
					for d := 0; d < hd; d++ {
						o[d] += scores[i] * vh[d]
					}
				}
			}
			out := cpuOracleMatVec(wo, concat, H, nH*hd)
			for i := 0; i < H; i++ {
				x[tt][i] += out[i]
			}
		}

		// --- FFN sublayer (PreNorm: x += ffn(norm(x))) ---
		postNorm := tensor(p + "post_attention_layernorm.weight")
		xm := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xm[tt] = minimaxOracleRMSNorm(x[tt], postNorm, eps, sp.normGain1p)
		}
		if sp.moe[l] {
			router := tensor(p + "mlp.gate.weight")
			corr := tensor(p + "mlp.gate.e_score_correction_bias")
			gateUp := tensor(p + "mlp.experts.gate_up_proj.weight")
			down := tensor(p + "mlp.experts.down_proj.weight")
			sGateUp := tensor(p + "mlp.shared_experts.gate_up_proj.weight")
			sDown := tensor(p + "mlp.shared_experts.down_proj.weight")
			for tt := 0; tt < seq; tt++ {
				d := minimaxOracleMoE(xm[tt], router, corr, gateUp, down, sGateUp, sDown, cfg, sp)
				for i := 0; i < H; i++ {
					x[tt][i] += d[i]
				}
			}
		} else {
			wg := tensor(p + "mlp.gate_proj.weight")
			wu := tensor(p + "mlp.up_proj.weight")
			wd := tensor(p + "mlp.down_proj.weight")
			for tt := 0; tt < seq; tt++ {
				g := cpuOracleMatVec(wg, xm[tt], denseI, H)
				u := cpuOracleMatVec(wu, xm[tt], denseI, H)
				var act []float32
				if sp.denseOAI {
					act = minimaxOracleSwigluOAI(g, u, sp.swigluAlpha, sp.swigluLimit)
				} else {
					act = make([]float32, denseI)
					for i := 0; i < denseI; i++ {
						act[i] = cpuOracleSilu(g[i]) * u[i]
					}
				}
				d := cpuOracleMatVec(wd, act, H, denseI)
				for i := 0; i < H; i++ {
					x[tt][i] += d[i]
				}
			}
		}
	}

	finalNorm := tensor("model.norm.weight")
	head := tensor("lm_head.weight") // tie_word_embeddings = false
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := minimaxOracleRMSNorm(x[tt], finalNorm, eps, sp.normGain1p)
		logits[tt] = cpuOracleMatVec(head, xf, V, H)
	}
	return logits
}

// minimaxOracleIDs is the prompt. len(ids)=9 at index_block_size 2 gives 5 causal
// blocks at the last position while index_topk_blocks is 2, so the block budget CANNOT
// cover the causal range and the sparse path is genuinely exercised — with a short
// prompt every query would attend its whole prefix and the oracle would be vacuous for
// the very topology it claims to cover. TestMiniMaxCPUNumericOracleSparsityIsNonVacuous
// pins that.
var minimaxOracleIDs = []int{3, 17, 5, 23, 41, 2, 19, 7, 31}

// ---- the witnesses -----------------------------------------------------------------

// TestMiniMaxCPUNumericOracle is the MiniMax-MSA x cpu M4 witness: the production
// cacheless Forward AND the cached Prefill/Step decode path must reproduce the
// independent HF-semantics reference at every position within cpuOracleTol. It covers
// the lightning-indexer block-sparse attention, the dense full_attention layer, partial
// rotary, per-head qk-norm, Gemma-style (1+w) norms, the sigmoid MoE router with its
// score-correction bias, the OAI-clamped SwiGLU experts, the always-on shared expert,
// the first-k dense OAI MLP at dense_intermediate_size, and the untied head.
//
// If this test reds, the honesty fence demotes the cell back to M3 (drop the covmatrix
// OracleInCI bit with it).
func TestMiniMaxCPUNumericOracle(t *testing.T) {
	f := newMinimaxOracleFixture(t)
	cfg := f.m.Cfg

	// The derivation must land the published MiniMax axes — the reference hardcodes them.
	if !cfg.isMiniMaxSparseAttn() {
		t.Fatal("minimax fixture did not derive isMiniMaxSparseAttn")
	}
	if !cfg.NormGain1p {
		t.Fatal("minimax derived NormGain1p = false, want true (MiniMaxM3VLRMSNorm is output*(1+weight))")
	}
	if !cfg.QKNorm {
		t.Fatal("minimax derived QKNorm = false, want true (M3 layers carry per-head q_norm/k_norm)")
	}
	if !cfg.NormTopKProb {
		t.Fatal("minimax derived NormTopKProb = false, want true (the router always renormalizes top-k)")
	}
	if cfg.NSharedExperts != 1 {
		t.Fatalf("minimax derived NSharedExperts = %d, want 1", cfg.NSharedExperts)
	}
	if cfg.BlockTopology != PreNorm {
		t.Fatalf("minimax derived topology = %v, want PreNorm", cfg.BlockTopology)
	}
	if cfg.rotaryDim() != f.sp.rotaryDim {
		t.Fatalf("minimax derived rotaryDim = %d, want %d", cfg.rotaryDim(), f.sp.rotaryDim)
	}
	// The dispatch the whole oracle depends on: layer 0 dense-OAI MLP, layers 1-2 MoE.
	for l := 0; l < cfg.NumLayers; l++ {
		if got, want := cfg.isMSALayer(l), f.sp.sparse[l]; got != want {
			t.Fatalf("layer %d isMSALayer = %v, want %v", l, got, want)
		}
		ffn := f.m.ffnForLayer(l)
		if f.sp.moe[l] {
			if _, ok := ffn.(minimaxMoeFFN); !ok {
				t.Fatalf("layer %d ffn = %T, want minimaxMoeFFN", l, ffn)
			}
		} else if _, ok := ffn.(minimaxDenseFFN); !ok {
			t.Fatalf("layer %d ffn = %T, want minimaxDenseFFN", l, ffn)
		}
	}
	// The fixture must present the REAL fused/batched checkpoint layout, so the loader's
	// splits are on the hot path of this comparison rather than bypassed.
	if !f.m.has(layerName(1, "mlp.experts.0.gate_proj.weight")) {
		t.Fatal("batched experts were not split — splitBatchedMoEExperts is not under test")
	}
	if !f.m.has(layerName(1, "mlp.shared_experts.gate_proj.weight")) {
		t.Fatal("fused shared expert was not split — materializeMiniMaxSharedExperts is not under test")
	}

	ids := minimaxOracleIDs
	ref := minimaxReference(t, f, ids, f.sp)

	act := f.m.Forward(ids)
	for tt := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > cpuOracleTol {
			t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", tt, d, cpuOracleTol)
		}
	}

	// Cached decode path: Prefill then Step must match the reference at the same
	// positions (the reference is cacheless, so Step(id) at position len(ids) is compared
	// against a reference run over the extended prompt).
	s := f.m.NewSession()
	pf := s.Prefill(ids)
	if d := cpuOracleMaxAbsDiff(pf, ref[len(ids)-1]); d > cpuOracleTol {
		t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
	next := 11
	st := s.Step(next)
	extRef := minimaxReference(t, f, append(append([]int(nil), ids...), next), f.sp)
	if d := cpuOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
		t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
}

// TestMiniMaxCPUNumericOracleSparsityIsNonVacuous proves the sparse topology the oracle
// claims to cover is actually exercised at this fixture's sequence length: on an MSA
// layer at least one query must admit a STRICT SUBSET of its causal prefix. Without
// this, a prompt short enough for the block budget to cover every causal block would
// make the MSA path numerically identical to dense GQA and the oracle would witness
// nothing family-specific. Computed from the reference selector, so production cannot
// satisfy it by agreeing with itself.
func TestMiniMaxCPUNumericOracleSparsityIsNonVacuous(t *testing.T) {
	f := newMinimaxOracleFixture(t)
	cfg := f.m.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	ids := minimaxOracleIDs

	// Feed the selector the same normalized layer-1 inputs the forward would see. Only
	// the SHAPE of the admitted sets is under test here, so driving it from the
	// embedding-level hidden is enough to prove the budget genuinely prunes.
	embed := f.tensor(t, "model.embed_tokens.weight")
	inNorm := f.tensor(t, layerPrefix(1)+"input_layernorm.weight")
	xn := make([][]float32, len(ids))
	for tt, id := range ids {
		xn[tt] = minimaxOracleRMSNorm(embed[id*H:(id+1)*H], inNorm, eps, f.sp.normGain1p)
	}
	sel := minimaxOracleSelect(t, f, 1, xn, f.sp)

	strict := 0
	for tt := range ids {
		if len(sel[tt]) == 0 {
			t.Fatalf("query %d admitted no keys", tt)
		}
		if sel[tt][len(sel[tt])-1] != tt {
			t.Errorf("query %d does not admit its own position (always-on local window is broken): %v", tt, sel[tt])
		}
		if len(sel[tt]) < tt+1 {
			strict++
		}
	}
	if strict == 0 {
		t.Fatalf("no query pruned any causal key at seq=%d, block=%d, topk=%d, local=%d — the sparse path is vacuous here",
			len(ids), f.sp.blockSize, f.sp.topKBlocks, f.sp.localBlocks)
	}
}

// minimaxOracleAssertAxisLive proves one MiniMax-specific axis is not numerically inert
// in this fixture: the reference under the HF spec must differ from the reference under
// a deliberately WRONG spec by more than the tolerance. This compares reference to
// reference, so it cannot be satisfied by anything production does.
func minimaxOracleAssertAxisLive(t *testing.T, f *minimaxOracleFixture, base [][]float32, wrong minimaxOracleSpec, axis string) {
	t.Helper()
	alt := minimaxReference(t, f, minimaxOracleIDs, wrong)
	var worst float64
	for tt := range minimaxOracleIDs {
		if d := cpuOracleMaxAbsDiff(base[tt], alt[tt]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Errorf("axis %q is inert in this fixture (max|delta| = %.3e <= tol %.0e): the oracle would stay green with that axis wrong",
			axis, worst, cpuOracleTol)
	}
}

// TestMiniMaxCPUNumericOracleAxesAreLive is the anti-vacuity gate for the family axes a
// size/scale choice could accidentally switch off. Each is mutated to the "wrong but
// plausible" value — including the two silent-bug shapes the MiniMax lane is most
// exposed to: the router normalization ORDER (selecting on the raw sigmoid instead of
// the correction-biased score, or taking the pick weight from the corrected score
// instead of the raw one) and the shared expert's placement (scaled by
// routed_scaling_factor instead of added unscaled).
func TestMiniMaxCPUNumericOracleAxesAreLive(t *testing.T) {
	f := newMinimaxOracleFixture(t)
	base := minimaxReference(t, f, minimaxOracleIDs, f.sp)

	mut := func(fn func(s *minimaxOracleSpec)) minimaxOracleSpec {
		s := f.sp
		s.sparse = append([]bool(nil), f.sp.sparse...)
		s.moe = append([]bool(nil), f.sp.moe...)
		fn(&s)
		return s
	}

	for _, tc := range []struct {
		axis string
		spec minimaxOracleSpec
	}{
		{"gemma (1+w) norm gain", mut(func(s *minimaxOracleSpec) { s.normGain1p = false })},
		{"per-head qk-norm", mut(func(s *minimaxOracleSpec) { s.qkNorm = false })},
		{"partial rotary width", mut(func(s *minimaxOracleSpec) { s.rotaryDim = 8 })},
		{"index_block_size", mut(func(s *minimaxOracleSpec) { s.blockSize = 4 })},
		{"index_topk_blocks", mut(func(s *minimaxOracleSpec) { s.topKBlocks = 64 })},
		{"index_local_blocks", mut(func(s *minimaxOracleSpec) { s.localBlocks = 0 })},
		{"block max-pool", mut(func(s *minimaxOracleSpec) { s.blockPoolMax = false })},
		{"MSA layer dispatch", mut(func(s *minimaxOracleSpec) { s.sparse[1], s.sparse[2] = false, false })},
		{"swiglu_limit clamp", mut(func(s *minimaxOracleSpec) { s.swigluLimit = 0 })},
		{"swiglu_alpha", mut(func(s *minimaxOracleSpec) { s.swigluAlpha = 1.702 })},
		{"dense layer OAI gate", mut(func(s *minimaxOracleSpec) { s.denseOAI = false })},
		{"norm_topk_prob", mut(func(s *minimaxOracleSpec) { s.normTopKProb = false })},
		{"routed_scaling_factor", mut(func(s *minimaxOracleSpec) { s.routedScaling = 1 })},
		{"e_score_correction_bias selects", mut(func(s *minimaxOracleSpec) { s.selectOnCorrected = false })},
		{"router weight is the RAW sigmoid", mut(func(s *minimaxOracleSpec) { s.weightFromCorrected = true })},
		{"shared expert present", mut(func(s *minimaxOracleSpec) { s.sharedExpert = false })},
		{"shared expert is UNSCALED", mut(func(s *minimaxOracleSpec) { s.sharedScaled = true })},
	} {
		t.Run(tc.axis, func(t *testing.T) {
			minimaxOracleAssertAxisLive(t, f, base, tc.spec, tc.axis)
		})
	}
}

// TestMiniMaxCPUNumericOracleIsSensitive proves the comparison is non-vacuous:
// perturbing ONE raw fixture element in each distinct weight ROLE must move the compared
// logits far beyond the tolerance. The lightning-indexer tensors, the router and its
// score-correction bias, the batched routed experts, the fused shared expert and the
// untied head are all included: a reference that ignored any of them — or that resolved
// the head to the tied embedding — would stay green without these.
//
// Two roles need a perturbation site chosen with care, and the reason is worth stating
// because it is a property of the model rather than of the test. Element 0 of
// model.embed_tokens.weight is row 0 — the embedding of token id 0, which this prompt
// never contains and which the untied lm_head does not reuse either, so it is genuinely
// dead weight and perturbing it SHOULD be a no-op; the probe uses the row of the
// prompt's first token instead. And the lightning indexer influences the output only
// DISCRETELY, through which blocks it selects: its per-head q vector is RMSNormed, so a
// perturbation cannot grow that head's score without bound, and a head that never wins
// the cross-head max-pool cannot change any block ranking however hard it is pushed. A
// sweep over all 12 rows of indexer q_proj at this fixture confirms the role is live
// (7 of 12 rows flip a selection at delta 4, 11 of 12 at delta 100) while row 0 — the
// element-0 site — is not among them, so the probe uses row 2, which flips a selection
// at every delta tried. Neither adjustment weakens an assertion: the tolerance, the
// comparison and the pass condition are unchanged.
func TestMiniMaxCPUNumericOracleIsSensitive(t *testing.T) {
	const H = 24 // minimaxOracleCfg().HiddenSize, i.e. the row stride of a [*, H] weight
	for _, tc := range []struct {
		name   string
		tensor string
		elem   int
		delta  float32
	}{
		// Row of token id 3, the prompt's first token — row 0 is a token this prompt
		// never uses, and with an untied head nothing else reads it.
		{"embed_tokens", "model.embed_tokens.weight", 3 * H, 0.5},
		{"lm_head", "lm_head.weight", 0, 0.5},
		{"final_norm", "model.norm.weight", 0, 0.5},
		{"l0_input_layernorm", layerPrefix(0) + "input_layernorm.weight", 0, 0.5},
		{"l0_q_proj", layerPrefix(0) + "self_attn.q_proj.weight", 0, 0.5},
		{"l0_k_proj", layerPrefix(0) + "self_attn.k_proj.weight", 0, 0.5},
		{"l0_v_proj", layerPrefix(0) + "self_attn.v_proj.weight", 0, 0.5},
		{"l0_o_proj", layerPrefix(0) + "self_attn.o_proj.weight", 0, 0.5},
		{"l0_q_norm", layerPrefix(0) + "self_attn.q_norm.weight", 0, 0.5},
		{"l0_k_norm", layerPrefix(0) + "self_attn.k_norm.weight", 0, 0.5},
		{"l0_post_attention_layernorm", layerPrefix(0) + "post_attention_layernorm.weight", 0, 0.5},
		{"l0_dense_gate_proj", layerPrefix(0) + "mlp.gate_proj.weight", 0, 0.5},
		{"l0_dense_up_proj", layerPrefix(0) + "mlp.up_proj.weight", 0, 0.5},
		{"l0_dense_down_proj", layerPrefix(0) + "mlp.down_proj.weight", 0, 0.5},
		// Row 2 of the indexer q projection: a row that does win the cross-head
		// max-pool, so the discrete block selection actually moves. See the note above.
		{"l1_indexer_q_proj", layerPrefix(1) + "self_attn.indexer.q_proj.weight", 2 * H, 4},
		{"l1_indexer_k_proj", layerPrefix(1) + "self_attn.indexer.k_proj.weight", 0, 4},
		{"l1_indexer_q_norm", layerPrefix(1) + "self_attn.indexer.q_norm.weight", 0, 4},
		{"l1_indexer_k_norm", layerPrefix(1) + "self_attn.indexer.k_norm.weight", 0, 4},
		{"l1_router", layerPrefix(1) + "mlp.gate.weight", 0, 4},
		{"l1_router_correction_bias", layerPrefix(1) + "mlp.gate.e_score_correction_bias", 0, 4},
		{"l1_experts_gate_up_proj", layerPrefix(1) + "mlp.experts.gate_up_proj.weight", 0, 0.5},
		{"l1_experts_down_proj", layerPrefix(1) + "mlp.experts.down_proj.weight", 0, 0.5},
		{"l1_shared_gate_up_proj", layerPrefix(1) + "mlp.shared_experts.gate_up_proj.weight", 0, 0.5},
		{"l1_shared_down_proj", layerPrefix(1) + "mlp.shared_experts.down_proj.weight", 0, 0.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newMinimaxOracleFixture(t)
			ids := minimaxOracleIDs
			ref := minimaxReference(t, f, ids, f.sp)

			base, ok := f.off[tc.tensor]
			if !ok {
				t.Fatalf("fixture tensor %q missing", tc.tensor)
			}
			if tc.elem >= f.n[tc.tensor] {
				t.Fatalf("probe element %d out of range for %q (%d elements)", tc.elem, tc.tensor, f.n[tc.tensor])
			}
			// Perturb one element of the role in the model's live bytes. The
			// batched/fused rows were removed from the manifest by the loader's zero-copy
			// splits, but the split views alias these very bytes, so this reaches them.
			off := base + tc.elem
			orig := math.Float32frombits(binary.LittleEndian.Uint32(f.m.raw[off*4:]))
			binary.LittleEndian.PutUint32(f.m.raw[off*4:], math.Float32bits(orig+tc.delta))

			act := f.m.Forward(ids)
			var worst float64
			for tt := range ids {
				if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > worst {
					worst = d
				}
			}
			if worst <= cpuOracleTol {
				t.Fatalf("perturbed fixture still within tolerance (max|delta|=%.3e) — the oracle is vacuous for role %s", worst, tc.tensor)
			}
		})
	}
}
