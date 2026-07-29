package model

// family_deepseek_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU
// numeric oracle for DeepSeek-V2/V3's Multi-head Latent Attention (MLA). Companion to
// family_cpu_oracle_test.go, which states the doctrine, and to
// family_cohere_cpu_oracle_test.go, the other "rotary convention is the whole ballgame"
// reference.
//
// Independence discipline (family_cpu_oracle_test.go:13-24): the reference below is a
// plain scalar transcription of DeepSeek's published HF modeling code
// (modeling_deepseek.py's DeepseekV2Attention/DeepseekV2DecoderLayer, and the identical
// upstream transformers models/deepseek_v3/modeling_deepseek_v3.py). It reuses NONE of
// the production MLA machinery — no glmDsaAttentionOutputFromTopKNormed, no
// glmDsaApplyInterleavedRoPE, no rmsnorm, no ropeRowForLayer, no cachedInvFreq, no
// residentMatMulBatch. Tensors come straight out of the fixture's own HF-order bytes,
// every matmul / norm / softmax / SiLU is a naive in-order scalar loop, the rotary table
// is rebuilt from theta here, and the block dataflow is hardcoded to DeepSeek's published
// topology instead of being routed through cfg.BlockTopology / normCfg / ffnForLayer.
//
// What DeepSeek MLA actually does, transcribed claim-by-claim from the HF source:
//
//   - QUERY, low-rank: q_lora_rank is set on V2/V3, so the query is
//     q_b_proj(q_a_layernorm(q_a_proj(x))) — a DOWN-projection to q_lora_rank, an RMSNorm
//     on the latent, then an UP-projection to num_heads*q_head_dim. Only the
//     q_lora_rank=None lineage uses a single q_proj, which fak does not need to serve for
//     V2/V3 and which this fixture therefore does not build.
//   - KEY/VALUE, low-rank + decoupled rope: kv_a_proj_with_mqa(x) produces ONE vector of
//     width kv_lora_rank + qk_rope_head_dim which is SPLIT:
//     `compressed_kv, k_pe = torch.split(compressed_kv, [kv_lora_rank, qk_rope_head_dim], -1)`.
//     Only the FIRST part goes through kv_a_layernorm, and it does so BEFORE the
//     up-projection: `kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))`. k_pe is
//     NOT normed. kv is then viewed as (heads, qk_nope_head_dim + v_head_dim) and split
//     into k_nope FIRST and value SECOND — that order is load-bearing and asymmetric
//     whenever qk_nope_head_dim != v_head_dim, which is the real V2/V3 geometry
//     (128 vs 128 on V3 — equal — but 128 vs 128 hides the axis, so this fixture uses
//     12 vs 10 to keep it observable).
//   - LATENT NORM EPSILON: DeepseekV2RMSNorm's signature is
//     `def __init__(self, hidden_size, eps=1e-6)`, and the attention block constructs its
//     two latent norms WITHOUT passing config.rms_norm_eps —
//     `self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)`,
//     `self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)` — while the decoder
//     layer constructs its own norms WITH it
//     (`DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)`). So MLA runs TWO
//     epsilons: 1e-6 inside attention, config.rms_norm_eps everywhere else. Published V2/V3
//     configs set rms_norm_eps to 1e-6, which makes the two coincide on every real
//     checkpoint and makes the axis untestable there; this fixture deliberately separates
//     them (see deepseekOracleCfg) so the split is live. This is exactly what production's
//     glmDsaInnerNormEps constant encodes, and the oracle pins it rather than assuming it.
//   - DECOUPLED ROPE: the rotary applies to the `_pe` SLICE ONLY. q is split
//     `q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], -1)`, only q_pe
//     is rotated, and the head vector is reassembled as [q_nope | q_pe]. On the key side
//     there is exactly ONE k_pe per position — shape (bsz, q_len, 1, qk_rope_head_dim) —
//     and it is BROADCAST across every head: `key_states[:, :, :, nope:] = k_pe`. So the
//     rope half of the key is head-independent while the nope half is per-head.
//   - ROPE TABLE: DeepseekV2RotaryEmbedding is constructed with
//     `dim=self.qk_rope_head_dim`, so inv_freq[j] = theta^(-2j/qk_rope_head_dim). The
//     denominator is qk_rope_head_dim, NOT head_dim and NOT q_head_dim.
//   - ROPE CONVENTION: DeepSeek ships its OWN apply_rotary_pos_emb which DE-INTERLEAVES
//     before applying Llama's rotate_half:
//     `q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)`, then
//     `q*cos + rotate_half(q)*sin` against `emb = cat((freqs, freqs), -1)`. Composing the
//     two gives, for the pe slice x of width d and half = d/2:
//     out[j]      = x[2j]*cos_j - x[2j+1]*sin_j
//     out[j+half] = x[2j+1]*cos_j + x[2j]*sin_j
//     i.e. the INPUT pairs are interleaved (2j, 2j+1) — Cohere/GPT-J style — while the
//     OUTPUT is in half-split layout. This is neither plain rotate_half nor plain
//     interleaved-in-interleaved-out, and getting it wrong is a silent wrong answer:
//     shapes all match, position 0 is exact (a length-1 causal softmax is 1.0 whatever q
//     and k are), and the error only grows with position. That is the shape of the bug
//     the Cohere lane just found, so it gets its own live-axis mutation below.
//   - SOFTMAX SCALE: `self.softmax_scale = self.q_head_dim ** (-0.5)` where
//     `self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim`. The SUM. Using
//     only the nope width or only the rope width preserves every shape and every tensor
//     name and is invisible to a structural test, so it gets three live-axis mutations
//     (nope-only, rope-only, and v_head_dim — the likeliest confusion, since the attention
//     OUTPUT is v_head_dim wide). YaRN multiplies the scale by mscale^2; rope_scaling is
//     absent here, so the reference uses the bare q_head_dim**-0.5 and the test pins
//     cfg.ropeAttentionFactor() == 1 so that omission cannot silently paper over a bug.
//   - BLOCK: DeepseekV2DecoderLayer is the plain Llama PreNorm sandwich —
//     x = x + attn(input_layernorm(x)); x = x + mlp(post_attention_layernorm(x)).
//   - MLP: DeepseekV2MLP is the standard SwiGLU down(silu(gate(x)) * up(x)). This fixture
//     builds the DENSE lineage (the first_k_dense_replace layers); the routed
//     DeepseekMoE block is NOT covered here — see the residuals note on the test below.
//   - HEAD: tie_word_embeddings is False on V2/V3, so lm_head is its own tensor. The
//     fixture keeps it untied so a head perturbation is separable from an embedding one.
//   - BIASES: attention_bias defaults False, so no projection in the fixture is biased.
//
// Fixture geometry is chosen so no MLA axis can cancel: qk_nope_head_dim (12) !=
// qk_rope_head_dim (8) != v_head_dim (10), q_head_dim (20) != v_head_dim (10),
// q_lora_rank (16) != kv_lora_rank (12) != hidden_size (24), num_heads*v_head_dim (36) !=
// hidden_size (24), and qk_rope_head_dim 8 makes the de-interleaved pairing
// (0,1),(2,3),(4,5),(6,7) distinguishable from rotate_half's (0,4),(1,5),(2,6),(3,7) — at
// width 2 the two conventions coincide and the rotary axis would be unobservable.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

// deepseekOracleSpec is the DeepSeek MLA semantics the reference hardcodes. The geometry
// fields are checkpoint dimensions (legitimately read off Config — they are what the
// tensors' shapes are), but every SEMANTIC field is the published HF rule written out
// here, never read from Config: the softmax-scale width, the inv_freq denominator, the
// 1e-6 latent-norm epsilon, and the four structural booleans. Mutating one and requiring
// the reference to move is what TestDeepSeekCPUNumericOracleAxesAreLive does.
type deepseekOracleSpec struct {
	// geometry
	qkNope, qkRope, vHead int
	qLora, kvLora         int
	theta                 float64

	// semantics — the HF rules, hardcoded
	scaleDim  int     // softmax_scale = scaleDim**-0.5; HF: q_head_dim = qkNope+qkRope
	ropeDenom int     // inv_freq[j] = theta^(-2j/ropeDenom); HF: qk_rope_head_dim
	innerEps  float32 // the MLA latent norms' eps; HF: DeepseekV2RMSNorm's 1e-6 default

	normQLatent  bool // HF true: q_a_layernorm between q_a_proj and q_b_proj
	normKVLatent bool // HF true: kv_a_layernorm BEFORE kv_b_proj, on the latent only
	ropeOnK      bool // HF true: k_pe is rotated too, not just q_pe
	ropeSliceAll bool // HF false: rotary touches the _pe slice ONLY, never _nope
	rotateHalf   bool // HF false: DeepSeek de-interleaves first, Llama does not
	kPEFirst     bool // HF false: the key head is [k_nope | k_pe], not [k_pe | k_nope]
	vFirst       bool // HF false: kv_b output splits as [k_nope | value], not [value | k_nope]
}

// deepseekOracleSpecFor writes out DeepSeek's published MLA semantics for a fixture's
// geometry. The scale width, the rope denominator and the latent epsilon are the three
// values a plausible implementation gets wrong without changing a single shape.
func deepseekOracleSpecFor(cfg Config) deepseekOracleSpec {
	return deepseekOracleSpec{
		qkNope: cfg.QKNopeHeadDim,
		qkRope: cfg.QKRopeHeadDim,
		vHead:  cfg.VHeadDim,
		qLora:  cfg.QLoraRank,
		kvLora: cfg.KVLoraRank,
		theta:  cfg.RopeTheta,

		scaleDim:  cfg.QKNopeHeadDim + cfg.QKRopeHeadDim, // HF: q_head_dim
		ropeDenom: cfg.QKRopeHeadDim,                     // HF: DeepseekV2RotaryEmbedding(dim=qk_rope_head_dim)
		innerEps:  1e-6,                                  // HF: DeepseekV2RMSNorm's default eps

		normQLatent:  true,
		normKVLatent: true,
		ropeOnK:      true,
		ropeSliceAll: false,
		rotateHalf:   false,
		kPEFirst:     false,
		vFirst:       false,
	}
}

// deepseekOracleCosSin is one (cos, sin) pair of DeepSeek's rotary table: angle
// pos * theta^(-2j/ropeDenom). Rebuilt from theta here rather than read from
// cachedInvFreq/ropeRowForLayer.
func deepseekOracleCosSin(j, pos int, spec deepseekOracleSpec) (float32, float32) {
	angle := float64(pos) / math.Pow(spec.theta, float64(2*j)/float64(spec.ropeDenom))
	return float32(math.Cos(angle)), float32(math.Sin(angle))
}

// deepseekOracleRopePE applies DeepSeek's apply_rotary_pos_emb to ONE slice at position
// pos and returns the rotated copy. The HF composition (view(d/2,2).transpose(4,3) then
// rotate_half against cat((freqs,freqs))) reduces to interleaved-in / half-split-out:
//
//	out[j]      = x[2j]*cos_j   - x[2j+1]*sin_j
//	out[j+half] = x[2j+1]*cos_j + x[2j]*sin_j
//
// spec.rotateHalf selects the WRONG-but-plausible Llama pairing (j, j+half) instead, for
// the live-axis test. Deliberately NOT glmDsaApplyInterleavedRoPE and not applyRopeRow.
func deepseekOracleRopePE(x []float32, pos int, spec deepseekOracleSpec) []float32 {
	half := len(x) / 2
	out := make([]float32, len(x))
	for j := 0; j < half; j++ {
		c, s := deepseekOracleCosSin(j, pos, spec)
		a, b := x[2*j], x[2*j+1]
		if spec.rotateHalf {
			a, b = x[j], x[j+half]
		}
		out[j] = a*c - b*s
		out[j+half] = b*c + a*s
	}
	return out
}

// deepseekOracleCfg is the tiny DeepSeek-V2/V3 dense-MLA fixture config.
//
// HeadDim is set to q_head_dim (qk_nope + qk_rope), which is what a DeepSeek export
// carries and what TestOptionalDeepSeekV2OracleDocumentsMLABoundary requires of a real
// checkpoint; deriveConfigAxes' hidden/heads fallback is covered separately by
// TestDeepSeekMLADerivedHeadDimCoversRopeSlice.
//
// RMSNormEps is deliberately 1e-4, NOT the 1e-6 every published DeepSeek config uses, so
// that the two-epsilon split (config eps outside attention, DeepseekV2RMSNorm's 1e-6
// default on the two MLA latent norms) is numerically observable instead of coincidental.
// The separation has to be this wide to clear cpuOracleTol: at the more familiar 1e-5 the
// axis moves the logits by only 3.9e-5 and TestDeepSeekCPUNumericOracleAxesAreLive
// correctly reports it inert. Nothing else in the fixture depends on the value.
func deepseekOracleCfg() Config {
	return Config{
		HiddenSize:       24,
		NumLayers:        3,
		NumHeads:         3,
		NumKVHeads:       3,
		HeadDim:          20, // q_head_dim = qk_nope 12 + qk_rope 8
		IntermediateSize: 40,
		VocabSize:        53,
		ModelType:        "deepseek2",
		Architectures:    []string{"DeepseekV2ForCausalLM"},
		HiddenAct:        "silu",
		RMSNormEps:       1e-4,
		RopeTheta:        10000,
		EOSTokenID:       -1,

		QLoraRank:     16,
		KVLoraRank:    12,
		QKNopeHeadDim: 12,
		QKRopeHeadDim: 8,
		VHeadDim:      10,
		IndexNHeads:   0, // DeepSeek is GLM-MLA MINUS the DSA lightning indexer

		TieWordEmbeddings: false,
	}
}

// deepseekOracleTensors is DeepSeek's REAL dense-MLA tensor roster at the fixture
// geometry: the two low-rank query tensors plus the query latent norm, the fused
// kv_a_proj_with_mqa (kv_lora_rank + qk_rope_head_dim wide) plus the kv latent norm sized
// to kv_lora_rank ONLY, the kv up-projection to heads*(qk_nope + v_head), o_proj reading
// heads*v_head, the Llama-shaped sandwich norms, a dense SwiGLU MLP, and an UNTIED
// lm_head. No projection biases (attention_bias False), no indexer tensors.
func deepseekOracleTensors(cfg Config) []synthTensor {
	nH := cfg.NumHeads
	qkNope, qkRope, vHead := cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.VHeadDim
	qkHead := qkNope + qkRope
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	type ts = synthTensor
	out := []ts{
		{"model.embed_tokens.weight", []int{V, H}},
		{"lm_head.weight", []int{V, H}},
	}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		ap := p + "self_attn."
		out = append(out,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{ap + "q_a_proj.weight", []int{cfg.QLoraRank, H}},
			ts{ap + "q_a_layernorm.weight", []int{cfg.QLoraRank}},
			ts{ap + "q_b_proj.weight", []int{nH * qkHead, cfg.QLoraRank}},
			ts{ap + "kv_a_proj_with_mqa.weight", []int{cfg.KVLoraRank + qkRope, H}},
			ts{ap + "kv_a_layernorm.weight", []int{cfg.KVLoraRank}},
			ts{ap + "kv_b_proj.weight", []int{nH * (qkNope + vHead), cfg.KVLoraRank}},
			ts{ap + "o_proj.weight", []int{H, nH * vHead}},
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{p + "mlp.gate_proj.weight", []int{I, H}},
			ts{p + "mlp.up_proj.weight", []int{I, H}},
			ts{p + "mlp.down_proj.weight", []int{H, I}},
		)
	}
	return append(out, ts{"model.norm.weight", []int{H}})
}

// newDeepSeekOracleModel builds the fixture and loads it through newHFCheckpointModel,
// the constructor every HF-source f32 loader funnels through. It also returns a PRISTINE
// copy of the checkpoint bytes in HF order, which the reference reads — so the reference
// sees the download exactly as HuggingFace wrote it and stays independent of any
// load-time layout normalization the loader does now or grows later.
func newDeepSeekOracleModel(t *testing.T) (*Model, []byte) {
	t.Helper()
	cfg := deepseekOracleCfg()
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	man, raw := synthBuildRaw(deepseekOracleTensors(cfg), func(name string, next func() float32) float32 {
		if isCPUOracleNormWeight(name) {
			return 1 + 0.25*next() // distinct NON-UNIT gains, well-conditioned
		}
		return synthMatmulFill(name, next)
	})
	hf := append([]byte(nil), raw...)
	m, err := newHFCheckpointModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newHFCheckpointModel(deepseek fixture): %v", err)
	}
	return m, hf
}

// deepseekReference runs the independent DeepSeek MLA forward: per-position logits for
// ids, under the semantics in spec. Every step is the HF DeepseekV2Attention /
// DeepseekV2DecoderLayer dataflow, hardcoded.
func deepseekReference(t *testing.T, m *Model, hf []byte, ids []int, spec deepseekOracleSpec) [][]float32 {
	t.Helper()
	cfg := m.Cfg
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH := cfg.NumHeads
	qkNope, qkRope, vHead := spec.qkNope, spec.qkRope, spec.vHead
	qkHead := qkNope + qkRope
	kvHead := qkNope + vHead
	qLora, kvLora := spec.qLora, spec.kvLora
	eps := float32(cfg.RMSNormEps) // the decoder-layer norms
	seq := len(ids)

	// tensor reads the PRISTINE HF-order bytes, not m.raw.
	tensor := func(name string) []float32 {
		t.Helper()
		meta, ok := m.manifest[name]
		if !ok {
			t.Fatalf("fixture tensor %q missing from manifest", name)
		}
		n := meta.Nbytes / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(hf[meta.Offset+i*4:]))
		}
		return out
	}

	embed := tensor("model.embed_tokens.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		ap := p + "self_attn."
		inNorm := tensor(p + "input_layernorm.weight")
		wQA := tensor(ap + "q_a_proj.weight")
		qANorm := tensor(ap + "q_a_layernorm.weight")
		wQB := tensor(ap + "q_b_proj.weight")
		wKVA := tensor(ap + "kv_a_proj_with_mqa.weight")
		kvANorm := tensor(ap + "kv_a_layernorm.weight")
		wKVB := tensor(ap + "kv_b_proj.weight")
		wO := tensor(ap + "o_proj.weight")
		postNorm := tensor(p + "post_attention_layernorm.weight")
		wg := tensor(p + "mlp.gate_proj.weight")
		wu := tensor(p + "mlp.up_proj.weight")
		wd := tensor(p + "mlp.down_proj.weight")

		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn := cpuOracleRMSNorm(x[tt], inNorm, eps)

			// Query: down-project to the latent, norm it, up-project to heads*q_head_dim.
			qLatent := cpuOracleMatVec(wQA, xn, qLora, H)
			if spec.normQLatent {
				qLatent = cpuOracleRMSNorm(qLatent, qANorm, spec.innerEps)
			}
			qFull := cpuOracleMatVec(wQB, qLatent, nH*qkHead, qLora)

			// Key/value: ONE projection to [kv_lora | k_pe]; only the latent is normed,
			// and the norm runs BEFORE the up-projection.
			ckv := cpuOracleMatVec(wKVA, xn, kvLora+qkRope, H)
			latent := append([]float32(nil), ckv[:kvLora]...)
			kPE := append([]float32(nil), ckv[kvLora:]...)
			if spec.normKVLatent {
				latent = cpuOracleRMSNorm(latent, kvANorm, spec.innerEps)
			}
			kv := cpuOracleMatVec(wKVB, latent, nH*kvHead, kvLora)

			// ONE k_pe for the position, rotated ONCE and broadcast to every head.
			kPERot := append([]float32(nil), kPE...)
			if spec.ropeOnK {
				kPERot = deepseekOracleRopePE(kPE, tt, spec)
			}

			q[tt] = make([]float32, nH*qkHead)
			k[tt] = make([]float32, nH*qkHead)
			v[tt] = make([]float32, nH*vHead)
			for h := 0; h < nH; h++ {
				qSrc := qFull[h*qkHead : (h+1)*qkHead]
				qDst := q[tt][h*qkHead : (h+1)*qkHead]
				kvSrc := kv[h*kvHead : (h+1)*kvHead]
				kDst := k[tt][h*qkHead : (h+1)*qkHead]

				kNope, vh := kvSrc[:qkNope], kvSrc[qkNope:]
				if spec.vFirst {
					vh, kNope = kvSrc[:vHead], kvSrc[vHead:]
				}

				switch {
				case spec.ropeSliceAll:
					// WRONG-but-plausible: rotate the whole head, _nope included.
					copy(qDst, deepseekOracleRopePE(qSrc, tt, spec))
					full := make([]float32, qkHead)
					copy(full[:qkNope], kNope)
					copy(full[qkNope:], kPE)
					copy(kDst, deepseekOracleRopePE(full, tt, spec))
				case spec.kPEFirst:
					// WRONG-but-plausible: the key head assembled as [k_pe | k_nope].
					copy(qDst[:qkNope], qSrc[:qkNope])
					copy(qDst[qkNope:], deepseekOracleRopePE(qSrc[qkNope:], tt, spec))
					copy(kDst[:qkRope], kPERot)
					copy(kDst[qkRope:], kNope)
				default:
					copy(qDst[:qkNope], qSrc[:qkNope])
					copy(qDst[qkNope:], deepseekOracleRopePE(qSrc[qkNope:], tt, spec))
					copy(kDst[:qkNope], kNope)
					copy(kDst[qkNope:], kPERot)
				}
				copy(v[tt][h*vHead:(h+1)*vHead], vh)
			}
		}

		// Causal attention, MHA (MLA materializes a full per-head key, so there is no GQA
		// grouping here), scaled by q_head_dim**-0.5.
		scale := float32(1.0 / math.Sqrt(float64(spec.scaleDim)))
		attnOut := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			concat := make([]float32, nH*vHead)
			for h := 0; h < nH; h++ {
				qh := q[tt][h*qkHead : (h+1)*qkHead]
				scores := make([]float32, tt+1)
				for j := 0; j <= tt; j++ {
					kh := k[j][h*qkHead : (h+1)*qkHead]
					var s float32
					for d := 0; d < qkHead; d++ {
						s += qh[d] * kh[d]
					}
					scores[j] = s * scale
				}
				cpuOracleSoftmax(scores)
				oh := concat[h*vHead : (h+1)*vHead]
				for j := 0; j <= tt; j++ {
					vh := v[j][h*vHead : (h+1)*vHead]
					for d := 0; d < vHead; d++ {
						oh[d] += scores[j] * vh[d]
					}
				}
			}
			attnOut[tt] = cpuOracleMatVec(wO, concat, H, nH*vHead)
		}
		for tt := 0; tt < seq; tt++ {
			for i := 0; i < H; i++ {
				x[tt][i] += attnOut[tt][i]
			}
		}

		// PreNorm MLP sublayer on the SECOND norm.
		for tt := 0; tt < seq; tt++ {
			xn := cpuOracleRMSNorm(x[tt], postNorm, eps)
			gate := cpuOracleMatVec(wg, xn, I, H)
			up := cpuOracleMatVec(wu, xn, I, H)
			for i := 0; i < I; i++ {
				gate[i] = cpuOracleSilu(gate[i]) * up[i]
			}
			down := cpuOracleMatVec(wd, gate, H, I)
			for i := 0; i < H; i++ {
				x[tt][i] += down[i]
			}
		}
	}

	norm := tensor("model.norm.weight")
	head := tensor("lm_head.weight") // untied on V2/V3
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		logits[tt] = cpuOracleMatVec(head, cpuOracleRMSNorm(x[tt], norm, eps), V, H)
	}
	return logits
}

var deepseekOracleIDs = []int{3, 17, 5, 23, 41, 2, 19}

// deepseekOracleAssertDerivedAxes pins the config derivation the reference depends on.
// The reference hardcodes DeepSeek's topology, so if derivation ever routed a deepseek2
// checkpoint somewhere else the comparison would be meaningless rather than red.
func deepseekOracleAssertDerivedAxes(t *testing.T, cfg Config) {
	t.Helper()
	if !cfg.usesMLAMoELayout() {
		t.Fatal("deepseek2 derived usesMLAMoELayout() = false — the MLA forward would not run")
	}
	if cfg.isGLMMoeDsa() {
		t.Fatal("deepseek2 derived isGLMMoeDsa() = true — it must NOT take the DSA lightning-indexer branch")
	}
	if cfg.BlockTopology != PreNorm {
		t.Fatalf("deepseek2 derived topology = %v, want PreNorm (DeepseekV2DecoderLayer is the Llama sandwich)", cfg.BlockTopology)
	}
	if cfg.LayerNorm {
		t.Fatal("deepseek2 derived LayerNorm = true, want false (DeepseekV2RMSNorm has no mean subtraction)")
	}
	if cfg.DenseMLP {
		t.Fatal("deepseek2 derived DenseMLP = true, want false (DeepseekV2MLP is SwiGLU gate/up/down)")
	}
	if cfg.IsMoE() {
		t.Fatal("dense-MLA fixture derived IsMoE() = true — this fixture builds no expert tensors")
	}
	// The rotary denominator is qk_rope_head_dim, not head_dim: the reference builds its
	// angles that way, so a derivation that disagreed would make the match accidental.
	if got := cfg.invFreqDenom(); got != cfg.QKRopeHeadDim {
		t.Fatalf("deepseek2 invFreqDenom() = %d, want qk_rope_head_dim = %d", got, cfg.QKRopeHeadDim)
	}
	// No rope_scaling on this fixture, so the MLA softmax scale must reduce to the bare
	// q_head_dim**-0.5 the reference uses (YaRN's mscale would multiply it).
	if got := cfg.ropeAttentionFactor(); got != 1 {
		t.Fatalf("deepseek2 ropeAttentionFactor() = %v, want 1 (no rope_scaling on this fixture)", got)
	}
	// A rope table shorter than the pe slice would index out of range in the MLA rotary.
	if cfg.HeadDim < cfg.QKRopeHeadDim {
		t.Fatalf("derived head_dim %d < qk_rope_head_dim %d: the rotary table cannot cover the pe slice",
			cfg.HeadDim, cfg.QKRopeHeadDim)
	}
}

// TestDeepSeekCPUNumericOracle is the DeepSeek-MLA x cpu M4 witness: the production
// forward (cacheless Forward, and the cached Prefill/Step decode path) must reproduce the
// independent HF-semantics reference on every position within cpuOracleTol.
//
// It covers, on both the prefill implementation (glm_dsa.go
// glmDsaAttentionOutputFromTopKNormed) and the independently written decode
// implementation (glm_dsa_session.go glmDsaAppendAttentionKV + glmDsaAttendCached): the
// low-rank q down/up projections around q_a_layernorm; the fused kv_a_proj_with_mqa split
// into latent + k_pe with kv_a_layernorm on the latent ONLY and BEFORE kv_b_proj; the
// [k_nope | value] split of the kv up-projection; decoupled RoPE on the _pe slice only,
// with a single shared k_pe broadcast to every head; DeepSeek's de-interleaving rotary
// convention; the inv_freq denominator qk_rope_head_dim; the q_head_dim**-0.5 softmax
// scale over the SUM of the nope and rope widths; the two distinct RMSNorm epsilons; the
// Llama PreNorm sandwich; the dense SwiGLU MLP; and the untied head.
//
// If this test reds, the honesty fence demotes the cell back to M3 (drop the covmatrix
// OracleInCI bit with it).
func TestDeepSeekCPUNumericOracle(t *testing.T) {
	m, hf := newDeepSeekOracleModel(t)
	deepseekOracleAssertDerivedAxes(t, m.Cfg)

	ids := deepseekOracleIDs
	spec := deepseekOracleSpecFor(m.Cfg)
	ref := deepseekReference(t, m, hf, ids, spec)

	act := m.Forward(ids)
	for tt := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > cpuOracleTol {
			t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", tt, d, cpuOracleTol)
		}
	}

	// Cached decode path: Prefill then Step must match the reference at the same
	// positions. The reference is cacheless, so Step(id) at position len(ids) is compared
	// against a reference run over the extended prompt.
	s := m.NewSession()
	pf := s.Prefill(ids)
	if d := cpuOracleMaxAbsDiff(pf, ref[len(ids)-1]); d > cpuOracleTol {
		t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
	next := 11
	st := s.Step(next)
	extRef := deepseekReference(t, m, hf, append(append([]int(nil), ids...), next), spec)
	if d := cpuOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
		t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
}

// deepseekOracleAssertAxisLive proves one DeepSeek-specific axis is not numerically inert
// in this fixture: the reference under the HF spec must differ from the reference under a
// deliberately WRONG spec by more than the tolerance. Without this, "production matches
// the reference" could be green simply because the axis does nothing at this scale — a
// vacuous witness. Reference-to-reference, so nothing production does can satisfy it.
func deepseekOracleAssertAxisLive(t *testing.T, m *Model, hf []byte, base [][]float32, wrong deepseekOracleSpec, axis string) {
	t.Helper()
	alt := deepseekReference(t, m, hf, deepseekOracleIDs, wrong)
	var worst float64
	for tt := range deepseekOracleIDs {
		if d := cpuOracleMaxAbsDiff(base[tt], alt[tt]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Errorf("axis %q is inert in this fixture (max|delta| = %.3e <= tol %.0e): the oracle would stay green with that axis wrong",
			axis, worst, cpuOracleTol)
	}
}

// TestDeepSeekCPUNumericOracleAxesAreLive is the anti-vacuity gate on the SEMANTIC axes —
// the ones a size or scale choice could accidentally switch off, and the ones a plausible
// MLA implementation gets wrong while preserving every tensor name and every shape. Each
// is mutated to a wrong-but-plausible value and the reference must move.
func TestDeepSeekCPUNumericOracleAxesAreLive(t *testing.T) {
	m, hf := newDeepSeekOracleModel(t)
	spec := deepseekOracleSpecFor(m.Cfg)
	base := deepseekReference(t, m, hf, deepseekOracleIDs, spec)

	// The softmax scale must use the SUM of the nope and rope widths. All three wrong
	// denominators below preserve every shape.
	nopeOnly := spec
	nopeOnly.scaleDim = spec.qkNope
	deepseekOracleAssertAxisLive(t, m, hf, base, nopeOnly, "softmax_scale over qk_nope_head_dim only")

	ropeOnly := spec
	ropeOnly.scaleDim = spec.qkRope
	deepseekOracleAssertAxisLive(t, m, hf, base, ropeOnly, "softmax_scale over qk_rope_head_dim only")

	vScale := spec
	vScale.scaleDim = spec.vHead
	deepseekOracleAssertAxisLive(t, m, hf, base, vScale, "softmax_scale over v_head_dim")

	// Decoupled rope: the rotary touches the _pe slice ONLY.
	allRope := spec
	allRope.ropeSliceAll = true
	deepseekOracleAssertAxisLive(t, m, hf, base, allRope, "rotary confined to the _pe slice")

	noKRope := spec
	noKRope.ropeOnK = false
	deepseekOracleAssertAxisLive(t, m, hf, base, noKRope, "k_pe is rotated, not just q_pe")

	// The rotary convention and its table.
	half := spec
	half.rotateHalf = true
	deepseekOracleAssertAxisLive(t, m, hf, base, half, "DeepSeek de-interleaving rotary vs Llama rotate_half")

	denom := spec
	denom.ropeDenom = spec.qkNope + spec.qkRope
	deepseekOracleAssertAxisLive(t, m, hf, base, denom, "inv_freq denominator qk_rope_head_dim vs q_head_dim")

	// The latent path: both norms, and the two split orders.
	noQNorm := spec
	noQNorm.normQLatent = false
	deepseekOracleAssertAxisLive(t, m, hf, base, noQNorm, "q_a_layernorm between q_a_proj and q_b_proj")

	noKVNorm := spec
	noKVNorm.normKVLatent = false
	deepseekOracleAssertAxisLive(t, m, hf, base, noKVNorm, "kv_a_layernorm before kv_b_proj")

	kFirst := spec
	kFirst.kPEFirst = true
	deepseekOracleAssertAxisLive(t, m, hf, base, kFirst, "key head assembled as [k_nope | k_pe]")

	vFirst := spec
	vFirst.vFirst = true
	deepseekOracleAssertAxisLive(t, m, hf, base, vFirst, "kv_b output split as [k_nope | value]")

	// The two-epsilon split: the MLA latent norms use DeepseekV2RMSNorm's 1e-6 default,
	// NOT config.rms_norm_eps. Real configs set rms_norm_eps to 1e-6 so the axis is
	// coincidental there; this fixture separates them (deepseekOracleCfg) to keep it
	// observable.
	cfgEps := spec
	cfgEps.innerEps = float32(m.Cfg.RMSNormEps)
	deepseekOracleAssertAxisLive(t, m, hf, base, cfgEps, "MLA latent-norm eps 1e-6 vs config rms_norm_eps")
}

// TestDeepSeekCPUNumericOracleIsSensitive proves the comparison is non-vacuous on the
// WEIGHT axis: perturbing ONE raw fixture element must move the compared logits far
// beyond the tolerance. Every distinct MLA weight role is listed, including both halves
// of the fused kv_a_proj_with_mqa (its first kv_lora_rank rows feed the latent; its last
// qk_rope_head_dim rows feed k_pe, which reaches attention only through the decoupled
// rotary) and both halves of a q_b_proj head row (its nope prefix bypasses the rotary; its
// pe suffix does not). A reference that ignored the latent norms, dropped the k_pe rows,
// or used a tied head would stay green without these.
func TestDeepSeekCPUNumericOracleIsSensitive(t *testing.T) {
	cfg := deepseekOracleCfg()
	qkNope, kvLora := cfg.QKNopeHeadDim, cfg.KVLoraRank
	H := cfg.HiddenSize

	for _, tc := range []struct {
		name   string
		tensor string
		elem   int // f32 index within the tensor
	}{
		{"input_layernorm", "model.layers.0.input_layernorm.weight", 0},
		{"post_attention_layernorm", "model.layers.0.post_attention_layernorm.weight", 0},
		{"final_norm", "model.norm.weight", 0},
		{"q_a_layernorm", "model.layers.0.self_attn.q_a_layernorm.weight", 0},
		{"kv_a_layernorm", "model.layers.0.self_attn.kv_a_layernorm.weight", 0},
		{"q_a_proj_down", "model.layers.0.self_attn.q_a_proj.weight", 0},
		{"q_b_proj_up_nope", "model.layers.0.self_attn.q_b_proj.weight", 0},
		{"q_b_proj_up_pe", "model.layers.0.self_attn.q_b_proj.weight", qkNope * cfg.QLoraRank},
		{"kv_a_proj_latent_rows", "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", 0},
		{"kv_a_proj_k_pe_rows", "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", kvLora * H},
		{"kv_b_proj_k_nope", "model.layers.0.self_attn.kv_b_proj.weight", 0},
		{"kv_b_proj_value", "model.layers.0.self_attn.kv_b_proj.weight", qkNope * kvLora},
		{"o_proj", "model.layers.0.self_attn.o_proj.weight", 0},
		{"mlp_down_proj", "model.layers.0.mlp.down_proj.weight", 0},
		{"lm_head", "lm_head.weight", 0},
		// Row deepseekOracleIDs[0], not row 0: the head is UNTIED, so an embedding row no
		// prompt token selects is genuinely dead weight and perturbing it proves nothing.
		{"embed_tokens", "model.embed_tokens.weight", deepseekOracleIDs[0] * H},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, hf := newDeepSeekOracleModel(t)
			ids := deepseekOracleIDs
			ref := deepseekReference(t, m, hf, ids, deepseekOracleSpecFor(m.Cfg))

			meta, ok := m.manifest[tc.tensor]
			if !ok {
				t.Fatalf("fixture tensor %q missing", tc.tensor)
			}
			if tc.elem*4 >= meta.Nbytes {
				t.Fatalf("fixture tensor %q has %d f32 values, element %d is out of range", tc.tensor, meta.Nbytes/4, tc.elem)
			}
			off := meta.Offset + tc.elem*4
			orig := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[off:]))
			binary.LittleEndian.PutUint32(m.raw[off:], math.Float32bits(orig+0.5))

			act := m.Forward(ids)
			var worst float64
			for tt := range ids {
				if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > worst {
					worst = d
				}
			}
			if worst <= cpuOracleTol {
				t.Fatalf("perturbed fixture still within tolerance (max|delta|=%.3e) — the oracle is vacuous", worst)
			}
		})
	}
}

// TestDeepSeekMLADerivedHeadDimCoversRopeSlice pins the ONE derived axis the numeric
// oracle above cannot reach, because the oracle fixture states head_dim explicitly.
//
// The HF semantics. DeepseekV2Config has NO head_dim field and DeepseekV2Attention never
// computes hidden_size/num_heads; it builds
//
//	self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
//
// and every per-head width in the MLA block is cut from that. For a real DeepSeek the two
// numbers are wildly different: DeepSeek-V3 is hidden_size 7168 / 128 heads = 56, while
// q_head_dim = 128 + 64 = 192. hidden_size/num_heads is not a lossy approximation of the
// MLA head width, it is unrelated to it.
//
// Why fak has to care. deriveConfigAxes' generic fallback fills an absent head_dim with
// hidden_size/num_heads. Nothing downstream re-derives it for MLA, and the MLA rotary
// consumes it: cachedInvFreq sizes the cos/sin row at rotaryDim()/2 == HeadDim/2, while
// glmDsaApplyInterleavedRoPE (glm_dsa.go, shared by the prefill and the decode paths)
// rotates the qk_rope_head_dim-wide _pe slice and therefore indexes cos[0 : qk_rope/2].
// So the forward requires HeadDim >= QKRopeHeadDim, and on DeepSeek-V3's real numbers the
// fallback yields 56 < 64 — a slice-bounds panic on the first token, not a wrong number.
//
// The path is reachable, and it is the ONLY way to run a stock DeepSeek checkpoint: fak
// gates the MLA forward on model_type == "deepseek2" (usesMLAMoELayout), which is
// llama.cpp's arch spelling, while a published DeepSeek config.json says "deepseek_v2" /
// "deepseek_v3". Anyone pointing fak at a real HF DeepSeek directory must therefore edit
// model_type to "deepseek2" — and the config they are editing has no head_dim key to
// carry over, because HF never emits one.
//
// This is a reference-to-production test: the wanted value is HF's rule written out here,
// not anything read back from the config.
func TestDeepSeekMLADerivedHeadDimCoversRopeSlice(t *testing.T) {
	// DeepSeek-V3's published config.json geometry, with model_type retyped to the
	// "deepseek2" spelling fak's MLA forward gates on. No head_dim: HF emits none.
	const cfgJSON = `{
		"model_type": "deepseek2",
		"architectures": ["DeepseekV3ForCausalLM"],
		"hidden_size": 7168,
		"num_attention_heads": 128,
		"num_key_value_heads": 128,
		"num_hidden_layers": 61,
		"intermediate_size": 18432,
		"vocab_size": 129280,
		"rms_norm_eps": 1e-6,
		"rope_theta": 10000,
		"q_lora_rank": 1536,
		"kv_lora_rank": 512,
		"qk_nope_head_dim": 128,
		"qk_rope_head_dim": 64,
		"v_head_dim": 128
	}`

	var cfg Config
	if err := json.Unmarshal([]byte(cfgJSON), &cfg); err != nil {
		t.Fatalf("decode deepseek config: %v", err)
	}
	if !cfg.usesMLAMoELayout() {
		t.Fatal("fixture config does not take the MLA forward — the axis under test is unreachable")
	}

	// HF: q_head_dim = qk_nope_head_dim + qk_rope_head_dim.
	wantHead := 128 + 64
	if cfg.HeadDim != wantHead {
		t.Errorf("derived head_dim = %d, want q_head_dim = qk_nope %d + qk_rope %d = %d "+
			"(hidden_size/num_heads = %d is not the MLA head width)",
			cfg.HeadDim, cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, wantHead,
			cfg.HiddenSize/cfg.NumHeads)
	}

	// The consequence the forward actually depends on: the rotary row must cover the half
	// of the _pe slice glmDsaApplyInterleavedRoPE indexes.
	inv := cachedInvFreq(cfg, 0)
	if need := cfg.QKRopeHeadDim / 2; len(inv) < need {
		t.Fatalf("rotary table has %d frequencies but the MLA rotary indexes %d of them "+
			"(qk_rope_head_dim %d / 2): glmDsaApplyInterleavedRoPE panics on the first token",
			len(inv), need, cfg.QKRopeHeadDim)
	}

	// The frequencies themselves must still be denominated by qk_rope_head_dim, exactly as
	// the oracle's reference builds them — widening head_dim must not disturb that.
	for j := 0; j < cfg.QKRopeHeadDim/2; j++ {
		want := 1.0 / math.Pow(cfg.RopeTheta, float64(2*j)/float64(cfg.QKRopeHeadDim))
		if math.Abs(inv[j]-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("inv_freq[%d] = %v, want theta^(-2j/qk_rope_head_dim) = %v", j, inv[j], want)
		}
	}
}
