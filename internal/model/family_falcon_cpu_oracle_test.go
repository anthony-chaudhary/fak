package model

// family_falcon_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU
// numeric oracle for the FALCON family (covmatrix "Falcon", topology
// ParallelResidual). It is the M4 witness style
// docs/standards/support-maturity-honesty-fence.md names for the cpu cell; the
// governing doctrine (independence discipline, fixture rules, "never flip the
// covmatrix bit without the reference") is stated at the top of
// family_cpu_oracle_test.go and is not repeated here.
//
// ---------------------------------------------------------------------------
// WHICH FALCON VARIANT THIS REFERENCE ENCODES — and why exactly one
// ---------------------------------------------------------------------------
// HuggingFace transformers/models/falcon/modeling_falcon.py is really THREE
// architectures behind one model_type, and they differ in the two places that
// matter numerically (the fused-qkv memory layout and the number of block
// LayerNorms). FalconAttention._split_heads enumerates them:
//
//	new_decoder_architecture=True   (Falcon-40B / 180B)
//	    qkv.view(batch, seq, num_kv_heads, num_heads//num_kv_heads + 2, head_dim)
//	    -> INTERLEAVED per KV group: [g q-heads | 1 k-head | 1 v-head] repeated
//	       num_kv_heads times; k/v then broadcast across the group.
//	    Block carries TWO norms, ln_attn and ln_mlp (num_ln_in_parallel_attn=2).
//
//	multi_query=True                (Falcon-7B — THE VARIANT ENCODED HERE)
//	    qkv.view(batch, seq, num_heads + 2, head_dim)
//	    -> CONTIGUOUS [num_heads q-heads | 1 shared k-head | 1 shared v-head];
//	       the single k/v head is broadcast to EVERY query head (num_kv_heads=1).
//	    Block carries ONE shared input_layernorm feeding both branches
//	    (FalconDecoderLayer.forward: mlp_layernorm_out = attention_layernorm_out).
//
//	neither                          (Falcon-RW)
//	    qkv.view(batch, seq, num_heads, 3, head_dim)
//	    -> INTERLEAVED per head: [q_h | k_h | v_h]. Also ALiBi, not RoPE.
//
// This file encodes the MULTI-QUERY (Falcon-7B) variant, and only it, because
// that is the one production actually implements: materialize.go's
// materializeFalconTensors aliases self_attention.query_key_value.weight to the
// canonical fused name and fused_split.go's splitFusedProjections cuts it as
// three CONTIGUOUS axis-0 ranges q(nH*hd) | k(nKV*hd) | v(nKV*hd) — bit-exactly
// HF's multi_query layout at nKV=1, and NOT either interleaved layout. The
// single shared input_layernorm is likewise the only per-layer norm the falcon
// resolver spec (tensor_resolver.go falconSpec) and materializeFalconTensors
// know; ln_attn / ln_mlp are not mapped at all. Writing a reference that
// straddled the variants would either be untrue to HF or silently green against
// a layout production does not implement.
//
// Config axes the fixture pins (each is derived, then asserted, by the test):
// parallel_attn -> ParallelResidual (config.go), model_type falcon -> LayerNorm
// (mean-subtracting, WITH bias) + DenseMLP (dense_h_to_4h -> act -> dense_4h_to_h,
// no up_proj / no gating), hidden_act "gelu" -> ActGeluErf (exact erf GELU, which
// is what HF's get_activation("gelu") == nn.GELU() computes), multi_query ->
// NumKVHeads 1. Falcon-7B/40B/180B all ship "bias": false, so the linear layers
// carry no bias; the LayerNorms always do (nn.LayerNorm), and the fixture gives
// every one of them a distinct non-unit gain AND a distinct non-zero bias so
// norm routing and the bias term are both numerically live.
//
// Independence: the fixture is built on Falcon's REAL HF tensor names and handed
// to the production loader (newModel) so the alias + fused-split passes actually
// run; the REFERENCE below then decodes the ORIGINAL, un-split
// transformer.h.<l>.self_attention.query_key_value.weight straight from the
// manifest bytes and performs its OWN HF _split_heads transcription. So a defect
// in the production split is a divergence, not a shared assumption. Every kernel
// below (LayerNorm, GELU, matvec, softmax, RoPE, the block dataflow) is a naive
// in-order scalar loop hardcoded to Falcon's published topology — nothing routes
// through cfg.BlockTopology, composeBlockAtLayer, normCfg, act, or ffnForLayer.

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// falconOracleCfg is the tiny Falcon-7B-shaped fixture config: multi_query
// (NumKVHeads 1), parallel_attn, RoPE, LayerNorm, dense GELU MLP, tied head.
//
// HiddenSize == NumHeads*HeadDim deliberately: HF's own multi_query projection
// width is `self.hidden_size + 2 * self.head_dim` and FalconAttention.dense is
// Linear(hidden_size, hidden_size), so Falcon BAKES IN head_dim ==
// hidden_size//num_heads. Making them differ (as the OLMo2/Qwen fixtures do)
// would be a config HF cannot express. IntermediateSize is NOT 4*HiddenSize, so
// a reference or kernel that silently assumed the ffn_hidden_size default
// instead of reading the config cannot cancel.
func falconOracleCfg() Config {
	return Config{
		HiddenSize:        24,
		NumLayers:         3,
		NumHeads:          6,
		NumKVHeads:        1, // multi_query=True: ONE shared k head and ONE shared v head
		HeadDim:           4,
		IntermediateSize:  40, // ffn_hidden_size, deliberately != 4*HiddenSize (96)
		VocabSize:         53,
		ModelType:         "falcon",
		Architectures:     []string{"FalconForCausalLM"},
		HiddenAct:         "gelu",
		RMSNormEps:        1e-5, // layer_norm_epsilon
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		ParallelAttention: true, // parallel_attn=True
	}
}

// falconOracleFill gives LayerNorm gains a distinct NON-UNIT value and LayerNorm
// biases a distinct NON-ZERO value, so neither the gain nor the bias term can be
// masked (a 1.0 gain / 0.0 bias would hide a norm-routing bug). Matmul weights
// get the deterministic shared-LCG draw.
//
// dense_h_to_4h is drawn wider on purpose. Falcon's MLP holds the block's ONLY
// nonlinearity, and at the default 0.1 scale the pre-activations cluster near
// zero, where every GELU variant agrees to ~1e-4 — an activation-class defect
// would then sit right at the tolerance floor instead of being caught outright.
// The wider draw pushes |pre-activation| into the ~1 range where exact GELU,
// tanh-GELU and SiLU visibly disagree. This changes only the fixture's operating
// point; the reference recomputes from the same bytes either way.
func falconOracleFill(name string, next func() float32) float32 {
	switch {
	case strings.HasSuffix(name, "layernorm.weight"), name == "transformer.ln_f.weight":
		return 1 + 0.25*next() // (0.75, 1.25): non-unit, well-conditioned
	case strings.HasSuffix(name, "layernorm.bias"), name == "transformer.ln_f.bias":
		return 0.25 + 0.1*next() // [0.15, 0.35): never zero
	case name == "transformer.word_embeddings.weight":
		return next() * 0.2 // wider so distinct ids separate cleanly
	case strings.HasSuffix(name, "mlp.dense_h_to_4h.weight"):
		return next() * 0.4 // drive GELU off its near-linear region
	default:
		return next() * 0.1
	}
}

// newFalconOracleModel builds the fixture on Falcon's REAL HF tensor roster —
// transformer.word_embeddings, per-layer transformer.h.<l>.{input_layernorm,
// self_attention.query_key_value, self_attention.dense, mlp.dense_h_to_4h,
// mlp.dense_4h_to_h}, transformer.ln_f — and runs it through the production
// loader (newModel) so materializeFalconTensors + splitFusedProjections actually
// execute. The fused qkv tensor is [(nH+2)*hd, H], i.e. HF's multi_query
// qkv_out_dim = hidden_size + 2*head_dim.
func newFalconOracleModel(t *testing.T) *Model {
	t.Helper()
	cfg := falconOracleCfg()
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize

	type ts = synthTensor
	tensors := []ts{{"transformer.word_embeddings.weight", []int{V, H}}}
	for l := 0; l < cfg.NumLayers; l++ {
		p := "transformer.h." + itoa(l) + "."
		tensors = append(tensors,
			// ONE shared LayerNorm per block (weight + bias) — Falcon-7B has no
			// ln_attn/ln_mlp pair and no post_attention_layernorm.
			ts{p + "input_layernorm.weight", []int{H}},
			ts{p + "input_layernorm.bias", []int{H}},
			// FUSED qkv: rows = nH q-heads ++ 1 shared k-head ++ 1 shared v-head.
			ts{p + "self_attention.query_key_value.weight", []int{(nH + 2*nKV) * hd, H}},
			ts{p + "self_attention.dense.weight", []int{H, nH * hd}},
			// Dense (non-gated) MLP: no up_proj tensor exists in a Falcon checkpoint.
			ts{p + "mlp.dense_h_to_4h.weight", []int{I, H}},
			ts{p + "mlp.dense_4h_to_h.weight", []int{H, I}},
		)
	}
	tensors = append(tensors,
		ts{"transformer.ln_f.weight", []int{H}},
		ts{"transformer.ln_f.bias", []int{H}},
	)

	man, raw := synthBuildRaw(tensors, falconOracleFill)
	m, err := newModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newModel(falcon fixture): %v", err)
	}
	return m
}

// falconOracleLayerNorm is the plain HF nn.LayerNorm, written out in scalar form:
// subtract the mean, divide by the biased std, scale by weight, add bias. Falcon
// uses LayerNorm everywhere (NOT RMSNorm) and every one of its LayerNorms carries
// a bias, so both the mean subtraction and the bias term are load-bearing.
func falconOracleLayerNorm(x, w, b []float32, eps float32) []float32 {
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(len(x))
	var ss float32
	for _, v := range x {
		d := v - mean
		ss += d * d
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = (v-mean)*inv*w[i] + b[i]
	}
	return out
}

// falconOracleGelu is the EXACT (erf) GELU that HF's get_activation("gelu") ==
// nn.GELU() applies in FalconMLP — not the tanh approximation, and not SiLU.
func falconOracleGelu(z float32) float32 {
	z64 := float64(z)
	return float32(0.5 * z64 * (1 + math.Erf(z64/math.Sqrt2)))
}

// falconReference runs the independent Falcon-7B forward and returns per-position
// logits for ids. Every step is transcribed from HF modeling_falcon.py and
// hardcoded — the block dataflow is NOT read off cfg.BlockTopology.
//
//	FalconDecoderLayer.forward (parallel_attn=True, num_ln_in_parallel_attn==1):
//	    residual = h
//	    attention_layernorm_out = self.input_layernorm(h)
//	    attention_output = self.self_attention(attention_layernorm_out)
//	    mlp_layernorm_out = attention_layernorm_out      # THE SAME normed tensor
//	    mlp_output = self.mlp(mlp_layernorm_out)
//	    mlp_output += attention_output                   # both deltas summed...
//	    output = mlp_output + residual                   # ...into ONE residual
//
// i.e. x = x + attn(ln(x)) + mlp(ln(x)), where BOTH branches read the SAME
// pre-block x through the SAME norm. The MLP never sees the attention output.
func falconReference(t *testing.T, m *Model, ids []int) [][]float32 {
	t.Helper()
	cfg := m.Cfg
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	eps := float32(cfg.RMSNormEps)
	theta := cfg.RopeTheta
	seq := len(ids)
	qkvWidth := (nH + 2*nKV) * hd

	// FalconModel.word_embeddings — no embedding scale in Falcon.
	embed := cpuOracleTensor(t, m, "transformer.word_embeddings.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := "transformer.h." + itoa(l) + "."
		lnW := cpuOracleTensor(t, m, p+"input_layernorm.weight")
		lnB := cpuOracleTensor(t, m, p+"input_layernorm.bias")
		// The ORIGINAL fused tensor, read straight from the manifest bytes: the
		// reference does its own _split_heads and never touches the canonical
		// q/k/v views the production fused-split produced.
		wqkv := cpuOracleTensor(t, m, p+"self_attention.query_key_value.weight")
		wdense := cpuOracleTensor(t, m, p+"self_attention.dense.weight")
		wh4 := cpuOracleTensor(t, m, p+"mlp.dense_h_to_4h.weight")
		w4h := cpuOracleTensor(t, m, p+"mlp.dense_4h_to_h.weight")

		// ONE shared norm, read by BOTH branches. Computed once per position and
		// reused, exactly as HF aliases mlp_layernorm_out = attention_layernorm_out.
		xn := make([][]float32, seq)
		q := make([][]float32, seq) // [seq][nH*hd]
		kShared := make([][]float32, seq)
		vShared := make([][]float32, seq) // [seq][nKV*hd], nKV==1
		for tt := 0; tt < seq; tt++ {
			xn[tt] = falconOracleLayerNorm(x[tt], lnW, lnB, eps)

			// FalconAttention.forward: fused_qkv = self.query_key_value(x), then
			// _split_heads. multi_query layout is qkv.view(num_heads + 2, head_dim):
			// heads [0, nH) are the queries, head nH is the ONE shared key, head
			// nH+1 is the ONE shared value.
			fused := cpuOracleMatVec(wqkv, xn[tt], qkvWidth, H)
			q[tt] = append([]float32(nil), fused[:nH*hd]...)
			kShared[tt] = append([]float32(nil), fused[nH*hd:nH*hd+nKV*hd]...)
			vShared[tt] = append([]float32(nil), fused[nH*hd+nKV*hd:qkvWidth]...)

			// apply_rotary_pos_emb: full rotary dim, rotate_half convention, on
			// every query head and on the single shared key head.
			for h := 0; h < nH; h++ {
				cpuOracleRope(q[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
			for h := 0; h < nKV; h++ {
				cpuOracleRope(kShared[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
		}

		scale := float32(1.0 / math.Sqrt(float64(hd))) // FalconAttention.inv_norm_factor
		for tt := 0; tt < seq; tt++ {
			concat := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				// MULTI-QUERY BROADCAST: with a single kv head every query head
				// reads kv head 0 — HF materializes this by broadcasting the k/v
				// head across all num_heads query heads before the SDPA call.
				kvh := h * nKV / nH
				qh := q[tt][h*hd : (h+1)*hd]
				scores := make([]float32, tt+1)
				for j := 0; j <= tt; j++ {
					kh := kShared[j][kvh*hd : (kvh+1)*hd]
					var s float32
					for d := 0; d < hd; d++ {
						s += qh[d] * kh[d]
					}
					scores[j] = s * scale
				}
				cpuOracleSoftmax(scores) // causal: keys 0..tt only
				o := concat[h*hd : (h+1)*hd]
				for j := 0; j <= tt; j++ {
					vh := vShared[j][kvh*hd : (kvh+1)*hd]
					for d := 0; d < hd; d++ {
						o[d] += scores[j] * vh[d]
					}
				}
			}
			attnOut := cpuOracleMatVec(wdense, concat, H, nH*hd) // self_attention.dense

			// FalconMLP: dense_4h_to_h(gelu(dense_h_to_4h(x))). NO gate/up split,
			// NO SwiGLU — and it reads xn[tt], the SAME normed block input the
			// attention read, never the attention-updated residual.
			h4 := cpuOracleMatVec(wh4, xn[tt], I, H)
			for i := 0; i < I; i++ {
				h4[i] = falconOracleGelu(h4[i])
			}
			mlpOut := cpuOracleMatVec(w4h, h4, H, I)

			// ONE residual, both deltas.
			for i := 0; i < H; i++ {
				x[tt][i] += attnOut[i] + mlpOut[i]
			}
		}
	}

	// FalconModel.ln_f + tied lm_head (Falcon-7B ties word_embeddings).
	normW := cpuOracleTensor(t, m, "transformer.ln_f.weight")
	normB := cpuOracleTensor(t, m, "transformer.ln_f.bias")
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := falconOracleLayerNorm(x[tt], normW, normB, eps)
		logits[tt] = cpuOracleMatVec(embed, xf, V, H)
	}
	return logits
}

var falconOraclePrompt = []int{3, 17, 5, 23, 41, 2, 19}

// TestFalconCPUNumericOracle is the Falcon×cpu M4 witness: the production forward
// (cacheless Forward, and the cached Prefill/Step decode path) must reproduce the
// independent HF-semantics reference on every position within cpuOracleTol. This
// is the CI numeric oracle covmatrix.OracleInCI cites for Falcon — if this test
// reds, the honesty fence demotes the cell back to M3 (drop the covmatrix bit
// with it).
func TestFalconCPUNumericOracle(t *testing.T) {
	m := newFalconOracleModel(t)

	// The derivation must land the published Falcon-7B axes — the reference
	// hardcodes every one of them.
	if m.Cfg.BlockTopology != ParallelResidual {
		t.Fatalf("falcon derived topology = %v, want ParallelResidual", m.Cfg.BlockTopology)
	}
	if !m.Cfg.LayerNorm {
		t.Fatal("falcon derived LayerNorm = false, want true (Falcon uses biased LayerNorm, not RMSNorm)")
	}
	if !m.Cfg.DenseMLP {
		t.Fatal("falcon derived DenseMLP = false, want true (dense_h_to_4h -> act -> dense_4h_to_h, no gating)")
	}
	if !m.Cfg.ActGeluErf {
		t.Fatal("falcon derived ActGeluErf = false, want true (hidden_act \"gelu\" is the exact erf GELU)")
	}
	if m.Cfg.NumKVHeads != 1 {
		t.Fatalf("falcon fixture NumKVHeads = %d, want 1 (multi_query: one shared k/v head)", m.Cfg.NumKVHeads)
	}
	// The fixture must have gone through the real alias + fused-split passes, and
	// the ORIGINAL fused tensor must still be readable for the reference.
	for _, name := range []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
		"transformer.h.0.self_attention.query_key_value.weight",
	} {
		if _, ok := m.manifest[name]; !ok {
			t.Fatalf("fixture manifest missing %q after the falcon load passes", name)
		}
	}
	// Falcon-7B has no separate MLP norm: both parallel branches must share the
	// one input_layernorm (the reference computes it exactly once per position).
	if _, ok := m.manifest["model.layers.0.post_attention_layernorm.weight"]; ok {
		t.Fatal("fixture grew a post_attention_layernorm — Falcon-7B shares ONE block norm")
	}

	ids := falconOraclePrompt
	ref := falconReference(t, m, ids)

	// Full-prefill Forward: every position must match the reference.
	act := m.Forward(ids)
	for tt := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > cpuOracleTol {
			t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", tt, d, cpuOracleTol)
		}
	}

	// Cached decode path: Prefill then Step must match the reference at the same
	// positions (the reference is cacheless, so Step(id) at position len(ids) is
	// compared against a reference run over the extended prompt).
	s := m.NewSession()
	pf := s.Prefill(ids)
	if d := cpuOracleMaxAbsDiff(pf, ref[len(ids)-1]); d > cpuOracleTol {
		t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
	next := 11
	st := s.Step(next)
	extRef := falconReference(t, m, append(append([]int(nil), ids...), next))
	if d := cpuOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
		t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
}

// TestFalconCPUNumericOracleIsSensitive proves the comparison is non-vacuous on
// the two tensors that carry Falcon's distinctive axes: a single element of the
// FUSED query_key_value block (which the production split has to cut correctly)
// and a single LayerNorm BIAS element (which only exists because Falcon norms are
// biased LayerNorms). Perturbing either in the raw fixture bytes, then re-running
// the production forward against the UNPERTURBED reference, must red the gate.
//
// This is the cheap in-package vacuity floor. The stronger teeth proof — that the
// oracle catches real SEMANTIC defects in production code (serializing the
// parallel residual, mis-cutting the multi-query fused qkv, dropping the
// LayerNorm mean/bias, swapping GELU for SiLU) — is run out-of-tree with
// `go test -overlay`, which perturbs a scratch copy of the production file so the
// shared trunk is never edited.
func TestFalconCPUNumericOracleIsSensitive(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tensor string
	}{
		{"fused_qkv", "transformer.h.0.self_attention.query_key_value.weight"},
		{"layernorm_bias", "transformer.h.0.input_layernorm.bias"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newFalconOracleModel(t)
			ids := falconOraclePrompt
			ref := falconReference(t, m, ids)

			meta := m.manifest[tc.tensor]
			orig := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[meta.Offset:]))
			binary.LittleEndian.PutUint32(m.raw[meta.Offset:], math.Float32bits(orig+0.5))

			act := m.Forward(ids)
			var worst float64
			for tt := range ids {
				if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > worst {
					worst = d
				}
			}
			if worst <= cpuOracleTol {
				t.Fatalf("perturbed %s still within tolerance (max|delta|=%.3e) — the oracle is vacuous", tc.tensor, worst)
			}
		})
	}
}
