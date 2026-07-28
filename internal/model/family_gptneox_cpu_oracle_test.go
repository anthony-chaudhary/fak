package model

// family_gptneox_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU
// numeric oracle for the GPT-NeoX family (#1271 Lane 1, support-maturity epic #1243).
// Companion to family_cpu_oracle_test.go, which states the doctrine and carries the
// OLMo2 (PostNorm) and Qwen2/3 (PreNorm) precedents. GPT-NeoX is the first
// ParallelResidual family to get a reference, so the block dataflow below is
// transcribed rather than adapted from either precedent.
//
// Independence discipline (family_cpu_oracle_test.go:13-24): the reference is a plain
// scalar transcription of HuggingFace transformers/models/gpt_neox/
// {modeling,modular,configuration}_gpt_neox.py. It reuses NONE of the production
// machinery — tensors come straight out of the manifest bytes via cpuOracleTensor,
// every matmul/norm/softmax/GELU is a naive in-order scalar loop, the fused
// query_key_value tensor is re-split here from the SOURCE bytes rather than read back
// out of materializeGPTNeoXQKVWeight's output, and the block dataflow is hardcoded to
// GPT-NeoX's published topology instead of being routed through cfg.BlockTopology.
//
// What GPT-NeoX actually does, transcribed from the HF source (not from the family's
// reputation):
//
//   - PARALLEL RESIDUAL (GPTNeoXLayer.forward, use_parallel_residual defaults True and
//     is True on every published Pythia/GPT-NeoX checkpoint):
//     hidden = mlp(post_attention_layernorm(x)) + attn(input_layernorm(x)) + x
//     BOTH branches read the SAME input x. The MLP does NOT see the attention output —
//     that is the serial PreNorm form and is a different model. The two norms are
//     DISTINCT tensors, so a reference that reuses one norm for both branches, or that
//     feeds attn's output into the MLP norm, diverges by O(0.1..1).
//   - NORM: mean-subtracting nn.LayerNorm with a LEARNED WEIGHT AND BIAS (nn.LayerNorm
//     defaults elementwise_affine=True and bias=True), NOT RMSNorm. All three sites —
//     input_layernorm, post_attention_layernorm, final_layer_norm — carry a .bias.
//     This is the axis config.go:627 turns on (cfg.LayerNorm), implemented by
//     arch.go:483 layernorm().
//   - FUSED QKV: one nn.Linear(hidden, 3*hidden) named attention.query_key_value.
//     GPTNeoXAttention.forward views its output as (..., num_heads, 3*head_size) and
//     then chunk(3, dim=-1), so the row layout is INTERLEAVED PER HEAD:
//     [h0_q(hd) h0_k(hd) h0_v(hd) h1_q(hd) h1_k(hd) h1_v(hd) ...], NOT the contiguous
//     [all_q | all_k | all_v] that Falcon/MPT-style fused tensors use. Head h's q rows
//     start at h*3*hd, k at h*3*hd+hd, v at h*3*hd+2*hd. This is the split
//     materialize.go:391 materializeGPTNeoXQKVWeight performs; the reference below
//     re-derives it independently from the source tensor. Because head_size is defined
//     as hidden_size // num_attention_heads and the projection is 3*hidden wide,
//     GPT-NeoX structurally forces hidden == num_heads*head_dim and num_kv == num_q
//     (no GQA) — unlike the OLMo2/Qwen fixtures, nH*hd cannot be made to differ from
//     HiddenSize here.
//   - PROJECTION BIAS: query_key_value and dense both carry bias=config.attention_bias,
//     which defaults True; dense_h_to_4h and dense_4h_to_h are plain biased nn.Linear.
//     Every projection in the fixture is therefore biased.
//   - PARTIAL ROTARY: rotary_ndims = int(head_size * partial_rotary_factor), where
//     configuration_gpt_neox.py:98 seeds partial_rotary_factor from the legacy
//     rotary_pct key with default 0.25. apply_rotary_pos_emb splits q/k at
//     rotary_ndims, rotates the leading slice with the non-interleaved rotate_half
//     convention, and concatenates the untouched tail (query_pass/key_pass) back.
//     TWO sub-facts matter: (a) the rotate_half pairing runs INSIDE the rotated slice
//     (element j pairs with j + rotary_ndims/2, not j + head_dim/2), and (b) the
//     inv_freq exponent DENOMINATOR is rotary_ndims, not head_dim —
//     modular_gpt_neox.py:90-99 computes dim = int(head_dim * partial_rotary_factor)
//     and then inv_freq = 1/base**(arange(0,dim,2)/dim), reusing the SAME dim for the
//     range and the denominator. See the note on TestGPTNeoXCPUNumericOraclePartialRotary
//     below: (b) is where production currently disagrees.
//   - ATTENTION: causal, no GQA, scores scaled by head_size**-0.5 (the FULL head dim,
//     not the rotary width), softmax in fp32.
//   - MLP: the DENSE (non-gated) form dense_4h_to_h(act(dense_h_to_4h(x))) with
//     act = hidden_act = "gelu" for the published checkpoints, which HF's ACT2FN maps
//     to the EXACT erf GELU 0.5*x*(1+erf(x/sqrt2)) — not the tanh approximation.
//     There is no up_proj and no SwiGLU multiply.
//   - HEAD: embed_out is a separate nn.Linear(hidden, vocab, bias=False); Pythia does
//     not tie it to embed_in.
//
// The fixture is built with synthBuildRaw on GPT-NeoX's REAL checkpoint names
// (gpt_neox.embed_in.weight, gpt_neox.layers.N.attention.query_key_value.weight,
// gpt_neox.final_layer_norm.bias, embed_out.weight, ...) and is loaded through
// newModel so the production side goes through materializeGPTNeoXTensors' aliasing AND
// the fused-QKV split — the reference never reads the split output. Every LayerNorm
// weight gets a distinct NON-UNIT gain and every LayerNorm bias a distinct NON-ZERO
// offset, so norm routing and bias application are numerically live rather than masked
// by 1.0 / 0.0.

import (
	"encoding/binary"
	"math"
	"strconv"
	"strings"
	"testing"
)

// gptneoxOracleLayerNorm is the plain HF nn.LayerNorm: (x-mean)/sqrt(var+eps)*w + b,
// with the biased (1/N) variance torch uses.
func gptneoxOracleLayerNorm(x, w, b []float32, eps float32) []float32 {
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

// gptneoxOracleGeluErf is HF ACT2FN["gelu"] (GELUActivation): the EXACT erf GELU
// 0.5*x*(1+erf(x/sqrt(2))). GPT-NeoX's hidden_act is "gelu", never "gelu_new".
func gptneoxOracleGeluErf(z float32) float32 {
	z64 := float64(z)
	return float32(0.5 * z64 * (1 + math.Erf(z64/math.Sqrt2)))
}

// gptneoxOracleRotaryDim is HF's rotary_ndims = int(head_size * partial_rotary_factor).
// A factor of 0 stands for "absent", which HF defaults to 1.0 (full rotary). Kept as a
// local scalar so the reference never consults cfg.rotaryDim() (production machinery).
func gptneoxOracleRotaryDim(hd int, factor float64) int {
	if factor <= 0 || factor >= 1 {
		return hd
	}
	return int(float64(hd) * factor)
}

// gptneoxOraclePartialRope rotates ONE head vector (length head_dim) in place at
// position pos: the leading rot entries take HF's non-interleaved rotate_half rotation
// with inv_freq[j] = base^-(2j/rot) — the denominator is the ROTARY width, exactly the
// dim that _compute_default_rope_parameters divides by — and hv[rot:] is left as it was
// (HF's query_pass / key_pass concatenated back unchanged).
func gptneoxOraclePartialRope(hv []float32, pos, rot int, theta float64) {
	half := rot / 2
	for j := 0; j < half; j++ {
		angle := float64(pos) / math.Pow(theta, float64(2*j)/float64(rot))
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		a, b := hv[j], hv[j+half]
		hv[j] = a*c - b*s
		hv[j+half] = b*c + a*s
	}
}

// isGPTNeoXOracleNormTensor reports whether a fixture tensor is a LayerNorm weight or
// bias (as opposed to a matmul weight or a projection bias). GPT-NeoX's final norm is
// spelled final_layer_norm, so the shared isCPUOracleNormWeight suffix set does not
// cover it.
func isGPTNeoXOracleNormTensor(name string) (norm, bias bool) {
	switch {
	case strings.HasSuffix(name, "layernorm.weight"), strings.HasSuffix(name, "final_layer_norm.weight"):
		return true, false
	case strings.HasSuffix(name, "layernorm.bias"), strings.HasSuffix(name, "final_layer_norm.bias"):
		return true, true
	}
	return false, false
}

// gptneoxOracleCfg is the tiny GPT-NeoX fixture config. head_dim 16 with the family's
// published rotary_pct 0.25 gives rotary_ndims 4 (half 2) — deliberately more than 2,
// because at rotary_ndims==2 the single frequency is base^0==1 and BOTH the pairing
// width and the inv_freq denominator become unobservable. HiddenSize == NumHeads*HeadDim
// and NumKVHeads == NumHeads are structural for GPT-NeoX (see the header), not a
// weakening of the fixture. partialRotary==0 builds the rotary_pct==1.0 lineage.
func gptneoxOracleCfg(partialRotary float64) Config {
	return Config{
		HiddenSize:          64,
		NumLayers:           3,
		NumHeads:            4,
		NumKVHeads:          4,
		HeadDim:             16,
		IntermediateSize:    40,
		VocabSize:           53,
		ModelType:           "gpt_neox",
		Architectures:       []string{"GPTNeoXForCausalLM"},
		HiddenAct:           "gelu",
		RMSNormEps:          1e-5,
		RopeTheta:           10000,
		PartialRotaryFactor: partialRotary,
	}
}

// gptneoxLayerSrc is the SOURCE (checkpoint) prefix for layer l — the names a real
// GPT-NeoX safetensors file carries, not the canonical model.layers.N.* the
// materializer aliases them to.
func gptneoxLayerSrc(l int) string { return "gpt_neox.layers." + strconv.Itoa(l) + "." }

// newGPTNeoXOracleModel builds the fixture on GPT-NeoX's REAL tensor roster and loads
// it through newModel, so the production side exercises materializeGPTNeoXTensors'
// aliasing AND materializeGPTNeoXQKVWeight/Bias' interleaved fused-QKV split.
func newGPTNeoXOracleModel(t *testing.T, partialRotary float64) *Model {
	t.Helper()
	cfg := gptneoxOracleCfg(partialRotary)
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	nH, hd := cfg.NumHeads, cfg.HeadDim
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize

	type ts = synthTensor
	var tensors []ts
	tensors = append(tensors, ts{"gpt_neox.embed_in.weight", []int{V, H}})
	for l := 0; l < cfg.NumLayers; l++ {
		p := gptneoxLayerSrc(l)
		tensors = append(tensors,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{p + "input_layernorm.bias", []int{H}},
			ts{p + "attention.query_key_value.weight", []int{3 * nH * hd, H}},
			ts{p + "attention.query_key_value.bias", []int{3 * nH * hd}},
			ts{p + "attention.dense.weight", []int{H, nH * hd}},
			ts{p + "attention.dense.bias", []int{H}},
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{p + "post_attention_layernorm.bias", []int{H}},
			ts{p + "mlp.dense_h_to_4h.weight", []int{I, H}},
			ts{p + "mlp.dense_h_to_4h.bias", []int{I}},
			ts{p + "mlp.dense_4h_to_h.weight", []int{H, I}},
			ts{p + "mlp.dense_4h_to_h.bias", []int{H}},
		)
	}
	tensors = append(tensors,
		ts{"gpt_neox.final_layer_norm.weight", []int{H}},
		ts{"gpt_neox.final_layer_norm.bias", []int{H}},
		ts{"embed_out.weight", []int{V, H}},
	)

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		norm, bias := isGPTNeoXOracleNormTensor(name)
		switch {
		case norm && bias:
			return 0.1 + 0.25*next() // distinct NON-ZERO LayerNorm offsets
		case norm:
			return 1 + 0.25*next() // distinct NON-UNIT LayerNorm gains
		case name == "gpt_neox.embed_in.weight":
			return next() * 0.2 // wider, so distinct ids separate cleanly
		}
		return synthMatmulFill(name, next)
	})

	m, err := newModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newModel: %v", err)
	}
	return m
}

// gptneoxReference runs the independent GPT-NeoX forward: per-position logits for ids.
// Every step is the HF GPTNeoXLayer dataflow, hardcoded — NOT routed through
// cfg.BlockTopology, normCfg, ffnFor, rotaryDim, applyRopeRow or any other production
// helper — and the fused query_key_value tensor is split here, from the source bytes.
func gptneoxReference(t *testing.T, m *Model, ids []int) [][]float32 {
	t.Helper()
	cfg := m.Cfg
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH, hd := cfg.NumHeads, cfg.HeadDim
	eps := float32(cfg.RMSNormEps)
	theta := cfg.RopeTheta
	rot := gptneoxOracleRotaryDim(hd, cfg.PartialRotaryFactor)
	seq := len(ids)

	embed := cpuOracleTensor(t, m, "gpt_neox.embed_in.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}
	addVec := func(y, b []float32) {
		for i := range y {
			y[i] += b[i]
		}
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := gptneoxLayerSrc(l)
		inW := cpuOracleTensor(t, m, p+"input_layernorm.weight")
		inB := cpuOracleTensor(t, m, p+"input_layernorm.bias")
		wqkv := cpuOracleTensor(t, m, p+"attention.query_key_value.weight")
		bqkv := cpuOracleTensor(t, m, p+"attention.query_key_value.bias")
		wo := cpuOracleTensor(t, m, p+"attention.dense.weight")
		bo := cpuOracleTensor(t, m, p+"attention.dense.bias")
		postW := cpuOracleTensor(t, m, p+"post_attention_layernorm.weight")
		postB := cpuOracleTensor(t, m, p+"post_attention_layernorm.bias")
		wUp := cpuOracleTensor(t, m, p+"mlp.dense_h_to_4h.weight")
		bUp := cpuOracleTensor(t, m, p+"mlp.dense_h_to_4h.bias")
		wDown := cpuOracleTensor(t, m, p+"mlp.dense_4h_to_h.weight")
		bDown := cpuOracleTensor(t, m, p+"mlp.dense_4h_to_h.bias")

		// --- fused QKV, per position, from input_layernorm(x). Head h occupies rows
		// [h*3hd, h*3hd+3hd): q | k | v, interleaved (HF view(..., nH, 3*hd).chunk(3)).
		qkv := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn := gptneoxOracleLayerNorm(x[tt], inW, inB, eps)
			row := cpuOracleMatVec(wqkv, xn, 3*nH*hd, H)
			addVec(row, bqkv)
			for h := 0; h < nH; h++ {
				base := h * 3 * hd
				gptneoxOraclePartialRope(row[base:base+hd], tt, rot, theta)      // q
				gptneoxOraclePartialRope(row[base+hd:base+2*hd], tt, rot, theta) // k
				_ = row[base+2*hd : base+3*hd]                                   // v is never rotated
			}
			qkv[tt] = row
		}

		// --- causal attention. No GQA: query head h reads kv head h.
		scale := float32(1.0 / math.Sqrt(float64(hd))) // head_size**-0.5, the FULL head dim
		attnOut := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			concat := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				base := h * 3 * hd
				qh := qkv[tt][base : base+hd]
				scores := make([]float32, tt+1)
				for j := 0; j <= tt; j++ {
					kh := qkv[j][base+hd : base+2*hd]
					var s float32
					for d := 0; d < hd; d++ {
						s += qh[d] * kh[d]
					}
					scores[j] = s * scale
				}
				cpuOracleSoftmax(scores)
				o := concat[h*hd : (h+1)*hd]
				for j := 0; j <= tt; j++ {
					vh := qkv[j][base+2*hd : base+3*hd]
					for d := 0; d < hd; d++ {
						o[d] += scores[j] * vh[d]
					}
				}
			}
			out := cpuOracleMatVec(wo, concat, H, nH*hd)
			addVec(out, bo) // attention.dense carries a bias
			attnOut[tt] = out
		}

		// --- MLP. PARALLEL RESIDUAL: this reads the ORIGINAL x, NOT x+attnOut.
		mlpOut := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn := gptneoxOracleLayerNorm(x[tt], postW, postB, eps)
			up := cpuOracleMatVec(wUp, xn, I, H)
			addVec(up, bUp)
			for i := 0; i < I; i++ {
				up[i] = gptneoxOracleGeluErf(up[i]) // dense, non-gated: no up_proj multiply
			}
			out := cpuOracleMatVec(wDown, up, H, I)
			addVec(out, bDown)
			mlpOut[tt] = out
		}

		// hidden = mlp_output + attn_output + hidden_states (GPTNeoXLayer.forward)
		for tt := 0; tt < seq; tt++ {
			for i := 0; i < H; i++ {
				x[tt][i] = mlpOut[tt][i] + attnOut[tt][i] + x[tt][i]
			}
		}
	}

	fnW := cpuOracleTensor(t, m, "gpt_neox.final_layer_norm.weight")
	fnB := cpuOracleTensor(t, m, "gpt_neox.final_layer_norm.bias")
	head := cpuOracleTensor(t, m, "embed_out.weight") // untied head, bias=False
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := gptneoxOracleLayerNorm(x[tt], fnW, fnB, eps)
		logits[tt] = cpuOracleMatVec(head, xf, V, H)
	}
	return logits
}

var gptneoxOracleIDs = []int{3, 17, 5, 23, 41, 2, 19}

// runGPTNeoXOracle compares the production cacheless Forward AND the cached
// Prefill/Step decode path against the independent reference at every position.
func runGPTNeoXOracle(t *testing.T, partialRotary float64) {
	t.Helper()
	m := newGPTNeoXOracleModel(t, partialRotary)

	// The derivation must land the published GPT-NeoX axes — the reference hardcodes them.
	if m.Cfg.BlockTopology != ParallelResidual {
		t.Fatalf("gptneox derived topology = %v, want ParallelResidual", m.Cfg.BlockTopology)
	}
	if !m.Cfg.LayerNorm {
		t.Fatal("gptneox derived LayerNorm = false, want true (GPT-NeoX norms are nn.LayerNorm with bias, not RMSNorm)")
	}
	if !m.Cfg.DenseMLP {
		t.Fatal("gptneox derived DenseMLP = false, want true (dense_h_to_4h -> gelu -> dense_4h_to_h, no SwiGLU gate)")
	}
	if !m.Cfg.ActGeluErf {
		t.Fatal("gptneox derived ActGeluErf = false, want true (hidden_act \"gelu\" is the exact erf GELU)")
	}
	// The fused query_key_value must actually have been split; otherwise the production
	// side would be reading a tensor the reference never modelled.
	if _, ok := m.manifest["model.layers.0.self_attn.q_proj.weight"]; !ok {
		t.Fatal("fused attention.query_key_value was not split into self_attn.q_proj.weight")
	}

	ids := gptneoxOracleIDs
	ref := gptneoxReference(t, m, ids)

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
	extRef := gptneoxReference(t, m, append(append([]int(nil), ids...), next))
	if d := cpuOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
		t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
}

// TestGPTNeoXCPUNumericOracle is the GPT-NeoX×cpu M4 witness for the rotary_pct==1.0
// lineage: the production forward must reproduce the independent HF-semantics
// reference on every position within cpuOracleTol. It covers every family-defining
// axis except the partial-rotary frequency table — the parallel residual, the two
// distinct LayerNorms with learned bias, the interleaved fused-QKV split, the biased
// dense/qkv/MLP projections, the exact-erf GELU dense MLP, and the untied embed_out
// head. If this test reds, the honesty fence demotes the cell back to M3 (drop the
// covmatrix OracleInCI bit with it).
func TestGPTNeoXCPUNumericOracle(t *testing.T) {
	runGPTNeoXOracle(t, 0) // partial_rotary_factor absent == HF default 1.0
}

// TestGPTNeoXCPUNumericOraclePartialRotary is the same witness at the family's
// PUBLISHED geometry: rotary_pct 0.25, i.e. rotary_ndims 4 out of head_dim 16.
//
// This is the test that CAUGHT the partial-rotary inv_freq bug and now holds it fixed.
// invFreqDenom special-cased only Qwen3.5-hybrid and MiniMax and otherwise returned
// cfg.HeadDim, so production built inv_freq[j] = 1/theta^(2j/head_dim) while HF
// (modular_gpt_neox.py:90-99, via modeling_rope_utils._compute_default_rope_parameters)
// builds 1/theta^(2j/rotary_ndims). The ROTATED RANGE was right on both sides; only the
// frequency table differed, by a factor of head_dim/rotary_ndims == 4 in the exponent,
// which is why it survived every full-rotary test. invFreqDenom now returns c.rotaryDim()
// on the fall-through, matching every HF partial-rotary family and llama.cpp's ggml_rope
// (which divides by n_dims == rope.dimension_count). Denominator-only coverage lives in
// rope_partial_rotary_denom_test.go; this test is the end-to-end half.
func TestGPTNeoXCPUNumericOraclePartialRotary(t *testing.T) {
	m := newGPTNeoXOracleModel(t, 0.25)
	if got := gptneoxOracleRotaryDim(m.Cfg.HeadDim, m.Cfg.PartialRotaryFactor); got != 4 {
		t.Fatalf("fixture rotary_ndims = %d, want 4 (a rotary_ndims of 2 makes the inv_freq denominator unobservable)", got)
	}
	runGPTNeoXOracle(t, 0.25)
}

// TestGPTNeoXCPUNumericOracleIsSensitive proves the comparison is non-vacuous:
// perturbing ONE raw fixture element must move the compared logits far beyond the
// tolerance. The tensors chosen are all ZERO-COPY ALIASES (materializeGPTNeoXTensors
// only rewrites the manifest key), so both sides read the same bytes and the
// perturbation is a genuine model change rather than an artefact of the fused-QKV
// split's copied output. A LayerNorm BIAS and the untied head are included because a
// reference that silently ignored the LN bias, or that tied the head to embed_in,
// would stay green without them.
func TestGPTNeoXCPUNumericOracleIsSensitive(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tensor string
	}{
		{"attention_dense_weight", "gpt_neox.layers.0.attention.dense.weight"},
		{"input_layernorm_bias", "gpt_neox.layers.0.input_layernorm.bias"},
		{"post_attention_layernorm_weight", "gpt_neox.layers.0.post_attention_layernorm.weight"},
		{"final_layer_norm_bias", "gpt_neox.final_layer_norm.bias"},
		{"embed_out_weight", "embed_out.weight"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newGPTNeoXOracleModel(t, 0)
			ids := gptneoxOracleIDs
			ref := gptneoxReference(t, m, ids)

			meta, ok := m.manifest[tc.tensor]
			if !ok {
				t.Fatalf("fixture tensor %q missing", tc.tensor)
			}
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
				t.Fatalf("perturbed fixture still within tolerance (max|delta|=%.3e) — the oracle is vacuous", worst)
			}
		})
	}
}
