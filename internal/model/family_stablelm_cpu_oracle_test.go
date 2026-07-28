package model

// family_stablelm_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU
// numeric oracle for the StableLM family (#1271 Lane 1, support-maturity epic #1243).
// Companion to family_cpu_oracle_test.go, which states the doctrine and carries the
// OLMo2 (PostNorm) and Qwen2/3 (PreNorm) precedents; StableLM lives in its own file
// so the shared one stops growing and concurrent family lanes do not collide.
//
// Independence discipline (family_cpu_oracle_test.go:13-24): the reference below is a
// plain scalar transcription of HuggingFace transformers/models/stablelm/
// modeling_stablelm.py. It reuses NONE of the production machinery — tensors come
// straight out of the manifest bytes via cpuOracleTensor, every matmul/norm/softmax is
// a naive in-order scalar loop, and the block dataflow is hardcoded to StableLM's
// published topology instead of being routed through cfg.BlockTopology.
//
// What StableLM actually does, transcribed from the HF source (not from the family's
// reputation):
//
//   - NORM: mean-subtracting nn.LayerNorm with a LEARNED WEIGHT AND BIAS, not RMSNorm.
//     StableLmDecoderLayer builds input_layernorm / post_attention_layernorm and
//     StableLmModel builds the final norm as nn.LayerNorm(hidden_size,
//     eps=config.layer_norm_eps) — bias defaults to True on nn.LayerNorm, so all three
//     carry a .bias. This is the axis config.go:657 turns on (cfg.LayerNorm) and the
//     one arch.go:483 layernorm() implements.
//   - PARTIAL ROTARY: only the leading rotary_dim = int(head_dim *
//     partial_rotary_factor) entries of EACH head are rotated; the tail
//     (query_pass / key_pass) is concatenated back unrotated. Two sub-facts matter and
//     are easy to get half-right:
//     (a) the rotate_half pairing runs INSIDE the rotated slice — element j pairs with
//     j + rotary_dim/2, NOT with j + head_dim/2; and
//     (b) the inv_freq exponent DENOMINATOR is rotary_dim, not head_dim:
//     StableLmRotaryEmbedding is constructed with dim = int(partial_rotary_factor *
//     head_dim) and computes inv_freq = 1/base**(arange(0,dim,2)/dim). Modern
//     transformers routes the same model through
//     modeling_rope_utils._compute_default_rope_parameters, which computes
//     dim = int(head_dim * partial_rotary_factor) and the identical expression — so
//     the denominator is rotary_dim on both the pre- and post-refactor code paths.
//   - ATTENTION: causal GQA, scores divided by sqrt(head_dim) (the FULL head dim, not
//     the rotary width), softmax in fp32, o_proj is bias=False.
//   - PROJECTION BIAS: q/k/v carry bias=config.use_qkv_bias (the StableLM-2 lineage
//     sets it; stablelm-3b-4e1t does not), o_proj never does. Both lineages run below.
//   - MLP: the standard gated form down(act(gate(x)) * up(x)) with act = hidden_act,
//     which is "silu" for every published StableLmForCausalLM checkpoint.
//   - TOPOLOGY: use_parallel_residual defaults False and is False on the published
//     checkpoints, giving the sequential PreNorm block x += attn(ln1(x)); x += mlp(ln2(x)).
//
// The fixture is built with synthBuildRaw on StableLM's REAL tensor roster — the
// canonical identity names the resolver's stableLMSpec (tensor_resolver.go:444) pins,
// PLUS the .bias twin of every LayerNorm — and gives every norm weight a distinct
// NON-UNIT gain and every norm bias a distinct NON-ZERO offset, so norm routing and
// bias application are numerically live rather than masked by 1.0 / 0.0.

import (
	"encoding/binary"
	"math"
	"testing"
)

// stablelmOracleLayerNorm is the plain HF nn.LayerNorm: (x-mean)/sqrt(var+eps)*w + b,
// with the biased (1/N) variance torch uses. bias may be nil (weight-only LayerNorm).
func stablelmOracleLayerNorm(x, w, b []float32, eps float32) []float32 {
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
		out[i] = (v - mean) * inv * w[i]
		if b != nil {
			out[i] += b[i]
		}
	}
	return out
}

// stablelmOraclePartialRope rotates ONE head vector (length head_dim) in place at
// position pos, StableLM style: the leading rot = int(head_dim*partial_rotary_factor)
// entries take HF's non-interleaved rotate_half rotation with inv_freq[j] =
// base^-(2j/rot) — the denominator is the ROTARY width — and hv[rot:] is left exactly
// as it was (HF's query_pass / key_pass concatenated back unchanged).
func stablelmOraclePartialRope(hv []float32, pos, rot int, theta float64) {
	half := rot / 2
	for j := 0; j < half; j++ {
		angle := float64(pos) / math.Pow(theta, float64(2*j)/float64(rot))
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		a, b := hv[j], hv[j+half]
		hv[j] = a*c - b*s
		hv[j+half] = b*c + a*s
	}
}

// stablelmOracleRotaryDim is int(head_dim * partial_rotary_factor), StableLM's
// rotary_ndims. Kept as a local scalar so the reference never consults
// cfg.rotaryDim() (production machinery).
func stablelmOracleRotaryDim(hd int, factor float64) int {
	return int(float64(hd) * factor)
}

// stablelmOracleCfg is the tiny StableLM fixture config. head_dim 16 with the family's
// published partial_rotary_factor 0.25 gives rotary_dim 4 (half 2) — deliberately more
// than 2, because at rotary_dim==2 the single frequency is base^0==1 and BOTH the
// pairing width and the inv_freq denominator become unobservable. nH*head_dim (64)
// also differs from HiddenSize (24) so a projection-width/hidden-width conflation
// cannot cancel, and nKV<nH keeps the GQA grouping live.
func stablelmOracleCfg() Config {
	return Config{
		HiddenSize:          24,
		NumLayers:           3,
		NumHeads:            4,
		NumKVHeads:          2,
		HeadDim:             16,
		IntermediateSize:    40,
		VocabSize:           53,
		ModelType:           "stablelm",
		RMSNormEps:          1e-5,
		RopeTheta:           10000,
		PartialRotaryFactor: 0.25,
		TieWordEmbeddings:   true,
	}
}

// isStableLMOracleNormTensor reports whether a fixture tensor is a LayerNorm weight or
// bias (as opposed to a matmul weight or a projection bias). Norm weights get distinct
// non-unit gains and norm biases distinct non-zero offsets.
func isStableLMOracleNormTensor(name string) (norm, bias bool) {
	switch {
	case hasCPUOracleSuffix(name, "layernorm.weight"), name == "model.norm.weight":
		return true, false
	case hasCPUOracleSuffix(name, "layernorm.bias"), name == "model.norm.bias":
		return true, true
	}
	return false, false
}

func hasCPUOracleSuffix(name, suffix string) bool {
	return len(name) >= len(suffix) && name[len(name)-len(suffix):] == suffix
}

// newStableLMOracleModel builds the fixture on StableLM's REAL tensor roster: the
// canonical split q/k/v/o projections and SwiGLU MLP (stableLMSpec is the identity
// name set), input_layernorm + post_attention_layernorm + model.norm each with BOTH a
// .weight and a .bias, and — for the use_qkv_bias lineage — q/k/v projection biases.
// o_proj is never biased (HF hardcodes bias=False).
func newStableLMOracleModel(qkvBias bool) *Model {
	cfg := stablelmOracleCfg()
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize

	type ts = synthTensor
	var tensors []ts
	tensors = append(tensors, ts{"model.embed_tokens.weight", []int{V, H}})
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		tensors = append(tensors,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{p + "input_layernorm.bias", []int{H}},
			ts{p + "self_attn.q_proj.weight", []int{nH * hd, H}},
			ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
		)
		if qkvBias {
			tensors = append(tensors,
				ts{p + "self_attn.q_proj.bias", []int{nH * hd}},
				ts{p + "self_attn.k_proj.bias", []int{nKV * hd}},
				ts{p + "self_attn.v_proj.bias", []int{nKV * hd}},
			)
		}
		tensors = append(tensors,
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{p + "post_attention_layernorm.bias", []int{H}},
			ts{p + "mlp.gate_proj.weight", []int{I, H}},
			ts{p + "mlp.up_proj.weight", []int{I, H}},
			ts{p + "mlp.down_proj.weight", []int{H, I}},
		)
	}
	tensors = append(tensors,
		ts{"model.norm.weight", []int{H}},
		ts{"model.norm.bias", []int{H}},
	)

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		norm, bias := isStableLMOracleNormTensor(name)
		switch {
		case norm && bias:
			return 0.1 + 0.25*next() // distinct NON-ZERO LayerNorm offsets
		case norm:
			return 1 + 0.25*next() // distinct NON-UNIT LayerNorm gains
		}
		return synthMatmulFill(name, next)
	})
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

// stablelmReference runs the independent StableLM forward: per-position logits for ids.
// Every step is the HF StableLm dataflow, hardcoded — NOT routed through
// cfg.BlockTopology, normCfg, rotaryDim, applyRopeRow or any other production helper.
func stablelmReference(t *testing.T, m *Model, ids []int, qkvBias bool) [][]float32 {
	t.Helper()
	cfg := m.Cfg
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	grp := nH / nKV
	eps := float32(cfg.RMSNormEps)
	theta := cfg.RopeTheta
	rot := stablelmOracleRotaryDim(hd, cfg.PartialRotaryFactor)
	seq := len(ids)

	embed := cpuOracleTensor(t, m, "model.embed_tokens.weight")
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
		p := layerPrefix(l)
		inW := cpuOracleTensor(t, m, p+"input_layernorm.weight")
		inB := cpuOracleTensor(t, m, p+"input_layernorm.bias")
		wq := cpuOracleTensor(t, m, p+"self_attn.q_proj.weight")
		wk := cpuOracleTensor(t, m, p+"self_attn.k_proj.weight")
		wv := cpuOracleTensor(t, m, p+"self_attn.v_proj.weight")
		wo := cpuOracleTensor(t, m, p+"self_attn.o_proj.weight")
		postW := cpuOracleTensor(t, m, p+"post_attention_layernorm.weight")
		postB := cpuOracleTensor(t, m, p+"post_attention_layernorm.bias")
		wg := cpuOracleTensor(t, m, p+"mlp.gate_proj.weight")
		wu := cpuOracleTensor(t, m, p+"mlp.up_proj.weight")
		wd := cpuOracleTensor(t, m, p+"mlp.down_proj.weight")

		// --- attention sub-layer, sequential Pre-Norm: x += attn(input_layernorm(x)) ---
		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn := stablelmOracleLayerNorm(x[tt], inW, inB, eps)
			q[tt] = cpuOracleMatVec(wq, xn, nH*hd, H)
			k[tt] = cpuOracleMatVec(wk, xn, nKV*hd, H)
			v[tt] = cpuOracleMatVec(wv, xn, nKV*hd, H)
			if qkvBias {
				// HF: q/k/v carry bias=config.use_qkv_bias; o_proj never does.
				addVec(q[tt], cpuOracleTensor(t, m, p+"self_attn.q_proj.bias"))
				addVec(k[tt], cpuOracleTensor(t, m, p+"self_attn.k_proj.bias"))
				addVec(v[tt], cpuOracleTensor(t, m, p+"self_attn.v_proj.bias"))
			}
			// Partial rotary: only hv[:rot] rotates; hv[rot:] is query_pass/key_pass.
			for h := 0; h < nH; h++ {
				stablelmOraclePartialRope(q[tt][h*hd:(h+1)*hd], tt, rot, theta)
			}
			for h := 0; h < nKV; h++ {
				stablelmOraclePartialRope(k[tt][h*hd:(h+1)*hd], tt, rot, theta)
			}
		}
		// HF divides the raw scores by sqrt(head_dim) — the FULL head dim.
		scale := float32(1.0 / math.Sqrt(float64(hd)))
		for tt := 0; tt < seq; tt++ {
			concat := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				kvh := h / grp // HF repeat_kv: query head h reads kv head h/num_key_value_groups
				qh := q[tt][h*hd : (h+1)*hd]
				scores := make([]float32, tt+1)
				for j := 0; j <= tt; j++ {
					kh := k[j][kvh*hd : (kvh+1)*hd]
					var s float32
					for d := 0; d < hd; d++ {
						s += qh[d] * kh[d]
					}
					scores[j] = s * scale
				}
				cpuOracleSoftmax(scores)
				o := concat[h*hd : (h+1)*hd]
				for j := 0; j <= tt; j++ {
					vh := v[j][kvh*hd : (kvh+1)*hd]
					for d := 0; d < hd; d++ {
						o[d] += scores[j] * vh[d]
					}
				}
			}
			attnOut := cpuOracleMatVec(wo, concat, H, nH*hd) // bias=False in HF
			for i := 0; i < H; i++ {
				x[tt][i] += attnOut[i]
			}
		}

		// --- MLP sub-layer, Pre-Norm: x += SwiGLU(post_attention_layernorm(x)) ---
		for tt := 0; tt < seq; tt++ {
			xn := stablelmOracleLayerNorm(x[tt], postW, postB, eps)
			gate := cpuOracleMatVec(wg, xn, I, H)
			up := cpuOracleMatVec(wu, xn, I, H)
			for i := 0; i < I; i++ {
				gate[i] = cpuOracleSilu(gate[i]) * up[i] // hidden_act == "silu"
			}
			mlpOut := cpuOracleMatVec(wd, gate, H, I)
			for i := 0; i < H; i++ {
				x[tt][i] += mlpOut[i]
			}
		}
	}

	normW := cpuOracleTensor(t, m, "model.norm.weight")
	normB := cpuOracleTensor(t, m, "model.norm.bias")
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := stablelmOracleLayerNorm(x[tt], normW, normB, eps)
		logits[tt] = cpuOracleMatVec(embed, xf, V, H) // tied head: logits = E @ xf
	}
	return logits
}

// TestStableLMCPUNumericOracle is the StableLM×cpu M4 witness: the production forward
// (Forward, and the cached Prefill/Step decode path) must reproduce the independent
// HF-semantics reference on every position within cpuOracleTol, for both published
// lineages (use_qkv_bias off — stablelm-3b-4e1t — and on — the StableLM-2 line).
// If this test reds, the honesty fence demotes the cell back to M3 (drop the
// covmatrix OracleInCI bit with it).
func TestStableLMCPUNumericOracle(t *testing.T) {
	for _, lineage := range []struct {
		name    string
		qkvBias bool
	}{
		{"no_qkv_bias", false},
		{"use_qkv_bias", true},
	} {
		t.Run(lineage.name, func(t *testing.T) {
			m := newStableLMOracleModel(lineage.qkvBias)
			if err := m.Cfg.deriveConfigAxes(configJSONHints{}); err != nil {
				t.Fatalf("deriveConfigAxes: %v", err)
			}
			// The derivation must land the published StableLM axes — the reference
			// hardcodes them.
			if m.Cfg.BlockTopology != PreNorm {
				t.Fatalf("stablelm derived topology = %v, want PreNorm", m.Cfg.BlockTopology)
			}
			if !m.Cfg.LayerNorm {
				t.Fatal("stablelm derived LayerNorm = false, want true (StableLM norms are nn.LayerNorm, not RMSNorm)")
			}
			if got := stablelmOracleRotaryDim(m.Cfg.HeadDim, m.Cfg.PartialRotaryFactor); got != 4 {
				t.Fatalf("fixture rotary_dim = %d, want 4 (a rotary_dim of 2 makes the RoPE axes unobservable)", got)
			}

			ids := []int{3, 17, 5, 23, 41, 2, 19}
			ref := stablelmReference(t, m, ids, lineage.qkvBias)

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
			extRef := stablelmReference(t, m, append(append([]int(nil), ids...), next), lineage.qkvBias)
			if d := cpuOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
				t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
			}
		})
	}
}

// TestStableLMCPUNumericOracleIsSensitive proves the StableLM reference comparison is
// non-vacuous: perturbing ONE raw fixture element must move the compared logits far
// beyond the tolerance. Two perturbations are exercised — a matmul weight and a
// LayerNorm BIAS — because a reference that silently ignored the LayerNorm bias would
// stay green under the second one.
func TestStableLMCPUNumericOracleIsSensitive(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tensor string
	}{
		{"q_proj_weight", "model.layers.0.self_attn.q_proj.weight"},
		{"input_layernorm_bias", "model.layers.0.input_layernorm.bias"},
		{"final_norm_bias", "model.norm.bias"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newStableLMOracleModel(false)
			if err := m.Cfg.deriveConfigAxes(configJSONHints{}); err != nil {
				t.Fatalf("deriveConfigAxes: %v", err)
			}
			ids := []int{3, 17, 5, 23, 41, 2, 19}
			ref := stablelmReference(t, m, ids, false)

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
