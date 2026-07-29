package model

// family_gptoss_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU
// numeric oracle for the gpt-oss MoE architecture (model_type "gpt_oss",
// GptOssForCausalLM), the M4 witness style described in family_cpu_oracle_test.go
// (#1271 Lane 1, support-maturity epic #1243).
//
// Independence discipline: gptossReference below is a plain in-order scalar
// transcription of HuggingFace's transformers/models/gpt_oss/modeling_gpt_oss.py +
// configuration_gpt_oss.py. It reuses NONE of the production machinery — not
// cfg.BlockTopology, not normCfg, not cfg.attnScale, not cfg.windowForLayer, not
// softmaxAttentionScores/softmaxDropSinkInPlace, not route/routeTopKSoftmax, not
// expertGPTOSS, and not the gate_up/down materializers. It reads the PRISTINE
// pre-load fixture bytes (the fused [E,H,2I] gate_up and [E,I,H] down tensors as
// the checkpoint ships them) and re-derives the de-interleave and the transpose
// itself, so a materializer bug cannot cancel against the reference.
//
// The three family-specific divergence risks it is built to catch:
//
//  1. ATTENTION SINKS. HF concatenates a learned per-head sink logit onto each
//     score row, softmaxes the concatenation, then DROPS the sink column
//     (eager_attention_forward in modeling_gpt_oss.py). The sink is therefore in
//     the DENOMINATOR only and contributes no value vector — an implementation that
//     forgets it, or that lets it contribute a value, diverges. gptossOracleRun
//     records the sink's share of the denominator so the fixture cannot make this
//     path vacuously small.
//
//  2. LAYER ALTERNATION. gpt-oss alternates sliding and full attention with
//     PERIOD 2 — GptOssConfig.__init__ synthesizes
//     ["sliding_attention" if bool((i + 1) % 2) else "full_attention" ...] when
//     config.json omits layer_types, i.e. EVEN layers windowed, ODD layers full.
//     The cadence must be DERIVED from the config, never defaulted; the same class
//     of bug was found in the Gemma2 lane (a synthesized schedule for gemma3 but
//     not gemma2, so every layer got windowed). Both directions are witnessed here:
//     the "derived_cadence" case omits layer_types and forces the derivation, the
//     "declared_cadence" case publishes the REVERSED list and forces production to
//     honor it rather than a family default.
//
//  3. THE CLAMPED SwiGLU. HF's GptOssExperts hardcodes limit=7.0 and alpha=1.702
//     (they are NOT config fields on this family — cfg.SwigluLimit/SwigluAlpha are
//     the MiniMax-M3 axes) and computes
//     gate.clamp(max=limit) -> glu = gate*sigmoid(alpha*gate) -> (up+1)*glu with
//     up clamped to +/-limit. Note the ASYMMETRY: the gate has no lower clamp. The
//     fixture plants gate_up biases of +/-9 so all three clamp branches fire on
//     every run, and the run counters below fail the test if any of them does not.

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// gptossOracleIDs is the shared prompt. It is longer than the sliding window so a
// windowed layer genuinely drops keys (without that, the cadence axis is vacuous).
var gptossOracleIDs = []int{3, 17, 5, 9, 21, 2, 19}

// gptossOracleSpec holds the HF RULES for the fixture — the cadence, the window,
// the attention scale denominator, the GQA mapping and the SwiGLU-OAI constants —
// hardcoded from the HF sources rather than read back out of Config. Comparing
// production against a reference driven by this spec therefore witnesses the
// DERIVATION (config.json -> Config axes) as well as the arithmetic.
type gptossOracleSpec struct {
	layers, hidden, inter, vocab int
	heads, kvHeads, headDim      int
	experts, topK                int
	window                       int      // HF sliding_window
	pattern                      int      // HF cadence period: layer l is full iff (l+1)%pattern == 0
	layerTypes                   []string // non-nil = the cadence config.json DECLARES
	theta                        float64  // rope_theta
	eps                          float64  // rms_norm_eps
	alpha                        float64  // GptOssExperts.alpha
	limit                        float64  // GptOssExperts.limit (the SwiGLU clamp)
	upBias                       float64  // HF's (up + 1) convention
	scaleDenom                   int      // attention scale is 1/sqrt(scaleDenom)
	sinks                        bool     // learned per-head sink in the softmax denominator
	gqaInterleaved               bool     // wrong-but-plausible GQA mapping (h%nKV instead of h/grp)
}

// gptossOracleBaseSpec is the tiny gpt-oss fixture. heads*head_dim (24) deliberately
// differs from hidden (16) — as it does on the real checkpoint, 64*64 vs 2880 — so a
// projection-width/hidden-width conflation cannot cancel; kvHeads < heads keeps GQA
// live; topK < experts keeps routing live.
func gptossOracleBaseSpec() gptossOracleSpec {
	return gptossOracleSpec{
		layers: 4, hidden: 16, inter: 10, vocab: 23,
		heads: 4, kvHeads: 2, headDim: 6,
		experts: 4, topK: 2,
		window: 3, pattern: 2,
		// The published rope_theta is 150000, but that is paired with head_dim 64 and a
		// 128k context. At head_dim 6 and 7 positions the same base leaves every rotation
		// angle < 0.3 rad, so RoPE is very nearly the identity and the numeric comparison
		// stops witnessing it at all (the axis-liveness test below measured exactly that:
		// theta 150000 -> 10000 moved the logits by only 3.5e-05, under the tolerance).
		// theta 100 puts this fixture's three frequency bands in the same regime the real
		// checkpoint's 32 bands occupy, which is what makes the RoPE path load-bearing.
		theta: 100, eps: 1e-5,
		alpha: 1.702, limit: 7, upBias: 1,
		scaleDenom: 6,
		sinks:      true,
	}
}

// isSliding is HF's per-layer attention type: the declared layer_types when
// config.json carries them, else the period-2 default GptOssConfig synthesizes.
func (s gptossOracleSpec) isSliding(l int) bool {
	if len(s.layerTypes) > 0 {
		return s.layerTypes[l] == "sliding_attention"
	}
	return s.pattern <= 0 || (l+1)%s.pattern != 0
}

// windowFor is the sliding bound for layer l: the window on a sliding layer, and
// -1 (full causal) on a full-attention layer.
func (s gptossOracleSpec) windowFor(l int) int {
	if s.isSliding(l) {
		return s.window
	}
	return -1
}

// kvHeadFor is HF repeat_kv's grouping: query head h reads KV head h/(nH/nKV).
func (s gptossOracleSpec) kvHeadFor(h int) int {
	if s.gqaInterleaved {
		return h % s.kvHeads
	}
	return h / (s.heads / s.kvHeads)
}

func gptossOracleCfg(spec gptossOracleSpec) Config {
	return Config{
		HiddenSize:       spec.hidden,
		NumLayers:        spec.layers,
		NumHeads:         spec.heads,
		NumKVHeads:       spec.kvHeads,
		HeadDim:          spec.headDim,
		IntermediateSize: spec.inter,
		VocabSize:        spec.vocab,
		NumExperts:       spec.experts,
		NumExpertsPerTok: spec.topK,
		ModelType:        "gpt_oss",
		Architectures:    []string{"GptOssForCausalLM"},
		RMSNormEps:       spec.eps,
		RopeTheta:        spec.theta,
		AttentionBias:    true, // gpt-oss config.json sets attention_bias: true
		LayerTypes:       append([]string(nil), spec.layerTypes...),
		// tie_word_embeddings is false on gpt-oss: the fixture carries a separate lm_head.
	}
}

// gptossOracleGateUpBias plants the SwiGLU clamp. k is the flat index into the
// fused [E, 2I] gate_up bias, whose last axis interleaves gate (even) and up (odd)
// exactly as HF slices it (gate_up[..., ::2] / gate_up[..., 1::2]). Biases of +/-9
// sit outside the +/-7 limit by more than the projection can move them, so every
// clamp branch — gate-high, up-high, up-low — fires deterministically on every run.
func gptossOracleGateUpBias(k, I int, draw float32) float32 {
	j := k % (2 * I)
	i := j / 2
	if j%2 == 0 { // gate half: HF clamps the gate ABOVE only (clamp(min=None, max=limit))
		if i%4 == 0 {
			return 9
		}
		return draw * 0.5
	}
	switch i % 4 { // up half: HF clamps both ends (clamp(min=-limit, max=limit))
	case 1:
		return 9
	case 2:
		return -9
	}
	return draw * 0.5
}

// newGPTOSSOracleModel builds the fixture on gpt-oss's REAL checkpoint roster —
// including the FUSED per-layer expert tensors (mlp.experts.gate_up_proj [E,H,2I],
// gate_up_proj_bias [E,2I], down_proj [E,I,H], down_proj_bias [E,H]) and the
// mlp.router.* names — and loads it through the production HF entry point so
// materializeGPTOSSTensors runs for real.
//
// It returns the loaded Model plus a PRISTINE snapshot of the pre-load manifest and
// bytes. That snapshot is not a convenience: materializeFusedExpertTensor deletes
// the fused sources from the manifest and appends the split per-expert tensors to
// raw, so after the load the checkpoint-as-shipped no longer exists inside the
// Model. The reference reads the snapshot, which is what makes the de-interleave
// and the transpose independently checked instead of assumed.
func newGPTOSSOracleModel(t *testing.T, spec gptossOracleSpec) (*Model, map[string]tensorMeta, []byte) {
	t.Helper()
	cfg := gptossOracleCfg(spec)
	window := spec.window
	if err := cfg.deriveConfigAxes(configJSONHints{SlidingWindow: &window}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V, E := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.NumExperts

	type ts = synthTensor
	var tensors []ts
	tensors = append(tensors, ts{"model.embed_tokens.weight", []int{V, H}})
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		tensors = append(tensors,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{p + "self_attn.q_proj.weight", []int{nH * hd, H}},
			ts{p + "self_attn.q_proj.bias", []int{nH * hd}},
			ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.k_proj.bias", []int{nKV * hd}},
			ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.v_proj.bias", []int{nKV * hd}},
			ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
			ts{p + "self_attn.o_proj.bias", []int{H}},
			ts{p + "self_attn.sinks", []int{nH}},
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{p + "mlp.router.weight", []int{E, H}},
			ts{p + "mlp.router.bias", []int{E}},
			ts{p + "mlp.experts.gate_up_proj", []int{E, H, 2 * I}},
			ts{p + "mlp.experts.gate_up_proj_bias", []int{E, 2 * I}},
			ts{p + "mlp.experts.down_proj", []int{E, I, H}},
			ts{p + "mlp.experts.down_proj_bias", []int{E, H}},
		)
	}
	tensors = append(tensors,
		ts{"model.norm.weight", []int{H}},
		ts{"lm_head.weight", []int{V, H}},
	)

	// synthBuildRaw fills each tensor's elements in flat order, so a per-name counter
	// recovers the flat index the interleaved gate_up bias pattern needs.
	seen := make(map[string]int, len(tensors))
	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		k := seen[name]
		seen[name]++
		switch {
		case isCPUOracleNormWeight(name):
			return 1 + 0.25*next() // distinct NON-UNIT gains, well-conditioned
		case strings.HasSuffix(name, "self_attn.sinks"):
			// Same magnitude as the attention scores, so exp(sink) is a MATERIAL share
			// of the softmax denominator. A near-zero sink would make the sink-drop
			// softmax numerically indistinguishable from a plain one.
			return next()
		case strings.HasSuffix(name, "mlp.router.weight"), strings.HasSuffix(name, "mlp.router.bias"):
			return next() * 0.8 // wide enough that the per-expert logits separate
		case strings.HasSuffix(name, "mlp.experts.gate_up_proj_bias"):
			return gptossOracleGateUpBias(k, I, next())
		case strings.HasSuffix(name, "mlp.experts.gate_up_proj"):
			return next() * 0.15
		}
		return synthMatmulFill(name, next)
	})

	srcMan := make(map[string]tensorMeta, len(man))
	for name, meta := range man {
		srcMan[name] = meta
	}
	srcRaw := append([]byte(nil), raw...)

	m, err := newHFCheckpointModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newHFCheckpointModel(gpt-oss fixture): %v", err)
	}
	return m, srcMan, srcRaw
}

// gptossOracleRun is the reference output plus the liveness counters that keep the
// comparison honest: which experts actually fired, how often each SwiGLU clamp
// branch engaged, the largest share of a softmax denominator the sink took, and how
// many (layer, position) pairs a sliding layer actually clipped.
type gptossOracleRun struct {
	logits      [][]float32
	fired       map[[2]int]bool
	firedExpert map[int]bool
	gateClamped int
	upClampHi   int
	upClampLo   int
	sinkShare   float64
	windowDrops int
}

// gptossOracleAddBias is HF's F.linear bias term.
func gptossOracleAddBias(y, b []float32) []float32 {
	for i := range y {
		y[i] += b[i]
	}
	return y
}

// gptossOracleTopK is torch.topk over the router logits: the k largest, descending,
// ties resolved to the LOWER expert index.
func gptossOracleTopK(logits []float32, k int) []int {
	if k > len(logits) {
		k = len(logits)
	}
	taken := make([]bool, len(logits))
	out := make([]int, 0, k)
	for n := 0; n < k; n++ {
		best := -1
		for e := range logits {
			if taken[e] {
				continue
			}
			if best < 0 || logits[e] > logits[best] {
				best = e
			}
		}
		taken[best] = true
		out = append(out, best)
	}
	return out
}

// gptossOracleRouteWeights is GptOssTopKRouter: softmax over ONLY the k selected
// logits — NOT a softmax over all experts followed by a top-k slice (which is what
// Mixtral/Qwen3-MoE do, and what this family must not do).
func gptossOracleRouteWeights(logits []float32, picks []int) []float32 {
	top := make([]float32, len(picks))
	for i, e := range picks {
		top[i] = logits[e]
	}
	cpuOracleSoftmax(top)
	return top
}

// gptossReference runs the independent gpt-oss forward over the PRISTINE fixture
// bytes: per-position logits for ids under spec. Every step is the HF
// GptOssDecoderLayer / GptOssAttention / GptOssTopKRouter / GptOssExperts dataflow,
// hardcoded.
func gptossReference(t *testing.T, srcMan map[string]tensorMeta, srcRaw []byte, ids []int, spec gptossOracleSpec) gptossOracleRun {
	t.Helper()
	H, I, V, E := spec.hidden, spec.inter, spec.vocab, spec.experts
	nH, nKV, hd := spec.heads, spec.kvHeads, spec.headDim
	eps := float32(spec.eps)
	limit := float32(spec.limit)
	alpha := float32(spec.alpha)
	upBias := float32(spec.upBias)
	seq := len(ids)
	run := gptossOracleRun{fired: map[[2]int]bool{}, firedExpert: map[int]bool{}}

	// tensor decodes straight from the pre-load snapshot — not m.tensor(), not
	// m.manifest, so the production materializers are on the other side of the gate.
	tensor := func(name string) []float32 {
		meta, ok := srcMan[name]
		if !ok {
			t.Fatalf("gpt-oss oracle: source tensor %q missing from the fixture", name)
		}
		out := make([]float32, meta.Nbytes/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(srcRaw[meta.Offset+i*4:]))
		}
		return out
	}

	embed := tensor("model.embed_tokens.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		// gpt-oss applies no embedding scale (unlike Gemma).
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	for l := 0; l < spec.layers; l++ {
		p := layerPrefix(l)
		inNorm := tensor(p + "input_layernorm.weight")
		wq, bq := tensor(p+"self_attn.q_proj.weight"), tensor(p+"self_attn.q_proj.bias")
		wk, bk := tensor(p+"self_attn.k_proj.weight"), tensor(p+"self_attn.k_proj.bias")
		wv, bv := tensor(p+"self_attn.v_proj.weight"), tensor(p+"self_attn.v_proj.bias")
		wo, bo := tensor(p+"self_attn.o_proj.weight"), tensor(p+"self_attn.o_proj.bias")
		sink := tensor(p + "self_attn.sinks")
		postNorm := tensor(p + "post_attention_layernorm.weight")
		wr, br := tensor(p+"mlp.router.weight"), tensor(p+"mlp.router.bias")
		gu := tensor(p + "mlp.experts.gate_up_proj")
		gub := tensor(p + "mlp.experts.gate_up_proj_bias")
		dp := tensor(p + "mlp.experts.down_proj")
		dpb := tensor(p + "mlp.experts.down_proj_bias")

		// ---- attention sub-layer, PreNorm: x += attn(rmsnorm(x)) ----
		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn := cpuOracleRMSNorm(x[tt], inNorm, eps)
			q[tt] = gptossOracleAddBias(cpuOracleMatVec(wq, xn, nH*hd, H), bq)
			k[tt] = gptossOracleAddBias(cpuOracleMatVec(wk, xn, nKV*hd, H), bk)
			v[tt] = gptossOracleAddBias(cpuOracleMatVec(wv, xn, nKV*hd, H), bv)
			// rotate_half RoPE on q and k, per head, after projection.
			for h := 0; h < nH; h++ {
				cpuOracleRope(q[tt][h*hd:(h+1)*hd], tt, hd, spec.theta)
			}
			for h := 0; h < nKV; h++ {
				cpuOracleRope(k[tt][h*hd:(h+1)*hd], tt, hd, spec.theta)
			}
		}
		scale := float32(1.0 / math.Sqrt(float64(spec.scaleDenom)))
		W := spec.windowFor(l)
		for tt := 0; tt < seq; tt++ {
			lo := 0
			if W >= 0 && tt-W+1 > 0 {
				lo = tt - W + 1
			}
			if lo > 0 {
				run.windowDrops++
			}
			concat := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				kvh := spec.kvHeadFor(h)
				qh := q[tt][h*hd : (h+1)*hd]
				scores := make([]float32, tt+1-lo)
				for j := lo; j <= tt; j++ {
					kh := k[j][kvh*hd : (kvh+1)*hd]
					var s float32
					for d := 0; d < hd; d++ {
						s += qh[d] * kh[d]
					}
					scores[j-lo] = s * scale
				}
				// HF: concat the sink logit onto the row, subtract the row max, softmax,
				// then drop the sink column. Net effect: the sink adds exp(sink-max) to
				// the DENOMINATOR and contributes no value vector.
				mx := scores[0]
				for _, s := range scores {
					if s > mx {
						mx = s
					}
				}
				var sinkTerm float32
				if spec.sinks {
					if sink[h] > mx {
						mx = sink[h]
					}
					sinkTerm = float32(math.Exp(float64(sink[h] - mx)))
				}
				denom := sinkTerm
				for i := range scores {
					e := float32(math.Exp(float64(scores[i] - mx)))
					scores[i] = e
					denom += e
				}
				if share := float64(sinkTerm) / float64(denom); share > run.sinkShare {
					run.sinkShare = share
				}
				for i := range scores {
					scores[i] /= denom
				}
				o := concat[h*hd : (h+1)*hd]
				for j := lo; j <= tt; j++ {
					vh := v[j][kvh*hd : (kvh+1)*hd]
					w := scores[j-lo]
					for d := 0; d < hd; d++ {
						o[d] += w * vh[d]
					}
				}
			}
			attn := gptossOracleAddBias(cpuOracleMatVec(wo, concat, H, nH*hd), bo)
			for i := 0; i < H; i++ {
				x[tt][i] += attn[i]
			}
		}

		// ---- MoE sub-layer, PreNorm: x += moe(rmsnorm(x)) ----
		for tt := 0; tt < seq; tt++ {
			xn := cpuOracleRMSNorm(x[tt], postNorm, eps)
			rlog := gptossOracleAddBias(cpuOracleMatVec(wr, xn, E, H), br)
			picks := gptossOracleTopK(rlog, spec.topK)
			weights := gptossOracleRouteWeights(rlog, picks)
			delta := make([]float32, H)
			for pi, e := range picks {
				run.fired[[2]int{l, e}] = true
				run.firedExpert[e] = true
				y := make([]float32, I)
				for i := 0; i < I; i++ {
					// gate_up_proj[e] is [H, 2I] and HF computes x @ W, so the gate row for
					// intermediate i is column 2i and the up row is column 2i+1.
					var g, u float32
					for h := 0; h < H; h++ {
						base := (e*H+h)*2*I + 2*i
						g += gu[base] * xn[h]
						u += gu[base+1] * xn[h]
					}
					g += gub[e*2*I+2*i]
					u += gub[e*2*I+2*i+1]
					if g > limit { // asymmetric: HF clamps the gate ABOVE only
						g = limit
						run.gateClamped++
					}
					if u > limit {
						u = limit
						run.upClampHi++
					} else if u < -limit {
						u = -limit
						run.upClampLo++
					}
					glu := g * float32(1/(1+math.Exp(-float64(alpha*g))))
					y[i] = (u + upBias) * glu
				}
				// down_proj[e] is [I, H] and HF computes y @ D.
				w := weights[pi]
				for h := 0; h < H; h++ {
					var s float32
					for i := 0; i < I; i++ {
						s += dp[(e*I+i)*H+h] * y[i]
					}
					delta[h] += w * (s + dpb[e*H+h])
				}
			}
			for h := 0; h < H; h++ {
				x[tt][h] += delta[h]
			}
		}
	}

	norm := tensor("model.norm.weight")
	head := tensor("lm_head.weight") // gpt-oss is UNTIED
	run.logits = make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := cpuOracleRMSNorm(x[tt], norm, eps)
		run.logits[tt] = cpuOracleMatVec(head, xf, V, H)
	}
	return run
}

// gptossOracleAssertLive fails the run if any of the family's distinctive paths did
// not actually engage on this fixture. Without these, a green comparison would be
// compatible with a forward that never clamps, never sinks and never windows.
func gptossOracleAssertLive(t *testing.T, run gptossOracleRun, spec gptossOracleSpec) {
	t.Helper()
	if run.gateClamped == 0 || run.upClampHi == 0 || run.upClampLo == 0 {
		t.Fatalf("SwiGLU clamp never engaged (gate-high=%d up-high=%d up-low=%d) — the clamp is untested",
			run.gateClamped, run.upClampHi, run.upClampLo)
	}
	if run.sinkShare < 0.05 {
		t.Fatalf("attention sink took at most %.4f of a softmax denominator — too small to witness", run.sinkShare)
	}
	if run.windowDrops == 0 {
		t.Fatalf("no sliding layer ever clipped a key — the layer cadence is untested")
	}
	if len(run.firedExpert) != spec.experts {
		t.Fatalf("router fired %d of %d experts — expert coverage is incomplete", len(run.firedExpert), spec.experts)
	}
}

// TestGPTOSSCPUNumericOracle is the gpt-oss x cpu M4 witness: the production forward
// (Forward, and the cached Prefill/Step decode path) must reproduce the independent
// HF-semantics reference at every position within cpuOracleTol.
//
// Two cadence cases, because "derived from the config" has two failure directions:
// derived_cadence omits layer_types (production must synthesize HF's period-2
// alternation), declared_cadence publishes the REVERSED list (production must honor
// it rather than a family default).
func TestGPTOSSCPUNumericOracle(t *testing.T) {
	for _, tc := range []struct {
		name       string
		layerTypes []string
	}{
		{"derived_cadence", nil},
		{"declared_cadence", []string{"full_attention", "sliding_attention", "full_attention", "sliding_attention"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spec := gptossOracleBaseSpec()
			spec.layerTypes = tc.layerTypes
			m, srcMan, srcRaw := newGPTOSSOracleModel(t, spec)

			if !m.Cfg.isGPTOSS() {
				t.Fatalf("fixture family key %q is not recognized as gpt-oss", m.Cfg.archFamilyKey())
			}
			if m.Cfg.BlockTopology != PreNorm {
				t.Fatalf("gpt-oss derived topology = %v, want PreNorm", m.Cfg.BlockTopology)
			}
			// The derivation must land HF's cadence; the reference hardcodes it.
			for l := 0; l < spec.layers; l++ {
				want := "full_attention"
				if spec.isSliding(l) {
					want = "sliding_attention"
				}
				if got := m.Cfg.layerType(l); got != want {
					t.Fatalf("layer %d type = %q, want %q (HF GptOssConfig alternates with period 2, "+
						"even layers sliding); derived layer_types = %v", l, got, want, m.Cfg.LayerTypes)
				}
				if got, want := m.Cfg.windowForLayer(l), spec.windowFor(l); got != want {
					t.Fatalf("layer %d window = %d, want %d; derived Window = %v", l, got, want, m.Cfg.Window)
				}
			}
			// The fused expert sources must have been consumed by the materializer, so the
			// production side really did run the de-interleave/transpose under test.
			for _, fused := range []string{"mlp.experts.gate_up_proj", "mlp.experts.down_proj"} {
				if _, ok := m.manifest[layerPrefix(0)+fused]; ok {
					t.Fatalf("fused %q survived the load — the materializer did not run", fused)
				}
			}
			if _, ok := m.manifest[expertName(0, spec.experts-1, "down_proj.weight")]; !ok {
				t.Fatal("materializer produced no per-expert down_proj for the last expert")
			}

			ids := gptossOracleIDs
			ref := gptossReference(t, srcMan, srcRaw, ids, spec)
			gptossOracleAssertLive(t, ref, spec)

			act := m.Forward(ids)
			for tt := range ids {
				if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref.logits[tt]); d > cpuOracleTol {
					t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", tt, d, cpuOracleTol)
				}
			}

			// Cached decode path: Prefill then Step against the same reference (the
			// reference is cacheless, so Step is compared against an extended-prompt run).
			s := m.NewSession()
			pf := s.Prefill(ids)
			if d := cpuOracleMaxAbsDiff(pf, ref.logits[len(ids)-1]); d > cpuOracleTol {
				t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
			}
			next := 11
			st := s.Step(next)
			extRef := gptossReference(t, srcMan, srcRaw, append(append([]int(nil), ids...), next), spec)
			if d := cpuOracleMaxAbsDiff(st, extRef.logits[len(ids)]); d > cpuOracleTol {
				t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
			}
		})
	}
}

// TestGPTOSSCPUNumericOracleIsSensitive proves the comparison is non-vacuous for
// EVERY distinct weight role gpt-oss carries: perturbing one raw f32 in each must
// move the compared logits far beyond the tolerance. A role that stays green is a
// role the oracle cannot see.
//
// The expert-weight rows are chosen deliberately: intermediate lanes whose planted
// bias saturates the SwiGLU clamp are numerically inert by construction (that is
// what a clamp does), so those perturbations target an UNCLAMPED lane instead.
func TestGPTOSSCPUNumericOracleIsSensitive(t *testing.T) {
	spec := gptossOracleBaseSpec()
	_, srcMan, srcRaw := newGPTOSSOracleModel(t, spec)
	ids := gptossOracleIDs
	ref := gptossReference(t, srcMan, srcRaw, ids, spec)
	gptossOracleAssertLive(t, ref, spec)

	// Perturb an expert the router actually fires, so an unrouted expert cannot mask
	// the perturbation.
	layer, expert := -1, -1
	for l := 0; l < spec.layers && layer < 0; l++ {
		for e := 0; e < spec.experts; e++ {
			if ref.fired[[2]int{l, e}] {
				layer, expert = l, e
				break
			}
		}
	}
	if layer < 0 {
		t.Fatal("no expert was routed on the fixture prompt")
	}

	H := spec.hidden
	for _, role := range []struct {
		role   string
		tensor string
		elem   int
	}{
		{"input norm", layerPrefix(0) + "input_layernorm.weight", 0},
		{"post-attention norm", layerPrefix(0) + "post_attention_layernorm.weight", 0},
		{"final norm", "model.norm.weight", 0},
		{"lm head", "lm_head.weight", 0},
		{"embedding", "model.embed_tokens.weight", ids[0] * H},
		{"q_proj", layerPrefix(0) + "self_attn.q_proj.weight", 0},
		{"q_proj bias", layerPrefix(0) + "self_attn.q_proj.bias", 0},
		{"k_proj", layerPrefix(0) + "self_attn.k_proj.weight", 0},
		{"v_proj", layerPrefix(0) + "self_attn.v_proj.weight", 0},
		{"o_proj", layerPrefix(0) + "self_attn.o_proj.weight", 0},
		{"o_proj bias", layerPrefix(0) + "self_attn.o_proj.bias", 0},
		{"attention sink", layerPrefix(0) + "self_attn.sinks", 0},
		{"router weight", routerName(0), 0},
		{"router bias", routerBiasName(0), 0},
		{"expert gate_proj", expertName(layer, expert, "gate_proj.weight"), 1 * H},
		{"expert up_proj", expertName(layer, expert, "up_proj.weight"), 0},
		{"expert down_proj", expertName(layer, expert, "down_proj.weight"), 0},
		{"expert gate bias", expertName(layer, expert, "gate_proj.bias"), 1},
		{"expert up bias", expertName(layer, expert, "up_proj.bias"), 0},
		{"expert down bias", expertName(layer, expert, "down_proj.bias"), 0},
	} {
		t.Run(strings.ReplaceAll(role.role, " ", "_"), func(t *testing.T) {
			m, _, _ := newGPTOSSOracleModel(t, spec)
			meta, ok := m.manifest[role.tensor]
			if !ok {
				t.Fatalf("perturbation target %q is absent from the loaded manifest", role.tensor)
			}
			if role.elem*4 >= meta.Nbytes {
				t.Fatalf("perturbation element %d is out of range for %q (%d bytes)", role.elem, role.tensor, meta.Nbytes)
			}
			at := meta.Offset + role.elem*4
			orig := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[at:]))
			binary.LittleEndian.PutUint32(m.raw[at:], math.Float32bits(orig+0.5))

			act := m.Forward(ids)
			var worst float64
			for tt := range ids {
				if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref.logits[tt]); d > worst {
					worst = d
				}
			}
			if worst <= cpuOracleTol {
				t.Fatalf("perturbing %s (%s[%d]) left the logits within tolerance (max|delta|=%.3e) — "+
					"the oracle does not see this weight role", role.role, role.tensor, role.elem, worst)
			}
		})
	}
}

// gptossOracleAssertAxisLive fails when moving a derived config axis to a
// wrong-but-plausible value leaves the reference unchanged. A dead axis means the
// numeric comparison above would pass just as well with the axis mis-derived, i.e.
// the oracle does not actually pin it.
func gptossOracleAssertAxisLive(t *testing.T, srcMan map[string]tensorMeta, srcRaw []byte, ids []int, base, wrong gptossOracleSpec, axis string) {
	t.Helper()
	b := gptossReference(t, srcMan, srcRaw, ids, base)
	w := gptossReference(t, srcMan, srcRaw, ids, wrong)
	var worst float64
	for tt := range ids {
		if d := cpuOracleMaxAbsDiff(b.logits[tt], w.logits[tt]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Errorf("axis %q is DEAD: the wrong value moved the reference by only %.3e (tol %.0e) — "+
			"the numeric oracle does not pin it", axis, worst, cpuOracleTol)
	}
}

// TestGPTOSSCPUNumericOracleAxesAreLive proves each gpt-oss axis the reference
// hardcodes is load-bearing: mutating it to a wrong-but-plausible value must move
// the reference. This is what makes TestGPTOSSCPUNumericOracle a derivation witness
// rather than only an arithmetic one.
func TestGPTOSSCPUNumericOracleAxesAreLive(t *testing.T) {
	base := gptossOracleBaseSpec()
	_, srcMan, srcRaw := newGPTOSSOracleModel(t, base)
	ids := gptossOracleIDs

	for _, tc := range []struct {
		axis   string
		mutate func(s *gptossOracleSpec)
	}{
		// Cadence period 1 = every layer full attention: the exact shape of the Gemma2
		// bug (a family whose alternation is never synthesized).
		{"cadence_pattern", func(s *gptossOracleSpec) { s.pattern = 1 }},
		// Cadence phase flipped: sliding and full layers swapped.
		{"cadence_phase", func(s *gptossOracleSpec) {
			s.layerTypes = []string{"full_attention", "sliding_attention", "full_attention", "sliding_attention"}
		}},
		{"sliding_window", func(s *gptossOracleSpec) { s.window = 5 }},
		{"attention_sinks", func(s *gptossOracleSpec) { s.sinks = false }},
		{"attn_scale_denom", func(s *gptossOracleSpec) { s.scaleDenom = s.headDim * 4 }},
		{"gqa_grouping", func(s *gptossOracleSpec) { s.gqaInterleaved = true }},
		{"rope_theta", func(s *gptossOracleSpec) { s.theta = 10000 }},
		{"expert_top_k", func(s *gptossOracleSpec) { s.topK = 3 }},
		{"swiglu_limit", func(s *gptossOracleSpec) { s.limit = 100 }},
		{"swiglu_alpha", func(s *gptossOracleSpec) { s.alpha = 1 }},
		{"swiglu_up_plus_one", func(s *gptossOracleSpec) { s.upBias = 0 }},
		{"rms_norm_eps", func(s *gptossOracleSpec) { s.eps = 1e-1 }},
	} {
		t.Run(tc.axis, func(t *testing.T) {
			wrong := base
			tc.mutate(&wrong)
			gptossOracleAssertAxisLive(t, srcMan, srcRaw, ids, base, wrong, tc.axis)
		})
	}
}
