package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

// arch.go — the Stage-2 mechanical-axis hooks (MODEL-ARCH-SEAM.md §2b class-1).
//
// Each helper here is a SINGLE source of truth for one mechanical architecture axis:
// a scalar/elementwise edit whose Llama value is the identity / current code path. The
// hot-path block functions (blockStep, the prefill/quant/batch twins, forward.layer)
// call these helpers instead of open-coding the axis, so an axis is defined once and
// the Llama no-op is uniform across every site — the §0 "arch change applied to 6 of 7"
// hazard cannot bite, because there is one definition.
//
// LLAMA INVARIANCE: every function below is provably a no-op under the Llama config
// (RopeScaling=="", QKNorm=false, NormGain1p=false, ActGeluTanh=false, all softcaps and
// scales 0, QueryPreAttnScalar==0). That is what TestArchLlamaNoOp / TestRefactorMatchesSerial
// assert by Float32bits equality.

// ---- RoPE scaling -----------------------------------------------------------------
//
// applyRopeScaling rescales an inv_freq array in place per the configured scheme. The
// default ("" / "none") returns the array untouched, so Llama's bare inv_freq is
// bit-for-bit preserved. "llama3" implements the piecewise low/high-freq-wavelength
// rescale of Llama-3.1/3.2/3.3 with the smooth interpolation band — the same formula
// HF's _compute_llama3_parameters uses.
func applyRopeScaling(cfg Config, inv []float64) {
	switch cfg.RopeScaling {
	case "", "none":
		return
	case "linear":
		factor := cfg.RopeFactor
		if factor == 0 {
			factor = cfg.effectiveRopeParameters().Factor
		}
		if factor <= 0 || math.IsNaN(factor) || math.IsInf(factor, 0) {
			return
		}
		for j := range inv {
			inv[j] /= factor
		}
	case "llama3":
		factor := cfg.RopeFactor
		lowFreqFactor := cfg.RopeLowFreqFactor
		highFreqFactor := cfg.RopeHighFreqFactor
		origCtx := float64(cfg.RopeOrigContext)
		if factor == 0 || lowFreqFactor == 0 || highFreqFactor == 0 || origCtx == 0 {
			// Misconfigured llama3 scaling: leave inv_freq bare rather than divide by zero.
			return
		}
		lowFreqWavelen := origCtx / lowFreqFactor
		highFreqWavelen := origCtx / highFreqFactor
		for j := range inv {
			wavelen := 2 * math.Pi / inv[j]
			switch {
			case wavelen > lowFreqWavelen:
				// long-wavelength (low-freq) band: fully scaled down by factor.
				inv[j] = inv[j] / factor
			case wavelen < highFreqWavelen:
				// short-wavelength (high-freq) band: untouched.
			default:
				// smooth interpolation band between the two.
				smooth := (origCtx/wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
				inv[j] = (1-smooth)*(inv[j]/factor) + smooth*inv[j]
			}
		}
	case "yarn":
		// Resolve the yarn block from EITHER HF shape (see effectiveRopeParameters), then
		// let each field fall through to its flat Config twin. Reading it opportunistically
		// — zero value when neither shape is present — instead of returning early is what
		// stops a yarn config from silently rotating with a BARE inv_freq (#4874); the
		// explicit all-params-zero guard below still fails safe for a misconfigured yarn.
		rp := cfg.effectiveRopeParameters()
		factor := rp.Factor
		if factor == 0 {
			factor = cfg.RopeFactor
		}
		origCtx := rp.OriginalMaxPositionEmbeddings
		if origCtx == 0 {
			origCtx = cfg.RopeOrigContext
		}
		base := rp.RopeTheta
		if base == 0 {
			base = cfg.RopeTheta
		}
		if factor == 0 || origCtx == 0 || base == 0 {
			return
		}
		dim := cfg.rotaryDim()
		betaFast := rp.BetaFast
		if betaFast == 0 {
			betaFast = 32
		}
		betaSlow := rp.BetaSlow
		if betaSlow == 0 {
			betaSlow = 1
		}
		truncate := true
		if rp.Truncate != nil {
			truncate = *rp.Truncate
		}
		low, high := yarnCorrectionRange(betaFast, betaSlow, dim, base, float64(origCtx), truncate)
		for j := range inv {
			ramp := yarnLinearRamp(float64(j), low, high)
			inv[j] = inv[j]*(1-ramp) + (inv[j]/factor)*ramp
		}
	default:
		// Unknown scheme: fail safe to bare inv_freq (an unsupported scaling is a load-time
		// concern surfaced elsewhere, not a silent mis-rotation here).
	}
}

func (c Config) defaultRopeParameters() (RopeScaling, bool) {
	rp, ok := c.RopeParameters["default"]
	return rp, ok
}

// effectiveRopeParameters is the SINGLE resolution of "which nested block carries this
// config's RoPE-scaling params", across the two HF shapes a checkpoint may use: the newer
// rope_parameters.default object, or the classic rope_scaling object (decoded into
// LongRope). The nested form wins when both are present; a LongRope carrying the longrope
// kind is NOT returned here (its per-dimension factor vectors run through ropeLongFactor,
// a different axis). The zero value when neither is present is deliberate — every caller
// falls its fields through to the flat Config twins, so an absent block degrades to the
// flat params rather than to "no scaling at all".
//
// Every yarn consumer resolves through THIS function — applyRopeScaling, the
// ropeAttentionFactor temperature, AND the cachedInvFreq key. That last one is load
// bearing: the cache key must fold in whichever block actually fed the table, or two
// configs that differ only in their classic-key yarn params would collide on one cached
// inv_freq.
func (c Config) effectiveRopeParameters() RopeScaling {
	if rp, ok := c.defaultRopeParameters(); ok {
		return rp
	}
	if c.LongRope != nil && c.LongRope.kind() == "yarn" {
		return *c.LongRope
	}
	return RopeScaling{}
}

func yarnCorrectionRange(lowRot, highRot float64, dim int, base, maxPos float64, truncate bool) (float64, float64) {
	correctionDim := func(rot float64) float64 {
		return (float64(dim) * math.Log(maxPos/(rot*2*math.Pi))) / (2 * math.Log(base))
	}
	low := correctionDim(lowRot)
	high := correctionDim(highRot)
	if truncate {
		low = math.Floor(low)
		high = math.Ceil(high)
	}
	if low < 0 {
		low = 0
	}
	hiMax := float64(dim - 1)
	if high > hiMax {
		high = hiMax
	}
	return low, high
}

func yarnLinearRamp(idx, low, high float64) float64 {
	if low == high {
		high += 0.001
	}
	v := (idx - low) / (high - low)
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func (c Config) ropeAttentionFactor() float64 {
	if c.RopeScaling != "yarn" {
		return 1
	}
	rp := c.effectiveRopeParameters()
	if rp.AttentionFactor != 0 {
		return rp.AttentionFactor
	}
	factor := rp.Factor
	if factor == 0 {
		factor = c.RopeFactor
	}
	mscale := rp.MScale
	mscaleAllDim := rp.MScaleAllDim
	getMScale := func(scale, m float64) float64 {
		if scale <= 1 {
			return 1
		}
		if m == 0 {
			m = 1
		}
		return 0.1*m*math.Log(scale) + 1
	}
	if mscale != 0 && mscaleAllDim != 0 {
		return getMScale(factor, mscale) / getMScale(factor, mscaleAllDim)
	}
	return getMScale(factor, 1)
}

// ---- per-projection bias ----------------------------------------------------------
//
// addBiasIfPresent adds a projection bias to y only when the named bias tensor exists
// in the checkpoint. This generalizes the all-or-nothing cfg.AttentionBias to
// presence-driven per-projection bias: a Llama checkpoint (no bias tensors) is the
// no-op it is today, Qwen2 (all of q/k/v present) reproduces the current addBias
// exactly, and a family carrying only some biases gets exactly those. The cfg.AttentionBias
// gate is preserved as a fast path / explicit opt-in; when it is set the tensors are
// assumed present (the historical contract). When it is unset we still honor any bias
// tensor that is physically present.
func (m *Model) addBiasIfPresent(y []float32, name string) {
	if m.has(name) {
		addBias(y, m.tensor(name))
	}
}

// fuseGatedMLPPanels finishes a batched gated MLP's gate/up panels IN PLACE: it adds the
// optional gate/up projection biases and then fuses the config's activation into G,
// G[i] = act(G[i], cfg) * U[i]. G and U are row-major [N,I]. For Llama act==silu, so this
// is the ordinary SwiGLU; for Gemma it is the GeGLU. Six batched lanes carried their own
// transcription of these two loops.
//
// The bias add is UNCONDITIONAL here — there is no withProjBias knob — and that is the
// point. It used to be a parameter, because prefillBatched (prefill_batch.go),
// verifyForwardBatched (verify.go), the batched f32 decode and the batched Q8 decode
// (both batch_step.go) dropped mlp.{gate,up,down}_proj.bias while every other lane applied
// it. That was a real wrong-numbers bug, not a design choice: the f32 lanes' own headers
// each contract them to be BIT-IDENTICAL to the per-token path, whose denseSwiGLU (moe.go)
// adds all three; the Q8 decode is licensed to differ from serial only by the tile GEMM's
// reduction order, which a dropped term is not; and none of their admission gates
// (q8PrefillNeedsTokenLoop, batchPreNormFastPathOK, verifyForwardBatchedOK, q8FastPreNormOK)
// has a bias term to keep a bias-carrying checkpoint out. Removing the parameter is what
// makes the divergence unreintroducible.
//
// addBiasIfPresent is a no-op when the tensor is absent, so a bias-free checkpoint (Llama,
// SmolLM2, Qwen2/3, Gemma, Mistral — every f32 CPU numeric oracle in the suite) is
// byte-for-byte unchanged. proj_bias_batched_lanes_test.go is the witness.
//
// The activation*up fusion is delegated to ffn.ApplyInPlace after the per-row bias add.
// The delegation is numerically identical: ApplyInPlace validates the same nonempty /
// equal-length / non-nil-activation contract and writes gate[i] = act(gate[i]) * up[i] in
// the same increasing-index order as the loop it replaces. A zero-row panel (N==0) is a
// no-op in both forms.
func (m *Model) fuseGatedMLPPanels(lp func(string) string, G, U []float32, N, I int, cfg Config) {
	// A zero-row (or zero-width) panel is a no-op, exactly as the pre-refactor
	// activation loop was: ApplyInPlace refuses an empty row, so it must not be
	// reached for an empty panel.
	if len(G) == 0 {
		return
	}
	for row := 0; row < N; row++ {
		m.addBiasIfPresent(G[row*I:(row+1)*I], lp("mlp.gate_proj.bias"))
		m.addBiasIfPresent(U[row*I:(row+1)*I], lp("mlp.up_proj.bias"))
	}
	activate := func(v float32) float32 { return act(v, cfg) }
	if err := ffn.ApplyInPlace(G, U, activate); err != nil {
		panic(err)
	}
}

// batchedGatedMLP runs one layer's f32 gated MLP over N already-post-normed rows Xn2 [N,H]
// and returns the [N,H] down projection: gate/up matMulBatch, fuseGatedMLPPanels, down
// matMulBatch, then the optional down bias. Every arithmetic call and its order is the one
// the four hand-copied lanes ran, and matMulBatch's reduction is in-order, so the bit-exact
// f32 rungs still hold. lp is the caller's layer-name closure. See fuseGatedMLPPanels for
// why the projection biases are unconditional.
func (m *Model) batchedGatedMLP(lp func(string) string, Xn2 []float32, N, H, I int, cfg Config) []float32 {
	G := matMulBatch(m.tensor(lp("mlp.gate_proj.weight")), Xn2, I, H, N)
	U := matMulBatch(m.tensor(lp("mlp.up_proj.weight")), Xn2, I, H, N)
	m.fuseGatedMLPPanels(lp, G, U, N, I, cfg)
	Down := matMulBatch(m.tensor(lp("mlp.down_proj.weight")), G, H, I, N)
	for row := 0; row < N; row++ {
		m.addBiasIfPresent(Down[row*H:(row+1)*H], lp("mlp.down_proj.bias"))
	}
	return Down
}

// applyProjBias applies q/k/v projection biases for layer l. With cfg.AttentionBias set
// it adds all three (the historical Qwen2 path, unchanged); otherwise it adds only the
// biases physically present. For Llama (no bias, flag off) it is a no-op.
func (m *Model) applyProjBias(l int, q, k, v []float32) {
	cfg := m.Cfg
	p := func(s string) string { return layerName(l, s) }
	if cfg.AttentionBias {
		addBias(q, m.tensor(p("self_attn.q_proj.bias")))
		addBias(k, m.tensor(p("self_attn.k_proj.bias")))
		addBias(v, m.tensor(p("self_attn.v_proj.bias")))
		return
	}
	m.addBiasIfPresent(q, p("self_attn.q_proj.bias"))
	m.addBiasIfPresent(k, p("self_attn.k_proj.bias"))
	m.addBiasIfPresent(v, p("self_attn.v_proj.bias"))
}

// applyLayerQKNorm runs per-head qk-norm on a packed q and k for layer l when QKNorm is
// on, using that layer's q_norm/k_norm weights. No-op when QKNorm is off (Llama).
func (m *Model) applyLayerQKNorm(l int, q, k []float32) {
	m.tpApplyQKNormBand(l, q, k, m.Cfg.NumHeads, m.Cfg.NumKVHeads)
}

// ---- per-head qk-norm -------------------------------------------------------------
//
// applyQKNorm RMS-normalizes each head slice of a packed q (or k) vector in place using
// a per-head weight (length HeadDim), when QKNorm is on. It is applied AFTER projection
// and BEFORE RoPE. qk-norm is position-independent, so it commutes with the single-
// rotation eviction re-derivation: the caller stashes Kraw POST-qk-norm / PRE-RoPE so
// Evict stays bit-exact (the §3 KV rider). Off (default) = no-op.
func applyQKNorm(hv, w []float32, nHeads, hd int, eps float32) {
	applyQKNormCfg(hv, w, nHeads, hd, eps, Config{})
}

func applyQKNormCfg(hv, w []float32, nHeads, hd int, eps float32, cfg Config) {
	if cfg.QKNormPerHeadWeight && hd > 0 && len(w) == nHeads*hd && len(hv) == nHeads*hd {
		// Cohere: CohereLayerNorm(hidden_size=(num_heads, head_dim)). The TUPLE hidden
		// size means one reduction PER HEAD over head_dim, scaled by THAT head's own row
		// of the (num_heads, head_dim) parameter — and, because every Cohere norm is a
		// CohereLayerNorm, the reduction is mean-subtracting. This weight has exactly the
		// same LENGTH as OLMo2's flat projection-width q_norm handled just below, so no
		// length check can tell the two apart; the derived family axis is what
		// discriminates. Guarding on len(hv) too keeps a tensor-parallel BAND (which
		// passes a band head count with the full-model weight) out of this branch.
		for h := 0; h < nHeads; h++ {
			applyNormInPlaceCfg(hv[h*hd:(h+1)*hd], w[h*hd:(h+1)*hd], eps, cfg)
		}
		return
	}
	if len(w) == len(hv) {
		applyRMSNormInPlaceCfg(hv, w, eps, cfg)
		return
	}
	if len(w) != hd {
		panic("model: qk-norm weight length does not match head_dim or projection width")
	}
	for h := 0; h < nHeads; h++ {
		head := hv[h*hd : (h+1)*hd]
		applyRMSNormInPlaceCfg(head, w, eps, cfg)
	}
}

// applyNormInPlaceCfg normalizes x in place with the model's norm KIND: bias-free
// mean-subtracting LayerNorm when cfg.LayerNorm, else RMSNorm. It is the in-place
// counterpart of normCfg, which allocates for the block norms.
func applyNormInPlaceCfg(x, w []float32, eps float32, cfg Config) {
	if !cfg.LayerNorm {
		applyRMSNormInPlaceCfg(x, w, eps, cfg)
		return
	}
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
	for i, v := range x {
		gain := w[i]
		if cfg.NormGain1p {
			gain = 1 + gain
		}
		x[i] = (v - mean) * inv * gain
	}
}

func applyRMSNormInPlace(x, w []float32, eps float32) {
	applyRMSNormInPlaceCfg(x, w, eps, Config{})
}

func applyRMSNormInPlaceCfg(x, w []float32, eps float32, cfg Config) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i := range x {
		gain := w[i]
		if cfg.NormGain1p {
			gain = 1 + gain
		}
		x[i] = x[i] * inv * gain
	}
}

// qkNormEps resolves the epsilon used by qk-norm: the explicit QKNormEps, else the
// model's RMSNormEps.
func (c Config) qkNormEps() float32 {
	if c.QKNormEps != 0 {
		return float32(c.QKNormEps)
	}
	return float32(c.RMSNormEps)
}

// ---- attention scale --------------------------------------------------------------
//
// attnScale is the per-head SDPA denominator. Default 1/sqrt(HeadDim); when
// QueryPreAttnScalar is set (Gemma) it is 1/sqrt(QueryPreAttnScalar). Phi longrope
// then applies its context-temperature multiplier; for non-longrope configs that
// multiplier is 1, so the Llama/Qwen path is unchanged.
func (c Config) attnScale() float32 {
	denom := c.HeadDim
	if c.QueryPreAttnScalar != 0 {
		denom = c.QueryPreAttnScalar
	}
	return float32((1.0 / math.Sqrt(float64(denom))) * longropeAttnScaleMul(c))
}

func (c Config) alibiScoreBias(head, key, keyLen int) float32 {
	if !c.Alibi {
		return 0
	}
	max := c.AlibiBiasMax
	if max == 0 {
		max = 8
	}
	pow2 := 1
	for pow2 < c.NumHeads {
		pow2 <<= 1
	}
	idx := head
	if pow2 != c.NumHeads {
		ordered := make([]int, 0, pow2)
		for i := 1; i < pow2; i += 2 {
			ordered = append(ordered, i)
		}
		for i := 0; i < pow2; i += 2 {
			ordered = append(ordered, i)
		}
		idx = ordered[head]
	}
	base := float64(idx+1) * (max / float64(pow2))
	slope := 1.0 / math.Pow(2, base)
	return float32(slope * float64(key-keyLen+1))
}

func (m *Model) softmaxAttentionScores(layer, head int, scores []float32) {
	name := layerName(layer, "self_attn.sinks")
	if !m.has(name) {
		softmaxInPlace(scores)
		return
	}
	sink := m.tensor(name)
	if head < 0 || head >= len(sink) {
		softmaxInPlace(scores)
		return
	}
	softmaxDropSinkInPlace(scores, sink[head])
}

func softmaxDropSinkInPlace(scores []float32, sink float32) {
	mx := sink
	for _, v := range scores {
		if v > mx {
			mx = v
		}
	}
	sum := float32(math.Exp(float64(sink - mx)))
	for i, v := range scores {
		e := float32(math.Exp(float64(v - mx)))
		scores[i] = e
		sum += e
	}
	for i := range scores {
		scores[i] /= sum
	}
}

// ---- soft-caps --------------------------------------------------------------------
//
// softcap maps z -> cap*tanh(z/cap). cap<=0 (default) is the identity. Used on attention
// scores (pre-softmax) and on final logits for Gemma2.
func softcap(z, cap float32) float32 {
	if cap <= 0 {
		return z
	}
	return cap * float32(math.Tanh(float64(z/cap)))
}

// softcapInPlace applies softcap to every element of s (no-op when cap<=0).
func softcapInPlace(s []float32, cap float32) {
	if cap <= 0 {
		return
	}
	for i := range s {
		s[i] = softcap(s[i], cap)
	}
}

// ---- embed / logit scale ----------------------------------------------------------
//
// embedScale is the multiplier applied to an embedding row at lookup. 0 or 1 = identity
// (Llama); Gemma uses sqrt(hidden).
func (c Config) embedScale() float32 {
	if c.EmbedScale == 0 || c.EmbedScale == 1 {
		return 1
	}
	return float32(c.EmbedScale)
}

// scaleEmbedInPlace multiplies an embedding row by the configured embed-scale (no-op at 1).
func scaleEmbedInPlace(x []float32, cfg Config) {
	s := cfg.embedScale()
	if s == 1 {
		return
	}
	for i := range x {
		x[i] *= s
	}
}

// embedRowsInto fills X — row-major [len(ids), H], and X may be a longer buffer whose
// FIRST len(ids) rows are written — with each token's embedding row, then applies the
// config's embedding scale in place (Gemma; a no-op for Llama). This is the embedding
// lookup every batched lane opens with; it used to be transcribed per lane.
func (m *Model) embedRowsInto(X []float32, ids []int, H int, cfg Config) {
	if m.Q2KEmbedding != nil {
		scale := cfg.embedScale()
		for row, id := range ids {
			if err := m.Q2KEmbedding.GatherRow(id, X[row*H:(row+1)*H], scale); err != nil {
				panic(err)
			}
		}
		return
	}
	embed := m.embedRows()
	for row, id := range ids {
		copy(X[row*H:(row+1)*H], embed[id*H:(id+1)*H])
		m.prismInverseEmbeddingRow("model.embed_tokens.weight", X[row*H:(row+1)*H])
		scaleEmbedInPlace(X[row*H:(row+1)*H], cfg)
	}
}

// embedRectRowsInto is embedRowsInto for a RECTANGULAR prompt batch: X is [B*P, H] and row
// b*P+t carries prompts[b][t], the row layout the rect-prefill lanes index with. Each
// prompt writes into its own row window, so the per-row work is identical to the flat case.
func (m *Model) embedRectRowsInto(X []float32, prompts [][]int, P, H int, cfg Config) {
	for b, p := range prompts {
		m.embedRowsInto(X[b*P*H:], p, H, cfg)
	}
}

// logitScaleInPlace multiplies final logits by the configured logit-scale (no-op at 1).
// Cohere uses 0.0625.
func logitScaleInPlace(logits []float32, cfg Config) {
	s := float32(1)
	if cfg.LogitScale != 0 && cfg.LogitScale != 1 {
		s = float32(cfg.LogitScale)
	}
	if s != 1 {
		for i := range logits {
			logits[i] *= s
		}
	}
	softcapInPlace(logits, float32(cfg.LogitSoftcap))
	// Forced token suppression (Gemma 4 masks its image/audio placeholder tokens). Applied
	// AFTER the soft-cap, mirroring the reference's post-lm_head logits bias. No-op when empty.
	if len(cfg.SuppressTokens) > 0 {
		negInf := float32(math.Inf(-1))
		for _, id := range cfg.SuppressTokens {
			if id >= 0 && id < len(logits) {
				logits[id] = negInf
			}
		}
	}
}

// ---- activation -------------------------------------------------------------------
//
// act applies the configured MLP gate activation. SiLU (default) for Llama; tanh-approx
// GELU for Gemma's GeGLU.
func act(z float32, cfg Config) float32 {
	if cfg.ActGeluTanh {
		return geluTanh(z)
	}
	if cfg.ActGeluErf {
		return geluErf(z)
	}
	return silu(z)
}

// geluTanh is the tanh-approximation GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
func geluTanh(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	x64 := float64(x)
	inner := c * (x64 + 0.044715*x64*x64*x64)
	return float32(0.5 * x64 * (1 + math.Tanh(inner)))
}

// geluErf is exact GELU: 0.5*x*(1+erf(x/sqrt(2))).
func geluErf(x float32) float32 {
	x64 := float64(x)
	return float32(0.5 * x64 * (1 + math.Erf(x64/math.Sqrt2)))
}

// ---- norm gain --------------------------------------------------------------------
//
// rmsnormCfg is the configured decoder/final norm without bias. The default is RMSNorm;
// NormGain1p makes RMSNorm read (1+w) for Gemma; LayerNorm switches to
// mean-subtracting LayerNorm. The Llama defaults lower to rmsnorm bit-for-bit.
func rmsnormCfg(x, w []float32, eps float32, cfg Config) []float32 {
	return normCfg(x, w, nil, eps, cfg)
}

func normCfg(x, w, bias []float32, eps float32, cfg Config) []float32 {
	if cfg.LayerNorm {
		return layernorm(x, w, bias, eps)
	}
	if !cfg.NormGain1p {
		return rmsnorm(x, w, eps)
	}
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * (1 + w[i])
	}
	return out
}

func layernorm(x, w, bias []float32, eps float32) []float32 {
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
		if bias != nil {
			out[i] += bias[i]
		}
	}
	return out
}

// arch.go — the block-topology axis (MODEL-ARCH-SEAM §2b class 3, Stage 4).
//
// Today the decoder block is PRE-norm (Llama: norm->sublayer->residual-add, for
// both the attention and the MLP sub-layer). Other families place the norm
// differently around each sub-layer, or change the residual *dataflow graph*
// entirely:
//
//   - PreNorm     (default; Llama/Qwen/Mistral): x = x + body(norm(x))
//   - PostNorm    (OLMo2):                        x = x + norm(body(x))
//   - SandwichNorm (Gemma2-style):                x = x + post(body(pre(x)))
//   - ParallelResidual (GPT-NeoX/Cohere):         x = x + attn(norm(x)) + mlp(norm(x))
//
// The first three keep the sequential graph (attn updates x, then mlp reads the
// updated x) and only move WHERE the norm sits relative to the sub-layer body.
// ParallelResidual is the one genuinely different graph: attention and the MLP
// BOTH read the SAME pre-block input through one shared norm, and both deltas are
// summed into one residual — the topology MODEL-ARCH-SEAM §2b flags as the only
// one not expressible in the common pre/post/sandwich template, and the one that
// re-touches the residual-add order R2 certifies.
//
// The axis is additive dispatch: the PreNorm value of every branch lowers to
// EXACTLY today's instruction stream, so the Llama path stays max|Δ|=0
// (TestBlockTopologyPreNormNoOp). Non-Llama topologies are a structural rewiring
// of the existing norm weights and sub-layer bodies — the *wiring* is proven here
// on a synthetic config (TestBlockTopology*); per-family NUMERIC correctness still
// needs a re-exported HF oracle per MODEL-ARCH-SEAM §7.1 (follow-on).
//
// Norm-placement here is purely block-level (where the norm sits in the residual
// graph). It is orthogonal to the norm FUNCTION (RMSNorm vs LayerNorm vs Gemma's
// (1+w) gain) and to qk-norm — those are separate mechanical axes (§2b class 1).

// BlockTopology selects how the per-sub-layer norm and the residual add are wired
// around the attention and MLP bodies. The zero value is PreNorm (Llama), so a
// Config that never sets the field — every existing call site and every existing
// oracle/synthetic export — is byte-for-byte the current path.
type BlockTopology int

const (
	// PreNorm is the Llama/Qwen/Mistral default: normalize the residual stream,
	// run the sub-layer on the normalized input, add the raw sub-layer output back
	// to the (un-normalized) residual. x = x + body(norm(x)). Zero value.
	PreNorm BlockTopology = iota
	// PostNorm is the OLMo2 placement: run the sub-layer on the RAW residual
	// stream, normalize the sub-layer output, then add. x = x + norm(body(x)).
	PostNorm
	// SandwichNorm normalizes both BEFORE and AFTER the sub-layer body (Gemma2's
	// pre/post-attention and pre/post-feedforward norms): x = x + post(body(pre(x))).
	SandwichNorm
	// ParallelResidual is the GPT-NeoX / Cohere graph: attention and the MLP both
	// read the SAME shared-normed block input and both deltas land in ONE residual.
	// x = x + attn(norm(x)) + mlp(norm(x)). The one topology whose dataflow graph
	// differs from the sequential pre/post/sandwich template (MODEL-ARCH-SEAM §2b).
	ParallelResidual
)

// String renders the BlockTopology as its name (PreNorm/PostNorm/SandwichNorm/
// ParallelResidual), falling back to "BlockTopology(n)" for an unknown value.
func (t BlockTopology) String() string {
	switch t {
	case PreNorm:
		return "PreNorm"
	case PostNorm:
		return "PostNorm"
	case SandwichNorm:
		return "SandwichNorm"
	case ParallelResidual:
		return "ParallelResidual"
	default:
		return "BlockTopology(" + itoa(int(t)) + ")"
	}
}

// IsParallel reports whether this topology runs attention and the MLP off the same
// shared-normed block input (a single shared norm, two deltas into one residual)
// rather than threading the MLP off the attention-updated residual.
func (t BlockTopology) IsParallel() bool { return t == ParallelResidual }

// sublayer is one residual sub-layer's compute: given an already-normalized input
// vector, it returns the raw sub-layer output (the attention output projection, or
// the SwiGLU MLP output) BEFORE any post-norm or residual add. Splitting the block
// into two such bodies lets the topology own only the norm placement and residual
// wiring, while the arithmetic inside each body is identical across topologies —
// which is what keeps the PreNorm lowering byte-for-byte the current path.
type sublayer func(normedIn []float32) []float32

type normWeights struct {
	pre      []float32
	preBias  []float32
	post     []float32
	postBias []float32
}

// composeBlockAtLayer forms one complete transformer block residual and then invokes
// the optional activation-space hook. Keeping the hook after the complete block makes
// its contract identical for sequential and parallel residual topologies.
func composeBlockAtLayer(layer int, t BlockTopology, x []float32, attnNorm, mlpNorm normWeights, eps float32, cfg Config, attn, mlp sublayer) {
	if t == ParallelResidual {
		// Attention and the MLP both read the ORIGINAL block input. GPT-NeoX uses
		// distinct input/post-attention LayerNorms for the two branches; Cohere lacks
		// a separate MLP norm and passes the same weights in both slots.
		xnAttn := normCfg(x, attnNorm.pre, attnNorm.preBias, eps, cfg)
		xnMLP := normCfg(x, mlpNorm.pre, mlpNorm.preBias, eps, cfg)
		o := attn(xnAttn)
		d := mlp(xnMLP)
		for i := range x {
			x[i] += o[i] + d[i]
		}
		invokeResidualHook(cfg, layer, x)
		return
	}

	// Sequential: attention updates x, then the MLP reads the updated x.
	composeSublayer(t, x, attnNorm, eps, cfg, attn)
	composeSublayer(t, x, mlpNorm, eps, cfg, mlp)
	invokeResidualHook(cfg, layer, x)
}

// composeSublayer applies ONE residual sub-layer (norm placement + body + add) to x
// in place under topology t, using pre/post norm weights. PreNorm:
// x += body(preNorm(x)). PostNorm: x += postNorm(body(x)). SandwichNorm:
// x += postNorm(body(preNorm(x))).
func composeSublayer(t BlockTopology, x []float32, n normWeights, eps float32, cfg Config, body sublayer) {
	switch t {
	case PostNorm:
		out := body(x)                                     // sub-layer reads the RAW residual stream
		nout := normCfg(out, n.post, n.postBias, eps, cfg) // ...then its output is normalized
		for i := range x {
			x[i] += nout[i]
		}
	case SandwichNorm:
		xn := normCfg(x, n.pre, n.preBias, eps, cfg) // pre-norm
		out := body(xn)
		nout := normCfg(out, n.post, n.postBias, eps, cfg) // post-norm (around the body)
		for i := range x {
			x[i] += nout[i]
		}
	default: // PreNorm — the verbatim Llama path
		xn := normCfg(x, n.pre, n.preBias, eps, cfg)
		out := body(xn)
		for i := range x {
			x[i] += out[i]
		}
	}
}

// ---- forward-requested cache-class registration ------------------------------------
//
// A hybrid checkpoint's forward pass requests more state than its weight tensors:
// per-layer token-indexed K/V (possibly window-bounded), compressed KV latents on a
// declared layer subset, a sparse-attention indexer's own KV in its own dtype, and an
// accumulated compressor/recurrent state. The serving engines enumerate exactly these
// classes up front — vLLM's DeepSeek-V4.1-Flash recipe pins the attention backend,
// the indexer KV dtype, the compressed-latent ratios and layers, and the per-layer
// sliding window in one recipe document, and TRT-LLM's DeepseekV4CacheManager maps
// the same classes per layer — so that loader coverage, memory admission, and
// device-op planning all DERIVE from that one listing instead of re-enumerating it
// ad hoc at each site.
//
// This section lands the fak arch-metadata half of that pattern: a CacheClassManifest
// that registers every forward-requested tensor/cache class once, plus a fail-closed
// check (CheckForwardRequests) that refuses a declared-but-unregistered class with
// ErrUnregisteredCacheClass BEFORE load. It is schema + gate only: no loader rewrite,
// no kernel, no hardware claim. Consumers (the offline tensor-coverage preflight and
// the admission/device-op planners) derive their class lists from the manifest in
// follow-ons; until then the manifest is the single source of truth for the listing.
//
// Provenance (ADAPT): the enumeration shape is adapted from the vLLM project's
// DeepSeek-V4.1-Flash H200 recipe (Apache-2.0),
// recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml at pin
// e740badc2e3df13a5c2abfca04ed0ac917985c5f — see THIRD_PARTY_NOTICES.md. No upstream
// code is copied; the recipe facts are re-expressed as Go registration data.

// CacheClassKind classifies what one forward-requested cache entry holds. It is the
// vocabulary the manifest budgets and checks against; it is orthogonal to which
// kernel serves the layer (two layers of one kind may run different kernels).
type CacheClassKind uint8

const (
	// CacheClassTokenKV is a token-indexed K/V span: full causal history when Window
	// is 0, or a bounded sliding window when Window > 0 (a position older than the
	// window is provably un-attended, so the resident footprint caps at the window).
	CacheClassTokenKV CacheClassKind = iota
	// CacheClassCompressedLatent is a compressed KV latent written at a declared
	// compression ratio — fewer latent rows than tokens, held in its own dtype.
	CacheClassCompressedLatent
	// CacheClassIndexerKV is a sparse-attention indexer's own KV, kept in the
	// indexer's dtype (distinct from the main KV) so the top-k selection can score
	// positions without touching the main cache.
	CacheClassIndexerKV
	// CacheClassRecurrentState is an ACCUMULATED O(1)-per-sequence state (compressor
	// or recurrent core); its footprint is constant in context length.
	CacheClassRecurrentState
)

// String renders the kind for witnesses and operator readouts.
func (k CacheClassKind) String() string {
	switch k {
	case CacheClassTokenKV:
		return "token-kv"
	case CacheClassCompressedLatent:
		return "compressed-latent"
	case CacheClassIndexerKV:
		return "indexer-kv"
	case CacheClassRecurrentState:
		return "recurrent-state"
	default:
		return fmt.Sprintf("cache-class-kind(%d)", int(k))
	}
}

// CacheClassSpec is one registered forward-requested tensor/cache class. Fields are
// closed metadata: empty/zero values mean "unspecified" (DType), "full history"
// (Window), or "every decoder layer" (Layers), never "unknown at runtime".
type CacheClassSpec struct {
	// Name is the stable class identity a forward request names (e.g. "kv", "indexer_kv").
	Name string
	// Kind classifies the state the class holds.
	Kind CacheClassKind
	// DType is the element dtype the class is stored in ("mxfp4", "fp4", …). Empty =
	// the recipe did not pin one.
	DType string
	// Window is the sliding-window size in tokens for CacheClassTokenKV classes; 0 =
	// full history. Ignored by the other kinds.
	Window int
	// Ratios are the declared compression ratios for CacheClassCompressedLatent
	// classes (e.g. {1, 2}). Empty for the other kinds.
	Ratios []int
	// Layers lists the decoder layers this class applies to; empty = every layer.
	Layers []int
}

// ErrUnregisteredCacheClass is the sentinel returned before load when a forward
// request names a cache class the arch manifest does not register. Callers match it
// with errors.Is; the wrapped error carries the offending class name.
var ErrUnregisteredCacheClass = errors.New("model: forward-requested cache class is not registered in the arch manifest")

// ErrDuplicateCacheClass is returned when a second spec tries to register a name a
// manifest already owns. Registration is one-class-per-name so a silent overwrite
// could not quietly change what a later admission check derives.
var ErrDuplicateCacheClass = errors.New("model: cache class is already registered")

// CacheClassManifest is the ordered registration table of a model's forward-requested
// tensor/cache classes. Loader coverage, admission fit, and device-op planning derive
// their class listings from it instead of keeping ad-hoc copies; CheckForwardRequests
// is the fail-closed gate that turns an unregistered declaration into a typed refusal
// before any weight I/O.
type CacheClassManifest struct {
	specs map[string]CacheClassSpec
	order []string // registration order, preserved for deterministic readouts
}

// NewCacheClassManifest builds a manifest from specs in the given order. It fails
// closed on an empty name, an out-of-range layer index, a non-positive compression
// ratio, or a duplicate name — a malformed listing must not become the thing every
// downstream check derives from.
func NewCacheClassManifest(specs ...CacheClassSpec) (*CacheClassManifest, error) {
	m := &CacheClassManifest{specs: make(map[string]CacheClassSpec, len(specs))}
	for _, s := range specs {
		if err := m.Register(s); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// Register adds one class to the manifest. The spec is validated by validateCacheClass
// and stored by value (the caller's slice headers are not retained beyond the copy).
func (m *CacheClassManifest) Register(spec CacheClassSpec) error {
	if err := validateCacheClass(spec); err != nil {
		return err
	}
	if _, exists := m.specs[spec.Name]; exists {
		return fmt.Errorf("%w: %q", ErrDuplicateCacheClass, spec.Name)
	}
	m.specs[spec.Name] = spec
	m.order = append(m.order, spec.Name)
	return nil
}

// Has reports whether name is a registered class.
func (m *CacheClassManifest) Has(name string) bool {
	_, ok := m.specs[name]
	return ok
}

// Spec returns the registered spec for name.
func (m *CacheClassManifest) Spec(name string) (CacheClassSpec, bool) {
	s, ok := m.specs[name]
	return s, ok
}

// Names returns the registered class names in registration order.
func (m *CacheClassManifest) Names() []string {
	out := make([]string, len(m.order))
	copy(out, m.order)
	return out
}

// LayersFor expands a class's layer scope against a concrete layer count: an empty
// Layers list means every layer [0, numLayers); an explicit list is returned as-is
// after each index is checked against numLayers. Out-of-range indices fail closed —
// a recipe that names layer 20 for a 16-layer checkpoint is a bad listing, not a
// silently-truncated one.
func (m *CacheClassManifest) LayersFor(name string, numLayers int) ([]int, error) {
	s, ok := m.specs[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnregisteredCacheClass, name)
	}
	if numLayers <= 0 {
		return nil, fmt.Errorf("model: cache class %q needs a positive layer count, got %d", name, numLayers)
	}
	if len(s.Layers) == 0 {
		all := make([]int, numLayers)
		for i := range all {
			all[i] = i
		}
		return all, nil
	}
	for _, l := range s.Layers {
		if l < 0 || l >= numLayers {
			return nil, fmt.Errorf("model: cache class %q declares layer %d outside the 0..%d range", name, l, numLayers-1)
		}
	}
	out := make([]int, len(s.Layers))
	copy(out, s.Layers)
	return out, nil
}

// CheckForwardRequests is the pre-load gate: every class name a forward request
// declares must be registered, else the first offender is refused with
// ErrUnregisteredCacheClass (wrapped, carrying the name). An empty request list is
// vacuously satisfied. This is the seam loader-coverage, admission, and device-op
// checks call so the refusal happens BEFORE weight I/O, not mid-load.
func (m *CacheClassManifest) CheckForwardRequests(requested []string) error {
	for _, name := range requested {
		if !m.Has(name) {
			return fmt.Errorf("%w: %q", ErrUnregisteredCacheClass, name)
		}
	}
	return nil
}

// validateCacheClass enforces the closed spec contract: a non-empty name, a known
// kind, non-negative window, positive compression ratios, and non-negative layer
// indices. (Layer counts are checked against the concrete model in LayersFor; the
// spec alone cannot know them.)
func validateCacheClass(s CacheClassSpec) error {
	if s.Name == "" {
		return errors.New("model: cache class spec needs a non-empty name")
	}
	if s.Kind > CacheClassRecurrentState {
		return fmt.Errorf("model: cache class %q has unknown kind %s", s.Name, s.Kind.String())
	}
	if s.Window < 0 {
		return fmt.Errorf("model: cache class %q has negative window %d", s.Name, s.Window)
	}
	for _, r := range s.Ratios {
		if r <= 0 {
			return fmt.Errorf("model: cache class %q has non-positive compression ratio %d", s.Name, r)
		}
	}
	for _, l := range s.Layers {
		if l < 0 {
			return fmt.Errorf("model: cache class %q has negative layer index %d", s.Name, l)
		}
	}
	return nil
}

// DeepSeekV41FlashCacheClasses registers the four forward-requested cache classes of
// the DeepSeek-V4.1-Flash H200 recipe in one pass (PF-06 one-pass forward-request
// listing): the per-layer 128-token sliding-window KV, the FP4 compressed latents at
// ratios 1 and 2 on layers 2/8/14/20, the mxfp4 sparse-indexer KV, and the
// compressor recurrent state. numLayers must exceed the highest declared layer;
// a smaller checkpoint is refused rather than silently losing a class.
func DeepSeekV41FlashCacheClasses(numLayers int) (*CacheClassManifest, error) {
	if numLayers <= 0 {
		return nil, errors.New("model: DeepSeek-V4.1-Flash cache manifest needs a positive layer count")
	}
	compressedLayers := []int{2, 8, 14, 20}
	for _, l := range compressedLayers {
		if l >= numLayers {
			return nil, fmt.Errorf("model: DeepSeek-V4.1-Flash compressed latents declare layer %d but the checkpoint has %d layers", l, numLayers)
		}
	}
	manifest, err := NewCacheClassManifest(
		CacheClassSpec{Name: "kv", Kind: CacheClassTokenKV, Window: 128},
		CacheClassSpec{Name: "compressed_latent", Kind: CacheClassCompressedLatent, DType: "fp4", Ratios: []int{1, 2}, Layers: compressedLayers},
		CacheClassSpec{Name: "indexer_kv", Kind: CacheClassIndexerKV, DType: "mxfp4"},
		CacheClassSpec{Name: "compressor_state", Kind: CacheClassRecurrentState},
	)
	if err != nil {
		return nil, err
	}
	return manifest, nil
}
