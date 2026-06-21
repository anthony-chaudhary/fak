package model

import "math"

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
		rp, ok := cfg.defaultRopeParameters()
		if !ok {
			return
		}
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
	rp, ok := c.defaultRopeParameters()
	if !ok {
		return 1
	}
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
	cfg := m.Cfg
	if !cfg.QKNorm {
		return
	}
	p := func(s string) string { return layerName(l, s) }
	eps := cfg.qkNormEps()
	applyQKNormCfg(q, m.tensor(p("self_attn.q_norm.weight")), cfg.NumHeads, cfg.HeadDim, eps, cfg)
	applyQKNormCfg(k, m.tensor(p("self_attn.k_norm.weight")), cfg.NumKVHeads, cfg.HeadDim, eps, cfg)
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

// composeBlock applies the residual/norm composition for one decoder block to x in
// place, dispatching on the topology. attnNorm/mlpNorm carry each sub-layer's
// pre/post norm weights; PreNorm reads only the pre weights, while Gemma2/3-style
// SandwichNorm can supply distinct post-attention / post-feedforward weights. attn
// and mlp are the two sub-layer bodies (each consuming a normalized input). For the
// sequential topologies the MLP reads the attention-updated residual; for
// ParallelResidual both read one shared norm of the original input.
//
// PreNorm lowers to the exact current instruction stream:
//
//	xn  := rmsnorm(x, attnNorm.pre); o := attn(xn); x += o
//	xn2 := rmsnorm(x, mlpNorm.pre);  d := mlp(xn2); x += d
//
// so a Config left at the zero value is bit-identical to the pre-axis block.
func composeBlock(t BlockTopology, x []float32, attnNorm, mlpNorm normWeights, eps float32, cfg Config, attn, mlp sublayer) {
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
		return
	}

	// Sequential: attention updates x, then the MLP reads the updated x.
	composeSublayer(t, x, attnNorm, eps, cfg, attn)
	composeSublayer(t, x, mlpNorm, eps, cfg, mlp)
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
