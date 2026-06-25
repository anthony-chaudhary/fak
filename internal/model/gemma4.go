package model

import (
	"math"
	"os"
	"strings"
)

// gemma4 debug knobs (default = the reference behavior). These let a diagnostic load the
// 12B weights ONCE and A/B the uncertain axes (softmax scale, rope_freqs, q/k/v norm gain)
// without a multi-minute reload per hypothesis. All default off => no behavior change.
func gemma4SkipRopeFreqs() bool { return os.Getenv("FAK_GEMMA4_NO_ROPEFREQS") == "1" }
func gemma4ScaleSqrt() bool     { return os.Getenv("FAK_GEMMA4_SCALE_SQRT") == "1" }

// gemma4.go — the dedicated cacheless forward for Google's Gemma 4 (GGUF arch
// "gemma4"). Gemma 4 does not fit the uniform-geometry SwiGLU path because it
// interleaves two attention regimes with DIFFERENT shapes per layer:
//
//   - local / sliding layers:  small head_dim, several kv heads, a short RoPE base,
//     a sliding window, and a real v_proj.
//   - global / full layers:    large head_dim, a SINGLE kv head whose k_proj output
//     also serves as V (no v_proj tensor), a long RoPE base scaled by a per-frequency
//     rope_freqs vector, and full causal range.
//
// On top of the per-layer geometry, every layer: RMS-norms Q and K per head (the
// baked-(1+w) GGUF weights, plain RMSNorm), RMS-normalizes V per head with NO learned
// gain and NO RoPE, uses a softmax scale of exactly 1.0 (no 1/sqrt(d)), wraps both
// sub-layers in sandwich norm, and multiplies the whole block output by a learned
// per-layer scalar (layer_output_scale). The block math mirrors llama.cpp build_gemma4.
//
// This path is reached only when Config.isGemma4(); every other family keeps the
// shared m.layer path bit-for-bit (the slices below are empty for them).

func (c Config) isGemma4() bool {
	return strings.Contains(c.archFamilyKey(), "gemma4")
}

// headDimForLayer returns layer l's attention head_dim, honoring the per-layer slice
// (gemma4 local vs global) and falling back to the scalar HeadDim otherwise.
func (c Config) headDimForLayer(l int) int {
	if l >= 0 && l < len(c.HeadDimPerLayer) && c.HeadDimPerLayer[l] > 0 {
		return c.HeadDimPerLayer[l]
	}
	return c.HeadDim
}

// numKVHeadsForLayer returns layer l's kv-head count, honoring the per-layer slice.
func (c Config) numKVHeadsForLayer(l int) int {
	if l >= 0 && l < len(c.NumKVHeadsPerLayer) && c.NumKVHeadsPerLayer[l] > 0 {
		return c.NumKVHeadsPerLayer[l]
	}
	return c.NumKVHeads
}

// ropeDimForLayer returns the rotary width for layer l (gemma4 rotates the full head,
// so this equals headDimForLayer(l) on that path), falling back to rotaryDim().
func (c Config) ropeDimForLayer(l int) int {
	if l >= 0 && l < len(c.RopeDimPerLayer) && c.RopeDimPerLayer[l] > 0 {
		return c.RopeDimPerLayer[l]
	}
	return c.rotaryDim()
}

func (c Config) gemma4LayerIsSliding(l int) bool {
	return c.layerType(l) == "sliding_attention"
}

// gemma4RopeFreqs reads the shared global-layer RoPE frequency-factor vector (the
// proportional/NTK rope of the full-attention layers) into f64, or nil when absent.
func (m *Model) gemma4RopeFreqs() []float64 {
	name := "model.rope_freqs.weight"
	if !m.has(name) {
		return nil
	}
	t := m.tensor(name)
	out := make([]float64, len(t))
	for i, v := range t {
		out[i] = float64(v)
	}
	return out
}

// layerGemma4 runs one Gemma 4 decoder block over the whole sequence in place: sandwich
// norm around the heterogeneous attention body and the GeGLU MLP, then a learned
// per-layer output scale on the block result.
func (m *Model) layerGemma4(l int, x [][]float32, ropeFreqs []float64) {
	cfg := m.Cfg
	eps := float32(cfg.RMSNormEps)
	attnSub := func(xn [][]float32) [][]float32 { return m.gemma4AttnSeq(l, xn, ropeFreqs) }
	mlpSub := func(xn [][]float32) [][]float32 { return m.mlpSeq(l, xn) }

	composeSeqSublayer(SandwichNorm, x, m.attentionNorms(l), eps, cfg, attnSub)
	composeSeqSublayer(SandwichNorm, x, m.mlpNorms(l), eps, cfg, mlpSub)

	// Per-layer output scale: the block output is multiplied by a single learned scalar
	// (layer_output_scale.weight, a [1] tensor) before it becomes the next layer's input.
	if name := layerName(l, "layer_output_scale.weight"); m.has(name) {
		if s := m.tensor(name); len(s) > 0 {
			scale := s[0]
			for t := range x {
				row := x[t]
				for i := range row {
					row[i] *= scale
				}
			}
		}
	}
}

// gemma4AttnSeq is the cacheless Gemma 4 attention body for a whole sequence of
// already-(pre)normalized inputs. It returns the per-position o_proj results (pre
// residual); the sandwich post-norm + residual add are owned by the caller.
func (m *Model) gemma4AttnSeq(l int, xn [][]float32, ropeFreqs []float64) [][]float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	hd := cfg.headDimForLayer(l)
	nH := cfg.NumHeads
	nKV := cfg.numKVHeadsForLayer(l)
	if nKV <= 0 {
		nKV = 1
	}
	grp := nH / nKV
	ropeDim := cfg.ropeDimForLayer(l)
	if ropeDim > hd {
		ropeDim = hd
	}
	seq := len(xn)
	eps := float32(cfg.RMSNormEps)
	p := func(s string) string { return layerName(l, s) }
	mat := residentKernel{m}

	hasV := m.has(p("self_attn.v_proj.weight"))
	qNorm := m.tensorOptional(p("self_attn.q_norm.weight"))
	kNorm := m.tensorOptional(p("self_attn.k_norm.weight"))

	// Per-layer RoPE inverse frequencies: full rotary over ropeDim. Global (full-
	// attention) layers additionally divide each frequency by the shared rope_freqs
	// factor (proportional rope); local layers do not. Memoized across forwards
	// (gemma4InvFreq) since the table is a pure function of pinned per-layer config;
	// the cached bytes are identical to this recompute.
	inv := m.gemma4InvFreq(l, ropeDim, ropeFreqs)

	q := make([][]float32, seq)
	k := make([][]float32, seq)
	v := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xp := mat.prep(xn[t])
		q[t] = mat.mul(p("self_attn.q_proj.weight"), xp, nH*hd, H)
		kt := mat.mul(p("self_attn.k_proj.weight"), xp, nKV*hd, H)
		if hasV {
			v[t] = mat.mul(p("self_attn.v_proj.weight"), xp, nKV*hd, H)
		} else {
			// Global layers carry no v_proj: V is the RAW k_proj output (before k-norm
			// and RoPE), copied off before K is normalized/rotated below.
			v[t] = append([]float32(nil), kt...)
		}
		k[t] = kt

		// Q/K per-head RMSNorm (baked (1+w) GGUF weights => plain RMSNorm, NormGain1p=false).
		if len(qNorm) > 0 {
			applyQKNormCfg(q[t], qNorm, nH, hd, eps, cfg)
		}
		if len(kNorm) > 0 {
			applyQKNormCfg(k[t], kNorm, nKV, hd, eps, cfg)
		}
		// V: weightless per-head RMSNorm, NO RoPE.
		gemma4NormVHeads(v[t], nKV, hd, eps)

		// RoPE (rotate_half) on Q and K per head; V is left un-rotated.
		cos, sin := ropeRowFromInvScaled(inv, t, 1)
		for h := 0; h < nH; h++ {
			applyRopeRow(q[t][h*hd:h*hd+ropeDim], cos, sin)
		}
		for h := 0; h < nKV; h++ {
			applyRopeRow(k[t][h*hd:h*hd+ropeDim], cos, sin)
		}
	}

	// Causal GQA attention with a per-layer window and a softmax scale of exactly 1.0
	// (Gemma 4 sets self.scaling = 1.0; the QK-norm carries the temperature).
	scale := float32(1.0)
	if gemma4ScaleSqrt() {
		scale = float32(1.0 / math.Sqrt(float64(hd)))
	}
	W := cfg.windowForLayer(l)
	attnOut := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		out := make([]float32, nH*hd)
		lo := 0
		if W >= 0 {
			if lo = t - W + 1; lo < 0 {
				lo = 0
			}
		}
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[t][h*hd : (h+1)*hd]
			scores := make([]float32, t+1-lo)
			for j := lo; j <= t; j++ {
				kh := k[j][kvh*hd : (kvh+1)*hd]
				scores[j-lo] = dot(qh, kh) * scale
			}
			softmaxInPlace(scores)
			o := out[h*hd : (h+1)*hd]
			for j := lo; j <= t; j++ {
				vh := v[j][kvh*hd : (kvh+1)*hd]
				w := scores[j-lo]
				for d := 0; d < hd; d++ {
					o[d] += w * vh[d]
				}
			}
		}
		attnOut[t] = mat.mul(p("self_attn.o_proj.weight"), mat.prep(out), H, nH*hd)
	}
	return attnOut
}

// gemma4NormVHeads applies a weightless per-head RMSNorm to a packed V vector in place
// (Gemma 4 normalizes value heads with no learned gain, unlike q/k which carry weights).
func gemma4NormVHeads(v []float32, nKV, hd int, eps float32) {
	for h := 0; h < nKV; h++ {
		head := v[h*hd : (h+1)*hd]
		var ss float32
		for _, x := range head {
			ss += x * x
		}
		inv := float32(1.0 / math.Sqrt(float64(ss/float32(hd)+eps)))
		for i := range head {
			head[i] *= inv
		}
	}
}
