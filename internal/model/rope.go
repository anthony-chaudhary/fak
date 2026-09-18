package model

import (
	"math"
	"sync"
)

// invFreq is the single RoPE inverse-frequency builder. The layer parameter selects
// layer-specific RoPE bases for families such as Gemma3; Llama/Qwen leave the
// per-layer override empty and use one shared table. After the bare Llama inv_freq is
// built, Phi longrope may divide each dimension by its pinned factor and Llama3
// rope_scaling may rescale the frequencies. Defaults are identity, so the Llama path
// is bit-identical.
func invFreq(cfg Config, layer int) []float64 {
	rotaryDim := cfg.rotaryDim()
	half := rotaryDim / 2
	factor := ropeLongFactor(cfg) // nil unless longrope
	theta := cfg.ropeThetaForLayer(layer)
	// HF: inv_freq[j] = 1 / theta^((2j)/dim) where dim is the rotated width. For full
	// rotary dim==head_dim; for Qwen3.5/Qwen3-Next partial rotary the HF default-rope init
	// uses dim = int(head_dim*partial_rotary_factor) (the rotary_dim), not head_dim.
	denom := cfg.invFreqDenom()
	inv := make([]float64, half)
	for j := 0; j < half; j++ {
		inv[j] = 1.0 / math.Pow(theta, float64(2*j)/float64(denom))
		if factor != nil && j < len(factor) && factor[j] != 0 {
			inv[j] /= factor[j]
		}
	}
	applyRopeScaling(cfg, inv)
	return inv
}

// invFreqDenom is the denominator dim of the RoPE inv_freq init: inv[j] = 1/theta^(2j/denom).
// Full rotary uses head_dim; partial-rotary (Qwen3.5/MiniMax) uses the rotary_dim; GLM-DSA /
// DeepSeek-MLA uses qk_rope_head_dim (NOT cfg.HeadDim, which the GGUF sets from the larger MLA
// latent attention.key_length). Centralized here so invFreq AND the cache key agree — a config
// whose denom differs must never collide on a cached inv-freq table.
func (c Config) invFreqDenom() int {
	// NOT redundant with the fall-through below: this branch must win over the MLA one,
	// so a config that is both a Qwen3.5 hybrid and an MLA/MoE layout keeps the rotary_dim.
	if c.IsQwen35Hybrid() || c.isMiniMax() {
		return c.rotaryDim()
	}
	if c.usesMLAMoELayout() && c.QKRopeHeadDim > 0 {
		return c.QKRopeHeadDim
	}
	// rotaryDim, not HeadDim: partial rotary is NOT a Qwen3.5/MiniMax-only axis. GPT-NeoX
	// (rotary_pct) and StableLM (partial_rotary_factor) rotate a prefix of each head too,
	// and HF inits their inv_freq over that prefix. Denominating by the full head_dim made
	// every frequency too high, so the rotation drifted with position -- silently, because
	// rotaryDim() already returns HeadDim whenever the factor is absent, <=0, or >=1, which
	// is every full-rotary family. So this is exactly identity for them and a fix for the
	// partial-rotary ones.
	return c.rotaryDim()
}

func (c Config) rotaryDim() int {
	if c.PartialRotaryFactor <= 0 || c.PartialRotaryFactor >= 1 {
		return c.HeadDim
	}
	dim := int(float64(c.HeadDim) * c.PartialRotaryFactor)
	if dim < 2 {
		dim = 2
	}
	if dim > c.HeadDim {
		dim = c.HeadDim
	}
	if dim%2 != 0 {
		dim--
	}
	return dim
}

type ropeInvKey struct {
	headDim   int
	rotaryDim int
	denom     int // the inv_freq denominator (head_dim / rotary_dim / qk_rope_head_dim) — see invFreqDenom
	thetaBits uint64
	layer     int
	// scaling axis: distinct scaling params must not collide in the cache. For Llama
	// (scaling=="") these are zero/empty, so the key reduces to the original triple.
	scaling                       string
	factorBits, lowBits, highBits uint64
	origCtx                       int
	betaFastBits, betaSlowBits    uint64
	truncate                      bool
	truncateSet                   bool
	longropeFactorHash            uint64
}

var ropeInvCache sync.Map // map[ropeInvKey][]float64

func cachedInvFreq(cfg Config, layer int) []float64 {
	rp := cfg.effectiveRopeParameters()
	factor := cfg.RopeFactor
	if factor == 0 {
		factor = rp.Factor
	}
	origCtx := cfg.RopeOrigContext
	if origCtx == 0 {
		origCtx = rp.OriginalMaxPositionEmbeddings
	}
	truncate := false
	truncateSet := false
	if rp.Truncate != nil {
		truncate = *rp.Truncate
		truncateSet = true
	}
	key := ropeInvKey{
		headDim:            cfg.HeadDim,
		rotaryDim:          cfg.rotaryDim(),
		denom:              cfg.invFreqDenom(),
		thetaBits:          math.Float64bits(cfg.ropeThetaForLayer(layer)),
		layer:              layer,
		scaling:            cfg.RopeScaling,
		factorBits:         math.Float64bits(factor),
		lowBits:            math.Float64bits(cfg.RopeLowFreqFactor),
		highBits:           math.Float64bits(cfg.RopeHighFreqFactor),
		origCtx:            origCtx,
		betaFastBits:       math.Float64bits(rp.BetaFast),
		betaSlowBits:       math.Float64bits(rp.BetaSlow),
		truncate:           truncate,
		truncateSet:        truncateSet,
		longropeFactorHash: ropeFactorHash(cfg),
	}
	if inv, ok := ropeInvCache.Load(key); ok {
		return inv.([]float64)
	}
	inv := invFreq(cfg, layer)
	actual, _ := ropeInvCache.LoadOrStore(key, inv)
	return actual.([]float64)
}

func (c Config) ropeThetaForLayer(layer int) float64 {
	if layer >= 0 && layer < len(c.RopeThetaPerLayer) && c.RopeThetaPerLayer[layer] != 0 {
		return c.RopeThetaPerLayer[layer]
	}
	return c.RopeTheta
}

// ropeFactorHash folds the PINNED longrope rescale vector into the inv-freq cache
// key so two configs that share head_dim+theta but differ in their (pinned) factor
// vector never collide on a cached table. 0 for the non-longrope path, so Llama's
// cache key is unchanged. Because the factor is pinned per session, this hash is
// constant for a session's lifetime — the same property Evict's re-rotation relies
// on to draw the identical inv_freq as the prefill.
func ropeFactorHash(cfg Config) uint64 {
	f := ropeLongFactor(cfg)
	if f == nil {
		return 0
	}
	// FNV-1a over the factor bits.
	h := uint64(1469598103934665603)
	for _, v := range f {
		b := math.Float64bits(v)
		for s := 0; s < 64; s += 8 {
			h ^= (b >> uint(s)) & 0xff
			h *= 1099511628211
		}
	}
	return h
}

// ropeRow returns cos/sin (length head_dim/2) for absolute position p.
func ropeRow(cfg Config, p int) (cos, sin []float32) {
	return ropeRowForLayer(cfg, 0, p)
}

func ropeRowForLayer(cfg Config, layer, p int) (cos, sin []float32) {
	return ropeRowFromInvScaled(cachedInvFreq(cfg, layer), p, cfg.ropeAttentionFactor())
}

func ropeRowFromInv(inv []float64, p int) (cos, sin []float32) {
	return ropeRowFromInvScaled(inv, p, 1)
}

func ropeRowFromInvScaled(inv []float64, p int, scale float64) (cos, sin []float32) {
	cos = make([]float32, len(inv))
	sin = make([]float32, len(inv))
	ropeRowInto(cos, sin, inv, p)
	if scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range cos {
			cos[i] *= s
			sin[i] *= s
		}
	}
	return
}

func ropeRowInto(cos, sin []float32, inv []float64, p int) {
	for j := range inv {
		a := float64(p) * inv[j]
		cos[j] = float32(math.Cos(a))
		sin[j] = float32(math.Sin(a))
	}
}

func applyRopeRow(hv, cos, sin []float32) {
	half := len(cos)
	for j := 0; j < half; j++ {
		a, b := hv[j], hv[j+half]
		// The explicit float32() conversions pin each product to f32 precision, blocking
		// the Go compiler's OPTIONAL fusion of a*c±b*s into an FMA. Without them the same
		// rotation fuses at one inlined call site (prefill) but not another (Evict
		// reposition) on arm64, so a repositioned K drifts 1 ULP from the prefill K and the
		// bit-exact KV rungs fail on Apple Silicon while passing on amd64 (which never
		// fuses). Pinning to the unfused value makes the rotation deterministic on every
		// architecture and call site — the "bit-identical kernel" guarantee the cache,
		// Clone, and Evict reposition all depend on.
		hv[j] = float32(a*cos[j]) - float32(b*sin[j])
		hv[j+half] = float32(b*cos[j]) + float32(a*sin[j])
	}
}

// applyRopeTailInterleaved rotates ONLY the last ropeDim components of hv, using the
// INTERLEAVED (adjacent-pair / GPT-J / complex) convention: the pair (hv[tail+2j],
// hv[tail+2j+1]) is treated as a complex number and multiplied by e^{i*angle_j}. The
// leading len(hv)-ropeDim components are left byte-identical. cos/sin must carry
// ropeDim/2 frequencies built over the rope width (see v41RopeTableForLayer), NOT over
// the full head width.
//
// This is the reference DeepSeek-V4.1-Flash contract:
//
//	freqs_cis = precompute_freqs_cis(self.rope_head_dim, ...)   # rope_head_dim freqs
//	apply_rotary_emb(x):  x[..., -rope_head_dim:] * complex(freqs_cis)
//
// with torch.view_as_complex(x.unflatten(-1, (-1, 2))) then view_as_real(...).flatten(-2)
// — i.e. the adjacent pair is the complex pair in BOTH directions. It deliberately does
// NOT reuse applyRopeRow, whose half-split whole-row pairing is the wrong convention for
// this path (and is load-bearing for the Llama/Qwen/GPT-NeoX/Cohere families and the
// DeepSeek-V2/V3 MLA de-interleave).
func applyRopeTailInterleaved(hv, cos, sin []float32, ropeDim int) {
	tail := len(hv) - ropeDim
	for j := 0; j < ropeDim/2; j++ {
		i := tail + 2*j
		a, b := hv[i], hv[i+1]
		// Explicit float32() conversions pin each product to f32 precision (no FMA
		// fusion), matching applyRopeRow's bit-determinism guarantee.
		hv[i] = float32(a*cos[j]) - float32(b*sin[j])
		hv[i+1] = float32(b*cos[j]) + float32(a*sin[j])
	}
}

// v41RopeTableForLayer builds the DeepSeek-V4.1 rotary table for one layer and
// position. Unlike ropeRowForLayer (whose table width and denominator follow
// cfg.rotaryDim()/invFreqDenom(), i.e. the full head width for a non-MLA
// layout), this table is sized to QKRopeHeadDim: len(cos)==len(sin)==QKRopeHeadDim/2
// with inv[j] = 1/theta^(2j/QKRopeHeadDim), exactly the reference
// precompute_freqs_cis(self.rope_head_dim, ...). It is the only table
// applyRopeTailInterleaved may consume. The layer-specific theta and the
// ropeAttentionFactor scaling are preserved so a scaled variant stays consistent
// with its own frequency build.
//
// QKRopeHeadDim must be positive and even; callers validate that against the head
// width and refuse with a typed error before calling. This builder assumes it.
func v41RopeTableForLayer(cfg Config, layer, p int) (cos, sin []float32) {
	n := cfg.QKRopeHeadDim / 2
	theta := cfg.ropeThetaForLayer(layer)
	scale := cfg.ropeAttentionFactor()
	cos = make([]float32, n)
	sin = make([]float32, n)
	for j := 0; j < n; j++ {
		a := float64(p) / math.Pow(theta, float64(2*j)/float64(cfg.QKRopeHeadDim))
		cv := float32(math.Cos(a))
		sv := float32(math.Sin(a))
		if scale != 0 && scale != 1 {
			s := float32(scale)
			cv *= s
			sv *= s
		}
		cos[j] = cv
		sin[j] = sv
	}
	return cos, sin
}
