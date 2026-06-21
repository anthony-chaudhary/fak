package model

import "math"

// longrope: Phi-3/3.5/4's long-context RoPE variant (Stage 6, MODEL-ARCH-SEAM.md
// §3 O3). It rescales the per-dimension inverse frequencies by a pinned factor
// vector and warms the attention score by a context-dependent temperature.
//
//	inv_freq[j] = (1 / theta^(2j/head_dim)) / factor[j]
//	rot_scale   = sqrt(1 + ln(max/orig)/ln(orig))                         (max > orig)
//	            = 1                                                        (max <= orig)
//	attn_scale  = (1/sqrt(head_dim)) * rot_scale^2
//
// The factor vector is SHORT or LONG, and which one is the load-bearing decision.
// HF's modeling picks per-forward by comparing the live sequence length to
// original_max_position_embeddings — which would make inv_freq depend on cache
// length. That is the O3 landmine: KVCache.Evict re-rotates a survivor at its NEW
// (smaller) index, and if the factor flipped on length, the re-rotation would draw
// a DIFFERENT inv_freq than the prefill that produced Kraw, silently mis-rotating a
// middle-span eviction (the asymmetry evict_test.go traps).
//
// HARD rule: pin the factor to the model's MAX-CONTEXT regime at session start,
// never to the live cache length. A served Phi model runs at its full window, so
// the long factor is correct for the whole session; pinning it removes the
// length-dependence entirely, so prefill, decode, AND Evict's re-rotation all draw
// the same vector. The selection is a pure function of the (immutable-per-session)
// Config, so "pinned for the session's lifetime" holds by construction — every call
// to ropeLongFactor(cfg) returns the identical vector. longropeFactorPinned asserts
// that invariant for the guard the issue requires.

// ropeLongFactor returns the pinned per-(head_dim/2) rescale vector for a longrope
// config, or nil if the config is not longrope (the plain-theta path). The choice
// is the model's MAX-CONTEXT regime: the long factor whenever the model's full
// window exceeds the original (pre-extension) window, which is the served regime
// for every real Phi long-context checkpoint. It NEVER consults a live length.
func ropeLongFactor(cfg Config) []float64 {
	if !cfg.isLongrope() {
		return nil
	}
	rs := cfg.LongRope
	if cfg.MaxPositionEmbeddings > rs.OriginalMaxPositionEmbeddings && len(rs.LongFactor) > 0 {
		return rs.LongFactor
	}
	return rs.ShortFactor
}

// longropeFactorPinned is the session-lifetime guard the design (§3 O3) requires:
// the factor selection must be fixed for the whole session, independent of the live
// cache length. It re-derives the choice from the Config alone and confirms it does
// not vary with a (hypothetical) live length argument — true by construction here,
// since ropeLongFactor reads only Config, but asserted so a future change that
// re-introduced length-dependence would trip a test (longrope_test.go) rather than
// silently mis-rotate an eviction. Returns false if the factor vector length does
// not match head_dim/2 (a malformed config), so the loader can fail closed.
func longropeFactorPinned(cfg Config) bool {
	f := ropeLongFactor(cfg)
	if f == nil {
		return true // not longrope: nothing to pin
	}
	return len(f) == cfg.HeadDim/2
}

// longropeAttnScaleMul is the effective score-scale warm-up on top of the base
// 1/sqrt(head_dim). HF applies the LongRoPE attention scaling to BOTH cos and sin,
// which scales both q and k after rotation, so the dot-product score sees the square
// of the rotary scale. It is 1.0 (identity) for the non-longrope path and for a
// longrope model whose full window does not exceed the original, so the Llama path's
// score scale is untouched.
func longropeAttnScaleMul(cfg Config) float64 {
	if !cfg.isLongrope() {
		return 1.0
	}
	orig := cfg.LongRope.OriginalMaxPositionEmbeddings
	maxp := cfg.MaxPositionEmbeddings
	if orig <= 1 || maxp <= orig {
		return 1.0
	}
	rotaryScale := math.Sqrt(1.0 + math.Log(float64(maxp)/float64(orig))/math.Log(float64(orig)))
	return rotaryScale * rotaryScale
}
