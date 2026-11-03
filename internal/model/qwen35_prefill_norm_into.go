package model

import "math"

// qwen35PrefillNormIntoDisjoint replaces copy(dst, rmsnormCfg(x, w, eps, cfg))
// only for the explicitly selected NormGain1p path. Its four production callers
// supply rows of fresh Xn/Xn2 panels, disjoint from live X and model weights.
// This is not an in-place or arbitrary-overlap API. x and w may alias each other.
// Short destinations receive a prefix; longer destinations keep their tail.
func qwen35PrefillNormIntoDisjoint(dst, x, w []float32, eps float32, cfg Config) {
	if !qwen35PrefillNormIntoQualification || cfg.LayerNorm || !cfg.NormGain1p {
		copy(dst, rmsnormCfg(x, w, eps, cfg))
		return
	}
	// Evaluating the original copy source fails at the first missing weight,
	// before any destination publication, even when dst is empty or short.
	if len(w) < len(x) {
		_ = w[len(w)]
	}
	// Preserve normCfg's serial float32 reduction, sqrt conversion, epsilon
	// placement and multiplication order. Do not substitute fdot or FMA here.
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i := range min(len(dst), len(x)) {
		dst[i] = x[i] * inv * (1 + w[i])
	}
}
