package model

import "math"

// MaxRepositionResidual is the read-only witness that a span eviction's re-RoPE was
// bit-exact. After KVCache.Evict compacts and renumbers the cache, every surviving
// position i must satisfy K[l][i,h] == RoPE(Kraw[l][i,h], i) for every layer l and KV
// head h: a survivor that moved down was re-derived from its PRE-RoPE Kraw in a SINGLE
// rotation at its new position i; one that did not move was rotated at i originally, so
// the identity holds there too. This returns the per-element
//
//	max | K[l][i,h] - RoPE(Kraw[l][i,h], i) |
//
// over the whole live cache. It is 0 on amd64 (which never auto-fuses the rotation) and
// <= fmaCrossPathTol on arches that fuse a*c±b*s into an FMA; a value far above that
// bound means the rotations were COMPOSED (old->new as two rotations) rather than
// applied once — the exact bug the pre-RoPE Kraw store exists to prevent.
//
// This is the same invariant TestKVQuarantineEqualsNeverSaw (rung-3 oracle) and
// TestEvictRepositionsWithLayerSpecificRopeTheta assert in-package; exposing it lets a
// demo or driver fire the mechanical reposition witness as a first-class number from the
// SAME run as the logit-level "evicted == never-saw" witness, instead of trusting that
// the two were measured together. It mutates nothing.
//
// ALiBi layers store K un-rotated (Kraw mirrors K), so the residual is the direct
// K-vs-Kraw delta there — exact by construction. The GLM-MoE-DSA cache repositions its
// survivor keys through a different path (glmDsaKVCache.rerotateSurvivor) and does not
// expose dense per-position K here, so this dense-K witness reports 0 (not asserted) for
// that architecture; its eviction is witnessed separately by glm_test.go.
func (c *KVCache) MaxRepositionResidual() float64 {
	if c == nil || c.cfg.NumLayers == 0 || len(c.pos) == 0 {
		return 0
	}
	if c.cfg.isGLMMoeDsa() {
		return 0
	}
	w := c.kvStride()
	hd, nKV := c.cfg.HeadDim, c.cfg.NumKVHeads
	var mx float64
	for i := 0; i < len(c.pos); i++ {
		for l := 0; l < c.cfg.NumLayers; l++ {
			if c.cfg.Alibi {
				for x := i * w; x < (i+1)*w; x++ {
					if d := math.Abs(float64(c.K[l][x] - c.Kraw[l][x])); d > mx {
						mx = d
					}
				}
				continue
			}
			cos, sin := ropeRowForLayer(c.cfg, l, i)
			for h := 0; h < nKV; h++ {
				want := append([]float32(nil), c.Kraw[l][i*w+h*hd:i*w+(h+1)*hd]...)
				applyRopeRow(want, cos, sin)
				got := c.K[l][i*w+h*hd : i*w+(h+1)*hd]
				for j := range want {
					if d := math.Abs(float64(want[j] - got[j])); d > mx {
						mx = d
					}
				}
			}
		}
	}
	return mx
}
