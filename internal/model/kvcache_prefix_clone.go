package model

// ClonePrefixExact returns the same independently owned state as Clone followed
// by TryEvict of the tail [positions, Len). Unsupported recurrent truncation
// returns nil. Out-of-range requests retain that existing Clone/TryEvict behavior.
//
// Ordinary dense canonical-position caches copy only the retained rows. The
// temporary view owns slice headers only and is consumed synchronously by Clone;
// no source payload is mutated or retained by the returned cache. Specialized,
// irregular or partially written caches keep the original lifecycle path.
func (c *KVCache) ClonePrefixExact(positions int) *KVCache {
	if cp, ok := c.TryCloneDensePrefixExact(positions); ok {
		return cp
	}
	cp := c.Clone()
	if positions < cp.Len() {
		if _, err := cp.TryEvict(positions, cp.Len()-positions); err != nil {
			return nil
		}
	}
	return cp
}

// TryCloneDensePrefixExact copies only retained payload when the strict dense
// prefix contract applies. A false result neither clones nor mutates anything;
// callers keep their existing specialized fallback and clone-cost accounting.
// The returned cache owns every retained payload and matches Clone+tail eviction.
func (c *KVCache) TryCloneDensePrefixExact(positions int) (*KVCache, bool) {
	if !c.canCloneDensePrefix(positions) {
		return nil, false
	}

	view := *c
	n := positions * c.kvStride() // admitted geometry bounds this by each live plane
	view.pos = c.pos[:positions:positions]
	view.lineage.ids = c.lineage.ids[:positions:positions]
	view.Kraw = make([][]float32, len(c.Kraw))
	for l := range view.Kraw {
		view.Kraw[l] = c.Kraw[l][:n:n]
	}
	if c.quantized() {
		view.kQ8 = make([]kvPackedRow, len(c.kQ8))
		view.vQ8 = make([]kvPackedRow, len(c.vQ8))
		for l := range view.kQ8 {
			view.kQ8[l], view.vQ8[l] = c.kQ8[l], c.vQ8[l]
			view.kQ8[l].codes = c.kQ8[l].codes[:n:n]
			view.vQ8[l].codes = c.vQ8[l].codes[:n:n]
			groups := n / kvQ8_0GroupSize
			view.kQ8[l].scales = c.kQ8[l].scales[:groups:groups]
			view.vQ8[l].scales = c.vQ8[l].scales[:groups:groups]
		}
	} else {
		view.K = make([][]float32, len(c.K))
		view.V = make([][]float32, len(c.V))
		for l := range view.K {
			view.K[l], view.V[l] = c.K[l][:n:n], c.V[l][:n:n]
		}
	}
	cp := view.Clone()
	// Clone starts a fresh mutation generation; successful tail eviction would
	// then bump it once even when no survivor needs positional re-RoPE.
	cp.bumpMutation()
	return cp, true
}

func (c *KVCache) canCloneDensePrefix(positions int) bool {
	if c == nil || positions < 0 || positions >= c.Len() || c.linear != nil ||
		c.glm != nil || c.msa != nil || c.cfg.usesMLAMoELayout() ||
		(c.prec != "" && c.prec != KVPrecisionFP32 && c.prec != KVPrecisionQ8_0) {
		return false
	}
	layers, rows, width := c.cfg.NumLayers, c.Len(), c.kvStride()
	if c.cfg.NumKVHeads <= 0 || c.cfg.HeadDim <= 0 || width <= 0 ||
		width/c.cfg.NumKVHeads != c.cfg.HeadDim || len(c.lineage.ids) != rows || len(c.K) != layers ||
		len(c.Kraw) != layers || len(c.V) != layers {
		return false
	}
	// Noncanonical surviving positions require the old re-RoPE path. Discarded
	// tail positions do not affect any survivor and need not be inspected.
	for i, p := range c.pos[:positions] {
		if i != p {
			return false
		}
	}
	fullPlane := func(n int) bool { return n/width == rows && n%width == 0 }
	if c.quantized() && (width%kvQ8_0GroupSize != 0 || len(c.kQ8) != layers || len(c.vQ8) != layers) {
		return false
	}
	for l := range c.Kraw {
		if !fullPlane(len(c.Kraw[l])) {
			return false
		}
		if !c.quantized() {
			if !fullPlane(len(c.K[l])) || !fullPlane(len(c.V[l])) {
				return false
			}
			continue
		}
		k, v := c.kQ8[l], c.vQ8[l]
		if len(c.K[l]) != 0 || len(c.V[l]) != 0 || k.width != width || v.width != width ||
			!fullPlane(len(k.codes)) || !fullPlane(len(v.codes)) ||
			len(k.scales) != len(k.codes)/kvQ8_0GroupSize || len(v.scales) != len(v.codes)/kvQ8_0GroupSize {
			return false
		}
	}
	return true
}
