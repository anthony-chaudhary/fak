package model

import "fmt"

// PruneAndCompactTreeKV keeps one accepted path from a tree verification panel.
// acceptedPath contains strictly increasing, panel-relative row indices. Tree
// verification must have written each selected row with the logical position it
// will occupy on the accepted linear path; this operation deliberately does not
// apply RoPE again.
func PruneAndCompactTreeKV(cache *KVCache, prefixLen int, acceptedPath []int, treeSize int) error {
	if cache == nil {
		return fmt.Errorf("model: cannot compact a nil KV cache")
	}
	if err := validateTreeSelection(prefixLen, acceptedPath, treeSize, cache.Len()); err != nil {
		return err
	}
	if err := cache.CanEvict(); err != nil {
		return err
	}
	if cache.glm != nil {
		return fmt.Errorf("model: tree KV compaction does not support GLM-DSA side state")
	}
	if cache.msa != nil {
		return fmt.Errorf("model: tree KV compaction does not support sparse-attention side state")
	}
	if cache.cfg.NumLayers < 0 || len(cache.K) != cache.cfg.NumLayers ||
		len(cache.Kraw) != cache.cfg.NumLayers || len(cache.V) != cache.cfg.NumLayers {
		return fmt.Errorf("model: tree KV cache layer geometry is inconsistent")
	}
	w := cache.kvStride()
	if w < 0 {
		return fmt.Errorf("model: tree KV cache has negative row width")
	}
	wantFloats := cache.Len() * w
	for l := 0; l < cache.cfg.NumLayers; l++ {
		if len(cache.K[l]) != wantFloats || len(cache.Kraw[l]) != wantFloats || len(cache.V[l]) != wantFloats {
			return fmt.Errorf("model: tree KV cache layer %d row geometry is inconsistent", l)
		}
	}
	if cache.lineage.fault != "" || len(cache.lineage.ids) != cache.Len() {
		return fmt.Errorf("model: tree KV cache token lineage is inconsistent")
	}
	for i := 0; i < prefixLen; i++ {
		if cache.pos[i] != i {
			return fmt.Errorf("model: tree KV prefix row %d carries position %d", i, cache.pos[i])
		}
	}
	for j, node := range acceptedPath {
		src := prefixLen + node
		if cache.pos[src] != prefixLen+j {
			return fmt.Errorf("model: accepted tree row %d carries position %d, want %d", node, cache.pos[src], prefixLen+j)
		}
	}

	for l := 0; l < cache.cfg.NumLayers; l++ {
		for j, node := range acceptedPath {
			dst, src := prefixLen+j, prefixLen+node
			if dst != src {
				copy(cache.K[l][dst*w:(dst+1)*w], cache.K[l][src*w:(src+1)*w])
				copy(cache.Kraw[l][dst*w:(dst+1)*w], cache.Kraw[l][src*w:(src+1)*w])
				copy(cache.V[l][dst*w:(dst+1)*w], cache.V[l][src*w:(src+1)*w])
			}
		}
	}
	for j, node := range acceptedPath {
		dst, src := prefixLen+j, prefixLen+node
		if dst != src {
			cache.lineage.ids[dst] = cache.lineage.ids[src]
		}
		cache.pos[dst] = dst
	}
	newLen := prefixLen + len(acceptedPath)
	for l := 0; l < cache.cfg.NumLayers; l++ {
		cache.K[l] = cache.K[l][:newLen*w]
		cache.Kraw[l] = cache.Kraw[l][:newLen*w]
		cache.V[l] = cache.V[l][:newLen*w]
	}
	cache.pos = cache.pos[:newLen]
	cache.lineage.ids = cache.lineage.ids[:newLen]
	return nil
}

// PruneAndCompactPagedTreeKV is the paged-layout companion. It preserves COW
// forks, copying a physical page only when an accepted row overwrites a shared
// destination page, then releases every now-unused trailing page.
func PruneAndCompactPagedTreeKV(cache *PagedKV, prefixLen int, acceptedPath []int, treeSize int) error {
	if cache == nil || cache.pool == nil {
		return fmt.Errorf("model: cannot compact a nil paged KV cache")
	}
	if err := validateTreeSelection(prefixLen, acceptedPath, treeSize, cache.nTokens); err != nil {
		return err
	}
	p := cache.pool
	if p.blockTokens <= 0 || p.stride < 0 || p.nLayers < 0 || p.planes != 3 {
		return fmt.Errorf("model: paged tree KV requires valid K/V/Kraw pool geometry")
	}
	if len(p.blocks) != len(p.ref) {
		return fmt.Errorf("model: paged tree KV pool block and reference tables differ")
	}
	for i, id := range p.free {
		if id < 0 || id >= len(p.blocks) || p.ref[id] != 0 || len(p.blocks[id]) != p.blockFloats() {
			return fmt.Errorf("model: paged tree KV free-list entry %d is invalid", i)
		}
		for j := 0; j < i; j++ {
			if p.free[j] == id {
				return fmt.Errorf("model: paged tree KV free list duplicates physical block %d", id)
			}
		}
	}
	requiredBlocks := 0
	if cache.nTokens > 0 {
		requiredBlocks = (cache.nTokens + p.blockTokens - 1) / p.blockTokens
	}
	if len(cache.table) < requiredBlocks {
		return fmt.Errorf("model: paged tree KV table does not cover its token count")
	}
	for i, id := range cache.table {
		if id < 0 || id >= len(p.blocks) || id >= len(p.ref) || p.ref[id] <= 0 || len(p.blocks[id]) != p.blockFloats() {
			return fmt.Errorf("model: paged tree KV table entry %d is invalid", i)
		}
		for j := 0; j < i; j++ {
			if cache.table[j] == id {
				return fmt.Errorf("model: paged tree KV table aliases physical block %d", id)
			}
		}
	}
	sharedDestinations := 0
	for j, node := range acceptedPath {
		dst, src := prefixLen+j, prefixLen+node
		li := dst / p.blockTokens
		if dst == src || p.ref[cache.table[li]] == 1 {
			continue
		}
		first := true
		for k := 0; k < j; k++ {
			if prefixLen+k != prefixLen+acceptedPath[k] && (prefixLen+k)/p.blockTokens == li {
				first = false
				break
			}
		}
		if first {
			sharedDestinations++
		}
	}
	if !p.canReserveBlocks(sharedDestinations) {
		return fmt.Errorf("model: paged tree KV compaction cannot reserve %d COW blocks", sharedDestinations)
	}

	// Claim every shared destination page before the first row write. Pages whose
	// rows stay in place need no write and remain shared.
	for j, node := range acceptedPath {
		dst, src := prefixLen+j, prefixLen+node
		if dst == src {
			continue
		}
		li := dst / p.blockTokens
		already := false
		for k := 0; k < j; k++ {
			if prefixLen+k != prefixLen+acceptedPath[k] && (prefixLen+k)/p.blockTokens == li {
				already = true
				break
			}
		}
		if !already {
			cache.ensureOwned(li)
		}
	}
	for j, node := range acceptedPath {
		dst, src := prefixLen+j, prefixLen+node
		if dst == src {
			continue
		}
		dstBlock := p.blocks[cache.table[dst/p.blockTokens]]
		srcBlock := p.blocks[cache.table[src/p.blockTokens]]
		dstOff, srcOff := dst%p.blockTokens, src%p.blockTokens
		for l := 0; l < p.nLayers; l++ {
			for plane := 0; plane < p.planes; plane++ {
				do, so := p.slot(l, plane, dstOff), p.slot(l, plane, srcOff)
				copy(dstBlock[do:do+p.stride], srcBlock[so:so+p.stride])
			}
		}
	}
	newLen := prefixLen + len(acceptedPath)
	keepBlocks := 0
	if newLen > 0 {
		keepBlocks = (newLen + p.blockTokens - 1) / p.blockTokens
	}
	for _, id := range cache.table[keepBlocks:] {
		p.release(id)
	}
	cache.table = cache.table[:keepBlocks]
	cache.nTokens = newLen
	return nil
}

func validateTreeSelection(prefixLen int, acceptedPath []int, treeSize, residentLen int) error {
	if prefixLen < 0 || treeSize < 0 || prefixLen > residentLen || treeSize > residentLen-prefixLen || prefixLen+treeSize != residentLen {
		return fmt.Errorf("model: invalid tree KV geometry prefix=%d tree=%d resident=%d", prefixLen, treeSize, residentLen)
	}
	previous := -1
	for i, node := range acceptedPath {
		if node < 0 || node >= treeSize {
			return fmt.Errorf("model: accepted tree index %d at path offset %d is out of range", node, i)
		}
		if node <= previous {
			return fmt.Errorf("model: accepted tree path must be strictly increasing and unique")
		}
		previous = node
	}
	return nil
}
