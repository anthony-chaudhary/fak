package model

import "unsafe"

// tryPrefillEmbedRectRows is private to the two rectangular prefill entries,
// before any projection worker dispatch. General/decode embedding helpers stay
// serial. Each admitted row keeps the original gather/inverse/scale order; the
// caller's worker budget and existing threshold decide whether to parallelize.
// A decline writes nothing, so the old serial loop keeps its original invalid
// token/shape/alias behavior, including partial writes before a panic.
func (m *Model) tryPrefillEmbedRectRows(X []float32, prompts [][]int, P, H int, cfg Config, workers int) bool {
	B := len(prompts)
	if m == nil || tensorLookupProbe != nil || m.attnObs != nil || m.lora != nil || B == 0 || P <= 0 || H <= 0 || workers <= 1 || P > len(X)/H/B {
		return false
	}
	rows := B * P // the division-first destination bound proves both products fit
	work := rows * H
	if work < parThreshold {
		return false
	}
	dst := X[:work]
	// In addition to embedding data, ids and their slice headers must not be
	// rewritten by destination stores. Address comparison never dereferences or
	// converts uintptr back to a pointer, including malformed cross-typed aliases.
	if prefillEmbeddingAliases(dst, unsafe.Pointer(unsafe.SliceData(prompts)), uintptr(B), unsafe.Sizeof([]int(nil))) {
		return false
	}
	for _, ids := range prompts {
		if len(ids) != P || prefillEmbeddingAliases(dst, unsafe.Pointer(unsafe.SliceData(ids)), uintptr(len(ids)), unsafe.Sizeof(int(0))) {
			return false
		}
	}
	var embed []float32
	var vocab int
	packed := m.Q2KEmbedding
	if packed != nil {
		if packed.hidden != H || packed.vocab <= 0 || packed.format < packedEmbeddingQ2K || packed.format > packedEmbeddingQ6K {
			return false
		}
		bw, bb := packed.format.blockWeights(), packed.format.blockBytes()
		if H%bw != 0 || H/bw > len(packed.raw)/bb {
			return false
		}
		rowBytes := (H / bw) * bb
		if rowBytes == 0 || len(packed.raw)%rowBytes != 0 || len(packed.raw)/rowBytes != packed.vocab ||
			prefillEmbeddingAliases(dst, unsafe.Pointer(unsafe.SliceData(packed.raw)), uintptr(len(packed.raw)), 1) {
			return false
		}
		vocab = packed.vocab
		if packed.prism != nil && !prefillEmbeddingPrismOK(dst, packed.prism, H, true) {
			return false
		}
	} else {
		// tensorLookupProbe was refused above: this is the existing immutable
		// manifest view, resolved once before workers rather than once per prompt.
		embed = m.embedRows()
		if len(embed)%H != 0 || prefillEmbeddingAliases(dst, unsafe.Pointer(unsafe.SliceData(embed)), uintptr(len(embed)), 4) {
			return false
		}
		vocab = len(embed) / H
		if m.prism != nil && !prefillEmbeddingPrismOK(dst, m.prism, H, false) {
			return false
		}
	}
	for _, ids := range prompts {
		for _, id := range ids {
			if id < 0 || id >= vocab {
				return false
			}
		}
	}
	scale := cfg.embedScale()
	parForRangeWorkers(rows, work, workers, func(lo, hi int) {
		for row := lo; row < hi; row++ {
			id := prompts[row/P][row%P]
			out := dst[row*H : (row+1)*H]
			if packed != nil {
				// All GatherRow error conditions and backing extents were admitted above.
				// No error replay is permitted after worker writes begin.
				if err := packed.GatherRow(id, out, scale); err != nil {
					panic(err)
				}
			} else {
				copy(out, embed[id*H:(id+1)*H])
				m.prismInverseEmbeddingRow("model.embed_tokens.weight", out)
				scaleEmbedInPlace(out, cfg)
			}
		}
	})
	return true
}

func prefillEmbeddingPrismOK(dst []float32, state *prismHadamardState, H int, packed bool) bool {
	width := state.inverseWidth["model.embed_tokens.weight"]
	if width == 0 && !packed {
		return true
	} // original f32 inverse is a no-op
	if width != H || !isPowerOfTwo(state.blockSize) || H%state.blockSize != 0 {
		return false
	}
	signs := state.signs[H]
	return len(signs) == H && !prefillEmbeddingAliases(dst, unsafe.Pointer(unsafe.SliceData(signs)), uintptr(len(signs)), 1)
}

// Compare complete live spans, not their capacities. Subtraction avoids an
// overflowing end-address addition; oversized byte products conservatively alias.
func prefillEmbeddingAliases(dst []float32, src unsafe.Pointer, n, size uintptr) bool {
	max := ^uintptr(0)
	if uintptr(len(dst)) > max/4 || n > max/size {
		return true
	}
	dbytes, sbytes := uintptr(len(dst))*4, n*size
	if dbytes == 0 || sbytes == 0 {
		return false
	}
	d, s := uintptr(unsafe.Pointer(unsafe.SliceData(dst))), uintptr(src)
	if d <= s {
		return s-d < dbytes
	}
	return d-s < sbytes
}
