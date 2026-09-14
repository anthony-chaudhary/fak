package model

// kvcache_q8.go — the Q8_0 KV-cache realization (#12981, epic #9430). This is the
// ENGINE half that compute/kvprecision.go explicitly defers ("The denser KVStore
// *realization* ... is the engine half ... and is not in this file"). It makes a
// cached position's POST-RoPE K and V rows physically q8_0-resident, while Kraw
// (pre-RoPE K) stays f32 because Evict re-positions a survivor by a single rotation
// of Kraw and that must stay bit-exact.
//
// WHY THIS SHAPE:
//   - The planner's KVPrecisionQ8 tier (compute/kvprecision.go) charges a MIXED
//     layout: one f32 row (Kraw) + two q8_0 rows (post-RoPE K, V). This file makes
//     that the realized layout, so the budget the admission gate uses equals the
//     bytes actually resident. Otherwise the realization is a planning fiction,
//     exactly the failure the #1045/#1047 split warned about.
//   - The default (zero-value KVPrecision) path is untouched: K/V stay f32 [][]float32
//     and every existing test is byte-for-byte identical. A session only pays the q8
//     path when an operator selects it.
//
// CORRECTNESS-FIRST TRADEOFF: the q8 path appends/stores packed blocks and
// dequantizes a row into a small scratch on attend (decodeRowInto). A fused
// quantized-attention kernel that scores directly over int8 codes is a follow-up;
// this realization is about correctness + residency, not decode throughput.

// kvQ8_0GroupSize is the number of contiguous elements sharing one f32 scale in the
// realized Q8_0 cache. It aliases KVQuantQ8_0BlockSize so the codec and the layout
// can never drift: a future block-size change moves both.
const kvQ8_0GroupSize = KVQuantQ8_0BlockSize

// kvPackedRow is one layer's K (or V) rows for a fixed position count, stored as
// concatenated Q8_0 blocks. Unlike KVQuantQ8_0 (a single vector), this holds MANY
// rows: row r occupies elements [r*width, (r+1)*width), and each block of
// kvQ8_0GroupSize elements carries its own f32 scale. A block never straddles two
// rows when width divides the group size, which holds for every supported head_dim
// (32 divides 64/128/256), so a group always pools one head's dynamic range.
type kvPackedRow struct {
	width  int       // elements per row (NumKVHeads*HeadDim)
	codes  []int8    // packed int8 codes, one per element
	scales []float32 // one scale per kvQ8_0GroupSize elements
}

func (p *kvPackedRow) len() int {
	if p == nil || p.width == 0 {
		return 0
	}
	return len(p.codes) / p.width
}

// appendRow encodes one row (width elements) and appends it.
func (p *kvPackedRow) appendRow(row []float32) {
	q := QuantizeKVQ8_0(row)
	p.codes = append(p.codes, q.Codes...)
	p.scales = append(p.scales, q.Scale...)
}

// appendPacked appends another packed row-set verbatim (used by Clone fast-paths
// and by batch prefill where rows were packed upstream).
func (p *kvPackedRow) appendPacked(other *kvPackedRow) {
	if other == nil {
		return
	}
	if p.width == 0 {
		p.width = other.width
	}
	p.codes = append(p.codes, other.codes...)
	p.scales = append(p.scales, other.scales...)
}

// decodeRowInto dequantizes row r into dst (len(dst) >= width). It is the
// dequantize-on-attend primitive: the attention kernels copy a row's contribution
// into a scratch dequantized row rather than reading f32 storage.
func (p *kvPackedRow) decodeRowInto(dst []float32, r int) {
	if p == nil || dst == nil || p.width == 0 {
		return
	}
	base := r * p.width
	n := p.width
	if base < 0 || base+n > len(p.codes) {
		return
	}
	dst = dst[:n]
	for i := 0; i < n; i++ {
		dst[i] = float32(p.codes[base+i]) * p.scales[(base+i)/kvQ8_0GroupSize]
	}
}

// encodeRowInto re-quantizes row and overwrites rows [i] in the packing. The scale
// groups covering row i are replaced wholesale, so re-encoding a survivor after
// Evict's re-RoPE yields the exact bytes a fresh append would have.
func (p *kvPackedRow) encodeRowInto(row []float32, i int) {
	if p == nil || i < 0 || p.width == 0 || len(row) < p.width {
		return
	}
	base := i * p.width
	if base+p.width > len(p.codes) {
		return
	}
	q := QuantizeKVQ8_0(row[:p.width])
	for j := 0; j < p.width; j++ {
		p.codes[base+j] = q.Codes[j]
	}
	fromGroup := base / kvQ8_0GroupSize
	for g := 0; g < len(q.Scale); g++ {
		if fromGroup+g < len(p.scales) {
			p.scales[fromGroup+g] = q.Scale[g]
		}
	}
}

// bytes is the resident footprint: 1 byte per int8 code + 4 bytes per f32 scale.
func (p *kvPackedRow) bytes() int64 {
	if p == nil {
		return 0
	}
	return int64(len(p.codes)) + 4*int64(len(p.scales))
}

// truncate keeps at most targetPositions rows (a slice-header adjustment that also
// keeps only the scales covering those rows).
func (p *kvPackedRow) truncate(targetPositions int) {
	if p == nil || targetPositions < 0 || p.width <= 0 {
		return
	}
	codes := targetPositions * p.width
	if codes > len(p.codes) {
		return
	}
	p.codes = p.codes[:codes]
	groups := codes / kvQ8_0GroupSize
	if groups > len(p.scales) {
		groups = len(p.scales)
	}
	p.scales = p.scales[:groups]
}

// dropSpan removes rows [from, end) from the packing, shifting the surviving codes
// and scales down — the compaction that parallels KVCache.evictSupported's slice
// append. Because width divides the group size, a whole number of scale groups is
// removed per row-span and the surviving group boundaries stay aligned.
func (p *kvPackedRow) dropSpan(from, end int) {
	if p == nil || p.width <= 0 || from < 0 || end <= from {
		return
	}
	fromElem := from * p.width
	endElem := end * p.width
	if endElem > len(p.codes) {
		endElem = len(p.codes)
	}
	p.codes = append(p.codes[:fromElem], p.codes[endElem:]...)
	fromGroup := fromElem / kvQ8_0GroupSize
	endGroup := endElem / kvQ8_0GroupSize
	if endGroup > len(p.scales) {
		endGroup = len(p.scales)
	}
	p.scales = append(p.scales[:fromGroup], p.scales[endGroup:]...)
}

// reserve grows the packing's spare capacity for extraPositions rows without
// changing len(). It mirrors reserveFloat32 for the packed representation.
func (p *kvPackedRow) reserve(extraPositions int) {
	if p == nil || p.width <= 0 || extraPositions <= 0 {
		return
	}
	extraCodes := extraPositions * p.width
	if cap(p.codes) < len(p.codes)+extraCodes {
		codes := make([]int8, len(p.codes), len(p.codes)+extraCodes)
		copy(codes, p.codes)
		p.codes = codes
	}
	extraGroups := (extraCodes + kvQ8_0GroupSize - 1) / kvQ8_0GroupSize
	if cap(p.scales) < len(p.scales)+extraGroups {
		scales := make([]float32, len(p.scales), len(p.scales)+extraGroups)
		copy(scales, p.scales)
		p.scales = scales
	}
}

// clonePacked deep-copies the packing, reserving extraPositions spare rows.
func clonePacked(src *kvPackedRow, extraPositions int) *kvPackedRow {
	if src == nil {
		return nil
	}
	extraCodes := extraPositions * src.width
	extraGroups := 0
	if extraCodes > 0 {
		extraGroups = (extraCodes + kvQ8_0GroupSize - 1) / kvQ8_0GroupSize
	}
	dst := &kvPackedRow{width: src.width}
	dst.codes = make([]int8, len(src.codes), len(src.codes)+extraCodes)
	copy(dst.codes, src.codes)
	dst.scales = make([]float32, len(src.scales), len(src.scales)+extraGroups)
	copy(dst.scales, src.scales)
	return dst
}

// KVPackedRowBytes reports the resident Q8_0 bytes for one row of `dim` elements:
// one int8 code per element plus one f32 scale per kvQ8_0GroupSize elements. It is
// the realized counterpart of KVVectorBytes(dim, KVPrecisionQ8_0) and is exported so
// tests and the fit estimator can compare budget math to realized bytes exactly.
func KVPackedRowBytes(dim int) int64 {
	if dim <= 0 {
		return 0
	}
	groups := int64((dim + kvQ8_0GroupSize - 1) / kvQ8_0GroupSize)
	return int64(dim) + 4*groups
}

// SupportsQuantizedKVCache reports whether this model geometry's ACTIVE forward path
// writes/reads the kernel-owned KVCache through the precision-aware helpers, so the
// realized Q8_0 tier is safe to enable. Unsupported today: the device HAL KV store
// (a separate compute.KVStore), the Metal resident prefill/decode paths (they append
// to c.K directly), hybrid Gated-DeltaNet (recurrent state), GLM-DSA, MiniMax-M3
// sparse, and gemma4 (cacheless/its own windowed path). Selecting q8 on any of them
// refuses loudly rather than silently mixing f32 and packed rows.
func (c Config) SupportsQuantizedKVCache() bool {
	return c.QuantizedKVUnsupportedReason() == ""
}

// QuantizedKVUnsupportedReason names the arch-specific forward path that blocks the
// realized Q8_0 KV tier, or "" when the geometry's active path uses the precision-aware
// helpers. It exists so the device/agent boundary can refuse with a concrete reason
// without reaching unexported predicates.
func (c Config) QuantizedKVUnsupportedReason() string {
	switch {
	case c.IsQwen35Hybrid():
		return "hybrid Gated-DeltaNet recurrent state"
	case c.usesMLAMoELayout():
		return "GLM-MoE-DSA layout"
	case c.isMiniMaxSparseAttn():
		return "MiniMax-M3 sparse attention"
	case c.isGemma4():
		return "gemma4 cacheless/windowed forward"
	default:
		return ""
	}
}

// ---- KVCache precision-aware accessors -------------------------------------
//
// On the f32 path these are thin wrappers over the historical K/V slices, so every
// call site that used to index c.K[l]/c.V[l] directly and now calls these is
// byte-for-byte identical. On the q8 path they route to the packed row-sets and the
// f32 slices stay empty. kvStride is the width w of one row either way.

// kvLen reports the number of cached positions on layer l. It replaces the
// `len(c.K[l]) / w` idiom the decode/prefill sites used, which cannot see the packed
// representation.
func (c *KVCache) kvLen(l int) int {
	if c.quantized() {
		if l < 0 || l >= len(c.kQ8) {
			return 0
		}
		return c.kQ8[l].len()
	}
	if l < 0 || l >= len(c.K) {
		return 0
	}
	w := c.kvStride()
	if w == 0 {
		return 0
	}
	return len(c.K[l]) / w
}

// appendKV appends one position's post-RoPE K row and V row to layer l.
func (c *KVCache) appendKV(l int, kRow, vRow []float32) {
	if c.quantized() {
		if l >= 0 && l < len(c.kQ8) {
			c.kQ8[l].appendRow(kRow)
			c.vQ8[l].appendRow(vRow)
		}
		return
	}
	if l >= 0 && l < len(c.K) {
		c.K[l] = append(c.K[l], kRow...)
		c.V[l] = append(c.V[l], vRow...)
	}
}

// appendK appends one position's post-RoPE K row to layer l (the packed twin of
// `c.K[l] = append(c.K[l], k...)`). Used by sites that append K before V.
func (c *KVCache) appendK(l int, kRow []float32) {
	if c.quantized() {
		if l >= 0 && l < len(c.kQ8) {
			c.kQ8[l].appendRow(kRow)
		}
		return
	}
	if l >= 0 && l < len(c.K) {
		c.K[l] = append(c.K[l], kRow...)
	}
}

// appendV appends one position's V row to layer l.
func (c *KVCache) appendV(l int, vRow []float32) {
	if c.quantized() {
		if l >= 0 && l < len(c.vQ8) {
			c.vQ8[l].appendRow(vRow)
		}
		return
	}
	if l >= 0 && l < len(c.V) {
		c.V[l] = append(c.V[l], vRow...)
	}
}

// decodeRowInto dequantizes layer l's cached row j (post-RoPE K when k true, else V)
// into dst. On the f32 path it copies the row; callers use it to read a row without
// caring about the representation.
func (c *KVCache) decodeRowInto(l, j int, k bool, dst []float32) {
	w := c.kvStride()
	if c.quantized() {
		var rows *kvPackedRow
		if k {
			if l < 0 || l >= len(c.kQ8) {
				return
			}
			rows = &c.kQ8[l]
		} else {
			if l < 0 || l >= len(c.vQ8) {
				return
			}
			rows = &c.vQ8[l]
		}
		rows.decodeRowInto(dst, j)
		return
	}
	src := c.K
	if !k {
		src = c.V
	}
	if l < 0 || l >= len(src) {
		return
	}
	row := src[l]
	if j < 0 || (j+1)*w > len(row) {
		return
	}
	copy(dst[:w], row[j*w:(j+1)*w])
}

// KVCacheResidentBytes reports the LIVE resident bytes of this cache's K/Kraw/V
// payload across all layers, counting the packed q8_0 sizes on the q8 path and the
// f32 rows otherwise. It is the measured counterpart of the planner's
// EstimateKVStoreBytes, used to witness that realized bytes match the budget math.
func (c *KVCache) KVCacheResidentBytes() int64 {
	if c == nil {
		return 0
	}
	var total int64
	n := c.cfg.NumLayers
	for l := 0; l < n; l++ {
		if l < len(c.Kraw) {
			total += int64(len(c.Kraw[l])) * 4
		}
		if c.quantized() {
			if l < len(c.kQ8) {
				total += c.kQ8[l].bytes()
			}
			if l < len(c.vQ8) {
				total += c.vQ8[l].bytes()
			}
			continue
		}
		if l < len(c.K) {
			total += int64(len(c.K[l])) * 4
		}
		if l < len(c.V) {
			total += int64(len(c.V[l])) * 4
		}
	}
	return total
}

// rewriteKRow overwrites layer l's cached post-RoPE K row i with row. On the f32 path
// it copies in place (the historical Evict re-RoPE write); on the q8 path it re-encodes
// row i block-by-block so Evict's survivor re-positioning lands the same packed bytes a
// fresh append at the new position would have produced. row must be w elements.
func (c *KVCache) rewriteKRow(l, i int, row []float32) {
	w := c.kvStride()
	if c.quantized() {
		if l < 0 || l >= len(c.kQ8) {
			return
		}
		c.kQ8[l].encodeRowInto(row, i)
		return
	}
	if l < 0 || l >= len(c.K) || i < 0 {
		return
	}
	if (i+1)*w > len(c.K[l]) {
		return
	}
	copy(c.K[l][i*w:(i+1)*w], row[:w])
}

// rewriteVRow overwrites layer l's cached V row i with row (the V twin of rewriteKRow).
func (c *KVCache) rewriteVRow(l, i int, row []float32) {
	w := c.kvStride()
	if c.quantized() {
		if l < 0 || l >= len(c.vQ8) {
			return
		}
		c.vQ8[l].encodeRowInto(row, i)
		return
	}
	if l < 0 || l >= len(c.V) || i < 0 {
		return
	}
	if (i+1)*w > len(c.V[l]) {
		return
	}
	copy(c.V[l][i*w:(i+1)*w], row[:w])
}

// appendBatchedKV appends P positions' post-RoPE K and V rows (row-major, P*w
// elements, as the batched prefill produces) to layer l. On the f32 path it is the
// historical single append; on the q8 path it encodes each row in turn.
func (c *KVCache) appendBatchedKV(l int, K, V []float32, p, w int) {
	if c.quantized() {
		if l < 0 || l >= len(c.kQ8) {
			return
		}
		for t := 0; t < p; t++ {
			c.kQ8[l].appendRow(K[t*w : (t+1)*w])
			c.vQ8[l].appendRow(V[t*w : (t+1)*w])
		}
		return
	}
	if l < 0 || l >= len(c.K) {
		return
	}
	c.K[l] = append(c.K[l], K...)
	c.V[l] = append(c.V[l], V...)
}

// attentionRows returns layer l's post-RoPE K and V as flat, densely-laid-out f32
// slices (row-major, w elements per position) for the attention kernels that read
// them directly. On the f32 path these ARE the cache's slices (zero copy,
// byte-identical). On the q8 path it dequantizes the whole layer into fresh buffers,
// which is the correctness-first cost a fused quantized-attention kernel will later
// remove. Callers must treat the result as read-only scratch.
func (c *KVCache) attentionRows(l int) ([]float32, []float32) {
	if !c.quantized() {
		if l < 0 || l >= len(c.K) {
			return nil, nil
		}
		return c.K[l], c.V[l]
	}
	if l < 0 || l >= len(c.kQ8) {
		return nil, nil
	}
	w := c.kvStride()
	n := c.kQ8[l].len()
	kOut := make([]float32, n*w)
	vOut := make([]float32, n*w)
	row := make([]float32, w)
	for j := 0; j < n; j++ {
		c.kQ8[l].decodeRowInto(row, j)
		copy(kOut[j*w:(j+1)*w], row)
		c.vQ8[l].decodeRowInto(row, j)
		copy(vOut[j*w:(j+1)*w], row)
	}
	return kOut, vOut
}

// ConvertToPrecision re-realizes the cache's attended K/V rows at prec in place,
// preserving Len() and pos[]. Converting f32 -> q8_0 (re-encode) and q8_0 -> f32
// (dequantize) are both lossy in the q8->f32 direction only via the already-stored
// codes, so it never fabricates precision: a q8 cache dequantized to f32 returns the
// same values a q8 attend would have computed. Kraw is untouched (always f32). A nil
// cache or a same-tier conversion is a no-op, so it is safe to call unconditionally.
func (c *KVCache) ConvertToPrecision(prec KVPrecision) {
	if c == nil || prec == "" || prec == c.prec {
		return
	}
	w := c.kvStride()
	n := c.kvLen(0)
	nLayers := c.cfg.NumLayers
	switch {
	case prec == KVPrecisionQ8_0:
		kQ8 := make([]kvPackedRow, nLayers)
		vQ8 := make([]kvPackedRow, nLayers)
		for l := 0; l < nLayers; l++ {
			kQ8[l].width = w
			vQ8[l].width = w
			ln := c.kvLen(l)
			for j := 0; j < ln; j++ {
				kQ8[l].appendRow(c.K[l][j*w : (j+1)*w])
				vQ8[l].appendRow(c.V[l][j*w : (j+1)*w])
			}
		}
		for l := 0; l < nLayers; l++ {
			c.K[l], c.V[l] = nil, nil
		}
		c.kQ8, c.vQ8 = kQ8, vQ8
		c.prec = KVPrecisionQ8_0
	case prec == KVPrecisionFP32:
		K := make([][]float32, nLayers)
		V := make([][]float32, nLayers)
		for l := 0; l < nLayers; l++ {
			ln := c.kvLen(l)
			K[l] = make([]float32, 0, ln*w)
			V[l] = make([]float32, 0, ln*w)
			row := make([]float32, w)
			for j := 0; j < ln; j++ {
				c.kQ8[l].decodeRowInto(row, j)
				K[l] = append(K[l], row...)
				c.vQ8[l].decodeRowInto(row, j)
				V[l] = append(V[l], row...)
			}
		}
		c.K, c.V = K, V
		c.kQ8, c.vQ8 = nil, nil
		c.prec = KVPrecisionFP32
	default:
		// fp16/q4_0 are declared tiers but not realized; leaving the cache unchanged is
		// fail-closed and honest (the witness test asserts the realized set).
		return
	}
	_ = n
}
