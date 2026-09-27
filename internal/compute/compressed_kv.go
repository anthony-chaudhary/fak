package compute

// compressed_kv.go — per-layer compressed KV geometry (#13555).
//
// A DeepSeek-V4 Flash checkpoint does not keep one uniform row per cached position per
// layer: every layer retains a bounded window of recent rows, and a layer whose
// compress_ratios entry r is positive ALSO retains one compressed row per r positions
// (ceil(tokens/r) of them), plus — on the indexed regime — one indexer key per compressed
// row. The uniform NumLayers x tokens x row product EstimateKVStoreBytes charges overcounts
// that schedule by orders of magnitude, and a window-only bound (KVConfig.WindowPerLayer)
// undercounts it by dropping the compressed history. This file is the charge for the
// schedule itself.
//
// This is PLANNING geometry only, with the same posture as KVConfig.WindowPerLayer: no
// KVStore implementation reads it, so it never changes what a backend allocates, only what
// the planner charges. It is charged only when a model projection sets
// ContextSizeConfig.CompressedKV (see model.Config.ContextSizeConfigWithPrecision for the
// gate that decides); a nil slice is the uniform case and every estimate stays
// byte-identical to before it existed. compute stays kvbudget-free: the rows are plain
// (window, ratio, row bytes) tuples the projection derives.

// CompressedKVLayer is one decoder layer of a per-layer compressed KV geometry (DeepSeek-V4
// Flash compress_ratios): window rows, plus on a compressing layer one compressed row per
// Ratio positions and an optional indexer key per compressed row.
type CompressedKVLayer struct {
	WindowRows    int   // >0: min(WindowRows, tokens) rows retained; <=0: every position (full attention)
	Ratio         int   // >0: ceil(tokens/Ratio) compressed rows in addition; <=0: none
	RowBytes      int64 // bytes of one window/compressed row
	IndexRowBytes int64 // bytes of one indexer key: one per compressed row when Ratio>0, else one per base row
}

// rows reports the layer's base (window-bounded) rows, compressed rows, and indexer keys at
// `tokens` cached positions. tokens must be positive.
func (l CompressedKVLayer) rows(tokens int64) (base, compressed, index int64) {
	base = tokens
	if l.WindowRows > 0 && int64(l.WindowRows) < tokens {
		base = int64(l.WindowRows)
	}
	index = base
	if l.Ratio > 0 {
		r := int64(l.Ratio)
		compressed = tokens / r
		if tokens%r != 0 {
			compressed++ // ceil: a partial trailing group still owns a compressed row
		}
		index = compressed
	}
	return base, compressed, index
}

// CompressedKVBytes = Σ_l [(base_l + ceil(t/r_l))·RowBytes + indexRows_l·IndexRowBytes],
// saturating at int64 max so an impossible geometry only grows more conservative; t<=0 ⇒ 0.
// Non-positive RowBytes/IndexRowBytes contribute nothing. The result is monotone
// non-decreasing in tokens (base and ceil rows both are), which is what lets the context
// auto-sizer invert it by binary search.
//
// For the pinned DeepSeek-V4 Flash fixture (43 layers, f32 512-wide rows, 128-row window,
// 128-wide indexer keys on the ratio-4 layers) this is 123,994,112 B at 8,192 tokens against
// the uniform 3-row charge's 2,164,260,864 B [SW-VERIFIED].
func CompressedKVBytes(layers []CompressedKVLayer, tokens int) int64 {
	if tokens <= 0 {
		return 0
	}
	t := int64(tokens)
	var total int64
	for _, l := range layers {
		base, compressed, index := l.rows(t)
		total = saturatingAddInt64(total,
			saturatingMulInt64(saturatingAddInt64(base, compressed), l.RowBytes),
			saturatingMulInt64(index, l.IndexRowBytes))
	}
	return total
}

// CompressedKVBound returns (fixed, perToken) with CompressedKVBytes(layers,t) <= fixed +
// perToken*t ∀ t>=0 — the linear envelope a per-token budget (warmup derivation, admission
// bytes-per-token) needs from a schedule whose true cost is a step function:
//
//	perToken = Σ_l ceil(slope_l), slope_l = (W<=0 ? Row : 0) + (r>0 ? (Row+Idx)/r : 0) + (W<=0 && r<=0 ? Idx : 0)
//	fixed    = Σ_l [(W>0 ? W·Row : 0) + (W>0 && r<=0 ? W·Idx : 0) + (r>0 ? Row+Idx : 0)]
//
// A windowed layer's rows are bounded by W, so they land in fixed; a compressing layer's
// ceil(t/r) <= t/r + 1, so its compressed row and key cost the r-th share per token plus one
// row of ceil slack in fixed. For the pinned DeepSeek-V4 Flash fixture this is fixed =
// 11,366,912 B and perToken = 13,760 B [SW-VERIFIED].
func CompressedKVBound(layers []CompressedKVLayer) (fixed, perToken int64) {
	for _, l := range layers {
		row, idx := l.RowBytes, l.IndexRowBytes
		if row < 0 {
			row = 0
		}
		if idx < 0 {
			idx = 0
		}
		if l.WindowRows > 0 {
			w := int64(l.WindowRows)
			fixed = saturatingAddInt64(fixed, saturatingMulInt64(w, row))
			if l.Ratio <= 0 {
				fixed = saturatingAddInt64(fixed, saturatingMulInt64(w, idx))
			}
		} else {
			perToken = saturatingAddInt64(perToken, row)
			if l.Ratio <= 0 {
				perToken = saturatingAddInt64(perToken, idx)
			}
		}
		if l.Ratio > 0 {
			r := int64(l.Ratio)
			share := saturatingAddInt64(row, idx)
			slope := share / r
			if share%r != 0 {
				slope++ // ceil keeps the envelope an upper bound at every t
			}
			perToken = saturatingAddInt64(perToken, slope)
			fixed = saturatingAddInt64(fixed, share)
		}
	}
	return fixed, perToken
}
