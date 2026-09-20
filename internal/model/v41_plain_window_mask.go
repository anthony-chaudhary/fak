package model

// v41_plain_window_mask.go — the #13303 configured causal-window seam for V4.1
// plain (uncompressed) attention.
//
// v41Layer's per-position plain sink contraction builds its KV row set as the
// FULL causal prefix [0, q] and never consults the layer's configured window
// (`Config.Window` / `windowForLayer`, config.go:1232). Every uncompressed layer
// therefore attends globally even when the parsed checkpoint declares a sliding
// window for it (GGUF maps `attention.sliding_window` in
// internal/ggufload/deepseek41.go). That is the correctness gap this leaf closes:
// leaves 06/07 cannot trust a bounded retained decode state while their
// full-prefix reference ignores the configured window.
//
// The seam is deliberately narrow and pure. `v41PlainWindowKeys` returns the
// ORDERED ABSOLUTE key-row IDs a query at position q attends, derived from the
// layer's window value and q alone:
//
//   - a positive window w: exactly the causal keys max(0, q-w+1) .. q
//   - -1 (or any non-positive value): the full causal prefix 0 .. q
//
// It duplicates no attention math: the caller feeds the returned IDs to the
// existing score/softmax/value contraction, so the arithmetic is unchanged and
// only WHICH rows are visible changes. No compressed-role masking, retained
// decode state, kernel fusion or configuration reinterpretation is introduced.

// v41PlainWindowKeys returns the ordered absolute key-row indices a plain
// (uncompressed) layer's query at absolute position q attends, for the layer's
// configured window w. The slice is ascending and always a contiguous causal
// interval ending at q:
//
//   - w <= 0 (and the -1 "full causal" sentinel): every causal key 0 .. q.
//   - w > 0: the trailing w causal keys, max(0, q-w+1) .. q.
//
// q < 0 is refused as an empty list (there is no query position). The returned
// slice is freshly allocated and owns no model state, so the caller may retain
// or mutate it freely.
func v41PlainWindowKeys(q, w int) []int32 {
	if q < 0 {
		return nil
	}
	lo := 0
	if w > 0 {
		lo = q - w + 1
		if lo < 0 {
			lo = 0
		}
	}
	keys := make([]int32, 0, q-lo+1)
	for i := lo; i <= q; i++ {
		keys = append(keys, int32(i))
	}
	return keys
}

// v41PlainWindowIndexList renders the [rows+1] int32 index list
// V41SparseAttentionSink consumes for a contract whose KV array is the
// window-selected rows FLATTENED in order. The sink's idx is a POSITIONAL index
// into that array (v41_sparse_attention.go validates each against opt.N), not an
// absolute history row: a caller that appends the selected rows in
// v41PlainWindowKeys order must therefore address them 0..rows-1. This is
// exactly the list the historical full-causal branch built (idx[i]=i over the
// whole prefix plus a trailing -1), now generalized to the window's row count,
// so a window layer's contraction stays byte-identical to a scalar oracle that
// selects the same ordered rows.
func v41PlainWindowIndexList(keys []int32) []int32 {
	idx := make([]int32, len(keys)+1)
	for i := range keys {
		idx[i] = int32(i)
	}
	idx[len(keys)] = -1
	return idx
}
