package model

// qwen35PrefillConv4 is the ordinary four-tap causal depthwise convolution.
// Each token resolves its four immutable source-row descriptors once, outside
// the channel loop. The channel-major, ascending-tap float32 expression and
// silu call are unchanged. Missing leading taps are skipped, not zero padded:
// multiplying a synthesized zero by a nonfinite weight would change results.
func qwen35PrefillConv4(out, mixed, weights []float32, history [][]float32, rows, width, workers int) bool {
	maxInt := int(^uint(0) >> 1)
	if rows <= 0 || rows > maxInt-3 || width <= 0 || width > maxInt/4 ||
		len(mixed)%width != 0 || len(mixed)/width != rows || len(out) != len(mixed) ||
		len(weights) != 4*width || len(history) > 3 ||
		uintptr(len(mixed)) > ^uintptr(0)/4 || uintptr(len(weights)) > ^uintptr(0)/4 {
		return false
	}
	// Complete admission precedes all writes. Fresh ordinary output is disjoint
	// from all operands; arbitrary aliases/malformed extents retain the old loop
	// and its original partial-write/panic behavior. Inputs can alias each other.
	if convPrefillRowsOverlap(out, mixed) || convPrefillRowsOverlap(out, weights) {
		return false
	}
	for _, row := range history {
		if len(row) != width || convPrefillRowsOverlap(out, row) {
			return false
		}
	}
	// Match the caller's existing partition and synchronous join. Four local
	// descriptors per active worker are the only new scratch; no numeric panel
	// or descriptor is retained after that worker's current token completes.
	parFor(rows, workers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			outRow := out[t*width : (t+1)*width]
			first := 3 - t - len(history)
			if first < 0 {
				first = 0
			}
			var tapRows [4][]float32
			for j := first; j < 4; j++ {
				src := t + j - 3
				if src >= 0 {
					tapRows[j] = mixed[src*width : (src+1)*width]
				} else {
					tapRows[j] = history[len(history)+src]
				}
			}
			for c := 0; c < width; c++ {
				var acc float32
				cb := c * 4
				for j := first; j < 4; j++ {
					row := tapRows[j]
					acc += weights[cb+j] * row[c]
				}
				outRow[c] = silu(acc)
			}
		}
	})
	return true
}
