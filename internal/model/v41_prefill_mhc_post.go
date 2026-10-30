package model

// v41PrefillMHCPostBF16On is only called by the serial multi-token layer
// epilogue, after routed/shared projection workers have joined. Device and
// malformed/single-token cases retain the ordinary implementation verbatim.
// It parallelizes private raw H-coordinate arithmetic, never token callbacks,
// BF16 validation, error selection or publication of next-layer streams.
func v41PrefillMHCPostBF16On(device *v41MHCDevice, layer int, x []float32, streams [][]float32, mix v41MHCMix, rows int) ([][]float32, error) {
	return v41PrefillMHCPostBF16WithReuse(device, layer, x, streams, mix, rows, false)
}

// v41PrefillMHCPostBF16Owned consumes only the private host residuals produced
// by v41LayerFFNInputs. On failure those residuals may be partly overwritten;
// the caller still owns the unchanged incoming streams until every row passes.
// Device, single-row and malformed requests keep the non-mutating contract.
func v41PrefillMHCPostBF16Owned(device *v41MHCDevice, layer int, x []float32, streams [][]float32, mix v41MHCMix, rows int) ([][]float32, error) {
	return v41PrefillMHCPostBF16WithReuse(device, layer, x, streams, mix, rows, true)
}

func v41PrefillMHCPostBF16WithReuse(device *v41MHCDevice, layer int, x []float32, streams [][]float32, mix v41MHCMix, rows int, owned bool) ([][]float32, error) {
	width := len(x)
	work, workOK := checkedMulInt(width, 20) // four destinations, four sources plus post
	if device != nil || rows <= 1 || !workOK || width == 0 || len(streams) != 4 || len(mix.post) != 4 || len(mix.comb) != 16 {
		return v41MHCPostBF16On(device, layer, x, streams, mix)
	}
	for _, stream := range streams {
		if len(stream) != width {
			// Preserve the old input-rounding-vs-shape-error precedence too.
			return v41MHCPostBF16On(device, layer, x, streams, mix)
		}
	}
	// Exact original first boundary: input finite/rounding failure must occur
	// before any raw post-mix output or output-side BF16 operation.
	row, err := v41MHCBF16(layer, x)
	if err != nil {
		return nil, err
	}
	next := streams
	if !owned {
		next = make([][]float32, 4)
		for dst := range next {
			next[dst] = make([]float32, width)
		}
	}
	parForRange(width, work, func(lo, hi int) {
		for d := lo; d < hi; d++ {
			var column [4]float32
			for dst := 0; dst < 4; dst++ {
				var value float32
				// Keep src0..3, then post, including the explicit F32
				// product boundaries of v41MHCPostObserved. Never reduce
				// across coordinates or merge/reassociate these terms.
				for src := 0; src < 4; src++ {
					value += float32(mix.comb[src*4+dst] * streams[src][d])
				}
				value += float32(mix.post[dst] * row[d])
				column[dst] = value
			}
			// Every destination depends on all four original source values.
			// Finish this coordinate before recycling any source; workers own
			// disjoint coordinates and no callback can observe the private rows.
			for dst := range column {
				next[dst][d] = column[dst]
			}
		}
	})
	// Workers have joined; preserve destination0..3 and element-order BF16
	// checks. The first original output fault still wins, with no partial
	// next-layer stream/carry publication and no callback-visible raw panel.
	for h := range next {
		next[h], err = v41MHCBF16Into(layer, next[h], next[h])
		if err != nil {
			return nil, err
		}
	}
	return next, nil
}
