package model

// DownsampleGLM5NextDSAKeys downsamples keys along the sequence length by stride (default 4).
// Input keys: flattened [seqLen * dim].
// Output downsampled keys: flattened [ceil(seqLen/stride) * dim].
// For each block b: output is the average of keys in [b*stride, min((b+1)*stride, seqLen)).
func DownsampleGLM5NextDSAKeys(keys []float32, seqLen, dim, stride int) []float32 {
	if stride <= 0 {
		stride = 4
	}
	if seqLen <= 0 || dim <= 0 || len(keys) < seqLen*dim {
		return nil
	}

	numBlocks := (seqLen + stride - 1) / stride
	out := make([]float32, numBlocks*dim)

	for b := 0; b < numBlocks; b++ {
		start := b * stride
		end := start + stride
		if end > seqLen {
			end = seqLen
		}
		count := float32(end - start)
		outRow := out[b*dim : (b+1)*dim]

		for t := start; t < end; t++ {
			inRow := keys[t*dim : (t+1)*dim]
			for d := 0; d < dim; d++ {
				outRow[d] += inRow[d]
			}
		}

		invCount := 1.0 / count
		for d := 0; d < dim; d++ {
			outRow[d] *= invCount
		}
	}

	return out
}
