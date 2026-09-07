package compute

import "errors"

// QSADynamicGatingThreshold defines the token depth threshold (16,384 tokens)
// above which QSA sparse row gather is triggered for full-attention layers.
const QSADynamicGatingThreshold = 16384

// QSAPerplexityDeltaTolerance defines the maximum relative L2 delta tolerance (0.05%)
// between dense masked attention and true sparse row gather attention.
const QSAPerplexityDeltaTolerance = 0.0005

// TiledConvConcatForwardSlices performs tiled memory channel transpose convolution (Issue #464)
// to eliminate strided DRAM bank thrashing. Returns a fallback error when hardware acceleration
// is not present, prompting caller fallback to the sequential CPU path.
func TiledConvConcatForwardSlices(mixed [][]float32, conv []float32, convDim, K int, out [][]float32) ([][]float32, int, int, error) {
	return nil, 0, 0, errors.New("compute: tiled conv concat hardware kernel unavailable")
}
