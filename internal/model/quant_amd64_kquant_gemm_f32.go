//go:build amd64

package model

func init() {
	kQuantF32BlockPairArch = kQuantF32BlockPairAVX2
}

func kQuantF32BlockPairAVX2() kQuantF32BlockPair {
	if qtier < tierAVX2 || qgemmMode == qgemmModeLegacy || qgemmMode == qgemmModeAccel {
		return nil
	}
	// Same in-tree exact block-dot implementation as resident Q4_K prefill.
	// Decoding and serial block combination stay in the generic caller.
	return q4kBlockDot256x2AVX2
}
