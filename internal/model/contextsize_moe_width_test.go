package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// moeOnlyWidthConfig declares the FFN width only as the routed-expert width, as real
// DeepSeek-V4 checkpoints do (IntermediateSize == 0).
func moeOnlyWidthConfig() Config {
	return Config{
		HiddenSize: 64, MoEIntermediateSize: 96, VocabSize: 128, NumLayers: 2,
		NumHeads: 4, NumKVHeads: 2, HeadDim: 16, MaxPositionEmbeddings: 256,
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestContextSizeScratchPricesMoEOnlyWidth(t *testing.T) {
	moe := moeOnlyWidthConfig()
	dense := moe
	dense.IntermediateSize, dense.MoEIntermediateSize = moe.MoEIntermediateSize, 0

	got := compute.EstimateHALScratchpadBytes(moe.ContextSizeConfig().Scratch)
	want := compute.EstimateHALScratchpadBytes(dense.ContextSizeConfig().Scratch)
	if want <= 0 || got != want {
		t.Fatalf("MoE-only width scratch = %d, want %d (priced at the expert width)", got, want)
	}

	// A declared dense width keeps precedence over the expert width.
	both := moe
	both.IntermediateSize = 256
	if w := both.ContextSizeConfig().Scratch.IntermediateSize; w != 256 {
		t.Fatalf("declared dense width scratch IntermediateSize = %d, want 256", w)
	}
}
