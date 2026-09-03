package model

import (
	"testing"
)

func TestExtractGLM5NextVisionPatches(t *testing.T) {
	const imgH = 28
	const imgW = 28
	const patchSize = 14
	const patchPixels = 14 * 14 * 3

	// 28x28 with patch size 14 -> 2x2 = 4 patches
	img := make([]float32, imgH*imgW*3)
	for i := range img {
		img[i] = float32(i)
	}

	patches := ExtractGLM5NextVisionPatches(img, imgH, imgW, patchSize)
	if len(patches) != 4*patchPixels {
		t.Fatalf("len(patches) = %d, want %d", len(patches), 4*patchPixels)
	}

	// Verify top-left pixel of top-left patch matches img[0]
	if patches[0] != img[0] {
		t.Fatalf("patches[0] = %g, want %g", patches[0], img[0])
	}
}

func TestMergeGLM5NextVisionTokens(t *testing.T) {
	// Full canonical GLM-5.3-Flash scale:
	// 32 x 32 patches, visionDim = 1024
	// Merged 2x2 -> 16 x 16 = 256 tokens, mergedDim = 4 * 1024 = 4096
	const gridH = 32
	const gridW = 32
	const visionDim = 1024
	const mergeSize = 2

	patchFeatures := make([]float32, gridH*gridW*visionDim)
	// Fill patch (0, 0) with 1.0, patch (0, 1) with 2.0, patch (1, 0) with 3.0, patch (1, 1) with 4.0
	for d := 0; d < visionDim; d++ {
		patchFeatures[(0*gridW+0)*visionDim+d] = 1.0
		patchFeatures[(0*gridW+1)*visionDim+d] = 2.0
		patchFeatures[(1*gridW+0)*visionDim+d] = 3.0
		patchFeatures[(1*gridW+1)*visionDim+d] = 4.0
	}

	merged := MergeGLM5NextVisionTokens(patchFeatures, gridH, gridW, visionDim, mergeSize)

	// 256 tokens * 4096 dimensions
	expectedTokens := 256
	expectedDim := 4096
	if len(merged) != expectedTokens*expectedDim {
		t.Fatalf("len(merged) = %d, want %d (%d tokens of %d dims)",
			len(merged), expectedTokens*expectedDim, expectedTokens, expectedDim)
	}

	// Token 0 must contain concatenated patches (0,0), (0,1), (1,0), (1,1)
	tok0 := merged[:expectedDim]
	for d := 0; d < visionDim; d++ {
		if tok0[0*visionDim+d] != 1.0 ||
			tok0[1*visionDim+d] != 2.0 ||
			tok0[2*visionDim+d] != 3.0 ||
			tok0[3*visionDim+d] != 4.0 {
			t.Fatalf("token 0 subpatch feature mismatch at d=%d", d)
		}
	}
}
