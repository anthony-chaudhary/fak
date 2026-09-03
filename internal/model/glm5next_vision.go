package model

// GLM5NextVisionConfig carries the vision tower geometry for GLM-5.3-Flash.
type GLM5NextVisionConfig struct {
	ImageSize        int // 448
	PatchSize        int // 14
	SpatialMergeSize int // 2
	VisionDim        int // 1024
	OutHiddenSize    int // 4096
}

// DefaultGLM5NextVisionConfig returns the canonical vision geometry.
func DefaultGLM5NextVisionConfig() GLM5NextVisionConfig {
	return GLM5NextVisionConfig{
		ImageSize:        448,
		PatchSize:        14,
		SpatialMergeSize: 2,
		VisionDim:        1024,
		OutHiddenSize:    4096,
	}
}

// ExtractGLM5NextVisionPatches extracts non-overlapping 14x14 RGB patches from a 448x448x3 image.
// Returns [1024 * (14*14*3)] = [1024 * 588] floats.
func ExtractGLM5NextVisionPatches(imgRGB []float32, imgH, imgW, patchSize int) []float32 {
	if patchSize <= 0 {
		patchSize = 14
	}
	gridH := imgH / patchSize
	gridW := imgW / patchSize
	numPatches := gridH * gridW
	patchPixels := patchSize * patchSize * 3
	if len(imgRGB) < imgH*imgW*3 || numPatches == 0 {
		return nil
	}

	out := make([]float32, numPatches*patchPixels)
	patchIdx := 0

	for ph := 0; ph < gridH; ph++ {
		for pw := 0; pw < gridW; pw++ {
			patchSlice := out[patchIdx*patchPixels : (patchIdx+1)*patchPixels]
			pPixel := 0
			for y := 0; y < patchSize; y++ {
				imgY := ph*patchSize + y
				for x := 0; x < patchSize; x++ {
					imgX := pw*patchSize + x
					pixelOff := (imgY*imgW + imgX) * 3
					patchSlice[pPixel+0] = imgRGB[pixelOff+0]
					patchSlice[pPixel+1] = imgRGB[pixelOff+1]
					patchSlice[pPixel+2] = imgRGB[pixelOff+2]
					pPixel += 3
				}
			}
			patchIdx++
		}
	}
	return out
}

// MergeGLM5NextVisionTokens takes the [gridH * gridW * visionDim] patch features and merges
// them 2x2 spatially:
// (gridH/2) * (gridW/2) = (32/2) * (32/2) = 16 * 16 = 256 tokens.
// Each merged token concatenates 4 patch vectors: 4 * visionDim = 4096 features.
func MergeGLM5NextVisionTokens(
	patchFeatures []float32,
	gridH, gridW, visionDim, mergeSize int,
) []float32 {
	if mergeSize <= 0 {
		mergeSize = 2
	}
	if gridH%mergeSize != 0 || gridW%mergeSize != 0 || visionDim <= 0 {
		return nil
	}
	outGridH := gridH / mergeSize
	outGridW := gridW / mergeSize
	numMergedTokens := outGridH * outGridW
	mergedDim := mergeSize * mergeSize * visionDim

	if len(patchFeatures) < gridH*gridW*visionDim {
		return nil
	}

	out := make([]float32, numMergedTokens*mergedDim)
	outTokenIdx := 0

	for mh := 0; mh < outGridH; mh++ {
		for mw := 0; mw < outGridW; mw++ {
			tokenSlice := out[outTokenIdx*mergedDim : (outTokenIdx+1)*mergedDim]
			subPatch := 0
			for dy := 0; dy < mergeSize; dy++ {
				py := mh*mergeSize + dy
				for dx := 0; dx < mergeSize; dx++ {
					px := mw*mergeSize + dx
					patchOff := (py*gridW + px) * visionDim
					copy(tokenSlice[subPatch*visionDim:(subPatch+1)*visionDim], patchFeatures[patchOff:patchOff+visionDim])
					subPatch++
				}
			}
			outTokenIdx++
		}
	}

	return out
}
