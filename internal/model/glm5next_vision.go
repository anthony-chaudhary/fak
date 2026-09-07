package model

import "math"

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

// GLM5NextVisionProjectorParams holds the weights for the vision-to-language projector MLP.
type GLM5NextVisionProjectorParams struct {
	InDim  int       // mergedDim (e.g. 4096)
	OutDim int       // hiddenSize (e.g. 4096)
	W1     []float32 // [OutDim * InDim]
	B1     []float32 // [OutDim]
	W2     []float32 // [OutDim * OutDim]
	B2     []float32 // [OutDim]
}

// NewGLM5NextVisionProjectorParams initializes a deterministic vision projector.
func NewGLM5NextVisionProjectorParams(inDim, outDim int) *GLM5NextVisionProjectorParams {
	p := &GLM5NextVisionProjectorParams{
		InDim:  inDim,
		OutDim: outDim,
		W1:     make([]float32, outDim*inDim),
		B1:     make([]float32, outDim),
		W2:     make([]float32, outDim*outDim),
		B2:     make([]float32, outDim),
	}
	for i := range p.W1 {
		p.W1[i] = float32((i%13)-6) * 0.005
	}
	for i := range p.W2 {
		p.W2[i] = float32((i%11)-5) * 0.005
	}
	return p
}

// ProjectGLM5NextVisionTokens projects merged 2x2 vision tokens into the language model's hidden dimension.
// Applies W1 -> GELU -> W2.
func ProjectGLM5NextVisionTokens(mergedTokens []float32, numTokens int, params *GLM5NextVisionProjectorParams) []float32 {
	if params == nil || numTokens <= 0 || len(mergedTokens) < numTokens*params.InDim {
		return nil
	}
	inDim := params.InDim
	outDim := params.OutDim
	out := make([]float32, numTokens*outDim)
	hidden := make([]float32, outDim)

	for t := 0; t < numTokens; t++ {
		inSlice := mergedTokens[t*inDim : (t+1)*inDim]
		outSlice := out[t*outDim : (t+1)*outDim]

		// Layer 1: hidden = GELU(W1 * in + B1)
		for o := 0; o < outDim; o++ {
			sum := params.B1[o]
			rowOff := o * inDim
			for i := 0; i < inDim; i++ {
				sum += params.W1[rowOff+i] * inSlice[i]
			}
			x := float64(sum)
			tanhArg := 0.7978845608 * (x + 0.044715*x*x*x)
			if tanhArg > 20.0 {
				tanhArg = 20.0
			} else if tanhArg < -20.0 {
				tanhArg = -20.0
			}
			hidden[o] = float32(0.5 * x * (1.0 + (math.Exp(2*tanhArg)-1.0)/(math.Exp(2*tanhArg)+1.0)))
		}

		// Layer 2: out = W2 * hidden + B2
		for o := 0; o < outDim; o++ {
			sum := params.B2[o]
			rowOff := o * outDim
			for i := 0; i < outDim; i++ {
				sum += params.W2[rowOff+i] * hidden[i]
			}
			outSlice[o] = sum
		}
	}

	return out
}
