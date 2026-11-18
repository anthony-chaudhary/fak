// Package quantmatrix — ModelOpt NVFP4 / W4A16 recipe semantics.
//
// This file is an INSPIRE-ONLY, recipe-semantics-only model of the NVIDIA
// Model-Optimizer NVFP4 / W4A16 quantization recipes. It pins the exact
// scale/exponent layout and late-tensor-scale semantics that fak's existing
// NVFP4 CPU backend queue items assume, and records the divergence between the
// two pinned references. It is a pure description: no checkpoint conversion
// runtime, no GPU, no network, no accuracy claim.
//
// Pinned references:
//   - NVIDIA/Model-Optimizer @ 91e31320a6853eceb16525bd09302fc404ca8e7d
//     (Apache-2.0): the quantization recipe — a packed E2M1 payload, one
//     E4M3FN block scale per 16 weights, one FP32 global (tensor) scale, and
//     the "early" decode that folds the global scale into every weight.
//   - flashinfer-ai/flashinfer @ 866c80d0c70a90624b4c2056d12d1543d709609e
//     (Apache-2.0): the W4A16 consumption recipe — BF16-rounded activations,
//     exact local E2M1*E4M3 products, FP32 serial accumulation, and the global
//     scale applied ONCE at the epilogue ("late" tensor scaling).
package quantmatrix

import (
	"errors"
	"fmt"
	"math"
)

// Pinned upstream revisions the recipe is studied against.
const (
	modeloptNVFP4Source   = "NVIDIA/Model-Optimizer@91e31320a6853eceb16525bd09302fc404ca8e7d"
	flashinferW4A16Source = "flashinfer-ai/flashinfer@866c80d0c70a90624b4c2056d12d1543d709609e"
)

// Fail-closed refusal reasons for the recipe model.
var (
	ErrRecipeUnknown   = errors.New("quantmatrix: unknown quantization recipe")
	ErrRecipeShape     = errors.New("quantmatrix: invalid quantization recipe shape")
	ErrRecipeScale     = errors.New("quantmatrix: invalid quantization recipe scale")
	ErrRecipeInput     = errors.New("quantmatrix: invalid quantization recipe input")
	ErrRecipePlacement = errors.New("quantmatrix: unsupported scale placement")
)

// e2m1Values is the OCP MX E2M1 finite value table indexed by one unpacked
// nibble: bit 3 is the sign, bits 2:0 the magnitude. It is byte-identical to
// the table the NVFP4 CPU backend consumes, so a recipe computed here decodes
// to the same weights the backend would.
var e2m1Values = [16]float32{
	0, 0.5, 1, 1.5, 2, 3, 4, 6,
	float32(math.Copysign(0, -1)), -0.5, -1, -1.5, -2, -3, -4, -6,
}

// decodeE4M3FN decodes one OCP float8_e4m3fn byte exactly into float32. It is
// byte-identical to internal/mathx.DecodeE4M3 so the recipe model and the
// backend agree on every block scale, including the reserved NaN byte 0x7f.
func decodeE4M3FN(b byte) float32 {
	sign := 1.0
	if b&0x80 != 0 {
		sign = -1
	}
	exp, mantissa := int(b>>3)&0x0f, int(b)&0x07
	switch {
	case exp == 0x0f && mantissa == 0x07:
		return float32(math.NaN())
	case exp == 0:
		return float32(sign * math.Ldexp(float64(mantissa)/8, 1-7))
	default:
		return float32(sign * math.Ldexp(1+float64(mantissa)/8, exp-7))
	}
}

// roundBF16RNE rounds a finite FP32 value to BF16 round-to-nearest-even,
// matching the activation staging the W4A16 backend performs before the local
// weight product.
func roundBF16RNE(value float32) float32 {
	bits := math.Float32bits(value)
	bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
	return math.Float32frombits(bits)
}

func finiteF32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

// NVFP4Layout pins the exact scale/exponent layout the NVFP4 CPU backend queue
// items assume. It is a pure description: it names the payload, the block-scale
// dtype and stride, the per-tensor global-scale dtype and count, and the device
// swizzle geometry. It does not encode or decode any value.
type NVFP4Layout struct {
	// WeightBits is the bits per weight (E2M1).
	WeightBits int
	// PackedPerByte is the weights packed per payload byte (two E2M1 nibbles).
	PackedPerByte int
	// LowNibbleIsEvenK records that even K occupies the low nibble of a byte.
	LowNibbleIsEvenK bool
	// BlockSize is the number of consecutive K values sharing one block scale.
	BlockSize int
	// BlockScaleBits is the block-scale dtype (E4M3FN, 8 bits).
	BlockScaleBits int
	// BlockScalePerBlock is the block scales per block (exactly one).
	BlockScalePerBlock int
	// GlobalScaleBits is the global (tensor) scale dtype (FP32).
	GlobalScaleBits int
	// GlobalScaleCount is the global scales per tensor (exactly one: the
	// per-tensor scale placement).
	GlobalScaleCount int
	// SwizzleTileRows / SwizzleTileCols pin the device 128x4 scale tiling.
	SwizzleTileRows int
	SwizzleTileCols int
}

// NVFP4Layout returns the pinned NVFP4 scale/exponent layout.
func NVFP4LayoutSpec() NVFP4Layout {
	return NVFP4Layout{
		WeightBits:         4,
		PackedPerByte:      2,
		LowNibbleIsEvenK:   true,
		BlockSize:          16,
		BlockScaleBits:     8,
		BlockScalePerBlock: 1,
		GlobalScaleBits:    32,
		GlobalScaleCount:   1,
		SwizzleTileRows:    128,
		SwizzleTileCols:    4,
	}
}

// NVFP4PayloadShape returns the packed-weight bytes, the linear block-scale
// bytes, and the swizzled block-scale bytes for a row-major [rows, cols] NVFP4
// weight. It refuses a shape that is not positive or whose K is not a multiple
// of the block size, and a count that overflows int.
func NVFP4PayloadShape(rows, cols int) (packedBytes, scaleBytes, swizzledBytes int, err error) {
	lay := NVFP4LayoutSpec()
	if rows <= 0 || cols <= 0 {
		return 0, 0, 0, fmt.Errorf("%w: [%d %d] needs positive dimensions", ErrRecipeShape, rows, cols)
	}
	if cols%lay.BlockSize != 0 {
		return 0, 0, 0, fmt.Errorf("%w: K=%d is not a multiple of block %d", ErrRecipeShape, cols, lay.BlockSize)
	}
	elements, ok := mulInt(rows, cols)
	if !ok {
		return 0, 0, 0, fmt.Errorf("%w: [%d %d] overflows element count", ErrRecipeShape, rows, cols)
	}
	packedBytes = elements / lay.PackedPerByte
	scaleCols := cols / lay.BlockSize
	scaleCount, ok := mulInt(rows, scaleCols)
	if !ok {
		return 0, 0, 0, fmt.Errorf("%w: [%d %d] overflows block-scale count", ErrRecipeShape, rows, cols)
	}
	scaleBytes = scaleCount * lay.BlockScalePerBlock
	// Swizzle: pad rows to a multiple of 128 and scale columns to a multiple of 4.
	paddedRows := ((rows + lay.SwizzleTileRows - 1) / lay.SwizzleTileRows) * lay.SwizzleTileRows
	paddedCols := ((scaleCols + lay.SwizzleTileCols - 1) / lay.SwizzleTileCols) * lay.SwizzleTileCols
	swizzled, ok := mulInt(paddedRows, paddedCols)
	if !ok {
		return 0, 0, 0, fmt.Errorf("%w: [%d %d] overflows swizzled scale layout", ErrRecipeShape, rows, cols)
	}
	return packedBytes, scaleBytes, swizzled, nil
}

// ScalePlacement names where the FP32 global (tensor) scale is applied.
type ScalePlacement string

const (
	// ScaleLate applies the global scale once to the accumulated sum (the
	// FlashInfer W4A16 epilogue).
	ScaleLate ScalePlacement = "late"
	// ScaleEarly folds the global scale into every local weight before
	// accumulation (the Model-Optimizer reference decode).
	ScaleEarly ScalePlacement = "early"
)

// RecipeID names a quantization recipe.
type RecipeID string

const (
	// RecipeNVFP4W4A16 is the NVFP4 weight-only-4-bit, 16-bit-activation
	// recipe consumed by fak's NVFP4 CPU backend.
	RecipeNVFP4W4A16 RecipeID = "recipe.modelopt.nvfp4.w4a16.v1"
)

// Recipe describes a pinned quantization recipe.
type Recipe struct {
	ID             RecipeID
	Source         string
	WeightBits     int
	ActBits        int
	Layout         NVFP4Layout
	ScalePlacement ScalePlacement
}

// recipes is the pinned recipe registry.
var recipes = map[RecipeID]Recipe{
	RecipeNVFP4W4A16: {
		ID:             RecipeNVFP4W4A16,
		Source:         flashinferW4A16Source,
		WeightBits:     4,
		ActBits:        16,
		Layout:         NVFP4LayoutSpec(),
		ScalePlacement: ScaleLate,
	},
}

// RecipeRequest asks to resolve a quantization recipe for a given shape and
// scale placement.
type RecipeRequest struct {
	ID             RecipeID
	Rows, Cols     int
	ScalePlacement ScalePlacement
}

// ResolvedRecipe is the pinned, validated recipe for a request.
type ResolvedRecipe struct {
	Recipe
	Layout        NVFP4Layout
	PackedBytes   int
	ScaleBytes    int
	SwizzledBytes int
}

// ResolveRecipe returns the pinned recipe for req, refusing fail-closed when
// the recipe is unknown, the scale placement is unsupported, or the shape is
// invalid. It never guesses a replacement recipe or layout.
func ResolveRecipe(req RecipeRequest) (ResolvedRecipe, error) {
	recipe, ok := recipes[req.ID]
	if !ok {
		return ResolvedRecipe{}, fmt.Errorf("%w: %s", ErrRecipeUnknown, req.ID)
	}
	if req.ScalePlacement != recipe.ScalePlacement {
		return ResolvedRecipe{}, fmt.Errorf("%w: %s wants %s, got %s", ErrRecipePlacement, req.ID, recipe.ScalePlacement, req.ScalePlacement)
	}
	packed, scale, swizzled, err := NVFP4PayloadShape(req.Rows, req.Cols)
	if err != nil {
		return ResolvedRecipe{}, err
	}
	return ResolvedRecipe{
		Recipe:        recipe,
		Layout:        recipe.Layout,
		PackedBytes:   packed,
		ScaleBytes:    scale,
		SwizzledBytes: swizzled,
	}, nil
}

// validateW4A16 checks the host-visible W4A16 contract before any product runs:
// K is a positive multiple of the block size, the packed payload and block
// scales have the exact byte counts, the global scale is finite and
// non-negative, every block scale decodes to a finite non-negative value, and
// every activation is finite and survives BF16 rounding.
func validateW4A16(packed, blockScales []byte, globalScale float32, x []float32) (int, error) {
	cols := len(x)
	if cols <= 0 || cols%NVFP4LayoutSpec().BlockSize != 0 {
		return 0, fmt.Errorf("%w: K=%d must be positive and a multiple of 16", ErrRecipeShape, cols)
	}
	if len(packed) != cols/2 {
		return 0, fmt.Errorf("%w: packed has %d bytes, want %d", ErrRecipeShape, len(packed), cols/2)
	}
	if len(blockScales) != cols/NVFP4LayoutSpec().BlockSize {
		return 0, fmt.Errorf("%w: block scales have %d bytes, want %d", ErrRecipeShape, len(blockScales), cols/NVFP4LayoutSpec().BlockSize)
	}
	if !finiteF32(globalScale) || globalScale < 0 {
		return 0, fmt.Errorf("%w: global scale must be finite and non-negative", ErrRecipeScale)
	}
	for i, raw := range blockScales {
		value := decodeE4M3FN(raw)
		if !finiteF32(value) || value < 0 {
			return 0, fmt.Errorf("%w: block scale %d must be finite and non-negative", ErrRecipeScale, i)
		}
	}
	for i, value := range x {
		if !finiteF32(value) {
			return 0, fmt.Errorf("%w: activation %d is non-finite", ErrRecipeInput, i)
		}
		if !finiteF32(roundBF16RNE(value)) {
			return 0, fmt.Errorf("%w: activation %d overflows BF16", ErrRecipeInput, i)
		}
	}
	return cols, nil
}

// w4a16LocalProduct computes one output row of a W4A16 dot product. Each
// activation is rounded to BF16 RNE, each local weight is the exact
// E2M1*E4M3 product, and the products accumulate serially in FP32. When late is
// set the global scale is applied once to the finished sum (FlashInfer W4A16
// epilogue); otherwise it is folded into every local weight before
// accumulation (Model-Optimizer reference decode).
func w4a16LocalProduct(packed, blockScales []byte, globalScale float32, x []float32, late bool) (float32, error) {
	cols, err := validateW4A16(packed, blockScales, globalScale, x)
	if err != nil {
		return 0, err
	}
	blocks := cols / NVFP4LayoutSpec().BlockSize
	var sum float32
	for block := 0; block < blocks; block++ {
		scale := decodeE4M3FN(blockScales[block])
		for j := 0; j < NVFP4LayoutSpec().BlockSize; j++ {
			k := block*NVFP4LayoutSpec().BlockSize + j
			pair := packed[k/2]
			code := pair & 0x0f
			if k&1 != 0 {
				code = pair >> 4
			}
			weight := e2m1Values[code] * scale
			if !late {
				weight *= globalScale
			}
			sum += weight * roundBF16RNE(x[k])
			if !finiteF32(sum) {
				return 0, fmt.Errorf("%w: FP32 accumulation overflow at K %d", ErrRecipeInput, k)
			}
		}
	}
	if late {
		sum *= globalScale
	}
	if !finiteF32(sum) {
		return 0, fmt.Errorf("%w: FP32 epilogue overflow", ErrRecipeInput)
	}
	return sum, nil
}

// W4A16Product computes one output row of a W4A16 (weight-4-bit,
// activation-16-bit) dot product under the FlashInfer late-tensor-scale
// recipe: the single FP32 global scale is applied ONCE to the accumulated sum.
// packed holds the E2M1 payload (even K in the low nibble) and blockScales holds
// one E4M3FN byte per 16 K values.
func W4A16Product(packed, blockScales []byte, globalScale float32, x []float32) (float32, error) {
	return w4a16LocalProduct(packed, blockScales, globalScale, x, true)
}

// W4A16ProductEarlyScale computes the same W4A16 dot product under the
// Model-Optimizer "early" decode: the global scale is folded into every local
// weight before accumulation instead of applied once at the epilogue.
func W4A16ProductEarlyScale(packed, blockScales []byte, globalScale float32, x []float32) (float32, error) {
	return w4a16LocalProduct(packed, blockScales, globalScale, x, false)
}

// W4A16ScaleDivergence computes the late and early W4A16 products for the same
// payload and activations and reports whether they differ. Both round
// activations to BF16 RNE, so any difference is the scale placement alone: late
// tensor scaling is not numerically identical to scaling every weight.
func W4A16ScaleDivergence(packed, blockScales []byte, globalScale float32, x []float32) (late, early float32, diverges bool, err error) {
	late, err = W4A16Product(packed, blockScales, globalScale, x)
	if err != nil {
		return 0, 0, false, err
	}
	early, err = W4A16ProductEarlyScale(packed, blockScales, globalScale, x)
	if err != nil {
		return 0, 0, false, err
	}
	return late, early, late != early, nil
}

func mulInt(a, b int) (int, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	maxInt := int(^uint(0) >> 1)
	if a > 0 && b > 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}
