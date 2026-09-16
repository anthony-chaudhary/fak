package v41

import (
	"math"

	model "github.com/anthony-chaudhary/fak/internal/model"
)

// These tiny fixtures are DUPLICATED from internal/model test helpers so the
// moved v41 tests compile without importing core test code (test-only helpers
// are not importable across packages, and exporting them is forbidden).
//   - e4m3Max, encodeE4M3 <- internal/model/fp8_blockscale_parity_test.go
// Bodies are kept identical; encodeE4M3 decodes through model.FP8E4M3ToF32 so
// the encoder can never disagree with the production decoder.

// e4m3Max is the largest finite float8_e4m3fn magnitude (S.1111.110 = 448); block
// scaling maps a tile's absmax onto this so the tile uses e4m3's full dynamic range.
const e4m3Max = 448.0

// encodeE4M3 rounds one f32 to the NEAREST representable float8_e4m3fn byte by scanning
// the 254 finite codes and decoding each with the production decoder (FP8E4M3ToF32), so
// the encoder can never disagree with the decoder about what a code means. The two NaN
// codes (0x7F / 0xFF) are excluded; inputs are pre-scaled into [-448,448] by the block
// quantizer so no saturation branch is needed here. O(256) per element is irrelevant at
// test fixture sizes and buys exactness over hand-rolled exponent arithmetic.
func encodeE4M3(x float32) byte {
	best := byte(0)
	bestErr := math.Inf(1)
	for c := 0; c < 256; c++ {
		b := byte(c)
		if b&0x7f == 0x7f {
			continue // 0x7f / 0xff are the sole NaN pattern in e4m3fn
		}
		v := float64(model.FP8E4M3ToF32(b))
		e := math.Abs(v - float64(x))
		if e < bestErr {
			bestErr = e
			best = b
		}
	}
	return best
}
