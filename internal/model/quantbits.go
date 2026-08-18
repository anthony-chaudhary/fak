package model

import "github.com/anthony-chaudhary/fak/internal/kquantbits"

// quantbits.go is the shared home for the low-level numeric bit-twiddling that the
// k-quant (Q4_K/Q5_K/…) dequant paths depend on. These helpers MUST be bit-exact:
// the resident decode path here and the ggufload reference loader both call them, so any
// drift would silently desync the two dequant results. They lived as byte-identical
// copies in internal/model and internal/ggufload; this file is the single canonical
// definition, exported so ggufload (which already imports internal/model) can share it.

// F16BitsToF32Bits converts an IEEE binary16 bit pattern to the binary32 bit pattern.
// It handles the three half-float regimes — subnormal/zero (exp==0), inf/NaN (exp==0x1f),
// and the normal range — by re-biasing the exponent and shifting the 10-bit mantissa into
// the 23-bit f32 field, preserving the sign. The result is the bit pattern, not the float;
// callers wrap it in math.Float32frombits.
func F16BitsToF32Bits(h uint16) uint32 { return kquantbits.F16BitsToF32Bits(h) }

// GetScaleMinK4 unpacks the j-th (scale, min) 6-bit pair from the 12-byte scales field of a
// k-quant super-block. The 8 pairs are packed across 12 bytes: pairs 0..3 take their low 6
// bits straight from q[j]/q[j+4], while pairs 4..7 splice the high 2 bits of the lower bytes
// into the top of q[j+4]. This is the 6-bit packing every k-quant variant shares.
func GetScaleMinK4(j int, q []byte) (scale, min uint8) { return kquantbits.ScaleMinK4(j, q) }
