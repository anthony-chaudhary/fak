package compute

import "math"

const (
	// Ink2FP8Max is the maximum finite magnitude of OCP E4M3FN (0x7E / 0xFE).
	Ink2FP8Max float32 = 448.0
	// Ink2FP8MinAbsmax floors row absmax so all-zero rows encode cleanly with scale 0.
	Ink2FP8MinAbsmax float32 = 1e-30
)

// EncodeE4M3 encodes one finite float32 to OCP E4M3FN with round-to-nearest-even
// and signed saturation at +/-448.0 (never emitting NaN 0x7F/0xFF).
func EncodeE4M3(x float32) byte {
	return siluMulFp8EncodeE4M3(x)
}

// Ink2FP8QuantizeRows quantizes row-major [rows, cols] FP32 activations to OCP
// E4M3FN codes with per-row FP32 scales (absmax / 448.0).
func Ink2FP8QuantizeRows(src []float32, rows, cols int) (codes []byte, scales []float32) {
	n := rows * cols
	codes = make([]byte, n)
	scales = make([]float32, rows)
	for r := 0; r < rows; r++ {
		row := src[r*cols : (r+1)*cols]
		var amax float32
		nonFinite := false
		for _, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				nonFinite = true
				break
			}
			if v < 0 {
				v = -v
			}
			if v > amax {
				amax = v
			}
		}
		if nonFinite || amax < Ink2FP8MinAbsmax {
			scales[r] = 0
			continue
		}
		scale := amax / Ink2FP8Max
		inv := Ink2FP8Max / amax
		scales[r] = scale
		dst := codes[r*cols : (r+1)*cols]
		for c, v := range row {
			dst[c] = EncodeE4M3(v * inv)
		}
	}
	return codes, scales
}

// Ink2FP8DequantRows dequantizes row-major [rows, cols] OCP E4M3FN codes with
// per-row FP32 scales back to FP32.
func Ink2FP8DequantRows(codes []byte, scales []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		sc := scales[r]
		if sc == 0 {
			continue
		}
		src := codes[r*cols : (r+1)*cols]
		dst := out[r*cols : (r+1)*cols]
		for c, b := range src {
			dst[c] = DecodeE4M3(b) * sc
		}
	}
	return out
}

// Ink2FP8ArenaBytes returns the 64-byte-aligned byte footprint of elemCount
// E4M3FN elements plus rowCount per-row FP32 scales.
func Ink2FP8ArenaBytes(elemCount, rowCount int64) int64 {
	codeBytes := (elemCount + 63) &^ 63
	scaleBytes := (rowCount*4 + 63) &^ 63
	return codeBytes + scaleBytes
}
