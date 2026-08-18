// Package kquantbits decodes bit-packed k-quant metadata.
package kquantbits

// ScaleMinK4 unpacks the j-th 6-bit scale and minimum pair from a 12-byte scales field.
func ScaleMinK4(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}
