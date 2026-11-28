// Package kquantbits decodes bit-packed k-quant metadata.
package kquantbits

import (
	"errors"
	"fmt"
)

// ScaleMinK4 unpacks the j-th 6-bit scale and minimum pair from a 12-byte scales field.
func ScaleMinK4(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// NUQCodebookMeta describes a non-uniform quantization (NUQ) lookup-table
// codebook with sensitivity-weighted signposts (kvquant-2).
type NUQCodebookMeta struct {
	BitWidth  int
	Signposts []float32
}

// DecodeNUQChannel decodes quantized codes through the per-channel NUQ LUT
// signposts scaled by (scale, zero).
func DecodeNUQChannel(codes []uint8, meta NUQCodebookMeta, scale, zero float32) ([]float32, error) {
	if meta.BitWidth <= 0 || meta.BitWidth > 8 {
		return nil, fmt.Errorf("kquantbits: invalid NUQ bit width %d", meta.BitWidth)
	}
	wantLen := 1 << meta.BitWidth
	if len(meta.Signposts) != wantLen {
		return nil, fmt.Errorf("kquantbits: NUQ codebook has %d signposts, want %d", len(meta.Signposts), wantLen)
	}
	if scale == 0 {
		return nil, errors.New("kquantbits: NUQ channel scale must be non-zero")
	}
	out := make([]float32, len(codes))
	for i, c := range codes {
		if int(c) >= wantLen {
			return nil, fmt.Errorf("kquantbits: NUQ code %d out of range [0,%d)", c, wantLen)
		}
		out[i] = meta.Signposts[c]*scale + zero
	}
	return out, nil
}
