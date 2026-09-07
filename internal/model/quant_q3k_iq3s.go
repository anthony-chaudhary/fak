package model

import (
	"encoding/binary"
	"math"
)

// Shared scalar GGUF decoders keep load-time expansion and resident native math identical.
// Layout oracle: ggml-org/llama.cpp@6fe74980162af0ed5e559870d5deccafaa034e7c,
// ggml/src/ggml-quants.c dequantize_row_q3_K and dequantize_row_iq3_s (MIT).

// DequantQ3K expands complete 256-weight/110-byte GGUF Q3_K blocks into out.
// Callers validate payload geometry before dispatch, as for the other raw quant decoders.
func DequantQ3K(out []float32, raw []byte) {
	for i := 0; i < len(out); i += qkK {
		q3kDequantSuperBlock(out[i:i+qkK], raw[:q3kBlockBytes])
		raw = raw[q3kBlockBytes:]
	}
}

func q3kDequantSuperBlock(out []float32, raw []byte) {
	hmask := raw[:qkK/8]
	q := raw[qkK/8 : qkK/8+qkK/4]
	scales := unpackQ3KScales(raw[qkK/8+qkK/4 : qkK/8+qkK/4+12])
	d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[q3kBlockBytes-2:])))
	qi := 0
	is := 0
	mask := byte(1)
	for n := 0; n < qkK; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			dl := d * float32(scales[is]-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+l] >> shift) & 3)
				if hmask[l]&mask == 0 {
					code -= 4
				}
				out[n+j*32+l] = dl * float32(code)
			}

			dl = d * float32(scales[is]-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+16+l] >> shift) & 3)
				if hmask[16+l]&mask == 0 {
					code -= 4
				}
				out[n+j*32+16+l] = dl * float32(code)
			}
			shift += 2
			mask <<= 1
		}
		qi += 32
	}
}

func unpackQ3KScales(raw []byte) [16]int8 {
	const (
		kmask1 = uint32(0x03030303)
		kmask2 = uint32(0x0f0f0f0f)
	)
	aux0 := binary.LittleEndian.Uint32(raw[0:4])
	aux1 := binary.LittleEndian.Uint32(raw[4:8])
	aux2 := binary.LittleEndian.Uint32(raw[8:12])
	tmp := aux2
	words := [4]uint32{
		(aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4),
		(aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4),
		((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
		((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
	}
	var scales [16]int8
	for i, word := range words {
		for j := 0; j < 4; j++ {
			scales[i*4+j] = int8(byte(word >> (8 * j)))
		}
	}
	return scales
}

// DequantIQ3S expands complete 256-weight/110-byte GGUF IQ3_S blocks into out.
// Callers validate payload geometry before dispatch.
func DequantIQ3S(out []float32, raw []byte) {
	for i := 0; i < len(out); i += qkK {
		iq3sDequantSuperBlock(out[i:i+qkK], raw[:iq3sBlockBytes])
		raw = raw[iq3sBlockBytes:]
	}
}

func iq3sDequantSuperBlock(out []float32, raw []byte) {
	const (
		qsOffset     = 2
		qhOffset     = 66
		signsOffset  = 74
		scalesOffset = 106
	)

	d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(raw)))
	qs := raw[qsOffset:qhOffset]
	qh := raw[qhOffset:signsOffset]
	signs := raw[signsOffset:scalesOffset]
	scales := raw[scalesOffset:iq3sBlockBytes]
	for pair := 0; pair < 4; pair++ {
		scaleByte := scales[pair]
		for half := 0; half < 2; half++ {
			ib32 := pair*2 + half
			nibble := scaleByte & 0x0f
			if half != 0 {
				nibble = scaleByte >> 4
			}
			db := d * float32(1+2*int(nibble))
			qBase := ib32 * 8
			high := qh[ib32]
			signBase := ib32 * 4
			outBase := ib32 * 32
			for group := 0; group < 4; group++ {
				idx1 := uint16(qs[qBase+2*group]) | (uint16(high)<<uint(8-2*group))&0x100
				idx2 := uint16(qs[qBase+2*group+1]) | (uint16(high)<<uint(7-2*group))&0x100
				grid1, grid2 := iq3SGrid[idx1], iq3SGrid[idx2]
				sign := signs[signBase+group]
				for j := 0; j < 4; j++ {
					v1 := float32(byte(grid1 >> uint(8*j)))
					v2 := float32(byte(grid2 >> uint(8*j)))
					if sign&(1<<uint(j)) != 0 {
						v1 = -v1
					}
					if sign&(1<<uint(j+4)) != 0 {
						v2 = -v2
					}
					out[outBase+group*8+j] = db * v1
					out[outBase+group*8+j+4] = db * v2
				}
			}
		}
	}
}
