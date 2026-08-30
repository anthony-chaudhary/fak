// Copyright 2026 The fak Authors
// SPDX-License-Identifier: Apache-2.0

package ggufload

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

// These compact fixtures were decoded by llama.cpp's actual dequantize_row_iq*
// functions at revision 6fe74980162af0ed5e559870d5deccafaa034e7c. The raw blocks
// are built deterministically below; the expected digest covers all 256 f32 output
// bit patterns without checking in a 1 KiB golden vector per format.
func TestIQ12Q2XLReferenceVectors(t *testing.T) {
	tests := []struct {
		name   string
		typ    TensorType
		raw    []byte
		digest string
		checks map[int]uint32
	}{
		{
			name:   "IQ2_XXS",
			typ:    TensorIQ2_XXS,
			raw:    iq2XXSReferenceBlock(),
			digest: "bf2e376e329dccbeb25ea7f802bb95b06bde2539c8db5b3e75ce20b370e73fa1",
			checks: map[int]uint32{0: 0xbf800000, 32: 0x41810000, 191: 0xc2098000, 255: 0xc2a14000},
		},
		{
			name:   "IQ2_XS",
			typ:    TensorIQ2_XS,
			raw:    iq2XSReferenceBlock(),
			digest: "d414860b71ea968bb3912bce0b927394694cfd1fdd67e4b03cb4bccb0622da37",
			checks: map[int]uint32{0: 0xc0480000, 31: 0x41f80000, 128: 0xc1100000, 255: 0x41880000},
		},
		{
			name:   "IQ2_S",
			typ:    TensorIQ2_S,
			raw:    iq2SReferenceBlock(),
			digest: "3d535584e36c5c4cd863606f3546a8338dbb4b6afe693a8614b512b50409773f",
			checks: map[int]uint32{0: 0xc1f80000, 32: 0xc31be000, 127: 0x40e00000, 255: 0x42a14000},
		},
		{
			name:   "IQ1_S",
			typ:    TensorIQ1_S,
			raw:    iq1SReferenceBlock(),
			digest: "d128b9b2b2bb686a37bfa871b6d1738997a35032ac627fc1d9e0adf15736e786",
			checks: map[int]uint32{0: 0x3e000000, 32: 0x40280000, 128: 0x3f900000, 255: 0x41520000},
		},
		{
			name:   "IQ1_M",
			typ:    TensorIQ1_M,
			raw:    iq1MReferenceBlock(),
			digest: "7d6b0472f61c9d9e7d49ab8c97494e07a12ba2a795735953fd4cd62bd9b67988",
			checks: map[int]uint32{0: 0x3ec00000, 32: 0xc1460000, 128: 0x41460000, 255: 0x411a0000},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dequantF32(TensorInfo{Name: tc.name, Dims: []uint64{qkK}, Type: tc.typ}, tc.raw)
			if err != nil {
				t.Fatalf("dequantF32: %v", err)
			}
			if digest := f32BitsDigest(got); digest != tc.digest {
				t.Fatalf("f32 digest = %s, want pinned llama.cpp digest %s", digest, tc.digest)
			}
			for index, want := range tc.checks {
				if bits := math.Float32bits(got[index]); bits != want {
					t.Fatalf("out[%d] bits = %#08x (%v), want %#08x", index, bits, got[index], want)
				}
			}
		})
	}
}

func TestIQ12CanonicalEnumsNamesAndBlockSizing(t *testing.T) {
	tests := []struct {
		typ        TensorType
		id         uint32
		name       string
		blockBytes uint64
	}{
		{TensorIQ2_XXS, 16, "IQ2_XXS", 66},
		{TensorIQ2_XS, 17, "IQ2_XS", 74},
		{TensorIQ1_S, 19, "IQ1_S", 50},
		{TensorIQ2_S, 22, "IQ2_S", 82},
		{TensorIQ1_M, 29, "IQ1_M", 56},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := uint32(tc.typ); got != tc.id {
				t.Fatalf("enum ID = %d, want canonical GGML ID %d", got, tc.id)
			}
			if got := tc.typ.String(); got != tc.name {
				t.Fatalf("String() = %q, want %q", got, tc.name)
			}
			info := TensorInfo{Name: tc.name, Dims: []uint64{2 * qkK}, Type: tc.typ}
			got, err := tensorPayloadBytes(info)
			if err != nil {
				t.Fatalf("tensorPayloadBytes: %v", err)
			}
			if want := 2 * tc.blockBytes; got != want {
				t.Fatalf("tensorPayloadBytes = %d, want %d", got, want)
			}
		})
	}

	// Q2_K_XL and UD-Q2_K_XL are quantization recipes, not GGML tensor tags.
	if got := TensorType(31).String(); got != "TensorType(31)" {
		t.Fatalf("unassigned enum 31 rendered as %q, want TensorType(31)", got)
	}
}

func TestIQ12MalformedPayloadsFailClosed(t *testing.T) {
	tests := []struct {
		name       string
		typ        TensorType
		blockBytes int
	}{
		{"IQ2_XXS", TensorIQ2_XXS, blockIQ2XXSBytes},
		{"IQ2_XS", TensorIQ2_XS, blockIQ2XSBytes},
		{"IQ1_S", TensorIQ1_S, blockIQ1SBytes},
		{"IQ2_S", TensorIQ2_S, blockIQ2SBytes},
		{"IQ1_M", TensorIQ1_M, blockIQ1MBytes},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			info := TensorInfo{Name: tc.name, Dims: []uint64{qkK}, Type: tc.typ}
			if _, err := dequantF32(info, make([]byte, tc.blockBytes-1)); err == nil {
				t.Fatal("accepted short payload")
			}
			if _, err := dequantF32(info, make([]byte, tc.blockBytes+1)); err == nil {
				t.Fatal("accepted payload with trailing bytes")
			}
			badShape := TensorInfo{Name: tc.name, Dims: []uint64{qkK - 1}, Type: tc.typ}
			if _, err := tensorPayloadBytes(badShape); err == nil {
				t.Fatal("accepted non-block-multiple element count")
			}
		})
	}
}

func iq2XXSReferenceBlock() []byte {
	raw := make([]byte, blockIQ2XXSBytes)
	binary.LittleEndian.PutUint16(raw, 0x3c00)
	for ib := 0; ib < 8; ib++ {
		aux0 := uint32(uint8(ib*31)) |
			uint32(uint8(ib*31+7))<<8 |
			uint32(uint8(ib*31+13))<<16 |
			uint32(uint8(ib*31+29))<<24
		aux1 := uint32((ib*17+1)&127) |
			uint32((ib*17+9)&127)<<7 |
			uint32((ib*17+33)&127)<<14 |
			uint32((ib*17+65)&127)<<21 |
			uint32(ib)<<28
		binary.LittleEndian.PutUint32(raw[2+8*ib:], aux0)
		binary.LittleEndian.PutUint32(raw[2+8*ib+4:], aux1)
	}
	return raw
}

func iq2XSReferenceBlock() []byte {
	raw := make([]byte, blockIQ2XSBytes)
	binary.LittleEndian.PutUint16(raw, 0x3c00)
	for i := 0; i < 32; i++ {
		q := uint16((i*73+5)&511) | uint16((i*29+3)&127)<<9
		binary.LittleEndian.PutUint16(raw[2+2*i:], q)
	}
	for ib := 0; ib < 8; ib++ {
		raw[66+ib] = byte(ib | (15-ib)<<4)
	}
	return raw
}

func iq2SReferenceBlock() []byte {
	raw := make([]byte, blockIQ2SBytes)
	binary.LittleEndian.PutUint16(raw, 0x3c00)
	for i := 0; i < 32; i++ {
		raw[2+i] = byte(i*37 + 11)
		raw[34+i] = byte(i*53 + 7)
	}
	for ib := 0; ib < 8; ib++ {
		raw[66+ib] = byte(ib*29 + 3)
		raw[74+ib] = byte((15 - ib) | ib<<4)
	}
	return raw
}

func iq1SReferenceBlock() []byte {
	raw := make([]byte, blockIQ1SBytes)
	binary.LittleEndian.PutUint16(raw, 0x3c00)
	for i := 0; i < 32; i++ {
		raw[2+i] = byte(i*61 + 17)
	}
	for ib := 0; ib < 8; ib++ {
		qh := uint16((ib+1)&7) |
			uint16((ib+3)&7)<<3 |
			uint16((ib+5)&7)<<6 |
			uint16((ib+7)&7)<<9 |
			uint16(ib&7)<<12
		if ib&1 != 0 {
			qh |= 0x8000
		}
		binary.LittleEndian.PutUint16(raw[34+2*ib:], qh)
	}
	return raw
}

func iq1MReferenceBlock() []byte {
	raw := make([]byte, blockIQ1MBytes)
	for i := 0; i < 32; i++ {
		raw[i] = byte(i*47 + 23)
	}
	for ib := 0; ib < 8; ib++ {
		qh0 := byte((ib+1)&7) | byte((ib+4)&7)<<4
		if ib&1 != 0 {
			qh0 |= 0x08
		}
		if ib&2 != 0 {
			qh0 |= 0x80
		}
		qh1 := byte((ib+2)&7) | byte((ib+6)&7)<<4
		if ib&4 != 0 {
			qh1 |= 0x08
		}
		if ib&1 != 0 {
			qh1 |= 0x80
		}
		raw[32+2*ib] = qh0
		raw[32+2*ib+1] = qh1
	}
	top := [...]uint16{0x0000, 0x0000, 0xc000, 0x3000}
	for pair := 0; pair < 4; pair++ {
		low := uint16((2*pair+1)&7) |
			uint16((2*pair+3)&7)<<3 |
			uint16((2*pair+5)&7)<<6 |
			uint16((2*pair+7)&7)<<9
		binary.LittleEndian.PutUint16(raw[48+2*pair:], top[pair]|low)
	}
	return raw
}

func f32BitsDigest(values []float32) string {
	raw := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(value))
	}
	return fmt.Sprintf("%x", sha256.Sum256(raw))
}
