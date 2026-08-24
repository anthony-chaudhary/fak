// Copyright 2026 The fak Authors
// SPDX-License-Identifier: Apache-2.0

package ggufload

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDequantIQ3SMatchesReferenceLayout(t *testing.T) {
	raw := make([]byte, blockIQ3SBytes)
	binary.LittleEndian.PutUint16(raw, 0x3c00) // f16(1)
	// Low indices 0 and 1 select packed grids [1,1,1,1] and [3,1,1,1].
	raw[2] = 0
	raw[3] = 1
	// Set the high index bit for the first grid in the second 32-value group.
	raw[2+64+1] = 1
	// Negate outputs 0 and 7 in the first eight-value group.
	raw[2+64+8] = 0x81
	// First pair subscales: db1=3, db2=5. Remaining groups use scale 1.
	raw[2+64+8+32] = 0x21

	tns := TensorInfo{Name: "iq3s.test", Dims: []uint64{qkIQ3S}, Type: TensorIQ3_S}
	got, err := dequantF32(tns, raw)
	if err != nil {
		t.Fatalf("dequantF32 IQ3_S: %v", err)
	}
	wantFirst := []float32{-3, 3, 3, 3, 9, 3, 3, -3}
	for i, want := range wantFirst {
		if got[i] != want {
			t.Fatalf("got[%d]=%v, want %v (first grid/sign/scale layout)", i, got[i], want)
		}
	}
	// qh[1] bit 0 promotes group 1's first index from 0 to 256.
	grid := iq3SGrid[256]
	for j := 0; j < 4; j++ {
		want := 5 * float32(byte(grid>>uint(8*j)))
		if got[32+j] != want {
			t.Fatalf("got[%d]=%v, want %v (high index layout)", 32+j, got[32+j], want)
		}
	}
}

func TestIQ3SPayloadContract(t *testing.T) {
	tns := TensorInfo{Name: "iq3s.test", Dims: []uint64{qkIQ3S}, Type: TensorIQ3_S}
	if got, err := tensorPayloadBytes(tns); err != nil || got != blockIQ3SBytes {
		t.Fatalf("tensorPayloadBytes IQ3_S = %d, %v; want %d", got, err, blockIQ3SBytes)
	}
	if _, err := dequantF32(tns, make([]byte, blockIQ3SBytes-1)); err == nil {
		t.Fatal("dequantF32 IQ3_S accepted a short payload")
	}
	bad := TensorInfo{Name: "iq3s.bad", Dims: []uint64{qkIQ3S - 1}, Type: TensorIQ3_S}
	if _, err := tensorPayloadBytes(bad); err == nil {
		t.Fatal("tensorPayloadBytes IQ3_S accepted a non-block element count")
	}
	if got := TensorIQ3_S.String(); got != "IQ3_S" {
		t.Fatalf("TensorIQ3_S.String()=%q, want IQ3_S", got)
	}
}

func TestIQ3SScaleIsFiniteForAllNibbles(t *testing.T) {
	raw := make([]byte, blockIQ3SBytes)
	binary.LittleEndian.PutUint16(raw, 0x3c00)
	for i := 0; i < 4; i++ {
		raw[106+i] = 0xff
	}
	out := make([]float32, qkIQ3S)
	dequantIQ3S(out, raw)
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("out[%d]=%v, want finite", i, v)
		}
	}
}
