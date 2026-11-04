// SPDX-License-Identifier: Apache-2.0

package model

import (
	"errors"
	"math"
	"testing"
)

// TestV41TwoPoolLayoutCheck pins the two-pool fp8 pair contract: one shared row
// index must address both pools for every row, so a deliberately shorter second
// pool is refused rather than walked off the end. Failing cases assert the
// closed sentinel; passing cases assert no error.
func TestV41TwoPoolLayoutCheck(t *testing.T) {
	if V41TwoPoolNopeRowBytes != 512 || V41TwoPoolQuantTile != 64 {
		t.Fatalf("published two-pool constants moved: nope row %d B (want 512), quant tile %d (want 64)",
			V41TwoPoolNopeRowBytes, V41TwoPoolQuantTile)
	}

	nope := func(rows int) V41TwoPool {
		return V41TwoPool{Rows: rows, RowBytes: V41TwoPoolNopeRowBytes, ElementBytes: 1, Bytes: rows * V41TwoPoolNopeRowBytes}
	}
	rope := func(rows, rowBytes, elementBytes, bytes int) *V41TwoPool {
		return &V41TwoPool{Rows: rows, RowBytes: rowBytes, ElementBytes: elementBytes, Bytes: bytes}
	}

	cases := []struct {
		name      string
		nope      V41TwoPool
		rope      *V41TwoPool
		ropeWidth int
		wantErr   bool
	}{
		{name: "valid pair", nope: nope(8), rope: rope(8, 64, 2, 8*64), ropeWidth: 64},
		{name: "empty pair", nope: nope(0), rope: rope(0, 64, 2, 0), ropeWidth: 64},
		{name: "shorter second pool", nope: nope(8), rope: rope(7, 64, 2, 7*64), ropeWidth: 64, wantErr: true},
		{name: "longer second pool", nope: nope(8), rope: rope(9, 64, 2, 9*64), ropeWidth: 64, wantErr: true},
		{name: "missing rope pool", nope: nope(8), rope: nil, ropeWidth: 64, wantErr: true},
		{
			name: "nope row not the packed width",
			nope: V41TwoPool{Rows: 8, RowBytes: 510, ElementBytes: 1, Bytes: 8 * 510},
			rope: rope(8, 64, 2, 8*64), ropeWidth: 64, wantErr: true,
		},
		{
			name: "nope element size not packed",
			nope: V41TwoPool{Rows: 8, RowBytes: 512, ElementBytes: 2, Bytes: 8 * 512},
			rope: rope(8, 64, 2, 8*64), ropeWidth: 64, wantErr: true,
		},
		{name: "rope row disagrees with declared width", nope: nope(8), rope: rope(8, 66, 2, 8*66), ropeWidth: 64, wantErr: true},
		{name: "zero declared rope width", nope: nope(8), rope: rope(8, 64, 2, 8*64), ropeWidth: 0, wantErr: true},
		{name: "rope storage too short", nope: nope(8), rope: rope(8, 64, 2, 448), ropeWidth: 64, wantErr: true},
		{name: "nope storage too long", nope: V41TwoPool{Rows: 8, RowBytes: 512, ElementBytes: 1, Bytes: 8*512 + 1}, rope: rope(8, 64, 2, 8*64), ropeWidth: 64, wantErr: true},
		{
			name: "row product overflow",
			nope: V41TwoPool{Rows: int(math.MaxInt)/V41TwoPoolNopeRowBytes + 1, RowBytes: V41TwoPoolNopeRowBytes, ElementBytes: 1, Bytes: 0},
			rope: rope(8, 64, 2, 8*64), ropeWidth: 64, wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := CheckV41TwoPoolPair(tc.nope, tc.rope, tc.ropeWidth)
			if tc.wantErr {
				if !errors.Is(err, ErrV41TwoPoolLayout) {
					t.Fatalf("CheckV41TwoPoolPair() error = %v, want ErrV41TwoPoolLayout", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("CheckV41TwoPoolPair() unexpected error = %v", err)
			}
		})
	}
}
