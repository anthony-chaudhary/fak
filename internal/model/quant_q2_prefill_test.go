package model

import (
	"encoding/binary"
	"math"
	"slices"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestQ2PrefillBlockReuseMatchesTokenRows(t *testing.T) {
	const out, in = 9, 512
	raw := make([]byte, out*(in/qBlk2G128)*q2G128BlockBytes)
	lcgBytes(raw, 0x5145045)
	for off := 0; off < len(raw); off += q2G128BlockBytes {
		binary.LittleEndian.PutUint16(raw[off:], uint16(0x1800+(off/q2G128BlockBytes)%4*0x400))
	}
	weights := make([]float32, out*in)
	for i := range weights {
		weights[i] = float32(i%31-15) / 23
	}
	for name, qt := range map[string]*q2Tensor{
		"g128": wrapQ2G128FromRaw(raw, out, in),
		"g32":  quantizeQ2(weights, out, in),
	} {
		t.Run(name, func(t *testing.T) {
			rawBefore := append([]byte(nil), qt.raw...)
			qBefore := append([]byte(nil), qt.q...)
			dBefore := append([]float32(nil), qt.d...)
			for _, rows := range []int{1, 3, q2PrefillTileRows + 3} {
				X := make([]float32, rows*in)
				for i := range X {
					X[i] = float32((i*17)%43-21) / 19
				}
				xBefore := append([]float32(nil), X...)
				want := make([]float32, rows*out)
				for tok := 0; tok < rows; tok++ {
					q2MatRowsRange(qt, X[tok*in:(tok+1)*in], want[tok*out:(tok+1)*out], 0, out)
				}
				got := q2MatRowsBatch(qt, X, rows)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("rows=%d output[%d]=%v, per-token=%v", rows, i, got[i], want[i])
					}
				}
				if !slices.Equal(X, xBefore) || !slices.Equal(qt.raw, rawBefore) || !slices.Equal(qt.q, qBefore) || !slices.Equal(qt.d, dBefore) {
					t.Fatal("prefill modified immutable weights or input")
				}
			}
		})
	}
}
