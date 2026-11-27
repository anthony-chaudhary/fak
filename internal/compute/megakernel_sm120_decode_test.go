package compute

import (
	"errors"
	"testing"
)

// Closed in-memory oracle for the sm_120a persistent decode structure (SamuelReeder/megakernels,
// ADAPT). [SW-VERIFIED] structure and oracle only: no hardware, no tuning, no speed claim.
//
// A producer fills ring slots with int8 weight rows; each slot carries a sequence flag IN the data
// (flag-in-data handoff) and the consumer dequantizes fused with the dot product, accepting a slot
// only when its flag equals the expected sequence number.
type mk120Slot struct {
	seq   int32 // flag in data: sequence number the producer wrote last
	row   [8]int8
	scale float32
}

var errMK120Stale = errors.New("megakernel sm120: slot flag does not match expected sequence (stale or unpublished)")

func mk120Dot(s *mk120Slot, want int32, x []float32) (float32, error) {
	if s.seq != want {
		return 0, errMK120Stale
	}
	var acc float32
	for i, w := range s.row { // fused dequant: int8 * scale applied inside the accumulate
		acc += float32(w) * s.scale * x[i]
	}
	return acc, nil
}

func TestMegakernelSM120Decode(t *testing.T) {
	ring := cuhDefine(t, "megakernel_sm120_decode.cuh", "FAK_MK120_RING_SLOTS")
	width := cuhDefine(t, "megakernel_sm120_decode.cuh", "FAK_MK120_ROW_WIDTH")
	if cuhDefine(t, "megakernel_sm120_decode.cuh", "FAK_MK120_CTAS_PER_SM") != 1 || width != len(mk120Slot{}.row) {
		t.Fatal("header constants drifted from the oracle fixture")
	}
	rows := [][8]int8{{1, -2, 3, -4, 5, -6, 7, -8}, {8, 7, 6, 5, 4, 3, 2, 1}, {-1, -1, -1, -1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 127}, {9, 9, 9, 9, 9, 9, 9, 9}, {2, 4, 6, 8, -2, -4, -6, -8}}
	scales := []float32{0.5, 0.25, 1, 0.125, 0.0625, 2}
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	ref := func(r int) float32 { // dequantize first, then dot, in the same accumulation order
		var acc float32
		for i, w := range rows[r] {
			acc += float32(w) * scales[r] * x[i]
		}
		return acc
	}
	t.Run("RingHandoffMatchesDequantThenDot", func(t *testing.T) {
		slots := make([]mk120Slot, ring)
		for r := range rows { // producer wraps the ring; consumer follows with sequence r
			s := &slots[r%ring]
			s.row, s.scale, s.seq = rows[r], scales[r], int32(r+1)
			got, err := mk120Dot(s, int32(r+1), x)
			if err != nil {
				t.Fatalf("row %d: %v", r, err)
			}
			if got != ref(r) {
				t.Fatalf("row %d = %v, want %v", r, got, ref(r))
			}
		}
	})
	t.Run("StaleSlotRefusedNotConsumed", func(t *testing.T) {
		slots := make([]mk120Slot, ring)
		slots[0] = mk120Slot{seq: 1, row: rows[0], scale: scales[0]}
		// Consumer lapped the producer: it expects the wrap-around sequence ring+1 but slot 0 still holds 1.
		if _, err := mk120Dot(&slots[0], int32(ring+1), x); !errors.Is(err, errMK120Stale) {
			t.Fatalf("err = %v, want errMK120Stale", err)
		}
		// An unpublished slot (zero flag) is also refused for any real sequence.
		if _, err := mk120Dot(&slots[1], 1, x); !errors.Is(err, errMK120Stale) {
			t.Fatalf("unpublished err = %v, want errMK120Stale", err)
		}
	})
}
