package compute

import (
	"errors"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestSmallmDispatchWindow(t *testing.T) {
	for _, tc := range []struct {
		name               string
		m, n, k, threshold int
		want               SmallmPath
	}{
		{"below", 3, 64, 1024, 4, SmallmVector}, {"threshold", 4, 64, 1024, 4, SmallmVector},
		{"above", 5, 64, 1024, 4, SmallmGeneral}, {"wide_n", 4, 65, 1024, 4, SmallmGeneral},
		{"k_tail", 4, 64, 1025, 4, SmallmGeneral}, {"larger_k", 32, 1, 2048, 32, SmallmVector},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := SelectSmallmDispatch(tc.m, tc.n, tc.k, tc.threshold)
			if err != nil || got != tc.want {
				t.Fatalf("got %v,%v want %v", got, err, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestSmallmDispatchRefuses(t *testing.T) {
	for _, shape := range [][4]int{{0, 1, 1024, 1}, {1, 0, 1024, 1}, {1, 1, 0, 1}, {1, 1, 1024, 0}, {1, 1, 1024, 33}, {-1, 1, 1024, 1}} {
		got, err := SelectSmallmDispatch(shape[0], shape[1], shape[2], shape[3])
		if got != SmallmGeneral || !errors.Is(err, ErrSmallmGeometry) {
			t.Fatalf("shape %v got %v,%v", shape, got, err)
		}
	}
}

// fak-test:runtime fast est=1ms
func TestSmallmDispatchKUint32Boundary(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("requires int wider than uint32")
	}
	largest := uint64(^uint32(0)) &^ uint64(1023)
	got, err := SelectSmallmDispatch(1, 1, int(largest), 1)
	if err != nil || got != SmallmVector {
		t.Fatalf("largest aligned K: %v,%v", got, err)
	}
	overflow := uint64(1) << 32
	got, err = SelectSmallmDispatch(1, 1, int(overflow), 1)
	if err != nil || got != SmallmGeneral {
		t.Fatalf("K truncating to zero: %v,%v", got, err)
	}
}
