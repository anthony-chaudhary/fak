package kvbudget

import (
	"errors"
	"math"
	"testing"
)

// Software arithmetic only: this helper has no production graph-probe caller.
// fak-test:runtime fast est=10ms lane=default
func TestGraphMemFootprintOverflow(t *testing.T) {
	cases := []struct {
		name    string
		probe   GraphMemProbe
		buckets int
		want    uint64
	}{
		{"addition overflow", GraphMemProbe{Deltas: []uint64{1 << 63, 1 << 63}}, 2, math.MaxUint64},
		{"multiplication overflow", GraphMemProbe{Deltas: []uint64{0, 1 << 63}}, 3, math.MaxUint64},
		{"exact maximum", GraphMemProbe{Deltas: []uint64{math.MaxUint64 - MinPerGraphBytes, MinPerGraphBytes}}, 2, math.MaxUint64},
		{"below maximum", GraphMemProbe{Deltas: []uint64{math.MaxUint64 - MinPerGraphBytes - 1, MinPerGraphBytes}}, 2, math.MaxUint64 - 1},
		{"single sample unchanged", GraphMemProbe{Deltas: []uint64{17}}, 9, 17},
		{"one bucket unchanged", GraphMemProbe{Deltas: []uint64{17, math.MaxUint64}}, 1, 17},
		{"empty unchanged", GraphMemProbe{}, 9, 0},
		{"negative buckets unchanged", GraphMemProbe{Deltas: []uint64{math.MaxUint64}}, -1, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.probe.ExtrapolatedPoolFootprint(tc.buckets); got != tc.want {
				t.Fatalf("got %d, want %d", got, tc.want)
			}
		})
	}
	reserve := (GraphMemProbe{Deltas: []uint64{1 << 63, 1 << 63}}).ExtrapolatedPoolFootprint(2)
	if got, err := DeriveMaxBlocksWithGraphReserve(1<<30, 0, reserve, 1, 1<<20); got != 0 || !errors.Is(err, ErrNonKVExceedsBudget) {
		t.Fatalf("overflowed footprint admitted blocks: %d, %v", got, err)
	}
	// The uint64 API cannot distinguish a saturated estimate from exact MaxUint64.
	// At that exact budget the preserved contract admits zero blocks, not an error.
	if got, err := DeriveMaxBlocksWithGraphReserve(math.MaxUint64, 0, reserve, 1, 1); got != 0 || err != nil {
		t.Fatalf("saturated reserve at exact maximum budget: %d, %v", got, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGraphReserveChargeOverflowAndBoundary(t *testing.T) {
	for _, charge := range [][2]uint64{{math.MaxUint64, 1}, {math.MaxUint64, 2}, {math.MaxUint64 - 1, 2}} {
		if got, err := DeriveMaxBlocksWithGraphReserve(math.MaxUint64, charge[0], charge[1], 1, 1); got != 0 || !errors.Is(err, ErrNonKVExceedsBudget) {
			t.Fatalf("overflow charge %d + %d: %d, %v", charge[0], charge[1], got, err)
		}
	}
	// Exactly the budget is valid and leaves no blocks, unlike overflow.
	if got, err := DeriveMaxBlocksWithGraphReserve(math.MaxUint64, math.MaxUint64-1, 1, 1, 1); got != 0 || err != nil {
		t.Fatalf("exact maximum: %d, %v", got, err)
	}
	if got, err := DeriveMaxBlocksWithGraphReserve(math.MaxUint64, math.MaxUint64-2, 1, 1, 1); got != 1 || err != nil {
		t.Fatalf("one block remains: %d, %v", got, err)
	}
	for _, free := range []uint64{0, 1024, math.MaxUint64} {
		want, wantErr := DeriveMaxBlocks(free, 0, 1, 1)
		got, err := DeriveMaxBlocksWithGraphReserve(free, 0, 0, 1, 1)
		if got != want || !errors.Is(err, wantErr) {
			t.Fatalf("zero-reserve parity free=%d: %d/%v != %d/%v", free, got, err, want, wantErr)
		}
	}
	// Existing input validation retains precedence over combined-charge refusal.
	if _, err := DeriveMaxBlocksWithGraphReserve(math.MaxUint64, math.MaxUint64, 1, 0, 1); !errors.Is(err, ErrUtilizationOutOfRange) {
		t.Fatal(err)
	}
	if _, err := DeriveMaxBlocksWithGraphReserve(math.MaxUint64, math.MaxUint64, 1, 1, 0); !errors.Is(err, ErrZeroBlockSize) {
		t.Fatal(err)
	}
}
