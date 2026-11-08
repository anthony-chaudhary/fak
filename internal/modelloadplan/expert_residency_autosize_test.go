package modelloadplan

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestExpertResidencyAutosize(t *testing.T) {
	for _, tc := range []struct {
		name                        string
		free, reserved, bytes       uint64
		layers, experts, top, slots int
		used                        uint64
	}{
		{"kv_first", 1000, 400, 100, 2, 8, 2, 3, 600},
		{"all_experts_cap", 10000, 0, 100, 2, 8, 2, 8, 1600},
		{"floor_partial_slot", 999, 400, 100, 2, 8, 2, 2, 400},
		{"exact_minimum", 800, 400, 100, 2, 8, 2, 2, 400},
		{"max_bytes", ^uint64(0), 0, ^uint64(0), 1, 1, 1, 1, ^uint64(0)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			slots, used, err := ExpertResidencyAutosize(tc.free, tc.reserved, tc.bytes, tc.layers, tc.experts, tc.top)
			if err != nil || slots != tc.slots || used != tc.used {
				t.Fatalf("got %d/%d %v; want %d/%d", slots, used, err, tc.slots, tc.used)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestExpertResidencyAutosizeRefuses(t *testing.T) {
	for _, tc := range []struct {
		free, reserved, bytes uint64
		layers, experts, top  int
	}{
		{1, 2, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1}, {1, 0, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1},
		{1, 0, 1, 1, 1, 0}, {1, 0, 1, 1, 1, 2}, {799, 400, 100, 2, 8, 2}, {^uint64(0), 0, ^uint64(0), 2, 2, 1},
	} {
		slots, used, err := ExpertResidencyAutosize(tc.free, tc.reserved, tc.bytes, tc.layers, tc.experts, tc.top)
		if slots != 0 || used != 0 || !errors.Is(err, ErrExpertResidencyAutosize) {
			t.Fatalf("got %d/%d %v for %+v", slots, used, err, tc)
		}
	}
}
