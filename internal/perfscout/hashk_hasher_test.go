package perfscout

import (
	"testing"
)

func TestSplitMix64Determinism(t *testing.T) {
	in := uint64(123456789)
	out1 := SplitMix64(in)
	out2 := SplitMix64(in)
	if out1 != out2 {
		t.Fatalf("SplitMix64 non-deterministic: %d vs %d", out1, out2)
	}
	if out1 == 0 {
		t.Fatalf("SplitMix64 output unexpected 0")
	}
}

func TestHashKRouterDualSubtableDispersion(t *testing.T) {
	router := NewHashKRouter(320000000, 4, 160)
	if router.NumSlotsPerSub != 80000000 {
		t.Fatalf("expected 80,000,000 slots per subtable, got %d", router.NumSlotsPerSub)
	}

	// Route 1000 sequential tokens and verify subtable 0 != subtable 1
	collisions := 0
	for i := uint64(0); i < 1000; i++ {
		slot0, slot1 := router.RouteToken(i, 0)
		if slot0 >= router.NumSlotsPerSub {
			t.Fatalf("slot0 %d exceeds slot count %d", slot0, router.NumSlotsPerSub)
		}
		if slot1 >= router.NumSlotsPerSub {
			t.Fatalf("slot1 %d exceeds slot count %d", slot1, router.NumSlotsPerSub)
		}
		if slot0 == slot1 {
			collisions++
		}
	}
	if collisions > 5 {
		t.Errorf("unexpected high collision rate between subtables: %d / 1000", collisions)
	}
}
