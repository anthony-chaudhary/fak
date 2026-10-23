package compute

import (
	"fmt"
	"strings"
	"testing"
)

// This exercises software slab ownership, without physical interconnect activity.
// fak-test:runtime fast est=1s
func TestUSB4AllReduceConstructionReturnsPartialSlots(t *testing.T) {
	const borrowedBytes int64 = 4096
	stages := []string{"send", "recv", "peer send", "peer recv"}
	for acquired := 0; acquired < TBV2_NSLOTS*len(stages); acquired++ {
		t.Run(fmt.Sprintf("after_%d_slots", acquired), func(t *testing.T) {
			capacity := borrowedBytes + int64(acquired)*int64(TBV2MaxPayloadBytes)
			conn, err := NewDualStrixHaloInterconnect(DualStrixHaloConfig{DirectSlabBytes: capacity})
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()
			slab := conn.Slab()
			bindTestDirectSlabRegistration(t, slab)
			unrelated, err := slab.Allocate(borrowedBytes)
			if err != nil {
				t.Fatal(err)
			}
			unrelated.Data[0] = 0x5a
			wantError := fmt.Sprintf("failed to allocate %s slot %d", stages[acquired%len(stages)], acquired/len(stages))
			for attempt := 0; attempt < 2; attempt++ {
				engine, err := NewUSB4DualAPUAllReduceEngine(conn, 0, 1)
				if err == nil || engine != nil || !strings.Contains(err.Error(), wantError) {
					t.Fatalf("attempt %d: engine=%v err=%v, want %s", attempt, engine, err, wantError)
				}
				if slab.Used() != borrowedBytes || slab.Available() != capacity-borrowedBytes {
					t.Fatalf("partial construction leaked slots: used=%d available=%d", slab.Used(), slab.Available())
				}
				if unrelated.Data[0] != 0x5a {
					t.Fatal("unrelated allocation was modified")
				}
				if _, err := unrelated.GetSGE(); err != nil {
					t.Fatalf("borrowed slab or unrelated allocation closed: %v", err)
				}
			}
			if err := unrelated.Free(); err != nil {
				t.Fatal(err)
			}
			whole, err := slab.Allocate(capacity)
			if err != nil {
				t.Fatalf("returned slots did not coalesce for reuse: %v", err)
			}
			if err := whole.Free(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestUSB4AllReduceConstructionTransfersSuccessfulOwnership(t *testing.T) {
	const borrowedBytes int64 = 4096
	engineBytes := int64(TBV2_NSLOTS) * 4 * int64(TBV2MaxPayloadBytes)
	conn, err := NewDualStrixHaloInterconnect(DualStrixHaloConfig{DirectSlabBytes: borrowedBytes + engineBytes})
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	slab := conn.Slab()
	bindTestDirectSlabRegistration(t, slab)
	unrelated, err := slab.Allocate(borrowedBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer unrelated.Free()
	for attempt := 0; attempt < 2; attempt++ {
		engine, err := NewUSB4DualAPUAllReduceEngine(conn, 0, 1)
		if err != nil {
			t.Fatal(err)
		}
		if slab.Used() != borrowedBytes+engineBytes {
			t.Fatal("successful engine lost its slots")
		}
		left, right := []float32{1, -2}, []float32{3, 5}
		outLeft, outRight := make([]float32, len(left)), make([]float32, len(right))
		if err := engine.AllReduceTP2(left, right, outLeft, outRight); err != nil {
			t.Fatal(err)
		}
		for i := range left {
			if outLeft[i] != left[i]+right[i] || outRight[i] != left[i]+right[i] {
				t.Fatal("successful software reduction changed")
			}
		}
		if err := engine.Close(); err != nil {
			t.Fatal(err)
		}
		if err := engine.Close(); err != nil {
			t.Fatal(err)
		}
		if slab.Used() != borrowedBytes {
			t.Fatal("engine close affected borrowed allocation accounting")
		}
		if _, err := unrelated.GetSGE(); err != nil {
			t.Fatal(err)
		}
	}
}
