package compute

import "testing"

// TestMachineFit pins the offline machine-fit seam: a registered capacity
// snapshot (a registry row or a saved probe) answers "does an N-GiB plan fit
// an M-GiB card" with the same verdicts as the live FitsOnDevice check. The
// 32 GB owned desktop RTX 5090 (sm_120 GDDR7) is the motivating card.
func TestMachineFit(t *testing.T) {
	const gib = int64(1) << 30

	// 30 GiB plan on a 32 GB card: FIT_OK, with the full ceiling available.
	if v, avail := FitsCapacity(32*gib, FreeUnknown, true, 30*gib, 0); v != FitOK || avail != 32*gib {
		t.Fatalf("30 GiB @ 32 GB = %v (avail %d), want FitOK (avail %d)", v, avail, 32*gib)
	}
	// 34 GiB plan on a 32 GB card: FIT_TOO_BIG.
	if v, _ := FitsCapacity(32*gib, FreeUnknown, true, 34*gib, 0); v != FitTooBig {
		t.Fatalf("34 GiB @ 32 GB = %v, want FitTooBig", v)
	}
	// Unknown registered capacity fails open, never refuses.
	if v, _ := FitsCapacity(0, 0, false, 30*gib, 0); v != FitUnknown {
		t.Fatalf("unknown capacity = %v, want FitUnknown (fail open)", v)
	}
	// Headroom reserves its fraction of the registered ceiling.
	if v, _ := FitsCapacity(32*gib, FreeUnknown, true, 31*gib, 0.05); v != FitTooBig {
		t.Fatalf("31 GiB @ 32 GB with 5%% headroom = %v, want FitTooBig", v)
	}
	if v, _ := FitsCapacity(32*gib, FreeUnknown, true, 30*gib, 0.05); v != FitOK {
		t.Fatalf("30 GiB @ 32 GB with 5%% headroom = %v, want FitOK", v)
	}
	// A live backend and its registered snapshot must agree byte-for-byte.
	dev := capDevice{total: 32 * gib, free: FreeUnknown, known: true}
	for _, want := range []int64{30 * gib, 34 * gib} {
		live, liveAvail := FitsOnDevice(dev, want, 0)
		reg, regAvail := FitsCapacity(32*gib, FreeUnknown, true, want, 0)
		if live != reg || liveAvail != regAvail {
			t.Fatalf("live (%v, %d) != registered (%v, %d) for want=%d", live, liveAvail, reg, regAvail, want)
		}
	}
}
