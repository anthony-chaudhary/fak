package compute

import (
	"strings"
	"testing"
)

// capDevice is a fakeDevice that ALSO reports its capacity — the test stand-in for a real
// device backend (cuda/metal/vulkan) implementing the DeviceCapacity capability. It embeds
// fakeDevice for the full Backend surface and overrides only Caps (to advertise the probe)
// and adds DeviceMemory.
type capDevice struct {
	fakeDevice
	total, free int64
	known       bool
}

func (c capDevice) Caps() Caps {
	return Caps{Async: true, DeviceMemory: true, FusedAttn: true, CapacityProbe: true}
}
func (c capDevice) DeviceMemory() (int64, int64, bool) { return c.total, c.free, c.known }

// halfAdvertised implements DeviceMemory but does NOT set Caps().CapacityProbe — it must be
// treated as non-reporting (the two-part discovery idiom: cap AND interface, never one).
type halfAdvertised struct {
	fakeDevice
}

func (halfAdvertised) DeviceMemory() (int64, int64, bool) { return 24 << 30, 24 << 30, true }

func TestDeviceMemoryInfoUnknownForPlainBackends(t *testing.T) {
	// The CPU reference and a plain device that does not implement DeviceCapacity both
	// report unknown — the portable-floor contract.
	for _, b := range []Backend{cpu(), fakeDevice{}} {
		total, free, known := DeviceMemoryInfo(b)
		if known {
			t.Fatalf("%s: capacity should be unknown, got total=%d free=%d", b.Name(), total, free)
		}
		if free != FreeUnknown {
			t.Fatalf("%s: unknown capacity must report free=FreeUnknown, got %d", b.Name(), free)
		}
	}
	if total, free, known := DeviceMemoryInfo(nil); known || free != FreeUnknown || total != 0 {
		t.Fatalf("nil backend: want (0, FreeUnknown, false), got (%d, %d, %v)", total, free, known)
	}
}

func TestCapacityProbeCapGatesTheAssertion(t *testing.T) {
	// A backend that implements DeviceMemory but forgets the Caps().CapacityProbe flag is
	// NOT trusted to report — it half-advertised, so it reads as unknown (fail-open).
	if _, _, known := DeviceMemoryInfo(halfAdvertised{}); known {
		t.Fatal("a backend that omits Caps().CapacityProbe must read as unknown")
	}
}

func TestDeviceCapacityReportsAndFits(t *testing.T) {
	dev := capDevice{total: 24 << 30, free: 10 << 30, known: true} // 24 GiB device, 10 GiB free
	total, free, known := DeviceMemoryInfo(dev)
	if !known || total != 24<<30 || free != 10<<30 {
		t.Fatalf("report mismatch: total=%d free=%d known=%v", total, free, known)
	}
	// Fits within free headroom.
	if v, avail := FitsOnDevice(dev, 8<<30, 0); v != FitOK || avail != 10<<30 {
		t.Fatalf("8 GiB into 10 GiB free: want FitOK avail=10GiB, got %s avail=%d", v, avail)
	}
	// Exceeds free headroom (even though it fits the 24 GiB total) -> known too big.
	if v, _ := FitsOnDevice(dev, 20<<30, 0); v != FitTooBig {
		t.Fatalf("20 GiB into 10 GiB free: want FitTooBig, got %s", v)
	}
	// Headroom reserves part of the budget for KV/scratch: 50% of 10 GiB = 5 GiB.
	if v, avail := FitsOnDevice(dev, 6<<30, 0.5); v != FitTooBig || avail != 5<<30 {
		t.Fatalf("6 GiB into 50%%-of-10GiB budget: want FitTooBig avail=5GiB, got %s avail=%d", v, avail)
	}
	if v, _ := FitsOnDevice(dev, 4<<30, 0.5); v != FitOK {
		t.Fatalf("4 GiB into 50%%-of-10GiB budget: want FitOK, got %s", v)
	}
}

func TestFitsOnDeviceFailsOpenOnUnknown(t *testing.T) {
	// A non-reporting backend must NEVER return FitTooBig — capacity is unknown, so the
	// caller proceeds. This is the contract that keeps the capability strictly additive.
	for _, want := range []int64{1, 1 << 40, 1 << 50} {
		if v, _ := FitsOnDevice(cpu(), want, 0); v != FitUnknown {
			t.Fatalf("unknown-capacity backend must fail open; want FitUnknown for %d bytes, got %s", want, v)
		}
	}
}

func TestFreeUnknownFallsBackToTotalCeiling(t *testing.T) {
	// total known, free not yet probeable (the cuda-before-cudaMemGetInfo case): the fit
	// check still catches a model that does not fit the WHOLE device, using total as budget.
	dev := capDevice{total: 16 << 30, free: FreeUnknown, known: true}
	if v, avail := FitsOnDevice(dev, 10<<30, 0); v != FitOK || avail != 16<<30 {
		t.Fatalf("10 GiB with free=unknown: want FitOK avail=16GiB (total), got %s avail=%d", v, avail)
	}
	if v, _ := FitsOnDevice(dev, 20<<30, 0); v != FitTooBig {
		t.Fatalf("20 GiB into a 16 GiB device: want FitTooBig, got %s", v)
	}
}

func TestFitVerdictString(t *testing.T) {
	for v, want := range map[FitVerdict]string{FitOK: "ok", FitTooBig: "too_big", FitUnknown: "unknown"} {
		if got := v.String(); got != want {
			t.Fatalf("FitVerdict(%d).String() = %q, want %q", v, got, want)
		}
	}
}

// RefuseIfTooBig is the load-time admission decision the model loaders call with a pre-load
// footprint estimate (issue #709; capacity-bridge Plank 5). It MUST fail open on every verdict
// except FitTooBig, so wiring it in never blocks a path that worked before — including the
// cpu-ref floor (unknown capacity) and a model that fits.

func TestRefuseIfTooBigFailsOpen(t *testing.T) {
	// FitUnknown: the cpu-ref floor and a nil backend both proceed (no typed refusal).
	for _, be := range []Backend{cpu(), nil} {
		if err := RefuseIfTooBig(be, 1<<40, 0); err != nil {
			t.Fatalf("%v: unknown capacity must fail open (nil), got %v", be, err)
		}
	}
	// FitOK: a model that fits a known ceiling is not refused.
	dev := capDevice{total: 24 << 30, free: 10 << 30, known: true}
	if err := RefuseIfTooBig(dev, 8<<30, 0); err != nil { // 8 GiB into 10 GiB free
		t.Fatalf("FitOK must yield nil, got %v", err)
	}
}

func TestRefuseIfTooBigReturnsTypedSizingError(t *testing.T) {
	// FitTooBig: a capacity-reporting backend that KNOWS the request exceeds its ceiling yields
	// a *FitError carrying the sizing — the answerable form of the would-be OOM panic.
	dev := capDevice{total: 24 << 30, free: 1 << 30, known: true} // 1 GiB free
	err := RefuseIfTooBig(dev, 8<<30, 0)                          // 8 GiB into 1 GiB -> FitTooBig
	if err == nil {
		t.Fatal("oversize request on a known ceiling must yield a FitError, got nil")
	}
	fe, ok := err.(*FitError)
	if !ok {
		t.Fatalf("want *FitError, got %T (%v)", err, err)
	}
	if fe.Verdict != FitTooBig {
		t.Fatalf("Verdict = %s, want FitTooBig", fe.Verdict)
	}
	if fe.Want != 8<<30 || fe.Avail != 1<<30 {
		t.Fatalf("FitError sizing: Want=%d Avail=%d, want 8 GiB / 1 GiB", fe.Want, fe.Avail)
	}
	msg := err.Error()
	for _, want := range []string{"needs", "device has", "GiB", "FitTooBig"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("FitError message %q missing %q", msg, want)
		}
	}
}
