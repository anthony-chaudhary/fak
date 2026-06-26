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
	hostTotal   int64
	hostFree    int64
	hostKnown   bool
	hostProbe   bool
}

func (c capDevice) Caps() Caps {
	caps := Caps{Async: true, DeviceMemory: true, FusedAttn: true, CapacityProbe: true}
	if c.hostProbe {
		caps.HostCapacityProbe = true
	}
	return caps
}
func (c capDevice) DeviceMemory() (int64, int64, bool) { return c.total, c.free, c.known }
func (c capDevice) HostMemory() (int64, int64, bool) {
	return c.hostTotal, c.hostFree, c.hostKnown
}

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
	if _, _, known := HostMemoryInfo(capDevice{hostTotal: 128 << 30, hostFree: 64 << 30, hostKnown: true}); known {
		t.Fatal("a backend that omits Caps().HostCapacityProbe must read as unknown")
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

func TestHostCapacityReportsAndFits(t *testing.T) {
	dev := capDevice{
		total: 24 << 30, free: 10 << 30, known: true,
		hostTotal: 128 << 30, hostFree: 64 << 30, hostKnown: true, hostProbe: true,
	}
	total, free, known := HostMemoryInfo(dev)
	if !known || total != 128<<30 || free != 64<<30 {
		t.Fatalf("host report mismatch: total=%d free=%d known=%v", total, free, known)
	}
	if v, avail := FitsOnHost(dev, 32<<30, 0); v != FitOK || avail != 64<<30 {
		t.Fatalf("32 GiB into 64 GiB host free: want FitOK avail=64GiB, got %s avail=%d", v, avail)
	}
	if v, _ := FitsOnHost(dev, 80<<30, 0); v != FitTooBig {
		t.Fatalf("80 GiB into 64 GiB host free: want FitTooBig, got %s", v)
	}
}

func TestHostSystemMemoryProbeIsSaneWhenAvailable(t *testing.T) {
	total, free, known := hostSystemMemory()
	if !known {
		return
	}
	if total <= 0 {
		t.Fatalf("hostSystemMemory known with non-positive total=%d", total)
	}
	if free != FreeUnknown && (free < 0 || free > total) {
		t.Fatalf("hostSystemMemory free=%d outside [0,total=%d]", free, total)
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

func TestMemoryPlanTotalsByClass(t *testing.T) {
	plan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 8 << 30, Detail: "gguf weights"},
		{Class: MemoryKVCache, Bytes: 2 << 30, Detail: "prompt kv"},
		{Class: MemoryScratchpad, Bytes: 512 << 20, Detail: "decode scratch"},
		{Class: MemoryKVCache, Bytes: 1 << 30, Detail: "continuation kv"},
		{Class: MemoryOffload, Bytes: -1, Detail: "bad estimate ignored"},
	}
	if got, want := plan.Total(), int64(11<<30)+(512<<20); got != want {
		t.Fatalf("Total = %d, want %d", got, want)
	}
	by := plan.ByClass()
	if by[MemoryWeights] != 8<<30 || by[MemoryKVCache] != 3<<30 || by[MemoryScratchpad] != 512<<20 {
		t.Fatalf("ByClass = %+v", by)
	}
	if _, ok := by[MemoryOffload]; ok {
		t.Fatalf("negative demand must not be counted: %+v", by)
	}
}

func TestMemoryPlanDeviceTotalIgnoresHostOffload(t *testing.T) {
	plan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 2 << 30, Detail: "dense weights"},
		{Class: MemoryOffload, Bytes: 80 << 30, Detail: "expert weights", Scope: MemoryScopeHost},
		{Class: MemoryKVCache, Bytes: 512 << 20, Detail: "kv"},
	}
	if got, want := plan.Total(), int64(82<<30)+(512<<20); got != want {
		t.Fatalf("Total = %d, want all host+device bytes %d", got, want)
	}
	if got, want := plan.DeviceTotal(), int64(2<<30)+(512<<20); got != want {
		t.Fatalf("DeviceTotal = %d, want device-only bytes %d", got, want)
	}
	if got, want := plan.HostTotal(), int64(80<<30); got != want {
		t.Fatalf("HostTotal = %d, want host-only bytes %d", got, want)
	}
	dev := capDevice{total: 8 << 30, free: 8 << 30, known: true}
	if err := RefuseMemoryPlanIfTooBig(dev, plan, 0); err != nil {
		t.Fatalf("host-scoped offload bytes must not reject a fitting device plan: %v", err)
	}
}

func TestEstimateKVStoreMemoryPlan(t *testing.T) {
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 4, HeadDim: 8}
	// 2 layers * 16 positions * 4 kv heads * 8 dims * 3 rows (Kraw,K,V) * 4-byte f32.
	const want = int64(12288)
	if got := EstimateKVStoreBytes(cfg, 16); got != want {
		t.Fatalf("EstimateKVStoreBytes = %d, want %d", got, want)
	}
	plan := EstimateKVStoreMemoryPlan(cfg, 16)
	if len(plan) != 1 || plan[0].Class != MemoryKVCache || plan[0].Bytes != want || plan[0].Detail != "hal-kv-store" || plan[0].DType != F32.String() {
		t.Fatalf("EstimateKVStoreMemoryPlan = %+v, want one classed kv_cache demand", plan)
	}
}

func TestEstimateKVStoreMemoryPlanFailsOpenOnIncompleteGeometry(t *testing.T) {
	if got := EstimateKVStoreMemoryPlan(KVConfig{NumLayers: 2, NumKVHeads: 4, HeadDim: 8}, 0); got != nil {
		t.Fatalf("zero tokens should skip the KV estimate, got %+v", got)
	}
	if got := EstimateKVStoreMemoryPlan(KVConfig{NumLayers: 2, HeadDim: 8}, 16); got != nil {
		t.Fatalf("incomplete KV geometry should skip the estimate, got %+v", got)
	}
}

func TestEstimateHALTransientMemoryPlan(t *testing.T) {
	cfg := TransformerScratchConfig{
		HiddenSize:       32,
		IntermediateSize: 64,
		VocabSize:        128,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IncludeLogits:    true,
	}
	plan := EstimateHALTransientMemoryPlan(cfg)
	if len(plan) != 2 {
		t.Fatalf("EstimateHALTransientMemoryPlan = %+v, want activation + scratchpad", plan)
	}
	by := plan.ByClass()
	if got, want := by[MemoryActivation], int64(640); got != want {
		t.Fatalf("activation bytes = %d, want %d", got, want)
	}
	if got, want := by[MemoryScratchpad], int64(3584); got != want {
		t.Fatalf("scratchpad bytes = %d, want %d", got, want)
	}
	if plan[0].Detail != "hal-token-activation" || plan[1].Detail != "hal-token-scratch" {
		t.Fatalf("plan details = %+v", plan)
	}
	if plan[0].DType != F32.String() || plan[1].DType != F32.String() {
		t.Fatalf("plan dtypes = %+v, want f32 activation/scratchpad", plan)
	}
	if got := EstimateHALTransientMemoryPlan(TransformerScratchConfig{HiddenSize: 32, NumLayers: 2}); len(got) != 1 || got[0].Class != MemoryActivation {
		t.Fatalf("partial geometry should keep only the supported activation estimate, got %+v", got)
	}
	if got := EstimateHALTransientMemoryPlan(TransformerScratchConfig{}); got != nil {
		t.Fatalf("empty geometry should skip the transient estimate, got %+v", got)
	}
}

func TestRefuseMemoryPlanIfTooBigCarriesClassBreakdown(t *testing.T) {
	dev := capDevice{total: 24 << 30, free: 10 << 30, known: true}
	plan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 8 << 30, Detail: "weights"},
		{Class: MemoryKVCache, Bytes: 3 << 30, Detail: "kv"},
		{Class: MemoryScratchpad, Bytes: 512 << 20, Detail: "scratch"},
	}
	err := RefuseMemoryPlanIfTooBig(dev, plan, 0)
	if err == nil {
		t.Fatal("oversize classed memory plan must yield a FitError")
	}
	fe, ok := err.(*FitError)
	if !ok {
		t.Fatalf("want *FitError, got %T (%v)", err, err)
	}
	if fe.Want != plan.Total() || fe.Avail != 10<<30 {
		t.Fatalf("FitError sizing: Want=%d Avail=%d", fe.Want, fe.Avail)
	}
	if fe.Scope != MemoryScopeDevice {
		t.Fatalf("FitError scope = %s, want %s", fe.Scope, MemoryScopeDevice)
	}
	if len(fe.Demands) != len(plan) {
		t.Fatalf("FitError demands = %+v, want %+v", fe.Demands, plan)
	}
	msg := err.Error()
	for _, want := range []string{"weights", "kv_cache", "scratchpad", "needs", "device has"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("FitError message %q missing %q", msg, want)
		}
	}
}

func TestRefuseMemoryPlanIfTooBigChecksKnownHostScopedDemands(t *testing.T) {
	dev := capDevice{
		total: 8 << 30, free: 8 << 30, known: true,
		hostTotal: 128 << 30, hostFree: 16 << 30, hostKnown: true, hostProbe: true,
	}
	plan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 2 << 30, Detail: "dense weights"},
		{Class: MemoryOffload, Bytes: 80 << 30, Detail: "expert weights", Scope: MemoryScopeHost},
		{Class: MemoryDDRCache, Bytes: 8 << 30, Detail: "dram cache", Scope: MemoryScopeHost},
	}
	err := RefuseMemoryPlanIfTooBig(dev, plan, 0)
	if err == nil {
		t.Fatal("known-too-small host capacity must yield a FitError")
	}
	fe, ok := err.(*FitError)
	if !ok {
		t.Fatalf("want *FitError, got %T (%v)", err, err)
	}
	if fe.Scope != MemoryScopeHost {
		t.Fatalf("FitError scope = %s, want %s", fe.Scope, MemoryScopeHost)
	}
	if fe.Want != plan.HostTotal() || fe.Avail != 16<<30 {
		t.Fatalf("FitError sizing: Want=%d Avail=%d", fe.Want, fe.Avail)
	}
	msg := err.Error()
	for _, want := range []string{"offload(host)", "ddr_cache(host)", "host has", "FitTooBig"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("FitError message %q missing %q", msg, want)
		}
	}
}

func TestRefuseMemoryPlanHostScopeFailsOpenWhenHostCapacityUnknown(t *testing.T) {
	plan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 2 << 30, Detail: "dense weights"},
		{Class: MemoryOffload, Bytes: 80 << 30, Detail: "expert weights", Scope: MemoryScopeHost},
	}
	for _, dev := range []capDevice{
		{total: 8 << 30, free: 8 << 30, known: true},
		{total: 8 << 30, free: 8 << 30, known: true, hostTotal: 128 << 30, hostFree: 16 << 30, hostKnown: false, hostProbe: true},
	} {
		if err := RefuseMemoryPlanIfTooBig(dev, plan, 0); err != nil {
			t.Fatalf("unknown host capacity must fail open for host-scoped plan, got %v", err)
		}
	}
}

func TestRefuseMemoryPlanFailsOpenOnUnknown(t *testing.T) {
	plan := MemoryPlan{{Class: MemoryWeights, Bytes: 1 << 40}, {Class: MemoryKVCache, Bytes: 1 << 40}}
	for _, be := range []Backend{cpu(), nil} {
		if err := RefuseMemoryPlanIfTooBig(be, plan, 0); err != nil {
			t.Fatalf("%v: unknown capacity must fail open for memory plan, got %v", be, err)
		}
	}
}

func TestDeviceAllocErrorCarriesMemoryClass(t *testing.T) {
	err := &DeviceAllocError{Bytes: 64 << 20, Site: "evict-scratch", Class: MemoryScratchpad}
	if got := err.DemandClass(); got != MemoryScratchpad {
		t.Fatalf("DemandClass = %s, want %s", got, MemoryScratchpad)
	}
	msg := err.Error()
	for _, want := range []string{"scratchpad", "64.00 MiB", "evict-scratch"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("DeviceAllocError message %q missing %q", msg, want)
		}
	}
	if got := (&DeviceAllocError{Bytes: 1}).DemandClass(); got != MemoryUnknown {
		t.Fatalf("empty class = %s, want %s", got, MemoryUnknown)
	}
}
