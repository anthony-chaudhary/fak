package agent

import (
	"errors"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// These tests cover the recover boundary that turns an in-kernel device-allocation panic into
// a typed, actionable error instead of crashing the serving goroutine. They need NO GPU: the
// panic payload (*compute.DeviceAllocError) is an ordinary Go value, so recoverDevicePanic —
// the factored-out body of Complete's deferred recover — is exercised directly.

func TestRecoverDevicePanic_DeviceAllocBecomesTypedOOM(t *testing.T) {
	const want = 4 << 30 // 4 GiB — the kind of logits buffer a large prompt drives
	err, handled := recoverDevicePanic(&compute.DeviceAllocError{Bytes: want, Site: "dallocWeight", Class: compute.MemoryWeights})
	if !handled {
		t.Fatal("a *compute.DeviceAllocError panic must be handled (recovered into a clean error)")
	}
	var oom *InKernelOOMError
	if !errors.As(err, &oom) {
		t.Fatalf("want *InKernelOOMError, got %T (%v)", err, err)
	}
	if oom.Bytes != want {
		t.Fatalf("byte count lost across recovery: got %d, want %d", oom.Bytes, want)
	}
	if oom.Class != compute.MemoryWeights {
		t.Fatalf("memory class lost across recovery: got %s, want %s", oom.Class, compute.MemoryWeights)
	}
	if oom.Site != "dallocWeight" {
		t.Fatalf("site lost across recovery: got %q", oom.Site)
	}
	// The message must name the actionable condition so an operator/client can act on it.
	if msg := oom.Error(); msg == "" {
		t.Fatal("InKernelOOMError.Error() must not be empty")
	}
}

// A device-alloc error WRAPPED in another error is still recognized via errors.As — the
// recover does not depend on the panic value being the bare type.
func TestRecoverDevicePanic_WrappedDeviceAllocStillHandled(t *testing.T) {
	wrapped := fmt.Errorf("decode step 7: %w", &compute.DeviceAllocError{Bytes: 1 << 20, Site: "evict-scratch", Class: compute.MemoryScratchpad})
	err, handled := recoverDevicePanic(wrapped)
	if !handled {
		t.Fatal("a wrapped *compute.DeviceAllocError must still be handled")
	}
	var oom *InKernelOOMError
	if !errors.As(err, &oom) || oom.Bytes != 1<<20 || oom.Class != compute.MemoryScratchpad {
		t.Fatalf("wrapped device-alloc error not recovered with its byte count: %v", err)
	}
}

// Everything that is NOT an in-kernel device-allocation failure must report handled=false so
// Complete RE-PANICS it — a validation bug, a nil deref, a raw string panic must keep today's
// loud crash/stack behavior and never be silently swallowed as an OOM.
func TestRecoverDevicePanic_OtherPanicsAreNotHandled(t *testing.T) {
	cases := []struct {
		name string
		val  any
	}{
		{"a plain error (validation bug)", errors.New("compute: cuda MatMul supports F32/F16/Q8_0/Q4_K weights today")},
		{"a raw string panic", "index out of range [1] with length 1"},
		{"a non-error value", 42},
		{"nil-ish struct that is not a device error", struct{}{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, handled := recoverDevicePanic(tc.val)
			if handled {
				t.Fatalf("%s must NOT be handled (Complete must re-panic it)", tc.name)
			}
		})
	}
}

type oomRetryBackend struct {
	compute.Backend
	recycle   int
	trim      int
	trimLarge []int
}

func (b *oomRetryBackend) Recycle() { b.recycle++ }
func (b *oomRetryBackend) Trim()    { b.trim++ }
func (b *oomRetryBackend) TrimLarge(maxKeepBytes int) {
	b.trimLarge = append(b.trimLarge, maxKeepBytes)
}

func TestPrepareDeviceOOMRetryTrimsIdlePoolsOnce(t *testing.T) {
	be := &oomRetryBackend{Backend: compute.Default()}
	p := &InKernelPlanner{modelID: "retry-test", backend: be}
	err := &InKernelOOMError{Bytes: 1 << 20, Class: compute.MemoryScratchpad, Site: "transient"}
	if !p.prepareDeviceOOMRetry(err) {
		t.Fatal("typed in-kernel OOM on a trim-capable backend should prepare one retry")
	}
	if be.recycle != 1 || be.trim != 1 || len(be.trimLarge) != 1 || be.trimLarge[0] != 0 {
		t.Fatalf("retry cleanup = recycle %d trim %d trimLarge %+v, want 1/1/[0]", be.recycle, be.trim, be.trimLarge)
	}
	if p.prepareDeviceOOMRetry(errors.New("ordinary upstream error")) {
		t.Fatal("ordinary errors must not trigger device OOM retry cleanup")
	}
	if be.recycle != 1 || be.trim != 1 || len(be.trimLarge) != 1 {
		t.Fatalf("non-OOM changed cleanup counters: recycle %d trim %d trimLarge %+v", be.recycle, be.trim, be.trimLarge)
	}
	if (&InKernelPlanner{backend: compute.Default()}).prepareDeviceOOMRetry(err) {
		t.Fatal("backend without trim/recycle hooks must not claim a retry was prepared")
	}
}

type capacityProbeBackend struct {
	compute.Backend
	total int64
	free  int64
	known bool
}

func (b capacityProbeBackend) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, CapacityProbe: true}
}

func (b capacityProbeBackend) Name() string {
	if b.Backend != nil {
		return b.Backend.Name()
	}
	return "capacity-probe"
}

func (b capacityProbeBackend) DeviceMemory() (total, free int64, known bool) {
	return b.total, b.free, b.known
}

func TestInKernelRequestMemoryPlanSplitsRuntimeClasses(t *testing.T) {
	p := &InKernelPlanner{
		m:       model.NewSynthetic(tinyConcurrencyConfig()),
		backend: capacityProbeBackend{total: 1 << 30, free: 1 << 30, known: true},
	}

	byClass := p.requestMemoryPlan(10, 5).ByClass()
	byClassDType := map[compute.MemoryClass]string{}
	for _, row := range p.requestMemoryPlan(10, 5) {
		byClassDType[row.Class] = row.DType
	}
	for _, class := range []compute.MemoryClass{compute.MemoryKVCache, compute.MemoryActivation, compute.MemoryScratchpad} {
		if byClass[class] <= 0 {
			t.Fatalf("request plan missing %s demand: %#v", class, byClass)
		}
		if byClassDType[class] != compute.F32.String() {
			t.Fatalf("request plan %s dtype = %q, want f32", class, byClassDType[class])
		}
	}
	if byClass[compute.MemoryWeights] != 0 {
		t.Fatalf("request plan with known free device memory must not double-count resident weights: %#v", byClass)
	}

	p.backend = capacityProbeBackend{total: 1 << 30, free: compute.FreeUnknown, known: true}
	byClass = p.requestMemoryPlan(10, 5).ByClass()
	if byClass[compute.MemoryWeights] <= 0 {
		t.Fatalf("request plan with unknown free memory must include resident weights against the total ceiling: %#v", byClass)
	}
	for _, row := range p.requestMemoryPlan(10, 5) {
		if row.Class == compute.MemoryWeights && row.DType != "mixed" {
			t.Fatalf("resident weight dtype = %q, want mixed", row.DType)
		}
	}
}

func TestInKernelRequestCapacityPrecheckRefusesKnownTooLargeKV(t *testing.T) {
	p := &InKernelPlanner{
		m:       model.NewSynthetic(tinyConcurrencyConfig()),
		backend: capacityProbeBackend{total: 1 << 20, free: 1 << 20, known: true},
	}

	err := p.refuseOversizeRequest(100_000, 256)
	var capErr *InKernelCapacityError
	if !errors.As(err, &capErr) {
		t.Fatalf("refuseOversizeRequest error = %T (%v), want *InKernelCapacityError", err, err)
	}
	if capErr.Class != compute.MemoryKVCache {
		t.Fatalf("capacity error class = %s, want %s", capErr.Class, compute.MemoryKVCache)
	}
	if capErr.Scope != compute.MemoryScopeDevice {
		t.Fatalf("capacity error scope = %s, want %s", capErr.Scope, compute.MemoryScopeDevice)
	}
	if capErr.Site != "capacity-precheck" {
		t.Fatalf("capacity error site = %q, want capacity-precheck", capErr.Site)
	}
	if capErr.Want <= capErr.Avail || capErr.Avail <= 0 {
		t.Fatalf("capacity sizing = want %d avail %d, want positive refused budget", capErr.Want, capErr.Avail)
	}
	st := p.RequestMemoryStats()
	if !st.Observed || st.Backend != "capacity-probe" || st.PromptTokens != 100_000 || st.MaxNewTokens != 256 || st.PlannedTokens != 100_256 {
		t.Fatalf("request memory stats = %+v, want observed capacity-probe 100000+256", st)
	}
	if len(st.MemoryPlan) == 0 || st.MemoryPlan[0].Class == "" || st.MemoryPlan[0].DType == "" {
		t.Fatalf("request memory plan missing class/dtype rows: %+v", st.MemoryPlan)
	}
	if len(st.Capacities) != 2 || !st.Capacities[0].Known {
		t.Fatalf("request memory capacities = %+v, want device/host snapshot with known device", st.Capacities)
	}
}

func TestInKernelRequestCapacityPrecheckFailsOpenWhenCapacityUnknown(t *testing.T) {
	p := &InKernelPlanner{
		m:       model.NewSynthetic(tinyConcurrencyConfig()),
		backend: capacityProbeBackend{known: false},
	}

	if err := p.refuseOversizeRequest(100_000, 256); err != nil {
		t.Fatalf("unknown-capacity backend must fail open, got %v", err)
	}
}
