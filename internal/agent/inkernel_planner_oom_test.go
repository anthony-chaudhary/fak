package agent

import (
	"context"
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

func TestInKernelOOMRetryStatsBucketByClass(t *testing.T) {
	be := &oomRetryBackend{Backend: compute.Default()}
	p := &InKernelPlanner{modelID: "retry-test", backend: be}

	p.recordInKernelOOMRetry(&InKernelOOMError{Bytes: 1 << 20, Class: compute.MemoryScratchpad, Site: "scratch-a"}, true)
	p.recordInKernelOOMRetry(&InKernelOOMError{Bytes: 2 << 20, Class: compute.MemoryScratchpad, Site: "scratch-b"}, false)
	p.recordInKernelOOMRetry(&InKernelOOMError{Bytes: 3 << 20, Class: compute.MemoryKVCache, Site: "kv-a"}, true)

	st := p.InKernelOOMRetryStats()
	if st.Backend != be.Name() {
		t.Fatalf("retry backend = %q, want %q", st.Backend, be.Name())
	}
	if len(st.Rows) != 2 {
		t.Fatalf("retry rows = %+v, want scratchpad and kv_cache", st.Rows)
	}
	byClass := map[string]InKernelOOMRetryClassStats{}
	for _, row := range st.Rows {
		byClass[row.Class] = row
	}
	scratch := byClass[string(compute.MemoryScratchpad)]
	if scratch.Attempts != 2 || scratch.Successes != 1 || scratch.Failures != 1 ||
		scratch.LastFailedBytes != 2<<20 || scratch.LastSite != "scratch-b" {
		t.Fatalf("scratch retry row = %+v, want 2 attempts/1 success/1 failure/latest scratch-b", scratch)
	}
	kv := byClass[string(compute.MemoryKVCache)]
	if kv.Attempts != 1 || kv.Successes != 1 || kv.Failures != 0 ||
		kv.LastFailedBytes != 3<<20 || kv.LastSite != "kv-a" {
		t.Fatalf("kv retry row = %+v, want 1 successful latest kv-a", kv)
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

type pressureTrimBackend struct {
	compute.Backend
	total     int64
	free      int64
	trimFree  int64
	recycle   int
	trim      int
	trimLarge []int
}

func (b *pressureTrimBackend) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, CapacityProbe: true}
}

func (b *pressureTrimBackend) Name() string { return "pressure-trim" }

func (b *pressureTrimBackend) DeviceMemory() (total, free int64, known bool) {
	return b.total, b.free, true
}

func (b *pressureTrimBackend) Recycle() { b.recycle++ }

func (b *pressureTrimBackend) Trim() {
	b.trim++
	if b.trimFree > b.free {
		b.free = b.trimFree
	}
}

func (b *pressureTrimBackend) TrimLarge(maxKeepBytes int) {
	b.trimLarge = append(b.trimLarge, maxKeepBytes)
}

func TestInKernelRequestPressureTrimRescuesCapacityPrecheck(t *testing.T) {
	be := &pressureTrimBackend{Backend: compute.Default(), total: 1 << 30, free: 1, trimFree: 1 << 30}
	p := &InKernelPlanner{
		m:       model.NewSynthetic(tinyConcurrencyConfig()),
		backend: be,
		modelID: "pressure-test",
	}

	if err := p.refuseOversizeRequest(10, 5); err != nil {
		t.Fatalf("pressure trim should rescue stale-free capacity refusal, got %v", err)
	}
	if be.recycle != 1 || be.trim != 1 || len(be.trimLarge) != 1 || be.trimLarge[0] != 0 {
		t.Fatalf("pressure cleanup = recycle %d trim %d trimLarge %+v, want 1/1/[0]", be.recycle, be.trim, be.trimLarge)
	}
	st := p.InKernelMemoryPressureTrimStats()
	if st.Backend != "pressure-trim" || len(st.Rows) != 1 {
		t.Fatalf("pressure trim stats = %+v, want one pressure-trim row", st)
	}
	row := st.Rows[0]
	if row.Reason != "capacity_precheck" || row.Scope != string(compute.MemoryScopeDevice) ||
		row.Attempts != 1 || row.Trimmed != 1 || row.NoHooks != 0 || row.Resolved != 1 ||
		row.LastWantBytes == 0 || row.LastBudgetBytes == 0 || row.LastMarginBytes <= 0 {
		t.Fatalf("pressure trim row = %+v, want rescued capacity_precheck with positive post-trim margin", row)
	}
}

func TestInKernelRequestPressureTrimOnLowMargin(t *testing.T) {
	m := model.NewSynthetic(tinyConcurrencyConfig())
	p := &InKernelPlanner{m: m}
	plan := p.requestMemoryPlan(10, 5)
	want := plan.DeviceTotal()
	if want <= 0 {
		t.Fatalf("test request plan want = %d, want positive", want)
	}
	margin := int64(32 << 20)
	free := int64(float64(want+margin) / (1 - inKernelRequestDeviceHeadroom))
	be := &pressureTrimBackend{Backend: compute.Default(), total: free + (1 << 30), free: free, trimFree: free}
	p.backend = be
	p.modelID = "pressure-test"

	if err := p.refuseOversizeRequest(10, 5); err != nil {
		t.Fatalf("low-margin request should still fit after proactive trim, got %v", err)
	}
	if be.trim != 1 {
		t.Fatalf("low-margin request did not trim idle pools: recycle %d trim %d trimLarge %+v", be.recycle, be.trim, be.trimLarge)
	}
	st := p.InKernelMemoryPressureTrimStats()
	if len(st.Rows) != 1 {
		t.Fatalf("pressure trim stats = %+v, want one row", st)
	}
	row := st.Rows[0]
	if row.Reason != "low_margin" || row.Attempts != 1 || row.Trimmed != 1 || row.Resolved != 0 ||
		row.LastMarginBytes <= 0 || row.LastMarginBytes > inKernelRequestPressureTrimMinMarginBytes {
		t.Fatalf("low-margin trim row = %+v", row)
	}
}

// qwen35RuntimeExtraConfig builds a tiny Qwen3.8-hybrid model config: the linear_attention layer
// type makes IsQwen35Hybrid true, so the panel-width chunk helper is reachable when q4k is set.
func qwen35RuntimeExtraConfig() model.Config {
	cfg := tinyConcurrencyConfig()
	cfg.LayerTypes = []string{"linear_attention", "full_attention"}
	return cfg
}

// runtimeExtraBackend both reports a device capacity (so the fit path is exercised) and advertises
// the Qwen35 sequence prefill capability (so the simultaneous panel width is reachable).
type runtimeExtraBackend struct {
	compute.Backend
	total, free int64
	known       bool
}

func (b runtimeExtraBackend) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, CapacityProbe: true}
}

func (b runtimeExtraBackend) Name() string { return "runtime-extra-probe" }

func (b runtimeExtraBackend) DeviceMemory() (int64, int64, bool) { return b.total, b.free, b.known }

func (b runtimeExtraBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

func (b runtimeExtraBackend) Qwen35SequencePrefill(compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	panic("test capability marker must not execute")
}

func (b runtimeExtraBackend) Qwen35SequenceEmbeddingRowsPath() string {
	return compute.Qwen35SequenceEmbeddingRowsPath
}

// qwen35RuntimeExtraPlanner builds a planner whose panel width is deterministic: q4k + a hybrid
// config + a sequence-prefill backend make nativeInferencePrefillChunkTokens reachable, and the
// explicit chunk size pins the simultaneous panel width the extras estimator must price.
func qwen35RuntimeExtraPlanner(panelTokens int) *InKernelPlanner {
	return &InKernelPlanner{
		m:                         model.NewSynthetic(qwen35RuntimeExtraConfig()),
		q4k:                       true,
		quant:                     true,
		qwenQ4KPrefillChunkTokens: panelTokens,
	}
}

// TestInKernelQwenRuntimeMemoryComposesIntoRequestPlan proves the composed plan carries the two
// #13315 terms the base context plan omits: the device-scope prefill-panel peak (reduced by the
// already-priced HAL transient) and the host-scope retained-MTP hidden history when the request
// is eligible.
func TestInKernelQwenRuntimeMemoryComposesIntoRequestPlan(t *testing.T) {
	p := qwen35RuntimeExtraPlanner(64)

	base, err := p.requestMemoryPlanWithExtras(128, 32, false)
	if err != nil {
		t.Fatalf("base extras composition rejected: %v", err)
	}
	// The panel term is priced whenever the simultaneous panel width is bound; it is NOT
	// MTP-history-specific. Retained history, by contrast, is gated on request eligibility.
	if hasDemandDetail(base, "qwen35-mtp-retained-hidden-history") {
		t.Fatalf("retained-history-off plan carried an MTP history demand: %#v", base)
	}
	if !hasDemandDetail(base, "qwen35-vulkan-prefill-panel-additional") {
		t.Fatalf("bounded panel width must price the panel term: %#v", base)
	}

	withHistory, err := p.requestMemoryPlanWithExtras(128, 32, true)
	if err != nil {
		t.Fatalf("history-on extras composition rejected: %v", err)
	}
	panel := demandDetail(withHistory, "qwen35-vulkan-prefill-panel-additional")
	if panel.Bytes <= 0 || panel.ScopeOrDefault() != compute.MemoryScopeDevice {
		t.Fatalf("composed panel term = %+v, want positive device-scope demand", panel)
	}
	if !hasDemandDetail(withHistory, "qwen35-mtp-retained-hidden-history") {
		t.Fatalf("history-on plan missing retained MTP history: %#v", withHistory)
	}
	history := demandDetail(withHistory, "qwen35-mtp-retained-hidden-history")
	if history.ScopeOrDefault() != compute.MemoryScopeHost {
		t.Fatalf("retained history scope = %s, want host", history.ScopeOrDefault())
	}
	if len(withHistory) <= len(base) {
		t.Fatalf("history-on plan must add demands over history-off: on=%d off=%d", len(withHistory), len(base))
	}
	// The panel term must be the EXCESS over scratch the base plan already prices, never the
	// whole peak again — integration must not double count overlapping HAL transient scratch.
	rawPanel := int64(64) * int64(p.m.Cfg.HiddenSize) * 4
	if panel.Bytes >= rawPanel {
		t.Fatalf("panel term %d must subtract already-priced HAL transient from raw peak %d", panel.Bytes, rawPanel)
	}
}

// TestInKernelRefuseOversizeRequestRefusesOnUnknownExtrasBound proves the composed plan (base
// context plan + the priced runtime extras) is enforced through the real pre-admission guard: a
// device whose free budget is one byte short of the composed demand refuses with a typed capacity
// error, while the same request fits once extras stop being charged (retained history off).
func TestInKernelRefuseOversizeRequestRefusesOnUnknownExtrasBound(t *testing.T) {
	// The composed demand is base + panel + retained history. Size a device one byte below it.
	p := qwen35RuntimeExtraPlanner(64)
	p.vulkanMTP = true
	p.speculativeEngine = model.NewSpeculativeEngine(nil, nil, model.SpeculativeEngineConfig{})
	plan, err := p.requestMemoryPlanWithExtras(64, 16, true)
	if err != nil {
		t.Fatalf("composing retained-history plan: %v", err)
	}
	want := plan.DeviceTotal()
	if want <= 0 {
		t.Fatalf("composed device demand = %d, want positive", want)
	}

	// One byte short of the headroom-adjusted budget.
	headroom := inKernelRequestDeviceHeadroom
	budget := int64(float64(want) / (1 - headroom))
	if budget <= want {
		budget = want + 1
	}
	short := budget - 1
	be := runtimeExtraBackend{Backend: compute.Default(), total: short + (1 << 30), free: short, known: true}
	p.backend = be
	err = p.refuseOversizeRequest(64, 16)
	var capErr *InKernelCapacityError
	if !errors.As(err, &capErr) {
		t.Fatalf("one-byte-short composed plan error = %T (%v), want *InKernelCapacityError", err, err)
	}
	if capErr.Site != "capacity-precheck" {
		t.Fatalf("capacity error site = %q, want capacity-precheck", capErr.Site)
	}

	// A roomy device with the same request fits: the guard admits a well-bounded plan.
	p2 := qwen35RuntimeExtraPlanner(64)
	p2.vulkanMTP = true
	p2.speculativeEngine = model.NewSpeculativeEngine(nil, nil, model.SpeculativeEngineConfig{})
	p2.backend = runtimeExtraBackend{Backend: compute.Default(), total: 1 << 40, free: 1 << 40, known: true}
	if err := p2.refuseOversizeRequest(64, 16); err != nil {
		t.Fatalf("bounded runtime extras should fit, got %v", err)
	}
}

// TestInKernelQwenRuntimeMemoryMTPHistoryEligibleIsRequestLocal proves the retained-history predicate is a
// function of THIS request, not planner-wide MTP support alone: a configured planner with a zero
// output budget must not reserve retained-history bytes.
func TestInKernelQwenRuntimeMemoryMTPHistoryEligibleIsRequestLocal(t *testing.T) {
	p := qwen35RuntimeExtraPlanner(64)
	p.vulkanMTP = true
	p.speculativeEngine = model.NewSpeculativeEngine(nil, nil, model.SpeculativeEngineConfig{})

	if !p.requestMTPHistoryEligible(8) {
		t.Fatal("configured MTP support with a positive decode budget must be eligible")
	}
	if p.requestMTPHistoryEligible(0) {
		t.Fatal("a zero-output-budget request must not be MTP-history eligible")
	}
	p.DisableSpeculativeDecoding()
	if p.requestMTPHistoryEligible(8) {
		t.Fatal("a planner with speculative decoding disabled must not be MTP-history eligible")
	}
}

func hasDemandDetail(plan compute.MemoryPlan, detail string) bool {
	for _, d := range plan {
		if d.Detail == detail {
			return true
		}
	}
	return false
}

func demandDetail(plan compute.MemoryPlan, detail string) compute.MemoryDemand {
	for _, d := range plan {
		if d.Detail == detail {
			return d
		}
	}
	return compute.MemoryDemand{}
}

// TestInKernelRefuseOversizeRequestGuardRunsBeforeExecution proves the served entry point — not
// just the helper — reaches the runtime-extras guard BEFORE any model work. An intentionally
// uninitialized model makes every session build / forward fail, so receiving the typed capacity
// error instead of a model/session error is the reachability witness: the guard ran first.
func TestInKernelRefuseOversizeRequestGuardRunsBeforeExecution(t *testing.T) {
	cfg := qwen35RuntimeExtraConfig()
	cfg.MaxPositionEmbeddings = 4096
	tok := loadProbeTok(t)
	// A device with a near-zero free budget: the composed plan (KV + panel) cannot fit, so the
	// guard must refuse before the deliberately-empty model is asked to allocate anything. A
	// successful session build would instead panic (the model has no weights), so receiving the
	// typed capacity error is the reachability witness.
	be := runtimeExtraBackend{Backend: compute.Default(), total: 1 << 40, free: 1, known: true}
	p := NewInKernelPlanner(&model.Model{Cfg: cfg}, tok, "runtime-extra-reach", true, be, false)
	p.qwenQ4KPrefillChunkTokens = 8

	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, WithMaxTokens(8))
	if comp != nil {
		t.Fatalf("guarded request returned completion: %+v", comp)
	}
	var capErr *InKernelCapacityError
	if !errors.As(err, &capErr) {
		t.Fatalf("complete error = %T (%v), want *InKernelCapacityError from the runtime-extras guard", err, err)
	}
	if capErr.Site != "capacity-precheck" {
		t.Fatalf("capacity error site = %q, want capacity-precheck", capErr.Site)
	}
}
