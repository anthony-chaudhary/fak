package model

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_ring_hal_test.go — the R0 witnesses for #5611 (epic #5606): staging ROUTED expert weights
// through the bounded pagedRing instead of the never-evicting halW memoizer bounds the ACTIVATED
// working set without moving the math. The four claims the plan names, one test each:
//
//	bit-exactness — a ring-served forward is byte-identical to the resident-HAL forward;
//	boundedness   — peak device weight bytes stay <= budget across a decode window that activates
//	                more distinct experts than fit, with evictions > 0 proving the bound was hit;
//	recovery      — an evicted expert pages back IN on its next activation, not silently lost;
//	default-off   — ExpertRingBytes == 0 leaves every weight on the unchanged permanent path.
//
// Plus the two structural gates the wiring depends on: only ROUTED experts are bounded (dense,
// attention, router and SHARED experts keep permanent residency, the #3212 distinction), and one
// expert's three co-used projections are never evicted out from under each other.

// expertRingTestModel builds a synthetic single-layer MoE whose `experts` routed experts all carry a
// resident Q4_K gate/up/down — the representation expertSwiGLUHAL admits — so every expert has an
// identical, exactly-known resident footprint and a ring budget can be stated in whole experts.
func expertRingTestModel(t *testing.T, hidden, experts int) *Model {
	t.Helper()
	cfg := expertHALTestConfig(hidden)
	cfg.NumExperts = experts
	m := NewSyntheticMoE(cfg)
	m.q4kw = map[string]*q4kTensor{}
	for e := 0; e < experts; e++ {
		for i, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
			name := expertName(0, e, suffix)
			m.q4kw[name] = &q4kTensor{out: hidden, in: hidden, nblk: hidden / qkK, raw: buildRawQ4K(t, hidden, hidden, 101+e*3+i)}
		}
	}
	return m
}

// expertRingSession returns a device-capable Q4_K session with the given routed-expert ring budget.
// ringBytes == 0 is the DEFAULT: no ring, permanent halW residency, the pre-#5611 path.
func expertRingSession(m *Model, ringBytes int64) *Session {
	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	return &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}, ExpertRingBytes: ringBytes}
}

func expertRingTestInput(hidden int) []float32 {
	x := make([]float32, hidden)
	for i := range x {
		x[i] = float32((i%23)-11) / 23
	}
	return x
}

// expertRingWeightBytes is the resident footprint of one routed-expert projection in the test model —
// the unit a ring budget is stated in (3 of these == one whole expert).
func expertRingWeightBytes(t *testing.T, m *Model) int64 {
	t.Helper()
	n := q4kResidentBytes(m.q4kw[expertName(0, 0, "gate_proj.weight")])
	if n <= 0 {
		t.Fatalf("expert projection reports %d resident bytes; the budget would be meaningless", n)
	}
	return n
}

// TestExpertRingForwardIsBitEqualToResidentHAL is the bit-exactness + boundedness witness. Over a
// decode window that activates SIX distinct experts through a ring sized for TWO, every expert's
// output is byte-for-byte the unbounded resident-HAL session's output — residency changed, the math
// did not — while peak device weight bytes never exceed the budget and evictions prove the bound was
// actually exercised rather than merely never reached.
func TestExpertRingForwardIsBitEqualToResidentHAL(t *testing.T) {
	const H, E = 256, 6
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)

	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 6 // exactly two whole experts
	ringS := expertRingSession(m, budget)
	baseS := expertRingSession(m, 0)

	// More distinct experts than fit, with the short-range reuse real routing shows, so the window
	// exercises all three outcomes: hit, page-in and evict.
	window := []int{0, 1, 0, 1, 2, 3, 2, 3, 0, 4, 5, 4, 1, 1}
	for step, e := range window {
		want := expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: baseS})
		got := expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: ringS})
		if len(got) != len(want) {
			t.Fatalf("step %d expert %d: ring output len=%d, resident len=%d", step, e, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("step %d expert %d: ring output[%d]=%v, resident=%v — a bounded ring must change residency, never the math",
					step, e, i, got[i], want[i])
			}
			if math.IsNaN(float64(got[i])) {
				t.Fatalf("step %d expert %d: output[%d] is NaN", step, e, i)
			}
		}
	}

	st := ringS.ExpertRing()
	if !st.Enabled {
		t.Fatal("ring reports Enabled=false after staging routed experts under a budget")
	}
	if st.BudgetBytes != budget {
		t.Fatalf("ring budget=%d, want %d", st.BudgetBytes, budget)
	}
	if st.PeakBytes > st.BudgetBytes {
		t.Fatalf("peak resident %d exceeds budget %d — the activated set is not bounded", st.PeakBytes, st.BudgetBytes)
	}
	if st.ResidentBytes > st.BudgetBytes {
		t.Fatalf("resident %d exceeds budget %d", st.ResidentBytes, st.BudgetBytes)
	}
	if st.Evictions == 0 {
		t.Fatalf("no evictions over a %d-expert window under a 2-expert budget: the bound was never exercised", E)
	}
	if st.Hits == 0 {
		t.Fatalf("no ring hits: every activation paged in, so the ring is caching nothing")
	}
	// The point of the rung: routed experts must NOT accumulate in the permanent memoizer.
	for key := range ringS.halW {
		if isRoutedExpertWeight(key) {
			t.Fatalf("routed expert %q landed in the permanent halW memoizer; the ring must own its residency", key)
		}
	}
	// Contrast: the unbounded session holds every activated expert forever — the behaviour R0 replaces.
	permanent := 0
	for key := range baseS.halW {
		if isRoutedExpertWeight(key) {
			permanent++
		}
	}
	if permanent != E*3 {
		t.Fatalf("resident-HAL session holds %d routed expert weights, want all %d — the contrast is unwitnessed", permanent, E*3)
	}
	if baseS.ExpertRing().Enabled {
		t.Fatal("a session with no declared budget reports a ring")
	}
}

// TestExpertRingPagesEvictedExpertBackIn is the recovery witness: an expert dropped under budget
// pressure is not lost — its next activation pages it back in. It also pins the LRU victim choice
// end to end, which is what makes the residency BOUNDED rather than merely capped.
func TestExpertRingPagesEvictedExpertBackIn(t *testing.T) {
	const H = 256
	m := expertRingTestModel(t, H, 3)
	x := expertRingTestInput(H)
	perWeight := expertRingWeightBytes(t, m)
	s := expertRingSession(m, perWeight*6) // two whole experts

	gate0 := "q4k:" + expertName(0, 0, "gate_proj.weight")
	expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	expertSwiGLU(m, 0, 1, x, sessionQ4KKernel{s: s})
	if !s.expertRing.isResident(gate0) {
		t.Fatal("expert 0 is not resident after activation under a budget that fits it")
	}
	if st := s.ExpertRing(); st.PageIns != 6 || st.Evictions != 0 {
		t.Fatalf("after two experts under a two-expert budget: pageIns=%d evictions=%d, want 6/0", st.PageIns, st.Evictions)
	}

	// A third distinct expert cannot fit: the coldest expert's three weights are paged out.
	expertSwiGLU(m, 0, 2, x, sessionQ4KKernel{s: s})
	if s.expertRing.isResident(gate0) {
		t.Fatal("expert 0 survived admission of a third expert under a two-expert budget")
	}
	st := s.ExpertRing()
	if st.PageIns != 9 || st.Evictions != 3 {
		t.Fatalf("after the third expert: pageIns=%d evictions=%d, want 9/3", st.PageIns, st.Evictions)
	}

	// Re-activating the evicted expert pages it back IN — recovery, not silent loss.
	expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	if !s.expertRing.isResident(gate0) {
		t.Fatal("re-activated expert 0 did not page back in")
	}
	st = s.ExpertRing()
	if st.PageIns != 12 || st.Evictions != 6 {
		t.Fatalf("after re-activation: pageIns=%d evictions=%d, want 12/6", st.PageIns, st.Evictions)
	}
	if st.PeakBytes > st.BudgetBytes {
		t.Fatalf("peak resident %d exceeds budget %d", st.PeakBytes, st.BudgetBytes)
	}

	// Close pages the whole ring out: ring-served handles are NOT in halW, so they need their own
	// teardown or they would outlive the session that staged them.
	ring := s.expertRing
	s.Close()
	if s.expertRing != nil {
		t.Fatal("Close left the routed-expert ring attached to the session")
	}
	if ring.residentCount() != 0 || ring.used() != 0 {
		t.Fatalf("Close left %d weights / %d bytes resident on the device", ring.residentCount(), ring.used())
	}
}

// TestExpertRingHoldsOneExpertsProjectionsTogether is the co-residency witness. One expert is THREE
// weights used together, so a ring too small to hold all three must NOT evict gate to make room for
// down — that would Free a handle the GEMMs are about to run against. Under a two-weight budget the
// held pair survives, the third falls back to permanent residency, and the output stays bit-exact.
func TestExpertRingHoldsOneExpertsProjectionsTogether(t *testing.T) {
	const H = 256
	m := expertRingTestModel(t, H, 2)
	x := expertRingTestInput(H)
	perWeight := expertRingWeightBytes(t, m)
	s := expertRingSession(m, perWeight*2) // two weights: one expert does NOT fit
	base := expertRingSession(m, 0)

	want := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: base})
	got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("undersized ring changed the math at [%d]: %v vs %v", i, got[i], want[i])
		}
	}
	if st := s.ExpertRing(); st.Evictions != 0 {
		t.Fatalf("evictions=%d while a single expert's projections were held: a held weight was paged out from under its own GEMM", st.Evictions)
	}
	for _, suffix := range []string{"gate_proj.weight", "up_proj.weight"} {
		if !s.expertRing.isResident("q4k:" + expertName(0, 0, suffix)) {
			t.Fatalf("%s was not held resident for the span of its expert", suffix)
		}
	}
	// The weight that could not be admitted without dropping a held resident falls back to permanent
	// residency rather than failing the forward: correctness never depends on a generous budget.
	if _, ok := s.halW["q4k:"+expertName(0, 0, "down_proj.weight")]; !ok {
		t.Fatal("the unadmittable projection did not fall back to permanent residency")
	}
	if st := s.ExpertRing(); st.ResidentBytes > st.BudgetBytes {
		t.Fatalf("resident %d exceeds budget %d", st.ResidentBytes, st.BudgetBytes)
	}
}

// TestExpertRingBoundsOnlyRoutedExperts pins the gate. Dense, attention, router and lm_head weights
// are activated by EVERY token, and so is the SHARED expert (#3212) — evicting them could only cost.
// Only `.mlp.experts.N.*`, the router-selected minority, is bounded.
func TestExpertRingBoundsOnlyRoutedExperts(t *testing.T) {
	const H = 256
	m := expertRingTestModel(t, H, 1)
	s := expertRingSession(m, expertRingWeightBytes(t, m)*8)

	qt := m.q4kw[expertName(0, 0, "gate_proj.weight")]
	permanent := []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.mlp.gate.weight",
		"model.layers.0.mlp.shared_experts.gate_proj.weight",
		"lm_head.weight",
	}
	for _, name := range permanent {
		s.weightHALQ4K(name, qt)
		if _, ok := s.halW["q4k:"+name]; !ok {
			t.Fatalf("%s did not take permanent residency", name)
		}
	}
	if s.expertRing != nil {
		t.Fatalf("a non-routed weight built a routed-expert ring (resident=%d)", s.expertRing.residentCount())
	}

	routed := expertName(0, 0, "gate_proj.weight")
	s.weightHALQ4K(routed, qt)
	if s.expertRing == nil || !s.expertRing.isResident("q4k:"+routed) {
		t.Fatal("the routed expert weight did not take bounded ring residency")
	}
	if _, ok := s.halW["q4k:"+routed]; ok {
		t.Fatal("the routed expert weight ALSO landed in the permanent memoizer")
	}
}

// TestExpertRingDefaultOffIsUnchanged is the default-unchanged witness: at ExpertRingBytes == 0 —
// every session in the tree today — no ring is built, routed experts memoize permanently in halW and
// upload exactly once, and ExpertRing reports the honest "residency is whatever accumulated".
func TestExpertRingDefaultOffIsUnchanged(t *testing.T) {
	const H, E = 256, 3
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)
	s := expertRingSession(m, 0)
	be := s.Backend.(*expertHALRecordingBackend)

	for token := 0; token < 2; token++ {
		for e := 0; e < E; e++ {
			expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: s})
		}
	}
	if be.uploads[compute.Q4_K] != E*3 {
		t.Fatalf("Q4_K uploads=%d over two tokens, want %d (one permanent upload per projection)", be.uploads[compute.Q4_K], E*3)
	}
	if s.expertRing != nil {
		t.Fatal("a session with ExpertRingBytes=0 built a ring")
	}
	if st := s.ExpertRing(); st.Enabled || st.PageIns != 0 || st.Evictions != 0 {
		t.Fatalf("default session reports ring stats %+v, want the zero value", st)
	}
	staged := 0
	for key := range s.halW {
		if strings.HasPrefix(key, "q4k:") && isRoutedExpertWeight(key) {
			staged++
		}
	}
	if staged != E*3 {
		t.Fatalf("halW holds %d routed expert weights, want %d", staged, E*3)
	}
}

// --- dtype-based routed-expert admission (fak#13360) --------------------------------
//
// The routed-expert HAL route (expertSwiGLUHAL) used to be gated ONLY by the explicit
// SupportsRoutedExpertKQuant marker interface, which only CUDA advertises. Vulkan already
// reports per-dtype device support through SupportsDeviceWeightDtype, so a Q2_K/Q5_K/Q6_K
// expert slate that the backend CAN serve was refused solely because the marker was absent.
//
// The capfence below replaces the marker-only authority with a per-operand admission: every
// resolved projection must have a device MatMul case for its ACTUAL operand dtype. A marker
// alone never invents support, and a backend that lacks the marker is no longer barred from
// a dtype it genuinely serves.

// dtypeAdmissionBackend is a DeviceMemory recorder whose per-dtype device MatMul support is
// stated explicitly, so a test can model any backend (Vulkan's Q2_K/Q6_K set, ROCm's
// no-Q2_K set, or a liar that advertises the marker while refusing every dtype).
type dtypeAdmissionBackend struct {
	*expertHALRecordingBackend
	marker bool
	dtypes map[compute.Dtype]bool
	probed []compute.Dtype
}

func (b *dtypeAdmissionBackend) SupportsRoutedExpertKQuant() bool { return b.marker }

func (b *dtypeAdmissionBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	b.probed = append(b.probed, dt)
	if b.dtypes == nil {
		return false
	}
	return b.dtypes[dt]
}

// q2kExpertModel builds a single-layer MoE whose routed expert 0 carries resident Q2_K
// gate/up/down: the DeepSeek-V4.1 Flash expert slate the marker-only authority refused.
func q2kExpertModel(t *testing.T) (*Model, [3]string) {
	t.Helper()
	const H, I = 256, 256
	cfg := expertHALTestConfig(H)
	cfg.IntermediateSize = I
	cfg.MoEIntermediateSize = I
	m := NewSyntheticMoE(cfg)
	names := [3]string{
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 0, "up_proj.weight"),
		expertName(0, 0, "down_proj.weight"),
	}
	m.q4kw = map[string]*q4kTensor{}
	m.kqw = map[string]*kQuantTensor{
		names[0]: q2kFixtureTensor(I, H, 0x13360),
		names[1]: q2kFixtureTensor(I, H, 0x13361),
		names[2]: q2kFixtureTensor(H, I, 0x13362),
	}
	for _, name := range names {
		delete(m.manifest, name)
	}
	return m, names
}

func q2kExpertInput(h int) []float32 {
	x := make([]float32, h)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}
	return x
}

// TestExpertHALDtypeAdmissionWithoutMarker is the fak#13360 positive witness: a backend that
// does NOT advertise SupportsRoutedExpertKQuant but DOES serve Q2_K through its device MatMul
// must be admitted to the routed-expert HAL route. A full Q2_K expert slate runs gate, up,
// SwiGLU and down on the device (3 MatMuls / 1 SwiGLU) with the invocation counters proving
// it, and the result matches the host oracle.
func TestExpertHALDtypeAdmissionWithoutMarker(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, names := q2kExpertModel(t)
	x := q2kExpertInput(256)
	want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

	rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	be := &dtypeAdmissionBackend{
		expertHALRecordingBackend: rec,
		marker:                    false, // Vulkan-like: no SupportsRoutedExpertKQuant
		dtypes:                    map[compute.Dtype]bool{compute.F32: true, compute.Q2_K: true},
	}
	s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}

	got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})

	if rec.matmuls != 3 || rec.swiglu != 1 {
		t.Fatalf("marker-less Q2_K expert ran matmul=%d swiglu=%d, want 3/1 on the device route", rec.matmuls, rec.swiglu)
	}
	if rec.uploads[compute.Q2_K] < 3 {
		t.Fatalf("Q2_K uploads=%d, want one per projection", rec.uploads[compute.Q2_K])
	}
	for _, name := range names {
		if _, ok := s.halW["kquant-raw:"+name]; !ok {
			t.Fatalf("Q2_K %s not staged through the routed HAL route; halW holds %d entries", name, len(s.halW))
		}
	}
	routedExpertParity(t, "marker-less device Q2_K expert", got, want)
}

// TestExpertHALMarkerAloneDoesNotAdmit is the fak#13360 negative control: a backend that
// advertises the marker but whose device MatMul refuses the operand dtype must DECLINE before
// staging. A marker is not support; the per-operand dtype check is authoritative.
func TestExpertHALMarkerAloneDoesNotAdmit(t *testing.T) {
	m, _ := q2kExpertModel(t)
	x := q2kExpertInput(256)

	rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	be := &dtypeAdmissionBackend{
		expertHALRecordingBackend: rec,
		marker:                    true,
		dtypes:                    map[compute.Dtype]bool{compute.F32: true}, // no Q2_K case
	}
	s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}

	got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	if len(got) != 256 {
		t.Fatalf("declined expert produced len=%d, want the host fallback result", len(got))
	}
	if rec.matmuls != 0 {
		t.Fatalf("a marker-only backend ran %d device MatMuls; Q2_K has no device case", rec.matmuls)
	}
	for key := range s.halW {
		t.Fatalf("marker-only backend staged %q despite refusing the Q2_K dtype", key)
	}
}

// TestExpertHALMixedOperandSupportDeclines pins the per-operand rule at the HAL boundary: a
// backend that serves gate/up Q2_K but NOT the down projection's dtype must not admit the
// expert through the three-operand HAL route. Admission is all-or-nothing across the three
// operands, so the unsupported down projection is never staged under its HAL key. The
// incremental seam below may still run the two supported gate/up projections - that is the
// pre-existing #13129 contract, not the HAL route - so the witness is the absence of a staged
// unsupported operand, not a zero device-op count.
func TestExpertHALMixedOperandSupportDeclines(t *testing.T) {
	m, names := q2kExpertModel(t)
	x := q2kExpertInput(256)

	rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	// The down projection is swapped to Q6_K, which this backend cannot serve.
	m.kqw[names[2]] = deviceDownKQuant(t, kindQ6K, 256, 256, 0x13363)
	be := &dtypeAdmissionBackend{
		expertHALRecordingBackend: rec,
		marker:                    false,
		dtypes:                    map[compute.Dtype]bool{compute.F32: true, compute.Q2_K: true}, // Q6_K absent
	}
	s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}

	got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	if len(got) != 256 {
		t.Fatalf("mixed-unsupported expert produced len=%d, want a full expert output", len(got))
	}
	if rec.uploads[compute.Q6_K] != 0 {
		t.Fatalf("the unsupported Q6_K down projection was staged (%d uploads); one unsupported operand must decline the whole HAL route", rec.uploads[compute.Q6_K])
	}
	if _, ok := s.halW["kquant-raw:"+names[2]]; ok {
		t.Fatal("the unsupported Q6_K down projection took the HAL route's shared staging key")
	}
}

// TestExpertHALMalformedMetadataDeclines pins fail-closed admission for a k-quant operand whose
// kind has no registered HAL descriptor: the three-operand HAL route must decline (never panic,
// never stage), leaving the expert to the host/incremental fallback.
func TestExpertHALMalformedMetadataDeclines(t *testing.T) {
	m, names := q2kExpertModel(t)
	x := q2kExpertInput(256)

	// A down projection with an unregistered kind (no descriptor -> no HAL dtype).
	m.kqw[names[2]] = q3kFixtureTensor(256, 256)
	rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	be := &dtypeAdmissionBackend{
		expertHALRecordingBackend: rec,
		marker:                    true,
		dtypes:                    map[compute.Dtype]bool{compute.F32: true, compute.Q2_K: true},
	}
	s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}

	var got []float32
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("unregistered-kind operand PANICKED instead of declining: %v", r)
			}
		}()
		got = expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	}()
	if len(got) != 256 {
		t.Fatalf("malformed-metadata expert produced len=%d, want a full expert output", len(got))
	}
	if _, ok := s.halW["kquant-raw:"+names[2]]; ok {
		t.Fatal("the unregistered-kind down projection took the HAL route's shared staging key")
	}
}
