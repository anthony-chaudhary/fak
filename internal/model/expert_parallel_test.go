package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_parallel_test.go — the correctness gates for the expert-parallel (EP) MoE FFN
// decomposition (expert_parallel.go). The gate triad mirrors tensor_parallel.go's
// row-parallel rung exactly:
//
//	EP(ranks=1)           ==(bit-exact, max|Δ|=0)        the routed monolith (moeFFN/glmMoeFFN)
//	EP via Collective     ==(bit-exact, max|Δ|=0)        EP rank-order reference   [any ranks]
//	EP(ranks=N)           ==(reassociation round-off)    the routed monolith
//
// so EP is sharding-invariant up to the single AllReduceSum's documented reassociation,
// proven with no multi-GPU hardware (LocalCollective is the single-box bit-exact default).
//
// Two substrates: a generic 8-expert top-4 MoE (NewSyntheticMoE) drives the rich multi-rank
// proof (ranks 1/2/4/8, genuine multi-partial reduction) against the moeFFN monolith; the
// glm_moe_dsa safetensors fixture (with a shared expert) drives the GLM-specific wrapper
// (expertParallelGLMMoEDelta) against glmMoeFFN.

// epGenMoeModel builds a generic synthetic MoE model with E experts and top-K routing — the
// rich substrate for the multi-rank EP proof.
func epGenMoeModel(experts, topk int) *Model {
	cfg := moeCfgForTest()
	cfg.NumExperts = experts
	cfg.NumExpertsPerTok = topk
	return NewSyntheticMoE(cfg)
}

func epMaxAbs(a, b []float32) float64 {
	var mx float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > mx {
			mx = d
		}
	}
	return mx
}

func epCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// TestExpertParallelRanks1MatchesMonolith pins the bit-exact rung: with one rank owning all
// experts, the EP routed delta equals the moeFFN monolith delta byte-for-byte. This is the
// EP twin of ForwardTP(ranks=1)==Forward.
func TestExpertParallelRanks1MatchesMonolith(t *testing.T) {
	const E, K = 8, 4
	m := epGenMoeModel(E, K)
	mat := f32Kernel{m}
	layer := 0

	xn := make([]float32, m.Cfg.HiddenSize)
	for i := range xn {
		xn[i] = float32(math.Sin(float64(i)*0.3 + 1))
	}
	picks := route(m, layer, xn, mat)
	if len(picks) != K {
		t.Fatalf("router returned %d picks, want top-k=%d", len(picks), K)
	}

	mono := moeFFN{}.apply(m, layer, xn, mat)

	plan, err := ExpertParallelPlan(E, 1)
	if err != nil {
		t.Fatalf("ExpertParallelPlan: %v", err)
	}
	got, err := m.ExpertParallelDelta(layer, xn, mat, picks, plan, nil)
	if err != nil {
		t.Fatalf("ExpertParallelDelta: %v", err)
	}
	if mx := epMaxAbs(got, mono); mx != 0 {
		t.Fatalf("EP(ranks=1) vs monolith max|Δ|=%g, want 0 (bit-exact)", mx)
	}
	// With no shared expert (generic MoE), the full-FFN twin equals the routed delta.
	full, err := m.expertParallelGLMMoEDelta(layer, xn, mat, picks, plan, nil)
	if err != nil {
		t.Fatalf("expertParallelGLMMoEDelta: %v", err)
	}
	if mx := epMaxAbs(full, mono); mx != 0 {
		t.Fatalf("expertParallelGLMMoEDelta(ranks=1) vs monolith max|Δ|=%g, want 0", mx)
	}
	t.Logf("EP(ranks=1) == moeFFN monolith, max|Δ|=0 (%d picks over %d experts)", len(picks), E)
}

// TestExpertParallelCollectiveMatchesRankOrderRef pins, for every rank count, that the
// Collective reduces the per-rank partials in RANK ORDER: the LocalCollective AllReduceSum
// result equals the rank-order reference at max|Δ|=0. A reduction that reordered, dropped, or
// double-counted a rank would be caught here even though the loose vs-monolith round-off bound
// cannot see it.
func TestExpertParallelCollectiveMatchesRankOrderRef(t *testing.T) {
	const E, K = 8, 4
	m := epGenMoeModel(E, K)
	mat := f32Kernel{m}
	layer := 0

	xn := make([]float32, m.Cfg.HiddenSize)
	for i := range xn {
		xn[i] = float32(math.Cos(float64(i)*0.21 + 0.4))
	}
	picks := route(m, layer, xn, mat)

	for _, ranks := range []int{1, 2, 4, 8} {
		plan, err := ExpertParallelPlan(E, ranks)
		if err != nil {
			t.Fatalf("ranks=%d: ExpertParallelPlan: %v", ranks, err)
		}
		got, err := m.ExpertParallelDelta(layer, xn, mat, picks, plan, LocalCollective{})
		if err != nil {
			t.Fatalf("ranks=%d: ExpertParallelDelta: %v", ranks, err)
		}
		ref, err := m.expertParallelReference(layer, xn, mat, picks, plan)
		if err != nil {
			t.Fatalf("ranks=%d: expertParallelReference: %v", ranks, err)
		}
		if mx := epMaxAbs(got, ref); mx != 0 {
			t.Fatalf("ranks=%d: collective vs rank-order ref max|Δ|=%g, want 0", ranks, mx)
		}
		t.Logf("ranks=%d: collective == rank-order reference, max|Δ|=0", ranks)
	}
}

// TestExpertParallelShardingInvariant pins the cross-rank rung: EP at ranks N>1 matches the
// monolith within the AllReduceSum reassociation round-off (cosine ≈ 1), i.e. the expert
// partition changes only the reduction grouping, never the result beyond float
// non-associativity. This is the EP twin of ForwardTP(ranks=N) ≈ Forward.
func TestExpertParallelShardingInvariant(t *testing.T) {
	const E, K = 8, 4
	m := epGenMoeModel(E, K)
	mat := f32Kernel{m}
	layer := 0

	xn := make([]float32, m.Cfg.HiddenSize)
	for i := range xn {
		xn[i] = float32(math.Sin(float64(i)*0.17+0.9)) * 1.5
	}
	picks := route(m, layer, xn, mat)
	mono := moeFFN{}.apply(m, layer, xn, mat)

	for _, ranks := range []int{2, 4, 8} {
		plan, err := ExpertParallelPlan(E, ranks)
		if err != nil {
			t.Fatalf("ranks=%d: ExpertParallelPlan: %v", ranks, err)
		}
		got, err := m.ExpertParallelDelta(layer, xn, mat, picks, plan, nil)
		if err != nil {
			t.Fatalf("ranks=%d: ExpertParallelDelta: %v", ranks, err)
		}
		cos := epCosine(got, mono)
		mx := epMaxAbs(got, mono)
		if cos < 0.99999 {
			t.Fatalf("ranks=%d: EP vs monolith cosine=%.8f, want ≥ 0.99999 (reassociation only)", ranks, cos)
		}
		t.Logf("ranks=%d: EP vs monolith cosine=%.8f max|Δ|=%g (reassociation round-off)", ranks, cos, mx)
	}
}

// TestExpertParallelLoadImbalance pins the degenerate-but-correct case the real MoE router
// hits constantly: when every picked expert lands in ONE rank's band, the other ranks
// contribute all-zero partials and the AllReduceSum still produces the exact rank=1 result.
// It guards against an EP implementation that skips a zero-partial rank or mis-handles an
// empty band.
func TestExpertParallelLoadImbalance(t *testing.T) {
	const E = 8
	m := epGenMoeModel(E, 4)
	mat := f32Kernel{m}
	layer := 0

	xn := make([]float32, m.Cfg.HiddenSize)
	for i := range xn {
		xn[i] = float32(i%5) * 0.1
	}
	// Hand-pick all experts in the low band [0,4) so a ranks=2 plan ([0,4)|[4,8)) loads only
	// rank 0; rank 1 owns none of them.
	picks := []routePick{
		{expert: 0, weight: 0.4},
		{expert: 1, weight: 0.3},
		{expert: 2, weight: 0.2},
		{expert: 3, weight: 0.1},
	}
	plan1, _ := ExpertParallelPlan(E, 1)
	plan2, _ := ExpertParallelPlan(E, 2)

	d1, err := m.ExpertParallelDelta(layer, xn, mat, picks, plan1, nil)
	if err != nil {
		t.Fatalf("ranks=1: %v", err)
	}
	d2, err := m.ExpertParallelDelta(layer, xn, mat, picks, plan2, nil)
	if err != nil {
		t.Fatalf("ranks=2: %v", err)
	}
	// All picks live in rank 0's band, so rank 1's partial is all-zero and the rank-order sum
	// is bit-exact to the single-rank result.
	if mx := epMaxAbs(d1, d2); mx != 0 {
		t.Fatalf("load-imbalanced EP ranks=2 vs ranks=1 max|Δ|=%g, want 0 (zero-partial rank)", mx)
	}
	t.Logf("load-imbalanced EP (all picks on rank 0): ranks=2 == ranks=1, max|Δ|=0")
}

// TestExpertParallelFailsClosed pins the fail-closed boundary: a plan whose Dim is not the
// expert count is rejected (it would mis-map experts to ranks), and ExpertParallelPlan rejects
// more ranks than experts (a rank with no experts) and a non-MoE expert count.
func TestExpertParallelFailsClosed(t *testing.T) {
	const E = 8
	m := epGenMoeModel(E, 4)
	mat := f32Kernel{m}
	xn := make([]float32, m.Cfg.HiddenSize)
	picks := []routePick{{expert: 0, weight: 1}}

	// Plan sized to the wrong dimension (hidden, not experts) must be rejected.
	badPlan, err := NewTPPlan(m.Cfg.HiddenSize, 2)
	if err != nil {
		t.Fatalf("NewTPPlan(hidden): %v", err)
	}
	if _, err := m.ExpertParallelDelta(0, xn, mat, picks, badPlan, nil); err == nil {
		t.Fatalf("ExpertParallelDelta with plan.Dim=%d (!= NumExperts=%d) should fail closed", badPlan.Dim, E)
	}
	// More ranks than experts is rejected at plan construction.
	if _, err := ExpertParallelPlan(E, E+1); err == nil {
		t.Fatalf("ExpertParallelPlan(ranks > experts) should fail closed")
	}
	// A non-MoE expert count is rejected.
	if _, err := ExpertParallelPlan(0, 1); err == nil {
		t.Fatalf("ExpertParallelPlan(numExperts=0) should fail closed")
	}
}

// TestExpertParallelFailsClosedGPTOSS pins the arch-fidelity guard surfaced by the EP
// correctness audit: gptoss experts use the expertGPTOSS form (clamped gate/up, (up+1)*glu),
// which moeFFN dispatches but the EP path does not. EP uses expertSwiGLU, so it must REFUSE a
// gptoss config rather than silently serve the wrong expert arithmetic. The guard fires on the
// config alone (before any weight access), so a bare gptoss-arch model triggers it.
func TestExpertParallelFailsClosedGPTOSS(t *testing.T) {
	m := epGenMoeModel(8, 4)
	m.Cfg.Architectures = []string{"GptOssForCausalLM"} // -> archFamilyKey contains "gptoss"
	if !m.Cfg.isGPTOSS() {
		t.Fatalf("test setup: cfg should report isGPTOSS()==true")
	}
	mat := f32Kernel{m}
	xn := make([]float32, m.Cfg.HiddenSize)
	picks := []routePick{{expert: 0, weight: 1}}
	plan, _ := ExpertParallelPlan(m.Cfg.NumExperts, 2)
	if _, err := m.ExpertParallelDelta(0, xn, mat, picks, plan, nil); err == nil {
		t.Fatalf("ExpertParallelDelta on a gptoss config should fail closed (experts use expertGPTOSS, not expertSwiGLU)")
	}
}

// TestExpertParallelGLMSharedExpert pins the GLM-specific wrapper (expertParallelGLMMoEDelta):
// on a real glm_moe_dsa fixture WITH a shared expert, the EP delta (routed reduce + the
// replicated shared-expert term added once) equals the glmMoeFFN monolith — bit-exact at
// ranks=1, within reassociation round-off at the full expert count. This exercises the
// shared-expert branch the generic substrate cannot.
func TestExpertParallelGLMSharedExpert(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if cfg.NSharedExperts <= 0 {
		t.Fatalf("fixture NSharedExperts=%d, want a shared expert", cfg.NSharedExperts)
	}
	mat := residentKernel{m}
	// Find a routed MoE layer (router + experts present).
	layer := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if m.hasWeight(routerName(l)) && m.hasWeight(expertName(l, 0, "gate_proj.weight")) {
			layer = l
			break
		}
	}
	if layer < 0 {
		t.Fatalf("no routed MoE layer in the glm fixture")
	}

	xn := make([]float32, cfg.HiddenSize)
	for i := range xn {
		xn[i] = float32(math.Sin(float64(i)*0.11 + 0.3))
	}
	picks := glmRoute(m, layer, xn, mat)
	mono := glmMoeFFN{}.apply(m, layer, xn, mat) // routed sum + shared expert

	// ranks=1: bit-exact (the shared term is added once, after the identity reduce).
	plan1, err := ExpertParallelPlan(cfg.NumExperts, 1)
	if err != nil {
		t.Fatalf("ExpertParallelPlan(1): %v", err)
	}
	got1, err := m.expertParallelGLMMoEDelta(layer, xn, mat, picks, plan1, nil)
	if err != nil {
		t.Fatalf("expertParallelGLMMoEDelta(ranks=1): %v", err)
	}
	if mx := epMaxAbs(got1, mono); mx != 0 {
		t.Fatalf("GLM EP(ranks=1, +shared) vs glmMoeFFN max|Δ|=%g, want 0 (bit-exact)", mx)
	}

	// ranks == NumExperts (one expert per rank): matches the monolith within reassociation
	// round-off when more than one expert fires.
	plan2, err := ExpertParallelPlan(cfg.NumExperts, cfg.NumExperts)
	if err != nil {
		t.Fatalf("ExpertParallelPlan(%d): %v", cfg.NumExperts, err)
	}
	got2, err := m.expertParallelGLMMoEDelta(layer, xn, mat, picks, plan2, nil)
	if err != nil {
		t.Fatalf("expertParallelGLMMoEDelta(ranks=%d): %v", cfg.NumExperts, err)
	}
	if cos := epCosine(got2, mono); cos < 0.99999 {
		t.Fatalf("GLM EP(ranks=%d, +shared) vs glmMoeFFN cosine=%.8f, want ≥ 0.99999", cfg.NumExperts, cos)
	}
	t.Logf("GLM EP +shared: ranks=1 bit-exact, ranks=%d within round-off vs glmMoeFFN (%d experts)", cfg.NumExperts, cfg.NumExperts)
}

// TestGlmMoeEPFFNMatchesMonolith pins the LIVE-PATH wiring (Rung 0): ffnForLayer dispatches a
// routed glm_moe_dsa layer to glmMoeEPFFN exactly when Model.epRanks > 1 (and to the monolith
// glmMoeFFN otherwise), and the dispatched glmMoeEPFFN.apply equals glmMoeFFN.apply byte-for-byte
// at ranks=1. This is the per-ffnKind, through-the-dispatch analogue of TestExpertParallelGLMSharedExpert
// (which calls expertParallelGLMMoEDelta directly): it proves the EP twin is now reachable from the
// live decode path (mlpBody -> ffnForLayer) and is a no-op until ranks>1, so an existing serve
// (epRanks 0) is unchanged.
func TestGlmMoeEPFFNMatchesMonolith(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	mat := residentKernel{m}
	layer := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if m.hasWeight(routerName(l)) && m.hasWeight(expertName(l, 0, "gate_proj.weight")) {
			layer = l
			break
		}
	}
	if layer < 0 {
		t.Fatalf("no routed MoE layer in the glm fixture")
	}

	// Dispatch wiring: epRanks 0/1 keeps the monolith; epRanks>1 routes through glmMoeEPFFN.
	m.SetExpertParallelRanks(0)
	if _, ok := m.ffnForLayer(layer).(glmMoeFFN); !ok {
		t.Fatalf("epRanks=0: ffnForLayer = %T, want glmMoeFFN (no-op default)", m.ffnForLayer(layer))
	}
	m.SetExpertParallelRanks(1)
	if _, ok := m.ffnForLayer(layer).(glmMoeFFN); !ok {
		t.Fatalf("epRanks=1: ffnForLayer = %T, want glmMoeFFN (ranks=1 not yet sharded)", m.ffnForLayer(layer))
	}
	m.SetExpertParallelRanks(2)
	ep, ok := m.ffnForLayer(layer).(glmMoeEPFFN)
	if !ok {
		t.Fatalf("epRanks=2: ffnForLayer = %T, want glmMoeEPFFN (EP dispatched)", m.ffnForLayer(layer))
	}

	xn := make([]float32, cfg.HiddenSize)
	for i := range xn {
		xn[i] = float32(math.Sin(float64(i)*0.11 + 0.3))
	}
	mono := glmMoeFFN{}.apply(m, layer, xn, mat)

	// Bit-exact: the dispatched EP ffnKind at ranks=1 == the monolith. Build a ranks=1 plan
	// (the dispatch above made a ranks=2 plan; ranks=1 is the bit-exact rung).
	plan1, err := ExpertParallelPlan(cfg.NumExperts, 1)
	if err != nil {
		t.Fatalf("ExpertParallelPlan(1): %v", err)
	}
	got := glmMoeEPFFN{plan: plan1, coll: LocalCollective{}}.apply(m, layer, xn, mat)
	if mx := epMaxAbs(got, mono); mx != 0 {
		t.Fatalf("glmMoeEPFFN(ranks=1).apply vs glmMoeFFN.apply max|Δ|=%g, want 0 (bit-exact)", mx)
	}
	// And the ranks=2 ffnKind the dispatch actually returned matches within reassociation round-off.
	got2 := ep.apply(m, layer, xn, mat)
	if cos := epCosine(got2, mono); cos < 0.99999 {
		t.Fatalf("dispatched glmMoeEPFFN(ranks=2).apply vs glmMoeFFN cosine=%.8f, want ≥ 0.99999", cos)
	}
	t.Logf("glmMoeEPFFN wired into ffnForLayer: epRanks≤1 -> monolith, >1 -> EP; ranks=1 bit-exact, ranks=2 within round-off")
}

// TestGlmMoeEPFFNReducesThroughDeviceCollective pins the DEVICE-collective seam into the live
// decode path — the gap GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md named: serve.go gates
// `--expert-parallel N>1` on a backend advertising Caps().Collective (the NCCL CollectiveBackend),
// but ffnForLayer reduced glmMoeEPFFN through a HARDCODED LocalCollective, so the device communicator
// the serve initialized was dead code in decode. After SetExpertParallelCollective, the dispatched
// glmMoeEPFFN carries the wired Collective and reduces the routed-expert partials through it.
//
// Proven on cpu-ref, whose BackendCollective is byte-identical to LocalCollective
// (collective_bridge_test.go), so the device-collective decode path is exercised end-to-end with
// NO multi-GPU hardware and the reduction is bit-exact vs the LocalCollective-driven EP — the EP
// decode twin of TestForwardTPViaBackendCollective. On a real NCCL backend the SAME glmMoeEPFFN.apply
// issues a cross-GPU all-reduce per MoE layer (the first live multi-GPU decode path); only the
// reduction order (NCCL ring/tree) then differs, within the Approx round-off every device op carries.
func TestGlmMoeEPFFNReducesThroughDeviceCollective(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	mat := residentKernel{m}
	layer := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if m.hasWeight(routerName(l)) && m.hasWeight(expertName(l, 0, "gate_proj.weight")) {
			layer = l
			break
		}
	}
	if layer < 0 {
		t.Fatalf("no routed MoE layer in the glm fixture")
	}

	// The serve wires a BackendCollective over the device backend (cpu-ref here; the NCCL backend
	// on a multi-GPU box) — exactly what gateway.New does when ExpertParallelRanks > 1.
	bc, err := NewBackendCollective(compute.Default())
	if err != nil {
		t.Fatalf("NewBackendCollective(cpu-ref): %v", err)
	}
	m.SetExpertParallelRanks(2)
	m.SetExpertParallelCollective(bc)

	// The dispatched ffnKind must carry the WIRED collective, not the hardcoded LocalCollective —
	// the regression guard that the device-collective seam actually reaches the decode reduction.
	ep, ok := m.ffnForLayer(layer).(glmMoeEPFFN)
	if !ok {
		t.Fatalf("epRanks=2: ffnForLayer = %T, want glmMoeEPFFN", m.ffnForLayer(layer))
	}
	if _, isLocal := ep.coll.(LocalCollective); isLocal {
		t.Fatalf("dispatched glmMoeEPFFN still reduces through LocalCollective; want the wired BackendCollective (device communicator would be dead code in decode)")
	}
	if _, isBC := ep.coll.(*BackendCollective); !isBC {
		t.Fatalf("dispatched glmMoeEPFFN coll = %T, want *BackendCollective (the device-collective seam)", ep.coll)
	}

	xn := make([]float32, cfg.HiddenSize)
	for i := range xn {
		xn[i] = float32(math.Sin(float64(i)*0.07 + 0.2))
	}

	// Bit-exact vs the SAME EP reduced through LocalCollective: BackendCollective(cpu-ref) ==
	// LocalCollective byte-for-byte, so routing the decode reduction through the device seam
	// changes no host bytes (the no-regression rung for the wiring).
	plan2, err := ExpertParallelPlan(cfg.NumExperts, 2)
	if err != nil {
		t.Fatalf("ExpertParallelPlan(2): %v", err)
	}
	viaDevice := ep.apply(m, layer, xn, mat)
	viaLocal := glmMoeEPFFN{plan: plan2, coll: LocalCollective{}}.apply(m, layer, xn, mat)
	if mx := epMaxAbs(viaDevice, viaLocal); mx != 0 {
		t.Fatalf("EP via device collective vs via LocalCollective max|Δ|=%g, want 0 (bit-exact on cpu-ref)", mx)
	}

	// Clearing the collective restores the LocalCollective default (no device state leaks into a
	// subsequent host-only serve).
	m.SetExpertParallelCollective(nil)
	if ep2, ok := m.ffnForLayer(layer).(glmMoeEPFFN); !ok {
		t.Fatalf("epRanks=2 after clear: ffnForLayer = %T, want glmMoeEPFFN", m.ffnForLayer(layer))
	} else if _, isLocal := ep2.coll.(LocalCollective); !isLocal {
		t.Fatalf("after SetExpertParallelCollective(nil) coll = %T, want LocalCollective default", ep2.coll)
	}
	t.Logf("EP decode reduction flows through the wired device collective (cpu-ref BackendCollective), bit-exact vs LocalCollective; on NCCL the same apply all-reduces across GPUs")
}
