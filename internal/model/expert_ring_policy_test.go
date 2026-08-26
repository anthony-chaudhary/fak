package model

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_ring_policy_test.go — the R4 witnesses for #5615 (epic #5606): the ring's victim ranking is
// now a seam, the ring records the ordered trace that seam must be judged on, and the promotion is
// gated on measured regret.
//
// The claim the plan states — "GoodDecisionRatio of the shipped policy on a real routing trace,
// reported against LRU on the same trace; promote only on a positive delta with no hit regression"
// — is witnessed in both directions: the gate promotes on the workload the candidate was designed
// for and REFUSES on one where it merely ties, and the policy it promotes is the one that actually
// wins on the live ring at the same budget.

// expertPolicySession is a ring session with an explicit victim policy and no pin knobs, so the
// policy is the only variable between two runs.
func expertPolicySession(m *Model, ringBytes int64, policy ExpertRingEvictPolicy) *Session {
	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	return &Session{
		M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{},
		ExpertRingBytes: ringBytes,
		ExpertRingEvict: policy,
	}
}

// expertJitterWindow is the workload the value-aware policy targets and pure recency gets wrong: a
// small stable hot set ({0,1}) re-touched between a stream of cold one-off experts. LRU lets each
// one-off evict a hot resident purely because the hot one was touched slightly less recently; a
// frequency signal sheds the one-off instead. It mirrors GenerateHotSetJitterTrace's shape so the
// live ring and the offline corpus describe the same phenomenon.
func expertJitterWindow(hot, cold int) []int {
	var w []int
	touchHot := func() {
		for e := 0; e < hot; e++ {
			w = append(w, e)
		}
	}
	touchHot()
	touchHot()
	for c := 0; c < cold; c++ {
		w = append(w, hot+c)
		touchHot()
	}
	return w
}

// driveExpertWindow activates the window against a session and reports the ring ledger.
func driveExpertWindow(s *Session, m *Model, window []int) ExpertRingStats {
	x := expertRingTestInput(m.Cfg.HiddenSize)
	for _, e := range window {
		expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: s})
	}
	return s.ExpertRing()
}

// TestExpertRingValueAwarePolicyBeatsLRUOnJitter is the load-bearing witness: at the SAME budget on
// the SAME workload, the promoted policy pages in strictly less and hits strictly more than the LRU
// the ring inherited — on the live ring, not in the simulation that argued for it.
func TestExpertRingValueAwarePolicyBeatsLRUOnJitter(t *testing.T) {
	const H, E = 256, 8
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 6 // two whole experts: exactly the hot set, with nothing spare for jitter
	window := expertJitterWindow(2, 6)

	lruSession := expertPolicySession(m, budget, ExpertRingEvictLRU)
	defer lruSession.Close()
	lru := driveExpertWindow(lruSession, m, window)

	vaSession := expertPolicySession(m, budget, ExpertRingEvictValueAware)
	defer vaSession.Close()
	va := driveExpertWindow(vaSession, m, window)

	t.Logf("lru: page-ins=%d hits=%d evictions=%d", lru.PageIns, lru.Hits, lru.Evictions)
	t.Logf("value-aware: page-ins=%d hits=%d evictions=%d", va.PageIns, va.Hits, va.Evictions)

	if va.PageIns >= lru.PageIns {
		t.Fatalf("value-aware paged in %d weights vs LRU's %d — the policy seam bought nothing", va.PageIns, lru.PageIns)
	}
	if va.Hits <= lru.Hits {
		t.Fatalf("value-aware scored %d hits vs LRU's %d — fewer page-ins must not come from doing less work", va.Hits, lru.Hits)
	}
	// Both must still respect the bound: a policy that wins by exceeding the budget has not won.
	for name, st := range map[string]ExpertRingStats{"lru": lru, "value-aware": va} {
		if st.PeakBytes > st.BudgetBytes {
			t.Fatalf("%s peak resident %d exceeds budget %d", name, st.PeakBytes, st.BudgetBytes)
		}
	}
	// Same total work, so a hit the value-aware ring won is a page-in it did not pay.
	if va.Hits+va.PageIns != lru.Hits+lru.PageIns {
		t.Fatalf("accesses differ (%d vs %d); the two runs did not see the same workload",
			va.Hits+va.PageIns, lru.Hits+lru.PageIns)
	}
}

// TestExpertRingDefaultPolicyIsLRUUnchanged is the default-off gate: the zero value allocates no
// heat state and evicts exactly as the ring always has, so R0/R2 sessions do not move.
func TestExpertRingDefaultPolicyIsLRUUnchanged(t *testing.T) {
	const H, E = 256, 8
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	window := expertJitterWindow(2, 6)

	declared := expertPolicySession(m, perWeight*6, ExpertRingEvictLRU)
	defer declared.Close()
	withLRU := driveExpertWindow(declared, m, window)

	// A session that never mentions a policy at all — the R0 constructor.
	silent := expertRingSession(m, perWeight*6)
	defer silent.Close()
	byDefault := driveExpertWindow(silent, m, window)

	if withLRU != byDefault {
		t.Fatalf("declaring ExpertRingEvictLRU changed the ledger: %+v vs default %+v", withLRU, byDefault)
	}
	if silent.expertRing.heat != nil {
		t.Fatal("the default policy allocated a heat map; LRU must cost nothing it did not cost before")
	}
	if silent.expertRing.policy != ExpertRingEvictLRU {
		t.Fatalf("default ring policy = %v, want LRU", silent.expertRing.policy)
	}
}

// TestSelectExpertRingEvictPolicyPromotesOnlyOnEvidence pins the gate in both directions. The rule
// is a STRICT eviction win with no hit regression: anything else keeps the incumbent, and says why.
func TestSelectExpertRingEvictPolicyPromotesOnlyOnEvidence(t *testing.T) {
	// (a) the workload the candidate is for — promote.
	jitter := GenerateHotSetJitterTrace(2, 6, 1024, 2048)
	policy, decision, err := SelectExpertRingEvictPolicy(jitter, ExpertResidencyLFUOptions{})
	if err != nil {
		t.Fatalf("SelectExpertRingEvictPolicy(jitter): %v", err)
	}
	if policy != ExpertRingEvictValueAware || !decision.Promoted {
		t.Fatalf("jitter trace did not promote: %+v", decision)
	}
	if decision.EvictionDelta <= 0 || decision.HitDelta < 0 {
		t.Fatalf("promoted on the wrong evidence: eviction delta %d, hit delta %d", decision.EvictionDelta, decision.HitDelta)
	}
	if decision.CandidateGoodDecisionRatio < decision.LRUGoodDecisionRatio {
		t.Fatalf("promoted a policy further from the oracle than LRU: %.3f vs %.3f",
			decision.CandidateGoodDecisionRatio, decision.LRUGoodDecisionRatio)
	}

	// (b) a sweep longer than the budget: every access misses under either ranking, so the candidate
	// only TIES. A tie is not evidence, and the incumbent stays.
	var sweep []ExpertAccessTraceEvent
	for i := 0; i < 16; i++ {
		sweep = append(sweep, ExpertAccessTraceEvent{Layer: 0, Expert: i % 4, WeightBytes: 1024})
	}
	flat := ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: "round-robin-sweep", Source: "synthetic",
		BudgetBytes: 2048, Events: sweep,
	}
	policy, decision, err = SelectExpertRingEvictPolicy(flat, ExpertResidencyLFUOptions{})
	if err != nil {
		t.Fatalf("SelectExpertRingEvictPolicy(sweep): %v", err)
	}
	if policy != ExpertRingEvictLRU || decision.Promoted {
		t.Fatalf("a tie promoted the candidate: %+v", decision)
	}
	if decision.Reason == "" {
		t.Fatal("the gate kept LRU without saying why; an unexplained demotion is indistinguishable from a default")
	}

	// (c) an unmeasurable trace is an ERROR that keeps LRU — not a silent demotion, because "could
	// not measure" and "measured and lost" are different facts.
	policy, decision, err = SelectExpertRingEvictPolicy(ExpertAccessTrace{}, ExpertResidencyLFUOptions{})
	if err == nil {
		t.Fatal("an empty trace was scored rather than refused")
	}
	if policy != ExpertRingEvictLRU || decision.Promoted {
		t.Fatalf("an unmeasurable trace changed the policy: %+v", decision)
	}
}

// TestExpertRingTraceFeedsTheGate closes the loop the rung exists for: the trace the LIVE ring
// recorded, replayed through the offline gauge, recommends the policy that actually wins live.
func TestExpertRingTraceFeedsTheGate(t *testing.T) {
	const H, E = 256, 8
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 6
	window := expertJitterWindow(2, 6)

	s := expertPolicySession(m, budget, ExpertRingEvictLRU) // measure while running the incumbent
	defer s.Close()
	driveExpertWindow(s, m, window)

	trace := s.ExpertRingTrace()
	if trace.BudgetBytes != budget {
		t.Fatalf("trace budget %d, want the ring's own %d", trace.BudgetBytes, budget)
	}
	if len(trace.Events) != len(window) {
		t.Fatalf("trace has %d accesses for a %d-activation window; the three projections of one activation must coalesce into one access",
			len(trace.Events), len(window))
	}
	for i, e := range trace.Events {
		if e.Expert != window[i] || e.Layer != 0 {
			t.Fatalf("trace event %d = (layer %d, expert %d), want expert %d in order", i, e.Layer, e.Expert, window[i])
		}
		if e.WeightBytes != perWeight*3 {
			t.Fatalf("trace event %d sized %d, want one whole expert (%d)", i, e.WeightBytes, perWeight*3)
		}
	}
	if trace.UnsizedTouches != 0 {
		t.Fatalf("%d accesses were dropped from a window far under the trace limit", trace.UnsizedTouches)
	}

	policy, decision, err := s.SelectExpertRingEvictPolicy(ExpertResidencyLFUOptions{})
	if err != nil {
		t.Fatalf("gate over the live trace: %v", err)
	}
	if policy != ExpertRingEvictValueAware || !decision.Promoted {
		t.Fatalf("the gate did not recommend the policy that wins live on this workload: %+v", decision)
	}
	t.Logf("verdict: %s", decision.Reason)
}

// TestExpertRingValueAwareRespectsPins is the interaction gate with R2: a durable pin is exempt from
// the value-aware ranking exactly as it is from LRU, so promoting the evictor cannot quietly
// un-protect the warm-started hot set.
func TestExpertRingValueAwareRespectsPins(t *testing.T) {
	const H, E = 256, 8
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	usage := filepath.Join(t.TempDir(), "expert-usage.json")

	// Build a prior in which expert 0 is hottest, then restart with it pinned under the value-aware
	// evictor and run a window whose LATER experts are much hotter than expert 0.
	warm := expertPinSession(m, perWeight*6, 1, usage)
	driveExpertWindow(warm, m, []int{0, 0, 0, 1})
	if _, err := warm.ExpertRingEndTurn(0.9, 1); err != nil {
		t.Fatalf("ExpertRingEndTurn: %v", err)
	}
	warm.Close()

	s := expertPinSession(m, perWeight*6, 1, usage)
	s.ExpertRingEvict = ExpertRingEvictValueAware
	defer s.Close()
	// Expert 0 is touched once; experts 2..6 are hammered, so on heat alone expert 0 is the obvious
	// victim. Being pinned, it must survive anyway.
	window := []int{0}
	for i := 0; i < 4; i++ {
		window = append(window, 2, 3, 4, 5, 6)
	}
	driveExpertWindow(s, m, window)

	if got := s.ExpertRing().PinnedCount; got != 1 {
		t.Fatalf("PinnedCount=%d, want the warm-started pin", got)
	}
	if !s.expertRing.isExpertPinned(0, 0) {
		t.Fatal("the warm start did not pin expert 0; the rest of this witness proves nothing")
	}
	for _, proj := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
		key := "q4k:" + expertName(0, 0, proj)
		if !s.expertRing.isResident(key) {
			t.Fatalf("pinned expert 0's %s was evicted by the value-aware ranking", proj)
		}
	}
	if s.ExpertRing().PeakBytes > s.ExpertRing().BudgetBytes {
		t.Fatal("pinning under the value-aware policy breached the byte bound")
	}
}

// TestExpertRingValueAwareLeavesRefusalUntouched pins the all-or-nothing contract across the new
// eviction path: a weight the ring cannot admit must page NOTHING out on its way to being refused,
// or a misconfigured budget would cost residency it never got anything for.
func TestExpertRingValueAwareLeavesRefusalUntouched(t *testing.T) {
	const H, E = 256, 8
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)

	s := expertPolicySession(m, perWeight*3, ExpertRingEvictValueAware) // one whole expert
	defer s.Close()
	driveExpertWindow(s, m, []int{0, 0})
	before := s.ExpertRing()
	if before.ResidentCount != 3 {
		t.Fatalf("resident count %d, want expert 0's three projections", before.ResidentCount)
	}

	// A weight larger than the entire budget: polymodel's ErrTooLarge case. It must not trigger the
	// policy's eviction pass, because evicting for a weight that can never be admitted is pure loss.
	r := s.expertRing
	_, ok := r.stage("q4k:oversized", func() compute.Tensor {
		return compute.NewF32(compute.Default(), []int{1, 1}, []float32{1})
	}, compute.F32, r.budget()*4, false)
	if ok {
		t.Fatal("a weight larger than the whole budget was admitted")
	}
	after := s.ExpertRing()
	if after.ResidentCount != before.ResidentCount || after.ResidentBytes != before.ResidentBytes {
		t.Fatalf("a refused stage moved the resident set: %d/%d -> %d/%d",
			before.ResidentCount, before.ResidentBytes, after.ResidentCount, after.ResidentBytes)
	}
	if after.Evictions != before.Evictions {
		t.Fatalf("a refused stage evicted %d weights", after.Evictions-before.Evictions)
	}
}

func TestExpertRingPolicySwapPreservesResidencyAndStartsTraceEpoch(t *testing.T) {
	const H, E = 256, 8
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	s := expertPolicySession(m, perWeight*6, ExpertRingEvictLRU)
	defer s.Close()
	driveExpertWindow(s, m, []int{0, 1, 0})

	before := s.ExpertRing()
	beforeTrace := s.ExpertRingTrace()
	if before.Policy != ExpertRingEvictLRU || before.PolicyGeneration != 1 || len(beforeTrace.Events) == 0 {
		t.Fatalf("initial epoch stats=%+v trace=%+v, want LRU generation 1 with evidence", before, beforeTrace)
	}
	resident := make(map[string]any, len(s.expertRing.resident))
	for id, tensor := range s.expertRing.resident {
		resident[string(id)] = tensor.Buf()
	}
	pins, budget := s.expertRing.pins, s.expertRing.budget()

	receipt, err := s.SwapExpertRingEvictPolicy(ExpertRingEvictValueAware)
	if err != nil {
		t.Fatalf("SwapExpertRingEvictPolicy: %v", err)
	}
	if !receipt.Changed || receipt.PreviousPolicy != ExpertRingEvictLRU || receipt.Policy != ExpertRingEvictValueAware ||
		receipt.PreviousGeneration != 1 || receipt.PolicyGeneration != 2 {
		t.Fatalf("receipt=%+v, want LRU/1 -> value-aware/2", receipt)
	}
	if receipt.PriorStats != before || len(receipt.PriorTrace.Events) != len(beforeTrace.Events) {
		t.Fatalf("receipt did not seal prior epoch: stats=%+v trace=%+v", receipt.PriorStats, receipt.PriorTrace)
	}
	after := s.ExpertRing()
	afterTrace := s.ExpertRingTrace()
	if after.Policy != ExpertRingEvictValueAware || after.PolicyGeneration != 2 || len(afterTrace.Events) != 0 ||
		afterTrace.Policy != ExpertRingEvictValueAware || afterTrace.PolicyGeneration != 2 {
		t.Fatalf("new epoch stats=%+v trace=%+v", after, afterTrace)
	}
	if after.ResidentCount != before.ResidentCount || after.ResidentBytes != before.ResidentBytes ||
		after.PageIns != before.PageIns || after.Hits != before.Hits || after.Evictions != before.Evictions ||
		s.expertRing.pins != pins || s.expertRing.budget() != budget {
		t.Fatalf("swap changed durable/cumulative ring state: before=%+v after=%+v", before, after)
	}
	for id, tensor := range s.expertRing.resident {
		if resident[string(id)] != tensor.Buf() {
			t.Fatalf("resident %s handle changed across swap", id)
		}
	}
	if s.expertRing.heat != nil || s.expertRing.lastUse != nil || s.expertRing.clock != 0 || s.expertRing.accesses != 0 {
		t.Fatalf("policy-local state survived swap: heat=%v lastUse=%v clock=%d accesses=%d",
			s.expertRing.heat, s.expertRing.lastUse, s.expertRing.clock, s.expertRing.accesses)
	}

	driveExpertWindow(s, m, []int{0})
	if got := s.ExpertRingTrace(); got.Policy != ExpertRingEvictValueAware || got.PolicyGeneration != 2 || len(got.Events) == 0 {
		t.Fatalf("post-swap access was not recorded in new epoch: %+v", got)
	}
}

func TestExpertRingPolicySwapRejectsInvalidAndNoopsSamePolicy(t *testing.T) {
	m := expertRingTestModel(t, 256, 4)
	s := expertPolicySession(m, expertRingWeightBytes(t, m)*3, ExpertRingEvictLRU)
	defer s.Close()
	driveExpertWindow(s, m, []int{0, 0})

	beforeStats, beforeTrace := s.ExpertRing(), s.ExpertRingTrace()
	heatLen, lastUseLen, clock, accesses := len(s.expertRing.heat), len(s.expertRing.lastUse), s.expertRing.clock, s.expertRing.accesses
	if _, err := s.SwapExpertRingEvictPolicy(ExpertRingEvictPolicy(99)); err == nil {
		t.Fatal("invalid policy was accepted")
	}
	if got := s.ExpertRing(); got != beforeStats {
		t.Fatalf("invalid policy changed stats: before=%+v after=%+v", beforeStats, got)
	}
	receipt, err := s.SwapExpertRingEvictPolicy(ExpertRingEvictLRU)
	if err != nil || receipt.Changed {
		t.Fatalf("same-policy swap receipt=%+v err=%v", receipt, err)
	}
	if got := s.ExpertRing(); got != beforeStats || len(s.ExpertRingTrace().Events) != len(beforeTrace.Events) ||
		len(s.expertRing.heat) != heatLen || len(s.expertRing.lastUse) != lastUseLen || s.expertRing.clock != clock || s.expertRing.accesses != accesses {
		t.Fatalf("same-policy request changed epoch or policy-local state")
	}
}
