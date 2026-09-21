package model

// expert_checkpoint_warm_test.go — the CW-15 execution witnesses for #13343
// (parent #12640, epic #12952; performance frontier #13294).
//
// #13349 (landed at 3da8409123) owns the SELECTION claim: given a recorded
// demand profile, ExpertWarmPlanFor returns a deterministic, budget-bounded set
// of canonical routed-expert names. This leaf owns the separate EXECUTION claim:
// that a selected warm actually populates residency, so a later real
// v41ExpertF32Into demand for a retained projection is served without any source
// read and returns byte-identical weights — the first-token path's disk fault is
// removed for the routed experts a prefill will touch.
//
// The tests drive the real tier (fault + bounded pool admission), not a mock:
// the "avoids source reads" claim is read off ExpertCheckpointStats.Reads, the
// tier's own independent ledger, so a warm that quietly streamed and dropped
// bytes cannot masquerade as residency.

import (
	"bytes"
	"testing"
)

// expertWarmPlanProfile returns a demand profile naming every projection of
// experts [0, count) at layer 0, ordered so expert 0 has the highest demand.
func expertWarmPlanProfile(count int) ExpertWarmProfile {
	id := expertWarmTestIdentity()
	entries := make([]ExpertWarmDemandEntry, 0, count)
	for e := 0; e < count; e++ {
		entries = append(entries, ExpertWarmDemandEntry{Layer: 0, Expert: e, Count: float64(count - e)})
	}
	return ExpertWarmProfile{Identity: id, Entries: entries}
}

// expertWarmV41Model is the V4.1-arch twin of expertCheckpointTestModel: it builds
// the same routed-expert MoE and moves its projections into fused checkpoint
// slabs, but stamps the slabs with Arch "deepseek41" so the tier indexes the
// native ffn.experts.<e>.{w1,w3,w2}.weight leaves the real V4.1 demand door
// (v41ExpertF32) resolves. The shared GLM-spelled fixture cannot drive that door:
// its names live under mlp.experts.<e>.<proj>, so a v41ExpertF32 demand would
// miss the tier and the "warmed, then read" witness would prove nothing.
func expertWarmV41Model(t *testing.T, hidden, experts, topK int, hostBytes int64) (*Model, *ExpertCheckpointTier, int64) {
	t.Helper()
	resident := expertPrefetchModel(t, hidden, experts, topK)

	var blob []byte
	fused := make([]FusedExpertTensor, 0, 3)
	for _, proj := range []string{"gate", "up", "down"} {
		suffix := proj + "_proj.weight"
		offset := int64(len(blob))
		for e := 0; e < experts; e++ {
			blob = append(blob, resident.q4kw[expertName(0, e, suffix)].raw...)
		}
		fused = append(fused, FusedExpertTensor{
			Name:    "blk.0.ffn_" + proj + "_exps.weight",
			Layer:   0,
			Proj:    proj + "_proj",
			Arch:    "deepseek41",
			Quant:   ExpertCheckpointQ4K,
			Offset:  offset,
			Experts: experts,
			Rows:    hidden,
			Cols:    hidden,
		})
	}
	stride := int64(len(resident.q4kw[expertName(0, 0, "gate_proj.weight")].raw))

	m := &Model{Cfg: resident.Cfg, q4kw: map[string]*q4kTensor{routerName(0): resident.q4kw[routerName(0)]}}
	tier := NewExpertCheckpointTier(hostBytes)
	if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), fused); err != nil {
		t.Fatalf("AddShard over a well-formed %d-byte V4.1 checkpoint: %v", len(blob), err)
	}
	m.SetExpertCheckpoint(tier)
	return m, tier, stride
}

// TestV41ExpertCacheWarmPlan is the named witness required by #13343.
//
// It pins the three scoped acceptance criteria:
//  1. a later real demand for a retained projection avoids source reads and
//     returns identical weights;
//  2. the host-byte ceiling is respected and demand hot state is never evicted
//     to make room for a warm, and a zero/insufficient budget skips rather than
//     warming;
//  3. warm source reads are attributed separately from the recorded demand.
func TestV41ExpertCacheWarmPlan(t *testing.T) {
	const H, E, K = 256, 8, 4
	id := expertWarmTestIdentity()

	t.Run("warmed_projection_is_served_without_a_further_read", func(t *testing.T) {
		// A tier with room for a few projections, and the real V4.1 model whose
		// routed experts live ONLY in that tier. One projection's stride is the
		// byte unit both the tier budget and the warm budget are stated in.
		_, _, stride := expertWarmV41Model(t, H, E, K, 0)
		m, tier, _ := expertWarmV41Model(t, H, E, K, int64(3*E)*stride)

		profile := expertWarmPlanProfile(2)
		res := m.WarmExpertProfile(profile, id, int64(3*2)*stride)
		if res.Reason != ExpertWarmReasonPlanned {
			t.Fatalf("warm reason = %q; want planned", res.Reason)
		}
		if res.RetainedBytes <= 0 || res.Retained == 0 {
			t.Fatalf("warm retained nothing: %+v", res)
		}
		// The warm itself reads exactly the selected projections, once each.
		if res.ReadBytes != res.RetainedBytes {
			t.Fatalf("warm read %d bytes but retained %d; a warm read must be the residency it keeps",
				res.ReadBytes, res.RetainedBytes)
		}

		afterWarm := tier.Stats().Reads
		if afterWarm == 0 {
			t.Fatal("warm issued no reads; the later no-read claim would be vacuous")
		}

		// A REAL demand for a warmed projection (expert 0's w1) must be served
		// from residency: zero new tier reads.
		cold, err := m.v41ExpertF32(0, "ffn.experts.0.w1.weight")
		if err != nil {
			t.Fatalf("demand for a warmed projection: %v", err)
		}
		if got := tier.Stats().Reads; got != afterWarm {
			t.Fatalf("a warmed projection cost %d further reads (Reads %d -> %d); it was not retained",
				got-afterWarm, afterWarm, got)
		}

		// Identical weights: read the same projection from a cold tier and compare.
		want := coldExpertProjection(t, H, E, K, 0, "ffn.experts.0.w1.weight")
		if len(cold) != len(want) {
			t.Fatalf("warmed projection length %d, want %d", len(cold), len(want))
		}
		for i := range want {
			if cold[i] != want[i] {
				t.Fatalf("warmed projection differs from a cold read at %d: %v vs %v", i, cold[i], want[i])
			}
		}
		if st := tier.Stats(); st.BudgetBytes != int64(3*E)*stride {
			t.Fatalf("tier budget changed to %d; the warm must respect the declared ceiling", st.BudgetBytes)
		}
	})

	t.Run("budget_zero_skips_and_never_warms", func(t *testing.T) {
		m, _, _ := expertWarmV41Model(t, H, E, K, 0)
		res := m.WarmExpertProfile(expertWarmPlanProfile(2), id, 0)
		if res.Warmed() {
			t.Fatalf("zero-budget warm retained bytes: %+v", res)
		}
		if res.Reason != ExpertWarmReasonBudgetZero {
			t.Fatalf("reason = %q; want %q", res.Reason, ExpertWarmReasonBudgetZero)
		}
		if res.RetainedBytes != 0 || res.ReadBytes != 0 {
			t.Fatalf("skipped warm still moved bytes: %+v", res)
		}
	})

	t.Run("insufficient_budget_retains_nothing_and_reports_skipped", func(t *testing.T) {
		_, _, stride := expertWarmV41Model(t, H, E, K, 0)
		m, tier, _ := expertWarmV41Model(t, H, E, K, stride/2)
		// Tier budget holds less than one projection: nothing can be retained.
		res := m.WarmExpertProfile(expertWarmPlanProfile(1), id, stride)
		if res.Warmed() {
			t.Fatalf("warm retained bytes under a sub-projection budget: %+v", res)
		}
		if res.Reason != ExpertWarmReasonNoFit && res.Reason != ExpertWarmReasonBudgetZero {
			t.Fatalf("reason = %q; want no_fit or budget_zero", res.Reason)
		}
		if got := tier.Stats().Reads; got != 0 {
			t.Fatalf("a warm that cannot retain anything issued %d reads; a warm must not stream-and-drop", got)
		}
	})

	t.Run("warm_does_not_evict_demand_hot_state", func(t *testing.T) {
		// Tier room for exactly six projections. First establish a demanded hot
		// entry by a real demand (expert 4's w1), then warm experts 0..1 whose
		// three projections each (six total) exactly fill the pool. The warm must
		// retain its set AND leave the demanded expert 4 entry resident.
		_, _, stride := expertWarmV41Model(t, H, E, K, 0)
		m, tier, _ := expertWarmV41Model(t, H, E, K, int64(6)*stride)
		if _, err := m.v41ExpertF32(0, "ffn.experts.4.w1.weight"); err != nil {
			t.Fatalf("establishing the demanded hot entry: %v", err)
		}
		demandReads := tier.Stats().Reads

		res := m.WarmExpertProfile(expertWarmPlanProfile(2), id, int64(3*2)*stride)
		if res.Reason != ExpertWarmReasonPlanned {
			t.Fatalf("reason = %q; want planned", res.Reason)
		}

		// The demanded entry must still be resident: a second demand is a hit.
		if _, err := m.v41ExpertF32(0, "ffn.experts.4.w1.weight"); err != nil {
			t.Fatalf("re-demand for the hot entry: %v", err)
		}
		if got := tier.Stats().Reads; got != demandReads+res.Reads {
			t.Fatalf("re-demanding the hot entry cost a further read (Reads %d -> %d, warm read %d); "+
				"the warm evicted demanded hot state", demandReads, got, res.Reads)
		}
	})

	t.Run("stale_identity_warms_nothing", func(t *testing.T) {
		_, _, stride := expertWarmV41Model(t, H, E, K, 0)
		m, tier, _ := expertWarmV41Model(t, H, E, K, int64(3*E)*stride)
		stale := ExpertWarmProfileIdentity{Checkpoint: "other", Quantization: "Q2_K", Layout: "ffn.experts"}
		res := m.WarmExpertProfile(expertWarmPlanProfile(2), stale, int64(3*2)*stride)
		if res.Warmed() || res.Reason != ExpertWarmReasonStaleIdentity {
			t.Fatalf("stale profile: warmed=%v reason=%q; want skipped/stale_identity", res.Warmed(), res.Reason)
		}
		if got := tier.Stats().Reads; got != 0 {
			t.Fatalf("a stale-identity warm issued %d reads", got)
		}
	})

	t.Run("nil_tier_skips_without_panicking", func(t *testing.T) {
		var m *Model
		res := m.WarmExpertProfile(expertWarmPlanProfile(1), id, 1<<20)
		if res.Warmed() {
			t.Fatalf("nil model warmed bytes: %+v", res)
		}
	})
}

// coldExpertProjection resolves one projection from a fresh, unwarmed tier with
// the identical bytes, so a warm-vs-cold comparison is a data comparison, never
// a fixture comparison.
func coldExpertProjection(t *testing.T, hidden, experts, topK int, layer int, leaf string) []float32 {
	t.Helper()
	m, _, _ := expertWarmV41Model(t, hidden, experts, topK, 0)
	w, err := m.v41ExpertF32(layer, leaf)
	if err != nil {
		t.Fatalf("cold read of %s: %v", leaf, err)
	}
	return w
}

// TestV41ExpertCacheWarmPlanReadsAreAttributedSeparately pins acceptance item 3:
// the warm's source reads land in the tier gate's own counters (Reads/BytesRead)
// and never in the recorded-demand profile the selector consumes, so a warm can
// never inflate the demand evidence that justified it.
func TestV41ExpertCacheWarmPlanReadsAreAttributedSeparately(t *testing.T) {
	const H, E, K = 256, 8, 4
	id := expertWarmTestIdentity()
	_, _, stride := expertWarmV41Model(t, H, E, K, 0)
	m, tier, _ := expertWarmV41Model(t, H, E, K, int64(3*E)*stride)

	profile := expertWarmPlanProfile(2)
	res := m.WarmExpertProfile(profile, id, int64(3*2)*stride)
	if res.Reason != ExpertWarmReasonPlanned {
		t.Fatalf("reason = %q; want planned", res.Reason)
	}

	// The profile the caller handed in is a value; the warm must not mutate it
	// into demand evidence.
	if len(profile.Entries) != 2 {
		t.Fatalf("warm mutated the caller's profile: %d entries", len(profile.Entries))
	}
	// Every byte the warm moved is visible in the tier gate's ledger.
	st := tier.Stats()
	if st.Reads != res.Reads || st.BytesRead != res.ReadBytes {
		t.Fatalf("warm counters (%d reads/%d bytes) diverge from the tier ledger (%d/%d)",
			res.Reads, res.ReadBytes, st.Reads, st.BytesRead)
	}
	// And the retained set is real residency, not a read that dropped its bytes.
	if st.ResidentBytes < res.RetainedBytes {
		t.Fatalf("tier resident bytes %d < warm retained %d", st.ResidentBytes, res.RetainedBytes)
	}
}
