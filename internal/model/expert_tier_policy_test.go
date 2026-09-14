package model

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
	"github.com/anthony-chaudhary/fak/pkg/moecache"
)

// expert_tier_policy_test.go — the #1300 witnesses for the three-tier (VRAM/L2/NVMe) expert streaming
// policy. One test per acceptance criterion, plus the vocabulary mapping AC1 pins:
//
//	AC1 — ExpertCacheTier maps onto the EXISTING pkg/moecache tiers; no fourth tier.
//	AC2 — an NVMe miss fills L2 when L2 has room and PromoteOnMiss is set, and refuses when full.
//	AC3 — an aligned span covers a non-4096 stride with a bounded over-read.
//	AC4 — demotion never drops a pinned/warm expert; if all candidates are pinned nothing is demoted.
//	AC5 — the zero-value policy is a no-op: no fill, no demotion, no allocation.
//	AC6 — Session.ExpertTierStats folds the ring, checkpoint and policy ledgers per tier.

// tierPolicyTestSession builds the smallest session that can carry the policy: no model, no ring, no
// checkpoint. That is deliberate for the pure promotion/geometry witnesses, which must not depend on
// any decode path to be meaningful.
func tierPolicyTestSession(policy ExpertTierPolicy) *Session {
	return &Session{ExpertTier: policy}
}

// tierPolicyTestName is the canonical routed-expert tensor name for (layer, expert). The demotion
// half recovers the identity from this name, so the tests use the real naming rather than a stub.
func tierPolicyTestName(layer, expert int) string {
	return expertName(layer, expert, "gate_proj.weight")
}

// TestExpertCacheTierMoETierMappingAndValidity is AC1: the closed three-tier vocabulary, its string
// tokens, and the one-way mapping onto pkg/moecache. There is no fourth tier to invent, and the zero
// value is the fastest rung.
func TestExpertTierVocabularyMoETierMappingAndValidity(t *testing.T) {
	var zero ExpertCacheTier
	if zero != ExpertCacheTierVRAM {
		t.Fatalf("zero tier = %v, want VRAM", zero)
	}
	for _, tc := range []struct {
		tier    ExpertCacheTier
		name    string
		moetier moecache.Tier
	}{
		{ExpertCacheTierVRAM, "vram", moecache.TierDRAM},
		{ExpertCacheTierL2, "l2", moecache.TierCheckpoint},
		{ExpertCacheTierNVMe, "nvme", moecache.TierNVMe},
	} {
		if !tc.tier.valid() {
			t.Fatalf("%v is not valid", tc.tier)
		}
		if got := tc.tier.String(); got != tc.name {
			t.Fatalf("String(%d) = %q, want %q", int(tc.tier), got, tc.name)
		}
		if got := tc.tier.MoETier(); got != tc.moetier {
			t.Fatalf("MoETier(%v) = %q, want %q", tc.tier, got, tc.moetier)
		}
		if !tc.tier.MoETier().Valid() {
			t.Fatalf("MoETier(%v) = %q is not a member of the moecache vocabulary", tc.tier, tc.tier.MoETier())
		}
	}
	if ExpertCacheTier(99).valid() {
		t.Fatal("an out-of-range tier reported valid")
	}
	if got := ExpertCacheTier(99).MoETier(); got != moecache.TierDRAM {
		t.Fatalf("out-of-range MoETier = %q, want the VRAM fallback %q", got, moecache.TierDRAM)
	}
}

// TestExpertTierNVMeGeometryAlignsUnalignedStride is AC3: the aligned span covers the requested
// range, rounds down/up to the alignment, defaults to 4096, honors a custom alignment, and keeps the
// over-read strictly below one alignment unit.
func TestExpertTierNVMeGeometryAlignsUnalignedStride(t *testing.T) {
	var def ExpertNVMeGeometry
	if got := def.AlignBytes; got != 0 {
		t.Fatalf("zero geometry AlignBytes = %d, want 0 (the field is resolved lazily)", got)
	}

	// A stride that is not a multiple of 4096 and whose start is unaligned.
	const off, length = int64(1000), int64(5000)
	start, end := def.AlignedSpan(off, length)
	if start > off {
		t.Fatalf("aligned start %d does not cover offset %d", start, off)
	}
	if end < off+length {
		t.Fatalf("aligned end %d does not cover end %d", end, off+length)
	}
	if start%expertTierDefaultAlignBytes != 0 || end%expertTierDefaultAlignBytes != 0 {
		t.Fatalf("span [%d,%d) is not 4096-aligned", start, end)
	}
	overRead := end - start - length
	if overRead <= 0 || overRead >= expertTierDefaultAlignBytes {
		t.Fatalf("over-read %d not in (0,%d): the span must absorb the stride but stay bounded", overRead, expertTierDefaultAlignBytes)
	}
	if got := def.AlignedReadLen(off, length); got != end-start {
		t.Fatalf("AlignedReadLen = %d, want span %d", got, end-start)
	}

	// An already-aligned stride reads exactly its own bytes and over-reads nothing.
	if s, e := def.AlignedSpan(4096, 4096); s != 4096 || e != 8192 {
		t.Fatalf("aligned stride span = [%d,%d), want [4096,8192)", s, e)
	}
	if got := def.AlignedReadLen(4096, 4096); got != 4096 {
		t.Fatalf("aligned stride read length = %d, want 4096 (no over-read)", got)
	}

	// A custom alignment is honored and still bounds the over-read by the declared unit.
	custom := ExpertNVMeGeometry{AlignBytes: 512}
	cs, ce := custom.AlignedSpan(1000, 1000)
	if cs != 512 || ce != 2048 {
		t.Fatalf("512-aligned span = [%d,%d), want [512,2048)", cs, ce)
	}
	if got := custom.AlignedReadLen(1000, 1000); got != 1536 {
		t.Fatalf("512-aligned read length = %d, want 1536", got)
	}
	// Over-read is bounded by TWO alignment units: the down-round of the start plus the up-round of
	// the end. Both are < align, so the sum is < 2*align, and the span still covers the stride.
	if over := ce - cs - 1000; over <= 0 || over >= 2*512 {
		t.Fatalf("custom over-read %d not in (0,%d)", over, 2*512)
	}

	// A negative length is treated as zero, not as a backwards read.
	if s, e := def.AlignedSpan(4096, -1); s != 4096 || e != 4096 {
		t.Fatalf("negative length span = [%d,%d), want the empty span at 4096", s, e)
	}
}

// TestExpertTierPolicyPromotesOnMissUnderBudget is AC2: a checkpoint miss fills L2 while the declared
// L2Bytes budget has room, the fill is recorded as an NVMe->L2 promotion, and a fill with no room is
// REFUSED rather than made by evicting an existing L2 resident (so demotion stays meaningful).
func TestExpertTierPolicyPromotesOnMissUnderBudget(t *testing.T) {
	const align = 4096
	policy := ExpertTierPolicy{Enabled: true, L2Bytes: 2 * align, PromoteOnMiss: true}
	s := tierPolicyTestSession(policy)

	first, ok := s.PromoteExpertTier(tierPolicyTestName(0, 0), 0, 1000)
	if !ok || !first.Promoted {
		t.Fatalf("first promotion refused: %+v ok=%v", first, ok)
	}
	if first.From != ExpertCacheTierNVMe || first.To != ExpertCacheTierL2 {
		t.Fatalf("promotion ledger = %v -> %v, want NVMe -> L2", first.From, first.To)
	}
	if first.Bytes != align {
		t.Fatalf("promotion booked %d bytes, want the aligned span %d", first.Bytes, align)
	}

	second, ok := s.PromoteExpertTier(tierPolicyTestName(0, 1), 100, 2000)
	if !ok {
		t.Fatalf("second promotion refused with room to spare: %+v", second)
	}

	if used := s.expertTier.pool.Used(); used != 2*align {
		t.Fatalf("L2 used = %d, want the declared ceiling %d", used, 2*align)
	}
	if got := s.expertTier.pool.Budget(); got != policy.L2Bytes {
		t.Fatalf("L2 budget = %d, want the declared L2Bytes %d", got, policy.L2Bytes)
	}

	// Full: refuse, move nothing, and say why. A promotion that evicted an existing resident to fit
	// would make "full" unobservable and the AC4 demotion dead code.
	third, ok := s.PromoteExpertTier(tierPolicyTestName(0, 2), 0, 1000)
	if ok || third.Promoted {
		t.Fatalf("a promotion made room by eviction: %+v", third)
	}
	if third.Bytes != align || third.Reason == "" {
		t.Fatalf("refusal ledger = %+v, want the attempted span and a reason", third)
	}
	if used := s.expertTier.pool.Used(); used != 2*align {
		t.Fatalf("a refused promotion moved residency: used = %d, want %d", used, 2*align)
	}

	// Re-promoting a resident is a hit, not a second fill.
	again, ok := s.PromoteExpertTier(tierPolicyTestName(0, 0), 0, 1000)
	if ok || again.Promoted {
		t.Fatalf("re-promoting a resident recorded a promotion: %+v", again)
	}
	if !strings.Contains(again.Reason, "resident") {
		t.Fatalf("resident re-promotion reason = %q, want a hit explanation", again.Reason)
	}

	st := s.ExpertTierStats()
	if st.L2.Promotions != 2 || st.L2.Reads != 2 || st.L2.Bytes != 2*align {
		t.Fatalf("L2 counters = %+v, want 2 promotions / 2 reads / %d bytes", st.L2, 2*align)
	}
	if st.L2ResidentCount != 2 || st.L2ResidentBytes != 2*align {
		t.Fatalf("L2 residency = %d entries / %d bytes, want 2 / %d", st.L2ResidentCount, st.L2ResidentBytes, 2*align)
	}
	if st.PeakBytes > policy.L2Bytes {
		t.Fatalf("L2 peak %d exceeded the declared ceiling %d", st.PeakBytes, policy.L2Bytes)
	}
}

// TestExpertTierPolicyDemotionExcludesPinned is AC4: demotion walks L2 for the least-recently-used
// UNPINNED resident, never touches a resident the ring's pin-set protects, and when every candidate
// is pinned demotes nothing while reporting PreservedPinned > 0.
func TestExpertTierPolicyDemotionExcludesPinned(t *testing.T) {
	const align = 4096
	policy := ExpertTierPolicy{Enabled: true, L2Bytes: 4 * align, PromoteOnMiss: true}
	s := tierPolicyTestSession(policy)

	// Four unpinned L2 residents, in promotion order so their recency stamps are strictly increasing.
	for e := 0; e < 4; e++ {
		if _, ok := s.PromoteExpertTier(tierPolicyTestName(0, e), 0, 1000); !ok {
			t.Fatalf("promotion of expert %d refused", e)
		}
	}

	// A ring whose pin-set holds expert 0 — the OLDEST resident, so pinning is the only reason it
	// could survive an LRU demotion. This is the R2 warm-set seam the demotion must consult.
	ring := newPagedRing(nil, align)
	hist := NewExpertUsageHistogram()
	hist.Observe(0, 0, 5)
	ring.WarmStartPins(hist, 1)
	s.expertRing = ring
	if !ring.isExpertPinned(0, 0) {
		t.Fatal("test setup: expert 0 was not pinned; the witness proves nothing")
	}

	// Expert 0 is pinned, so the victim is the oldest UNPINNED resident: expert 1.
	d1, ok := s.DemoteExpertTier()
	if !ok || !d1.Demoted {
		t.Fatalf("first demotion did not happen: %+v ok=%v", d1, ok)
	}
	if d1.PreservedPinned != 1 {
		t.Fatalf("first demotion preserved %d pinned, want 1", d1.PreservedPinned)
	}
	if _, still := s.expertTier.l2[polymodel.ModelID(tierPolicyTestName(0, 1))]; still {
		t.Fatal("the victim expert 1 is still resident")
	}
	if _, live := s.expertTier.l2[polymodel.ModelID(tierPolicyTestName(0, 0))]; !live {
		t.Fatal("the pinned expert 0 was demoted; demotion dropped a warm expert")
	}

	// Drain the remaining unpinned residents.
	if _, ok := s.DemoteExpertTier(); !ok {
		t.Fatal("second demotion refused")
	}
	if _, ok := s.DemoteExpertTier(); !ok {
		t.Fatal("third demotion refused")
	}

	// Only the pinned expert remains: demote nothing, report PreservedPinned > 0.
	last, ok := s.DemoteExpertTier()
	if ok || last.Demoted {
		t.Fatalf("a demotion happened with every candidate pinned: %+v", last)
	}
	if last.PreservedPinned <= 0 {
		t.Fatalf("all-pinned pass reported PreservedPinned=%d, want > 0", last.PreservedPinned)
	}
	if !strings.Contains(last.Reason, "pinned") {
		t.Fatalf("all-pinned reason = %q, want it to name the pin", last.Reason)
	}
	if len(s.expertTier.l2) != 1 {
		t.Fatalf("L2 holds %d residents, want only the pinned expert", len(s.expertTier.l2))
	}
	st := s.ExpertTierStats()
	if st.L2.Demotions != 3 || st.L2.PinnedPreserved < 1 {
		t.Fatalf("L2 transition counters = %+v, want 3 demotions and a preserved pin", st.L2)
	}
}

// TestExpertTierPolicyDefaultOffIsNoop is AC5: the zero-value policy fills nothing, demotes nothing,
// allocates nothing, and leaves a ring session's ledger byte-for-byte identical to a session that
// never mentions the policy.
func TestExpertTierPolicyDefaultOffIsNoop(t *testing.T) {
	bare := &Session{}
	if l, ok := bare.PromoteExpertTier(tierPolicyTestName(0, 0), 0, 4096); ok || l != (ExpertTierLedger{}) {
		t.Fatalf("disabled promotion = %+v ok=%v, want the zero ledger and false", l, ok)
	}
	if l, ok := bare.DemoteExpertTier(); ok || l != (ExpertTierLedger{}) {
		t.Fatalf("disabled demotion = %+v ok=%v, want the zero ledger and false", l, ok)
	}
	if bare.expertTier != nil {
		t.Fatal("the zero-value policy allocated L2 state")
	}
	if got := bare.ExpertTierStats(); got != (ExpertTierStats{}) {
		t.Fatalf("zero session stats = %+v, want the zero value", got)
	}

	// An EXPLICIT zero-value policy must move nothing either, and a session carrying it must drive
	// the identical ring ledger to one that never mentions the field.
	const H, E = 256, 8
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 6
	window := []int{0, 1, 2, 3, 0, 1, 2, 3}

	silent := expertRingSession(m, budget)
	defer silent.Close()
	driveExpertWindow(silent, m, window)

	declared := expertRingSession(m, budget)
	defer declared.Close()
	declared.ExpertTier = ExpertTierPolicy{Enabled: false, L2Bytes: 1 << 20, PromoteOnMiss: true}
	declared.PromoteExpertTier(tierPolicyTestName(0, 0), 0, 4096)
	declared.DemoteExpertTier()
	driveExpertWindow(declared, m, window)

	if got, want := declared.ExpertRing(), silent.ExpertRing(); got != want {
		t.Fatalf("a disabled policy changed the ring ledger: %+v vs %+v", got, want)
	}
	if declared.expertTier != nil {
		t.Fatal("a disabled policy allocated L2 state on a ring session")
	}
}

// TestSessionExpertTierStatsFoldsLedgers is AC6: the session reading joins the device ring, the
// checkpoint tier and the policy counters into one per-tier record, without moving residency.
func TestSessionExpertTierStatsFoldsLedgers(t *testing.T) {
	const H, E, K = 256, 8, 4
	m, _, stride := expertCheckpointTestModel(t, H, E, K, 0)
	s, _ := expertPrefetchSession(m, stride*3*K)
	defer s.Close()
	s.ExpertTier = ExpertTierPolicy{Enabled: true, L2Bytes: 4 * 4096, PromoteOnMiss: true}

	// Drive one layer so the ring pages in and the checkpoint tier faults the activated set.
	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})

	// One explicit L2 fill, so the policy half of the fold has data too.
	if _, ok := s.PromoteExpertTier(tierPolicyTestName(0, 0), 0, 1000); !ok {
		t.Fatal("L2 promotion refused")
	}

	st := s.ExpertTierStats()
	if !st.Enabled {
		t.Fatal("stats from an enabled ring+checkpoint+policy reported Enabled=false")
	}
	if !st.Ring.Enabled || !st.Checkpoint.Enabled {
		t.Fatalf("folded ring/checkpoint not enabled: %+v", st)
	}
	if st.VRAM.Hits != st.Ring.Hits || st.VRAM.Reads != st.Ring.PageIns || st.VRAM.Bytes != st.Ring.PageInBytes {
		t.Fatalf("VRAM counters %+v do not fold the ring ledger %+v", st.VRAM, st.Ring)
	}
	if st.NVMe.Reads != st.Checkpoint.Reads || st.NVMe.Bytes != st.Checkpoint.BytesRead {
		t.Fatalf("NVMe counters %+v do not fold the checkpoint ledger %+v", st.NVMe, st.Checkpoint)
	}
	if st.Checkpoint.Reads <= 0 || st.Checkpoint.BytesRead <= 0 {
		t.Fatalf("checkpoint tier read nothing; the fold witness is vacuous: %+v", st.Checkpoint)
	}
	if st.L2.Promotions != 1 || st.L2ResidentCount != 1 {
		t.Fatalf("L2 fold = %+v (residents %d), want one promotion", st.L2, st.L2ResidentCount)
	}
	if st.L2ResidentBytes > s.ExpertTier.L2Bytes {
		t.Fatalf("L2 resident %d exceeds declared ceiling %d", st.L2ResidentBytes, s.ExpertTier.L2Bytes)
	}

	// The fold is a pure READ: reading it again moves nothing.
	before := st
	again := s.ExpertTierStats()
	if again != before {
		t.Fatalf("ExpertTierStats is not a pure read: %+v then %+v", before, again)
	}
}
