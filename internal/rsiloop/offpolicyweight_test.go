package rsiloop

import (
	"math"
	"testing"
)

// kept builds a kept archive row demonstrating a point-delta gain: the only shape
// WeighArchive mines. Before/After carry the objective points so gain = after-before.
func kept(id string, before, after float64) LoopVariantRecord {
	return LoopVariantRecord{
		Variant: LoopVariant{ID: id},
		Kept:    true,
		Before:  SpecFold{Points: before},
		After:   SpecFold{Points: after},
	}
}

func TestOffPolicyWeight(t *testing.T) {
	const k = 4
	cases := []struct {
		name                      string
		sampleGen, currentGen, kk int
		want                      float64
	}{
		// (iii) a fresh sample (delta <= 0) weighs EXACTLY 1 — the equal-weight fold.
		{"delta 0 is fresh", 5, 5, k, 1},
		// A future-generation sample (out-of-order append) is treated as fresh,
		// never weighted above 1.
		{"future gen clamps to 1", 9, 5, k, 1},
		// (i) a sample K OR MORE generations old contributes a HARD 0 (staleness-K).
		{"delta == K is hard 0", 0, k, k, 0},
		{"delta > K is hard 0", 0, 10, k, 0},
		// Linear decay strictly between fresh and the K bound.
		{"delta 1 of 4", 3, 4, k, 0.75},
		{"delta 2 of 4", 2, 4, k, 0.5},
		{"delta 3 of 4", 1, 4, k, 0.25},
		// k <= 0 DISABLES the decay: every sample weighs 1 (decay is opt-in).
		{"k=0 disables decay", 0, 100, 0, 1},
		{"negative k disables decay", 0, 100, -3, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := OffPolicyWeight(c.sampleGen, c.currentGen, c.kk); math.Abs(got-c.want) > 1e-9 {
				t.Errorf("OffPolicyWeight(%d,%d,%d) = %v, want %v", c.sampleGen, c.currentGen, c.kk, got, c.want)
			}
		})
	}
}

func TestCapInfluence(t *testing.T) {
	if got := CapInfluence(5, 2); got != 2 {
		t.Errorf("over +clip = %v, want 2", got)
	}
	if got := CapInfluence(-5, 2); got != -2 {
		t.Errorf("under -clip = %v, want -2 (symmetric about zero)", got)
	}
	if got := CapInfluence(1.5, 2); got != 1.5 {
		t.Errorf("within clip = %v, want passthrough 1.5", got)
	}
	if got := CapInfluence(999, 0); got != 999 {
		t.Errorf("clip<=0 = %v, want disabled passthrough", got)
	}
}

// TestAgeWeightedInfluenceFreshReducesToRaw is property (iii): at generation-delta 0
// with the cap disabled, the guard composes to EXACTLY the raw influence, so a fully
// fresh corpus folds identically to today's equal-weight behaviour.
func TestAgeWeightedInfluenceFreshReducesToRaw(t *testing.T) {
	for _, raw := range []float64{0, 1.5, 42, -3} {
		if got := AgeWeightedInfluence(raw, 7, 7, 5, 0); got != raw {
			t.Errorf("fresh+uncapped influence for raw %v = %v, want raw", raw, got)
		}
	}
}

// TestStaleHeavyTailCannotDominate is properties (i)+(ii): a single heavy-tailed
// off-policy sample is bounded two ways — a K+ old one decays to 0 (dropped), and a
// recent monster is clipped to the cap — so it can never dominate a fold the way the
// same sample would with equal weighting.
func TestStaleHeavyTailCannotDominate(t *testing.T) {
	const (
		k          = 5
		cap        = 2.0
		currentGen = 10
	)
	// (i) K+ generations old -> hard 0 contribution, regardless of raw magnitude.
	if stale := AgeWeightedInfluence(1e6, 0, currentGen, k, cap); stale != 0 {
		t.Fatalf("K+ stale heavy tail contributed %v, want 0 (bounded off-policy age)", stale)
	}
	// (ii) a RECENT monster is clipped to the cap.
	recentHuge := AgeWeightedInfluence(1e6, currentGen, currentGen, k, cap)
	if recentHuge != cap {
		t.Fatalf("recent heavy tail = %v, want clipped to cap %v", recentHuge, cap)
	}
	// A fold of modest fresh samples, each decaying linearly by age.
	fold := 0.0
	for g := 6; g <= currentGen; g++ {
		fold += AgeWeightedInfluence(1.0, g, currentGen, k, cap)
	}
	// The capped heavy tail is a bounded minority of the fold, not the whole of it.
	if recentHuge >= fold+recentHuge {
		t.Fatalf("capped heavy tail %v still dominates fold %v", recentHuge, fold+recentHuge)
	}
	// Contrast: with NO cap the same sample swamps the fold — proof the cap is what
	// bounds it, not the age decay alone.
	if uncapped := AgeWeightedInfluence(1e6, currentGen, currentGen, k, 0); uncapped <= fold {
		t.Fatalf("fixture error: uncapped heavy tail %v should swamp fold %v", uncapped, fold)
	}
}

// TestWeighArchiveMinesKeptGains witnesses which archive rows become stepping stones:
// only KEPT rows with a positive demonstrated gain, each carrying its age-decayed
// influence (a K+ old stone decays to 0 but is still weighed).
func TestWeighArchiveMinesKeptGains(t *testing.T) {
	archive := []LoopVariantRecord{
		kept("v0", 0, 100), // gen 0: huge gain but oldest
		{Variant: LoopVariant{ID: "v1"}, Kept: false}, // REVERT: skipped
		kept("v2", 3, 3), // zero gain: skipped
		kept("v3", 0, 5), // gen 3
		kept("v4", 0, 3), // gen 4
		kept("v5", 0, 1), // gen 5 (current, fresh)
	}
	p := AgeDecayArchiveProposer{StalenessK: 5, InfluenceCap: 0}
	stones := p.WeighArchive(archive)
	if len(stones) != 4 {
		t.Fatalf("mined %d stones, want 4 (v0,v3,v4,v5; revert+zero-gain excluded): %+v", len(stones), stones)
	}
	byID := map[string]SteppingStone{}
	for _, s := range stones {
		byID[s.Variant.ID] = s
	}
	// currentGen = len-1 = 5. v0 at gen 0 -> delta 5 >= K -> decays to 0.
	if s := byID["v0"]; s.Gain != 100 || s.Influence != 0 {
		t.Errorf("oldest stone v0 = gain %v influence %v, want gain 100 influence 0", s.Gain, s.Influence)
	}
	// v3 gen 3 -> delta 2, weight 0.6 -> 5*0.6 = 3.0 (cap disabled).
	if s := byID["v3"]; math.Abs(s.Influence-3.0) > 1e-9 {
		t.Errorf("v3 influence = %v, want 3.0", s.Influence)
	}
	// v5 gen 5 -> delta 0, weight 1 -> 1.0 (fresh, equal weight).
	if s := byID["v5"]; s.Influence != 1.0 {
		t.Errorf("v5 fresh influence = %v, want 1.0", s.Influence)
	}
}

// TestProposeDropsStaleAndOrders is the end-to-end wired proposer: a stone K+ lever
// generations old decays to 0 and is DROPPED from the mine regardless of the raw gain
// it once demonstrated, and the survivors come out ordered most-trusted (highest
// age-decayed influence) first. Archive length 6 => currentGen = 5, so delta = 5-gen.
func TestProposeDropsStaleAndOrders(t *testing.T) {
	archive := []LoopVariantRecord{
		kept("v0", 0, 100), // gen 0: delta 5 == K -> weight 0 -> DROPPED despite the huge gain
		kept("v1", 0, 10),  // gen 1: delta 4 -> weight 0.2 -> influence 2.0
		kept("v2", 0, 10),  // gen 2: delta 3 -> weight 0.4 -> influence 4.0
		kept("v3", 0, 5),   // gen 3: delta 2 -> weight 0.6 -> influence 3.0
		kept("v4", 0, 3),   // gen 4: delta 1 -> weight 0.8 -> influence 2.4
		kept("v5", 0, 1),   // gen 5: delta 0 -> weight 1.0 -> influence 1.0 (fresh)
	}
	p := AgeDecayArchiveProposer{StalenessK: 5, InfluenceCap: 0}
	got, err := p.ProposeLoopVariants(LoopConfig{}, archive)
	if err != nil {
		t.Fatalf("ProposeLoopVariants error: %v", err)
	}
	// v0 is dropped (too stale); survivors ordered by descending age-decayed influence.
	want := []string{"v2", "v3", "v4", "v1", "v5"}
	if len(got) != len(want) {
		t.Fatalf("proposed %d variants, want %d (v0 dropped as too stale): %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].ID != want[i] {
			gotIDs := make([]string, len(got))
			for j, v := range got {
				gotIDs[j] = v.ID
			}
			t.Fatalf("proposed order = %v, want %v (v0 dropped, rest most-trusted first)", gotIDs, want)
		}
	}
}

// TestProposeCapReordersHeavyTail witnesses the is_clip half in the wired proposer: a
// recent heavy-tailed stone that would otherwise top the order is clipped to the cap,
// so it can no longer dominate the proposal ranking.
func TestProposeCapReordersHeavyTail(t *testing.T) {
	archive := []LoopVariantRecord{
		kept("heavy", 0, 100), // gen 0: raw gain 100, but oldest
		kept("mid", 0, 5),     // gen 1
		kept("fresh", 0, 4),   // gen 2 (current, fresh)
	}
	// currentGen = 2, K large so nothing decays to 0; only the cap bites.
	p := AgeDecayArchiveProposer{StalenessK: 100, InfluenceCap: 3}
	got, err := p.ProposeLoopVariants(LoopConfig{}, archive)
	if err != nil {
		t.Fatalf("ProposeLoopVariants error: %v", err)
	}
	// Raw age-decayed influences: heavy gen0 delta2 -> 100*0.98 = 98 -> CAPPED to 3;
	// mid gen1 delta1 -> 5*0.99 = 4.95 -> CAPPED to 3; fresh gen2 delta0 -> 4 -> CAPPED to 3.
	// All three clip to 3, so the heavy tail no longer dominates: the tie breaks by
	// newest generation first (fresh, mid, heavy), NOT by raw gain.
	want := []string{"fresh", "mid", "heavy"}
	if len(got) != len(want) {
		t.Fatalf("proposed %d variants, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].ID != want[i] {
			gotIDs := make([]string, len(got))
			for j, v := range got {
				gotIDs[j] = v.ID
			}
			t.Fatalf("capped order = %v, want %v (heavy tail clipped, tie broken newest-first)", gotIDs, want)
		}
	}
}
