package ctxmmu

import (
	"testing"
)

// fakeAdvisor is an ExpertResidencyAdvisor with a controllable band and recorded membership calls,
// so the policy's seam behavior is testable without importing internal/model.
type fakeAdvisor struct {
	heat map[expertResidencyKey]float64
	// margin reproduces the learner's band: admit iff ch > v + v*margin + abs.
	margin float64
	abs    float64
	hotSet []ExpertAccess
	marks  []expertResidencyKey
	clears []expertResidencyKey
}

func newFakeAdvisor(margin, abs float64) *fakeAdvisor {
	return &fakeAdvisor{heat: map[expertResidencyKey]float64{}, margin: margin, abs: abs}
}

func (a *fakeAdvisor) Heat(layer, expert int) float64 {
	if a == nil {
		return 0
	}
	return a.heat[expertResidencyKey{layer, expert}]
}

func (a *fakeAdvisor) Admit(chHeat, vHeat float64) bool {
	if a == nil {
		return false
	}
	return chHeat > vHeat+vHeat*a.margin+a.abs
}

func (a *fakeAdvisor) HotSet() []ExpertAccess {
	if a == nil {
		return nil
	}
	return a.hotSet
}

func (a *fakeAdvisor) ObservedCount() int {
	if a == nil {
		return 0
	}
	return len(a.heat)
}

func (a *fakeAdvisor) MarkResident(layer, expert int) {
	a.marks = append(a.marks, expertResidencyKey{layer, expert})
}

func (a *fakeAdvisor) ClearResident(layer, expert int) {
	a.clears = append(a.clears, expertResidencyKey{layer, expert})
}

// TestAdvisorGateBypassesLukewarmChallenger witnesses decision 1: inside the displacement band a
// cold decode touch that would evict a resident is BYPASSED — served from backing, not made
// resident, no hot eviction, counted in DecodeBypasses.
func TestAdvisorGateBypassesLukewarmChallenger(t *testing.T) {
	backing := &countingBacking{bytes: 100}
	ad := newFakeAdvisor(0.25, 4)
	p, err := NewExpertResidencyPolicyWithAdvisor(
		ExpertResidencyOptions{HotBytes: 200, TransientBytes: 200}, backing, ad) // holds two 100-byte entries
	if err != nil {
		t.Fatalf("NewExpertResidencyPolicyWithAdvisor: %v", err)
	}

	// Seed two residents (the ring is now full); each carries heat 10 on the advisor side.
	for e := 0; e < 2; e++ {
		if _, ok := p.Decode(0, e); !ok {
			t.Fatalf("seed decode expert %d missed", e)
		}
		ad.heat[expertResidencyKey{0, e}] = 10
	}
	// The challenger clears nothing: heat 14 against victim heat 10 needs > 10+2.5+4 = 16.5.
	ad.heat[expertResidencyKey{0, 9}] = 14
	d, ok := p.Decode(0, 9)
	if !ok || d.Expert != 9 {
		t.Fatalf("bypassed touch must still be served, got ok=%v d=%+v", ok, d)
	}
	st := p.Stats()
	if st.DecodeBypasses != 1 {
		t.Fatalf("decode_bypasses=%d, want 1", st.DecodeBypasses)
	}
	if st.DemandMisses != 3 {
		t.Fatalf("demand_misses=%d, want 3 (two seeds + the bypassed serve)", st.DemandMisses)
	}
	if st.HotEvictions != 0 {
		t.Fatalf("hot_evictions=%d, want 0 (the incumbent stays)", st.HotEvictions)
	}
	if got := p.Tier(0, 9); got != TierCold {
		t.Fatalf("bypassed expert tier=%q, want cold (no residency earned)", got)
	}
	if got := p.Tier(0, 0); got != TierResident {
		t.Fatalf("incumbent tier=%q after bypass, want resident", got)
	}
	// Two seed loads plus exactly one load for the bypassed serve.
	if backing.loads != 3 {
		t.Fatalf("backing loads=%d, want 3 (two seeds + the bypassed serve)", backing.loads)
	}
}

// TestAdvisorGateAdmitsClearChallenger witnesses the other arm of decision 1: past the band the
// newcomer IS admitted, displacing the LRU victim, and the membership sync reaches the advisor.
func TestAdvisorGateAdmitsClearChallenger(t *testing.T) {
	backing := &countingBacking{bytes: 100}
	ad := newFakeAdvisor(0.25, 4)
	p, err := NewExpertResidencyPolicyWithAdvisor(
		ExpertResidencyOptions{HotBytes: 200, TransientBytes: 200}, backing, ad) // holds two 100-byte entries
	if err != nil {
		t.Fatalf("NewExpertResidencyPolicyWithAdvisor: %v", err)
	}
	for e := 0; e < 2; e++ {
		if _, ok := p.Decode(0, e); !ok {
			t.Fatalf("seed decode expert %d missed", e)
		}
		ad.heat[expertResidencyKey{0, e}] = 10
	}
	// Challenger heat 20 > 10 + 2.5 + 4 = 16.5: it clears the band and displaces the LRU victim.
	ad.heat[expertResidencyKey{0, 9}] = 20
	if _, ok := p.Decode(0, 9); !ok {
		t.Fatal("clear challenger missed")
	}
	st := p.Stats()
	if st.DecodeBypasses != 0 {
		t.Fatalf("decode_bypasses=%d, want 0 (challenger cleared the band)", st.DecodeBypasses)
	}
	if st.DemandMisses != 3 {
		t.Fatalf("demand_misses=%d, want 3 (two seeds + the admitted challenger)", st.DemandMisses)
	}
	if st.HotEvictions != 1 {
		t.Fatalf("hot_evictions=%d, want 1 (one resident displaced)", st.HotEvictions)
	}
	if got := p.Tier(0, 9); got != TierResident {
		t.Fatalf("admitted challenger tier=%q, want resident", got)
	}
	// Membership sync: the seed inserts marked both incumbents, the admission marked the
	// challenger and cleared exactly one evicted victim.
	var marked int
	for _, k := range ad.marks {
		if k == (expertResidencyKey{0, 9}) {
			marked++
		}
	}
	if marked != 1 {
		t.Fatalf("advisor saw %d marks for the admitted unit, want 1", marked)
	}
	if len(ad.clears) != 1 {
		t.Fatalf("advisor saw %d clears, want 1 (the evicted victim)", len(ad.clears))
	}
}

// TestAdvisorWarmStartSeedsLearnedHotSet witnesses the warm-start half of the seam: the advisor's
// committed hot-set seeds the resident ring, loading each descriptor from backing exactly once.
func TestAdvisorWarmStartSeedsLearnedHotSet(t *testing.T) {
	backing := &countingBacking{bytes: 100}
	ad := newFakeAdvisor(0.25, 4)
	ad.hotSet = []ExpertAccess{{Layer: 0, Expert: 3}, {Layer: 0, Expert: 4}}
	p, err := NewExpertResidencyPolicyWithAdvisor(
		ExpertResidencyOptions{HotBytes: 300, TransientBytes: 200}, backing, ad)
	if err != nil {
		t.Fatalf("NewExpertResidencyPolicyWithAdvisor: %v", err)
	}
	if seeded := p.WarmStartFromAdvisor(); seeded != 2 {
		t.Fatalf("warm start seeded %d experts, want 2", seeded)
	}
	if backing.loads != 2 {
		t.Fatalf("backing loads=%d, want 2 (one per seeded unit)", backing.loads)
	}
	for _, e := range []int{3, 4} {
		if got := p.Tier(0, e); got != TierResident {
			t.Fatalf("seeded expert (0,%d) tier=%q, want resident", e, got)
		}
	}
	// A second call seeds nothing new: the units are already resident.
	if seeded := p.WarmStartFromAdvisor(); seeded != 0 {
		t.Fatalf("repeat warm start seeded %d experts, want 0", seeded)
	}
	if backing.loads != 2 {
		t.Fatalf("backing loads=%d after repeat, want 2 (no re-read)", backing.loads)
	}
}

// TestAdvisorWarmStartStopsAtBudget pins the fit-stop rule: seeding halts at the budget rather
// than evicting an already-seeded hotter unit.
func TestAdvisorWarmStartStopsAtBudget(t *testing.T) {
	backing := &countingBacking{bytes: 100}
	ad := newFakeAdvisor(0.25, 4)
	ad.hotSet = []ExpertAccess{{Layer: 0, Expert: 0}, {Layer: 0, Expert: 1}, {Layer: 0, Expert: 2}}
	p, err := NewExpertResidencyPolicyWithAdvisor(
		ExpertResidencyOptions{HotBytes: 250, TransientBytes: 200}, backing, ad) // holds two 100-byte entries
	if err != nil {
		t.Fatalf("NewExpertResidencyPolicyWithAdvisor: %v", err)
	}
	if seeded := p.WarmStartFromAdvisor(); seeded != 2 {
		t.Fatalf("warm start seeded %d experts, want 2 (budget holds two)", seeded)
	}
	if got := p.Tier(0, 0); got != TierResident {
		t.Fatalf("first-seeded expert tier=%q, want resident (not churned by the third)", got)
	}
}

// TestAdvisorNilKeepsPlainLRU witnesses the default-off contract: a nil advisor leaves every
// decision byte-for-byte plain LRU — admissions always happen, DecodeBypasses stays 0, and
// WarmStartFromAdvisor seeds nothing.
func TestAdvisorNilKeepsPlainLRU(t *testing.T) {
	backing := &countingBacking{bytes: 100}
	p, err := NewExpertResidencyPolicyWithAdvisor(
		ExpertResidencyOptions{HotBytes: 200, TransientBytes: 200}, backing, nil)
	if err != nil {
		t.Fatalf("NewExpertResidencyPolicyWithAdvisor(nil): %v", err)
	}
	// Three 100-byte decodes into a 200-byte ring: the third evicts the first, no gate in between.
	for e := 0; e < 3; e++ {
		if _, ok := p.Decode(0, e); !ok {
			t.Fatalf("decode expert %d missed", e)
		}
	}
	st := p.Stats()
	if st.DecodeBypasses != 0 {
		t.Fatalf("decode_bypasses=%d with nil advisor, want 0", st.DecodeBypasses)
	}
	if st.DemandMisses != 3 {
		t.Fatalf("demand_misses=%d, want 3 (every touch loaded from backing)", st.DemandMisses)
	}
	if st.HotEvictions != 1 {
		t.Fatalf("hot_evictions=%d, want 1 (plain LRU eviction)", st.HotEvictions)
	}
	if got := p.WarmStartFromAdvisor(); got != 0 {
		t.Fatalf("warm start with nil advisor seeded %d, want 0", got)
	}
}

// TestSizeTransientRing pins the sizing arithmetic: observed breadth minus the committed hot set,
// priced at the average expert bytes, clamped to the cap, degenerate inputs yield zero.
func TestSizeTransientRing(t *testing.T) {
	cases := []struct {
		name     string
		observed int
		hotSet   int
		avg      int64
		cap      int64
		want     int64
	}{
		{"breadth-priced", 100, 40, 1024, 1 << 30, 60 * 1024},
		{"clamped-to-cap", 100, 40, 1 << 20, 5 << 20, 5 << 20},
		{"hot-set-covers-observed", 40, 40, 1024, 1 << 30, 0},
		{"hot-set-exceeds-observed", 30, 40, 1024, 1 << 30, 0},
		{"zero-average-bytes", 100, 40, 0, 1 << 30, 0},
		{"zero-cap", 100, 40, 1024, 0, 0},
		{"zero-observed", 0, 0, 1024, 1 << 30, 0},
		{"overflow-guard-clamps", 1<<30 - 1, 0, 1 << 30, 1 << 30, 1 << 30},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := SizeTransientRing(c.observed, c.hotSet, c.avg, c.cap); got != c.want {
				t.Fatalf("SizeTransientRing(%d, %d, %d, %d) = %d, want %d",
					c.observed, c.hotSet, c.avg, c.cap, got, c.want)
			}
		})
	}
}
