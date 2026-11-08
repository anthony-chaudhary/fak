package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// The model adapter satisfies the policy seam without a production upward import.
var _ ctxmmu.ExpertResidencyAdvisor = (*ExpertHotSetAdvisor)(nil)

// expert_hot_set_advisor_test.go — witnesses for ExpertHotSetAdvisor (expert_hot_set_advisor.go),
// the exported integration point that feeds ExpertHotSetLearner into ctxmmu's expert-residency
// decisions: the band Admit answers with must be the EXACT band Learn() commits under, and every
// other method must forward to the learner without drift.

// TestAdvisorAdmitMirrorsLearnBand pins the gate/commit agreement on both arms of the band: at
// victim heat 100 with the default 0.25 margin the threshold is 100 + 25 + 4 = 129, so a
// challenger at exactly 129 stays out (Learn keeps the incumbent) while 130 clears it.
func TestAdvisorAdmitMirrorsLearnBand(t *testing.T) {
	const vHeat = 100.0
	l := NewExpertHotSetLearner(1, DefaultExpertHotSetHysteresis)
	a := &ExpertHotSetAdvisor{l: l}

	// The incumbent: committed at heat 100.
	l.Observe(0, 0)
	for i := 0; i < 99; i++ {
		l.Observe(0, 0)
	}
	if swaps := l.Learn(); len(swaps) != 1 || swaps[0].InLayer != 0 || swaps[0].InExpert != 0 {
		t.Fatalf("seed learn = %+v, want the single fill of (0,0)", swaps)
	}

	// Challenger at exactly the threshold: Learn must keep the incumbent, Admit must refuse.
	l.Observe(0, 1)
	for i := 0; i < 128; i++ { // heat 129 total on (0,1): 1 + 128
		l.Observe(0, 1)
	}
	if got := l.heat.Count(0, 1); got != 129 {
		t.Fatalf("challenger heat=%v, want 129 (fixture sanity)", got)
	}
	if swaps := l.Learn(); len(swaps) != 0 {
		t.Fatalf("learn at threshold performed %d swaps, want 0 (the incumbent stays)", len(swaps))
	}
	if a.Admit(129, vHeat) {
		t.Fatal("Admit(129, 100) = true, want false (inside the band)")
	}
	// One past the threshold: Learn displaces, Admit admits.
	l.Observe(0, 1) // heat 130
	if swaps := l.Learn(); len(swaps) != 1 || swaps[0].OutExpert != 0 || swaps[0].InExpert != 1 {
		t.Fatalf("learn past threshold = %+v, want one displacement of (0,0) by (0,1)", swaps)
	}
	if !a.Admit(130, vHeat) {
		t.Fatal("Admit(130, 100) = false, want true (clears the band)")
	}
}

// TestAdvisorHotSetAdaptsOrderAndShape witnesses the HotSet adaptation: the learner's total order
// survives into the ctxmmu shape, and WeightBytes is 0 (the learner tracks activations, not bytes).
func TestAdvisorHotSetAdaptsOrderAndShape(t *testing.T) {
	l := NewExpertHotSetLearner(3, DefaultExpertHotSetHysteresis)
	a := &ExpertHotSetAdvisor{l: l}
	counts := map[int]int{0: 5, 1: 20, 2: 5, 3: 5}
	for e, n := range counts {
		for i := 0; i < n; i++ {
			a.Observe(0, e)
		}
	}
	if swaps := a.Learn(); len(swaps) == 0 {
		t.Fatal("learn performed no swaps")
	}
	got := a.HotSet()
	want := []ctxmmu.ExpertAccess{{Layer: 0, Expert: 1}, {Layer: 0, Expert: 0}, {Layer: 0, Expert: 2}}
	if len(got) != len(want) {
		t.Fatalf("hot-set length=%d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("hot-set[%d] = %+v, want %+v (learner order, zero bytes)", i, got[i], want[i])
		}
	}
}

// TestAdvisorObservedCountForwards pins ObservedCount to the learner's observed population, not
// the committed set.
func TestAdvisorObservedCountForwards(t *testing.T) {
	a := NewExpertHotSetAdvisor(2, DefaultExpertHotSetHysteresis)
	for e := 0; e < 7; e++ {
		a.Observe(0, e)
	}
	if got := a.ObservedCount(); got != 7 {
		t.Fatalf("observed count=%d, want 7", got)
	}
	if got := a.Learner().heat.Len(); got != 7 {
		t.Fatalf("learner observed=%d, want 7 (adapter must not fork the population)", got)
	}
}

// TestAdvisorResidentPartitionForwards witnesses MarkResident/ClearResident reaching the learner's
// ranking partition: a marked unit sorts ahead of a hotter unmarked one, and clearing it restores
// the heat-only order.
func TestAdvisorResidentPartitionForwards(t *testing.T) {
	a := NewExpertHotSetAdvisor(2, DefaultExpertHotSetHysteresis)
	for i := 0; i < 10; i++ {
		a.Observe(0, 0)
	}
	for i := 0; i < 4; i++ {
		a.Observe(0, 1)
	}
	a.Learn()
	if got := a.HotSet()[0]; got.Layer != 0 || got.Expert != 0 {
		t.Fatalf("hot-set head = (%d,%d), want (0,0) by heat before any resident mark", got.Layer, got.Expert)
	}
	// Mark the cooler unit resident: it must now lead even against higher heat.
	a.MarkResident(0, 1)
	if got := a.HotSet()[0]; got.Layer != 0 || got.Expert != 1 {
		t.Fatalf("hot-set head after MarkResident = (%d,%d), want (0,1) (resident first)", got.Layer, got.Expert)
	}
	a.ClearResident(0, 1)
	if got := a.HotSet()[0]; got.Layer != 0 || got.Expert != 0 {
		t.Fatalf("hot-set head after ClearResident = (%d,%d), want (0,0) (heat order restored)", got.Layer, got.Expert)
	}
}

// TestAdvisorNilSafety pins the closed contract on a nil advisor: every method degrades to the
// empty answer rather than panicking or fabricating a decision.
func TestAdvisorNilSafety(t *testing.T) {
	var a *ExpertHotSetAdvisor
	if a.Heat(0, 0) != 0 {
		t.Fatal("nil Heat = non-zero")
	}
	if a.Admit(1e6, 0) {
		t.Fatal("nil Admit admitted; a missing advisor must never open the gate")
	}
	if a.HotSet() != nil || a.ObservedCount() != 0 {
		t.Fatal("nil advisor reported a hot-set or observations")
	}
	a.Observe(0, 0) // must not panic
	a.Absorb(NewExpertUsageHistogram())
	a.ObserveTrace(ExpertAccessTrace{})
	a.MarkResident(0, 0)
	a.ClearResident(0, 0)
	if a.Learn() != nil {
		t.Fatal("nil Learn returned a non-nil swap slice")
	}
	if a.Learner() != nil {
		t.Fatal("nil Learner() returned a learner")
	}
}
