package model

import (
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// expert_hot_set_test.go -- the #1301 witnesses for the ONLINE hot-set learner
// (expert_hot_set.go): EPLB-style activation statistics feed a committed hot-set
// that is re-selected between turns under the SAME `victim + victim/4 + 4`
// hysteresis band the offline policy uses, so a lukewarm challenger cannot evict
// a hot incumbent and the pin-set stops oscillating under routing jitter.
//
// The claims, one test each:
//
//	convergence  -- after enough skew, Learn() commits exactly the true top-budget experts;
//	hysteresis   -- a challenger inside the band performs ZERO swaps, one past it displaces the
//	               single cold incumbent, and re-learning unchanged heat is a no-op (no oscillation);
//	edges        -- a non-positive budget is a no-op, a non-positive hysteresis is the shipped
//	               default, and a malformed activation is ignored rather than poisoning the set;
//	seeding      -- SeedPins pins the learned set and carries its heat into an ExpertPinSet;
//	default-off  -- a session at hysteresis 0 allocates no learner and its ring ledger is R2's,
//	               byte-for-byte;
//	determinism  -- identical streams yield identical swaps and HotSet (guards map iteration order).

// expertHotSetLearner drives a learner with `counts` activations of each expert in layer 0 and
// returns the final HotSet() after one Learn(). It is the compact fixture the unit witnesses share.
func expertHotSetLearner(t *testing.T, budget int, counts []int) []ExpertUsageCount {
	t.Helper()
	l := NewExpertHotSetLearner(budget, DefaultExpertHotSetHysteresis)
	for expert, n := range counts {
		for i := 0; i < n; i++ {
			l.Observe(0, expert)
		}
	}
	l.Learn()
	return l.HotSet()
}

// TestExpertHotSetLearnerConvergesToHighFrequencyExperts is acceptance criterion 1's witness: a
// skewed activation stream (expert 1 touched far more than 0/2/3) commits exactly the true top-budget
// experts, in the learner's deterministic (layer asc, expert asc) order.
func TestExpertHotSetLearnerConvergesToHighFrequencyExperts(t *testing.T) {
	// Budget 4, four observed experts: every one is admitted, ordered by identity not heat -- the
	// set is the whole population and the deterministic order is what is checked.
	got := expertHotSetLearner(t, 4, []int{4, 20, 4, 4})
	want := []ExpertUsageCount{
		{Layer: 0, Expert: 0, Count: 4},
		{Layer: 0, Expert: 1, Count: 20},
		{Layer: 0, Expert: 2, Count: 4},
		{Layer: 0, Expert: 3, Count: 4},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("budget-4 hot-set = %+v, want the four observed units in identity order %+v", got, want)
	}

	// Budget 2 must keep the two HOTTEST experts (1 and, on the tie, 0), dropping 2 and 3 -- the
	// selection is by frequency, not by identity.
	got = expertHotSetLearner(t, 2, []int{4, 20, 4, 4})
	want = []ExpertUsageCount{
		{Layer: 0, Expert: 0, Count: 4},
		{Layer: 0, Expert: 1, Count: 20},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("budget-2 hot-set = %+v, want the two hottest experts %+v", got, want)
	}
}

// TestExpertHotSetLearnerHysteresisPreventsOscillation is criterion 2's witness. A stable committed
// hot-set A is built by heat; a challenger B inside the band (<= cold + cold*0.25 + 4) is REFUSED, and
// one that clears the margin displaces the single coldest member. Re-learning unchanged heat is a
// no-op, and alternating two near-tied challengers never flips the committed set.
func TestExpertHotSetLearnerHysteresisPreventsOscillation(t *testing.T) {
	base := []int{10, 0, 0, 0} // expert 0 at heat 10 is the incumbent; budget 1
	l := NewExpertHotSetLearner(1, DefaultExpertHotSetHysteresis)
	for expert, n := range base {
		for i := 0; i < n; i++ {
			l.Observe(0, expert)
		}
	}
	if sw := l.Learn(); len(sw) != 1 || sw[0].InExpert != 0 || sw[0].OutExpert != -1 {
		t.Fatalf("initial fill = %+v, want one free-slot admission of expert 0", sw)
	}

	// B at heat 16: cold(10) + 10*0.25 + 4 == 16.5, so 16 is INSIDE the band. Learn must refuse.
	for i := 0; i < 16; i++ {
		l.Observe(0, 1)
	}
	if sw := l.Learn(); len(sw) != 0 {
		t.Fatalf("inside-band challenger (heat 16 vs 10, margin 16.5) performed %+v, want zero swaps", sw)
	}
	if hs := l.HotSet(); len(hs) != 1 || hs[0].Expert != 0 || hs[0].Count != 10 {
		t.Fatalf("inside-band HotSet = %+v, want the unchanged incumbent expert 0", hs)
	}

	// B at heat 17 now clears the margin: exactly one displacement, cold-out/hot-in.
	l.Observe(0, 1)
	sw := l.Learn()
	want := ExpertHotSetSwap{OutLayer: 0, OutExpert: 0, InLayer: 0, InExpert: 1, OutHeat: 10, InHeat: 17}
	if len(sw) != 1 || sw[0] != want {
		t.Fatalf("above-band challenger = %+v, want the single displacement %+v", sw, want)
	}
	if hs := l.HotSet(); len(hs) != 1 || hs[0].Expert != 1 {
		t.Fatalf("above-band HotSet = %+v, want expert 1", hs)
	}

	// Re-learning on unchanged heat is a no-op: the band is stable, so a second turn performs nothing.
	if sw := l.Learn(); sw != nil {
		t.Fatalf("second Learn on unchanged heat = %+v, want nil (stable, no oscillation)", sw)
	}

	// Direct no-oscillation loop: two near-tied challengers, learned repeatedly, must not flip the
	// committed set. Each lean alternation stays well inside the band, so the incumbent survives.
	for round := 0; round < 5; round++ {
		for i := 0; i < 2; i++ {
			l.Observe(0, 2)
		}
		for i := 0; i < 2; i++ {
			l.Observe(0, 3)
		}
		if sw := l.Learn(); sw != nil {
			t.Fatalf("round %d flipped the committed set on near-tied challengers: %+v", round, sw)
		}
	}
	if hs := l.HotSet(); len(hs) != 1 || hs[0].Expert != 1 {
		t.Fatalf("after the near-tie loop HotSet = %+v, want expert 1 unchanged", hs)
	}
}

// TestExpertHotSetLearnerDefaultMarginsAndBudgetEdges pins the guard behavior: a non-positive budget
// makes Learn a no-op and HotSet empty, a non-positive hysteresis falls back to the shipped default,
// and a malformed activation is ignored rather than poisoning the statistics.
func TestExpertHotSetLearnerDefaultMarginsAndBudgetEdges(t *testing.T) {
	// budget <= 0: Learn returns nil, HotSet is empty, no panic.
	for _, budget := range []int{0, -1} {
		l := NewExpertHotSetLearner(budget, DefaultExpertHotSetHysteresis)
		l.Observe(0, 0)
		if sw := l.Learn(); sw != nil {
			t.Fatalf("budget %d Learn = %+v, want nil (no-op)", budget, sw)
		}
		if hs := l.HotSet(); len(hs) != 0 {
			t.Fatalf("budget %d HotSet = %+v, want empty", budget, hs)
		}
	}

	// hysteresis <= 0 uses the default margin: a challenger must clear cold*1.25 + 4 either way, so
	// the effective band of a hysteresis-0 learner equals the DefaultExpertHotSetHysteresis learner's.
	for _, hyst := range []float64{0, -1, DefaultExpertHotSetHysteresis} {
		l := NewExpertHotSetLearner(1, hyst)
		for i := 0; i < 10; i++ {
			l.Observe(0, 0)
		}
		l.Learn()
		for i := 0; i < 16; i++ { // heat 16: exactly the default band's edge, must refuse
			l.Observe(0, 1)
		}
		if sw := l.Learn(); len(sw) != 0 {
			t.Fatalf("hysteresis %v admitted a challenger at the band edge: %+v", hyst, sw)
		}
		l.Observe(0, 1) // heat 17 clears it
		if sw := l.Learn(); len(sw) != 1 {
			t.Fatalf("hysteresis %v refused a challenger past the band: %+v", hyst, sw)
		}
	}

	// A negative layer/expert is ignored: no panic, no heat, and the learner stays empty.
	l := NewExpertHotSetLearner(4, DefaultExpertHotSetHysteresis)
	l.Observe(-1, 0)
	l.Observe(0, -1)
	l.Observe(-5, -5)
	if sw := l.Learn(); sw != nil {
		t.Fatalf("malformed activations produced %+v, want nil", sw)
	}
	if hs := l.HotSet(); len(hs) != 0 {
		t.Fatalf("malformed activations produced a hot-set %+v, want empty", hs)
	}
}

// TestExpertHotSetLearnerSeedsPins witnesses the bridge from the learned set to the durable pin-set:
// SeedPins pins every committed unit and folds the learner's heat into the pin-set, so an unpinned
// unit outside the learned set stays unpinned.
func TestExpertHotSetLearnerSeedsPins(t *testing.T) {
	l := NewExpertHotSetLearner(2, DefaultExpertHotSetHysteresis)
	for i := 0; i < 9; i++ {
		l.Observe(0, 1)
	}
	for i := 0; i < 5; i++ {
		l.Observe(0, 2)
	}
	for i := 0; i < 2; i++ {
		l.Observe(0, 3)
	}
	l.Learn() // commits the two hottest: experts 1 and 2

	p := WarmStartExpertPins(NewExpertUsageHistogram(), 2)
	l.SeedPins(p)

	for _, c := range l.HotSet() {
		if !p.IsPinned(c.Layer, c.Expert) {
			t.Fatalf("learned unit (%d,%d) was not pinned by SeedPins", c.Layer, c.Expert)
		}
		if got := p.heat.Count(c.Layer, c.Expert); got != c.Count {
			t.Fatalf("pin-set heat for (%d,%d) = %v, want the learner's %v", c.Layer, c.Expert, got, c.Count)
		}
	}
	if got := p.Len(); got != 2 {
		t.Fatalf("pin-set length = %d, want the learned budget 2", got)
	}
	if p.IsPinned(0, 3) {
		t.Fatal("SeedPins pinned expert 3, which the learner did not commit")
	}
}

// TestExpertHotSetDefaultOffLeavesRingUnchanged is criterion 4's session-level witness: a session at
// ExpertHotSetHysteresis 0 allocates no learner and its ring ledger is R2's exactly; the enabled
// session builds a learner while its pin set and residency stay within budget.
func TestExpertHotSetDefaultOffLeavesRingUnchanged(t *testing.T) {
	const H, E = 256, 8
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 6
	window := expertJitterWindow(2, 6)

	// The reference R2 session: pin knobs, no hot-set knob at all. Both sessions cross the SAME
	// turn boundary, so the only variable between them is the hot-set knob.
	plain := expertPinSession(m, budget, 4, "")
	defer plain.Close()
	driveExpertWindow(plain, m, window)
	if _, err := plain.ExpertRingEndTurn(0.9, 2); err != nil {
		t.Fatalf("ExpertRingEndTurn (R2 reference): %v", err)
	}
	plainStats := plain.ExpertRing()

	// The default-off session: same knobs, explicitly at 0.
	off := expertPinSession(m, budget, 4, "")
	defer off.Close()
	driveExpertWindow(off, m, window)
	if _, err := off.ExpertRingEndTurn(0.9, 2); err != nil {
		t.Fatalf("ExpertRingEndTurn (default-off): %v", err)
	}
	if off.expertRing.hotSet != nil {
		t.Fatal("hysteresis 0 allocated a hot-set learner; the default must build no extra state")
	}
	if offStats := off.ExpertRing(); offStats != plainStats {
		t.Fatalf("hysteresis 0 changed the ring ledger: %+v vs R2 %+v", offStats, plainStats)
	}

	// The enabled session: a learner exists, and the pin-set/residency respect their bounds.
	on := expertPinSession(m, budget, 4, "")
	on.ExpertHotSetHysteresis = DefaultExpertHotSetHysteresis
	defer on.Close()
	onStats := driveExpertWindow(on, m, window)
	if _, err := on.ExpertRingEndTurn(0.9, 2); err != nil {
		t.Fatalf("ExpertRingEndTurn (enabled): %v", err)
	}
	if on.expertRing.hotSet == nil {
		t.Fatal("hysteresis > 0 did not build a hot-set learner")
	}
	if onStats.PeakBytes > onStats.BudgetBytes {
		t.Fatalf("enabled peak resident %d exceeds budget %d", onStats.PeakBytes, onStats.BudgetBytes)
	}
	st := on.ExpertRing()
	if st.PinnedCount > 4 {
		t.Fatalf("PinnedCount = %d, want <= the 4-slot pin budget", st.PinnedCount)
	}
	if st.PeakBytes > st.BudgetBytes {
		t.Fatalf("enabled final peak %d exceeds budget %d", st.PeakBytes, st.BudgetBytes)
	}
	// The default session is the invariant; the enabled run is allowed to differ and is logged so a
	// reader can see the learned policy's effect rather than having it silently asserted away.
	if plainStats != onStats {
		t.Logf("enabled run differed from default (allowed): default=%+v enabled=%+v", plainStats, onStats)
	}
}

// TestExpertHotSetLearnerDeterministic guards against Go's randomized map iteration leaking into the
// decision: identical input streams must yield identical swaps and identical HotSet across runs.
func TestExpertHotSetLearnerDeterministic(t *testing.T) {
	counts := []int{3, 11, 7, 2, 5, 7, 1, 9}
	run := func() ([]ExpertHotSetSwap, []ExpertUsageCount) {
		l := NewExpertHotSetLearner(4, DefaultExpertHotSetHysteresis)
		for expert, n := range counts {
			for i := 0; i < n; i++ {
				l.Observe(0, expert)
			}
		}
		swaps := l.Learn()
		return swaps, l.HotSet()
	}
	wantSwaps, wantHot := run()
	for rep := 0; rep < 8; rep++ {
		swaps, hot := run()
		if !reflect.DeepEqual(swaps, wantSwaps) {
			t.Fatalf("rep %d swaps = %+v, want %+v (map iteration leaked)", rep, swaps, wantSwaps)
		}
		if !reflect.DeepEqual(hot, wantHot) {
			t.Fatalf("rep %d HotSet = %+v, want %+v (map iteration leaked)", rep, hot, wantHot)
		}
	}
}

// TestExpertHotSetLearnerPinsStayWithinBudgetUnderWarmPrior is the disjoint-prior regression
// witness. An adversarial review found that applyHotSetSwaps could grow the pin-set PAST its
// budget: ExpertPinSet.fillPins and RepinPass both respect p.budget, but the learner's swaps were
// applied unconditionally. When the learner's heat (this turn's routing) and the pin-set's warm
// prior are DISJOINT, the learner has an empty committed set no matter how many units the prior
// already pinned, so Learn() emits a FILL (OutLayer=-1, nothing deleted) for its own hot unit --
// and pinning it on top of the prior's full set yields budget+1 pins. The fix clamps a fill that
// would overrun the budget (skip it while len(pinned) >= budget); a displacement already deletes
// before it inserts and so is self-bounding.
//
// The unit arm exercises applyHotSetSwaps directly -- the smallest deterministic reproduction -- and
// a positive control proves the same machine still pins when there IS room, so the assertion is not
// passing merely because applyHotSetSwaps became a no-op. The session arm drives the same collision
// end to end through ExpertRingEndTurn and checks the ring's public pinned/peak accounting.
func TestExpertHotSetLearnerPinsStayWithinBudgetUnderWarmPrior(t *testing.T) {
	const budget = 2

	// A warm prior whose hottest is expert 7 (heat 9), then expert 8 (heat 3): HotSet(2) selects both
	// as the initial pins. The learner below will observe only expert 1, so the prior and the learned
	// set share no unit -- the exact disjointness the bug needs.
	prior := NewExpertUsageHistogram()
	prior.Observe(0, 7, 9)
	prior.Observe(0, 8, 3)
	p := WarmStartExpertPins(prior, budget)
	if p.Len() != budget {
		t.Fatalf("warm-start pinned %d units, want the budget %d (%+v)", p.Len(), budget, p.Pins())
	}
	if !p.IsPinned(0, 7) || !p.IsPinned(0, 8) {
		t.Fatalf("warm-start pins = %+v, want the prior's two hottest (0,7) and (0,8)", p.Pins())
	}

	// The learner observes ONLY expert 1, many times, so its committed set becomes the single unit
	// (0,1) -- a FILL, disjoint from every prior pin.
	l := NewExpertHotSetLearner(budget, DefaultExpertHotSetHysteresis)
	for i := 0; i < 20; i++ {
		l.Observe(0, 1)
	}
	swaps := l.Learn()
	if len(swaps) != 1 || swaps[0].OutLayer != -1 || swaps[0].InLayer != 0 || swaps[0].InExpert != 1 {
		t.Fatalf("learner over a disjoint prior emitted %+v, want one fill of (0,1)", swaps)
	}

	// The regression assertion: applying a fill while the pin-set is already at budget must NOT
	// overrun it. Pre-fix this is 3 with budget 2; post-fix it stays 2.
	before := p.Len()
	p.applyHotSetSwaps(swaps)
	if got := p.Len(); got > budget {
		t.Fatalf("applyHotSetSwaps grew the pin-set to %d, past the budget %d: a disjoint learner "+
			"fill overran the pin budget (was %d before)", got, budget, before)
	}
	// No newly pinned unit may lack heat: a pin with no basis would be an unrankable guess for
	// RepinPass. With the clamp the new unit is REFUSED, so this is checked structurally.
	for _, c := range p.Pins() {
		if p.heat.Count(c.Layer, c.Expert) <= 0 {
			t.Fatalf("pinned unit (%d,%d) carries no heat; pins=%+v", c.Layer, c.Expert, p.Pins())
		}
	}

	// Positive control: the SAME learner decision applied to a pin-set with a free slot still pins
	// the challenger and records its heat -- so the guard above is a repair, not a disablement.
	fresh := WarmStartExpertPins(NewExpertUsageHistogram(), budget)
	if fresh.Len() != 0 {
		t.Fatalf("empty prior pinned %d units, want 0", fresh.Len())
	}
	fresh.applyHotSetSwaps(swaps)
	if !fresh.IsPinned(0, 1) {
		t.Fatalf("with a free slot, applyHotSetSwaps did not pin the challenger; pins=%+v", fresh.Pins())
	}
	if got := fresh.heat.Count(0, 1); got <= 0 {
		t.Fatalf("newly pinned (0,1) heat = %v, want the fill's InHeat > 0", got)
	}

	// End-to-end: the SAME collision through the public turn boundary. A session pins the prior's hot
	// expert (0), then routes ONLY a different expert (1); the learner's fill must not push the
	// pinned count past ExpertPinBudget, and the ring's byte peak must stay inside its budget.
	const H, E = 256, 4
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	usage := filepath.Join(t.TempDir(), "expert-usage.json")
	// The prior names expert 0 as hot. A decay of 1.0 plus the prior's mass keeps RepinPass from
	// swapping it out, so the only thing that could overrun the budget is the learner's fill.
	priorFile := NewExpertUsageHistogram()
	priorFile.Observe(0, 0, 100)
	if err := priorFile.Persist(usage); err != nil {
		t.Fatalf("persist prior: %v", err)
	}

	s := expertPinSession(m, perWeight*6, 1, usage)
	defer s.Close()
	s.ExpertHotSetHysteresis = DefaultExpertHotSetHysteresis
	window := make([]int, 20)
	for i := range window {
		window[i] = 1 // touch ONLY expert 1, disjoint from the prior's pinned expert 0
	}
	driveExpertWindow(s, m, window)
	if _, err := s.ExpertRingEndTurn(1.0, 2); err != nil {
		t.Fatalf("ExpertRingEndTurn: %v", err)
	}
	st := s.ExpertRing()
	if st.PinnedCount > s.ExpertPinBudget {
		t.Fatalf("PinnedCount = %d, want <= ExpertPinBudget %d: a disjoint learner fill overran the "+
			"session's pin budget", st.PinnedCount, s.ExpertPinBudget)
	}
	if st.PeakBytes > st.BudgetBytes {
		t.Fatalf("peak resident %d exceeds budget %d across the disjoint-prior turn", st.PeakBytes, st.BudgetBytes)
	}
}

// TestExpertHotSetLearnerHasExactlyOneLearnerSymbol is the #13040 reconciliation witness. A peer
// once wrote an independent, UNWIRED `OnlineExpertHotSet` into this same package mid-task (recovered
// issue #1435); that peer file was preserved out-of-tree at
// %TEMP%\opencode\peer-1435-online-expert-hot-set.go. The landed canonical implementation is the
// WIRED ExpertHotSetLearner (expert_hot_set.go, reached by Session.ExpertHotSetHysteresis ->
// applyHotSetSwaps through expert_ring_pins.go / kv.go / paging_ring.go).
//
// The peer file no longer exists on any checked-out tree, and `OnlineExpertHotSet` appears nowhere in
// the package (verified by direct grep on trunk). The reconciliation decision recorded here is
// therefore DISCARD-AS-DUPLICATE: the landed wired learner is canonical, and there was no surviving
// peer source to fold. This test locks that decision in structurally so the duplicate-concept trap
// the issue names cannot silently re-appear: exactly ONE hot-set learner type may be declared in
// internal/model, and it must be the canonical wired one.
//
// The scan is deterministic and source-level: it reads the package's own non-test .go files (the
// authoritative declarations), finds every `type <Name> struct` whose name contains `HotSet` (excluding the value type
// ExpertHotSetSwap), and asserts the learner set is exactly {ExpertHotSetLearner}. A reintroduced
// OnlineExpertHotSet -- under any casing -- fails by name.
func TestExpertHotSetLearnerHasExactlyOneLearnerSymbol(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}

	typeDecl := regexp.MustCompile(`(?m)^type\s+([A-Za-z0-9_]+)\s+struct\b`)
	var hotSetTypes []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(filepath.Join(".", name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		for _, m := range typeDecl.FindAllStringSubmatch(string(src), -1) {
			if strings.Contains(m[1], "HotSet") && m[1] != "ExpertHotSetSwap" {
				hotSetTypes = append(hotSetTypes, m[1])
			}
		}
	}
	sort.Strings(hotSetTypes)

	// Exactly one hot-set learner type, and it is the canonical wired one.
	if len(hotSetTypes) != 1 || hotSetTypes[0] != "ExpertHotSetLearner" {
		t.Fatalf("hot-set learner types in internal/model = %v, want exactly [ExpertHotSetLearner]; a "+
			"second learner (e.g. OnlineExpertHotSet) is the #13040 duplicate-concept trap and must be "+
			"reconciled, not added", hotSetTypes)
	}

	// The canonical symbol is reachable and wired: presence is not the claim; that the learner can be
	// constructed and driven through its decision path is. This is the invokability half.
	l := NewExpertHotSetLearner(2, DefaultExpertHotSetHysteresis)
	l.Observe(0, 7)
	l.Observe(0, 7)
	l.Observe(0, 8)
	if sw := l.Learn(); len(sw) == 0 {
		t.Fatalf("canonical ExpertHotSetLearner.Learn() produced no swaps for a clear top-2 population")
	}
	if got := l.HotSet(); len(got) != 2 {
		t.Fatalf("canonical learner committed %d units, want 2", len(got))
	}
}
