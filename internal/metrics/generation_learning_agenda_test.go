package metrics

import (
	"strings"
	"testing"
)

// topics extracts the selected topics in plan order, for compact assertions.
func selectedTopics(p LearningAgendaPlan) []string {
	out := make([]string, 0, len(p.Selected))
	for _, s := range p.Selected {
		out = append(out, s.Topic)
	}
	return out
}

func hasTopic(p LearningAgendaPlan, topic string) bool {
	for _, s := range p.Selected {
		if s.Topic == topic {
			return true
		}
	}
	return false
}

func deferralFor(p LearningAgendaPlan, topic string) (AgendaDeferral, bool) {
	for _, d := range p.Deferred {
		if d.Topic == topic {
			return d.Reason, true
		}
	}
	return "", false
}

// urgentNowFlood is the adversarial candidate set the whole rule exists for: three
// gen/now items, each more urgent than the lone gen/future item and each big enough
// to eat the budget on its own.
func urgentNowFlood() []LearningItem {
	return []LearningItem{
		{Topic: "now-regression-a", Horizon: "now", Minutes: 30, Urgency: 0.99},
		{Topic: "now-regression-b", Horizon: "now", Minutes: 30, Urgency: 0.98},
		{Topic: "now-regression-c", Horizon: "now", Minutes: 30, Urgency: 0.97},
		{Topic: "future-standard", Horizon: "future", Minutes: 15, Urgency: 0.10},
	}
}

// TestLearningAgendaReserveProtectsOptionality is the core witness for #1675: with the
// optionality reserve OFF (the naive urgency-greedy selector), a flood of urgent gen/now
// items consumes the entire budget and the gen/future item is never learned. With the
// reserve ON, the same candidates and the same budget still fund the future item.
//
// This is the before/after readout the issue asks for, as one deterministic test.
func TestLearningAgendaReserveProtectsOptionality(t *testing.T) {
	items := urgentNowFlood()

	// BEFORE: reserve disabled -> pure urgency greedy.
	before := SelectLearningAgenda(items, AgendaBudget{TotalMinutes: 60, OptionalityReserve: 0})
	if hasTopic(before, "future-standard") {
		t.Fatalf("control: naive greedy should starve optionality, but selected it: %v", selectedTopics(before))
	}
	if got := before.OptionalityMinutes(); got != 0 {
		t.Fatalf("control: optionality minutes = %d, want 0", got)
	}
	if reason, ok := deferralFor(before, "future-standard"); !ok || reason != DeferBudgetExhausted {
		t.Fatalf("control: future item deferral = (%q,%v), want (%q,true)", reason, ok, DeferBudgetExhausted)
	}

	// AFTER: same items, same total budget, reserve enabled.
	after := SelectLearningAgenda(items, AgendaBudget{TotalMinutes: 60, OptionalityReserve: DefaultOptionalityReserve})
	if after.ReserveMinutes != 15 {
		t.Fatalf("reserve minutes = %d, want 15 (25%% of 60)", after.ReserveMinutes)
	}
	if !hasTopic(after, "future-standard") {
		t.Fatalf("reserve failed to fund the future item; selected: %v", selectedTopics(after))
	}
	if got := after.OptionalityMinutes(); got != 15 {
		t.Fatalf("optionality minutes = %d, want 15", got)
	}
	// The reserve must actually bind: the future item is bought with reserved attention,
	// not with leftover open-pool attention it would have won anyway.
	for _, s := range after.Selected {
		if s.Topic == "future-standard" && s.Pool != PoolReserve {
			t.Fatalf("future item pool = %q, want %q", s.Pool, PoolReserve)
		}
	}
	if after.ReserveUsed != 15 {
		t.Fatalf("reserve used = %d, want 15", after.ReserveUsed)
	}
	// The most urgent now item still lands — protecting optionality is not starving urgency.
	if !hasTopic(after, "now-regression-a") {
		t.Fatalf("most urgent now item was dropped; selected: %v", selectedTopics(after))
	}
	// Budget is never overspent.
	if after.MinutesUsed > 60 {
		t.Fatalf("minutes used = %d, exceeds budget 60", after.MinutesUsed)
	}
}

// TestLearningAgendaRankingIsHorizonBlind mechanizes the issue's non-goal: gen/future is
// a horizon label, not a value judgment. In the open pool (reserve off) a high-urgency
// gen/future item must outrank a low-urgency gen/now item when only one can fit.
func TestLearningAgendaRankingIsHorizonBlind(t *testing.T) {
	items := []LearningItem{
		{Topic: "sleepy-now", Horizon: "now", Minutes: 10, Urgency: 0.10},
		{Topic: "urgent-future", Horizon: "future", Minutes: 10, Urgency: 0.90},
	}
	p := SelectLearningAgenda(items, AgendaBudget{TotalMinutes: 10, OptionalityReserve: 0})
	if !hasTopic(p, "urgent-future") || hasTopic(p, "sleepy-now") {
		t.Fatalf("horizon leaked into the ranking; selected: %v", selectedTopics(p))
	}
	// And the symmetric case: a high-urgency now item beats a low-urgency future item.
	flip := []LearningItem{
		{Topic: "urgent-now", Horizon: "now", Minutes: 10, Urgency: 0.90},
		{Topic: "sleepy-future", Horizon: "future", Minutes: 10, Urgency: 0.10},
	}
	q := SelectLearningAgenda(flip, AgendaBudget{TotalMinutes: 10, OptionalityReserve: 0})
	if !hasTopic(q, "urgent-now") || hasTopic(q, "sleepy-future") {
		t.Fatalf("ranking should follow urgency, not horizon; selected: %v", selectedTopics(q))
	}
}

// TestLearningAgendaReserveIsFloorNotCeiling proves the reserve never wastes attention
// and never caps optionality:
//   - with no optionality candidates, the reserve releases its minutes to the open pool;
//   - optionality items may also win the open pool, spending more than the reserve.
func TestLearningAgendaReserveIsFloorNotCeiling(t *testing.T) {
	// No optionality candidates: the full budget must still be spendable on gen/now.
	noneOptional := []LearningItem{{Topic: "big-now", Horizon: "now", Minutes: 60, Urgency: 0.5}}
	p := SelectLearningAgenda(noneOptional, AgendaBudget{TotalMinutes: 60, OptionalityReserve: DefaultOptionalityReserve})
	if !hasTopic(p, "big-now") {
		t.Fatalf("reserve stranded budget: a 60m now item did not fit a 60m budget")
	}
	if p.ReserveUsed != 0 || p.MinutesUsed != 60 {
		t.Fatalf("reserve used = %d, minutes used = %d; want 0 and 60", p.ReserveUsed, p.MinutesUsed)
	}

	// Optionality can exceed the reserve by winning the open pool on urgency.
	mixed := []LearningItem{
		{Topic: "small-future", Horizon: "future", Minutes: 10, Urgency: 0.10},
		{Topic: "urgent-next", Horizon: "next", Minutes: 50, Urgency: 0.99},
		{Topic: "mid-now", Horizon: "now", Minutes: 40, Urgency: 0.50},
	}
	q := SelectLearningAgenda(mixed, AgendaBudget{TotalMinutes: 100, OptionalityReserve: 0.10})
	if q.ReserveMinutes != 10 {
		t.Fatalf("reserve minutes = %d, want 10", q.ReserveMinutes)
	}
	if got := q.OptionalityMinutes(); got != 60 {
		t.Fatalf("optionality minutes = %d, want 60 (reserve is a floor, not a cap)", got)
	}
	if q.MinutesUsed != 100 {
		t.Fatalf("minutes used = %d, want 100", q.MinutesUsed)
	}
	// The 50m optionality item could not fit the 10m reserve, so it must have been
	// bought in the open pool — the reserve skipped it without dropping it.
	for _, s := range q.Selected {
		if s.Topic == "urgent-next" && s.Pool != PoolOpen {
			t.Fatalf("urgent-next pool = %q, want %q", s.Pool, PoolOpen)
		}
	}
}

// TestLearningAgendaFailsClosedOnInvalidItems checks the closed vocabularies: an unknown
// horizon and a non-positive cost are named refusals, distinct from budget exhaustion,
// and never silently selected.
func TestLearningAgendaFailsClosedOnInvalidItems(t *testing.T) {
	items := []LearningItem{
		{Topic: "someday-item", Horizon: "someday", Minutes: 5, Urgency: 0.9},
		{Topic: "free-lunch", Horizon: "now", Minutes: 0, Urgency: 0.9},
		{Topic: "negative-cost", Horizon: "future", Minutes: -5, Urgency: 0.9},
		{Topic: "good-item", Horizon: "now", Minutes: 5, Urgency: 0.5},
	}
	p := SelectLearningAgenda(items, AgendaBudget{TotalMinutes: 100, OptionalityReserve: DefaultOptionalityReserve})

	if len(p.Selected) != 1 || p.Selected[0].Topic != "good-item" {
		t.Fatalf("invalid items leaked into the brief; selected: %v", selectedTopics(p))
	}
	for topic, want := range map[string]AgendaDeferral{
		"someday-item":  DeferUnknownHorizon,
		"free-lunch":    DeferInvalidCost,
		"negative-cost": DeferInvalidCost,
	} {
		got, ok := deferralFor(p, topic)
		if !ok || got != want {
			t.Fatalf("deferral for %q = (%q,%v), want (%q,true)", topic, got, ok, want)
		}
	}
	// Conservation: every input item lands in exactly one of Selected / Deferred.
	if n := len(p.Selected) + len(p.Deferred); n != len(items) {
		t.Fatalf("selected+deferred = %d, want %d (an item was silently dropped)", n, len(items))
	}
}

// TestLearningAgendaDeterministicTieBreak pins the tie-break: equal urgency resolves by
// topic ascending, never by input order, so a brief is reproducible.
func TestLearningAgendaDeterministicTieBreak(t *testing.T) {
	items := []LearningItem{
		{Topic: "zebra", Horizon: "now", Minutes: 10, Urgency: 0.5},
		{Topic: "alpha", Horizon: "now", Minutes: 10, Urgency: 0.5},
	}
	p := SelectLearningAgenda(items, AgendaBudget{TotalMinutes: 10, OptionalityReserve: 0})
	if !hasTopic(p, "alpha") || hasTopic(p, "zebra") {
		t.Fatalf("tie-break should prefer topic ascending; selected: %v", selectedTopics(p))
	}
	// Reversing input order must not change the outcome.
	q := SelectLearningAgenda([]LearningItem{items[1], items[0]}, AgendaBudget{TotalMinutes: 10, OptionalityReserve: 0})
	if len(q.Selected) != 1 || q.Selected[0].Topic != "alpha" {
		t.Fatalf("selection depends on input order; selected: %v", selectedTopics(q))
	}
}

// TestLearningAgendaRenderCarriesOrthogonalityAndMix proves the rendered agenda states
// the generation/priority/trunk/feature-gate orthogonality, shows every horizon in the
// mix (including starved ones), names its deferrals, and is byte-deterministic.
func TestLearningAgendaRenderCarriesOrthogonalityAndMix(t *testing.T) {
	p := SelectLearningAgenda(urgentNowFlood(), AgendaBudget{TotalMinutes: 60, OptionalityReserve: DefaultOptionalityReserve})
	out := p.Render()

	if !strings.Contains(out, OrthogonalityNote) {
		t.Fatalf("render missing orthogonality note:\n%s", out)
	}
	for _, kw := range []string{"priority", "shared trunk", "feature gate"} {
		if !strings.Contains(strings.ToLower(out), kw) {
			t.Fatalf("orthogonality note does not name %q:\n%s", kw, out)
		}
	}
	// Every horizon appears in the mix, so a starved horizon is visible, not absent.
	for _, s := range RoadmapGenerations {
		if !strings.Contains(out, s+"=") {
			t.Fatalf("render missing horizon %q in mix:\n%s", s, out)
		}
	}
	if !strings.Contains(out, string(PoolReserve)) {
		t.Fatalf("render should name the pool that funded the optionality item:\n%s", out)
	}
	if !strings.Contains(out, string(DeferBudgetExhausted)) {
		t.Fatalf("render should name the deferral reason:\n%s", out)
	}
	if p.Render() != out {
		t.Fatal("Render is not deterministic")
	}
}

// TestReserveWasBindingIsTheKillCriterion makes the demotion evidence checkable, not just
// prose: the reserve is "binding" when it changes the outcome, and "ceremony" (the retire
// signal) when it does not. This is the measurement an operator counts before dropping
// DefaultOptionalityReserve to zero.
func TestReserveWasBindingIsTheKillCriterion(t *testing.T) {
	// Binding: the urgent-now flood is exactly the case the reserve exists for — without
	// it the future item is starved, so keeping the reserve funds more optionality.
	if !ReserveWasBinding(urgentNowFlood(), AgendaBudget{TotalMinutes: 60, OptionalityReserve: DefaultOptionalityReserve}) {
		t.Fatal("reserve should be binding on the urgent-now flood")
	}
	// Ceremony: when optionality fits in the open pool anyway, the reserve changes
	// nothing — the exact "reserve never binding" demotion evidence the memo names.
	roomy := []LearningItem{
		{Topic: "future-standard", Horizon: "future", Minutes: 15, Urgency: 0.10},
		{Topic: "now-thing", Horizon: "now", Minutes: 15, Urgency: 0.99},
	}
	if ReserveWasBinding(roomy, AgendaBudget{TotalMinutes: 100, OptionalityReserve: DefaultOptionalityReserve}) {
		t.Fatal("reserve should NOT be binding when optionality fits without it")
	}
	// A zero reserve can never change the outcome, so it is never binding.
	if ReserveWasBinding(urgentNowFlood(), AgendaBudget{TotalMinutes: 60, OptionalityReserve: 0}) {
		t.Fatal("a zero reserve is never binding")
	}
}

// TestAgendaDeferralsClosed binds the deferral vocabulary; a drift here is a spec change.
func TestAgendaDeferralsClosed(t *testing.T) {
	want := []AgendaDeferral{DeferBudgetExhausted, DeferUnknownHorizon, DeferInvalidCost}
	if len(AgendaDeferrals) != len(want) {
		t.Fatalf("AgendaDeferrals = %v, want %v", AgendaDeferrals, want)
	}
	for i, d := range want {
		if AgendaDeferrals[i] != d {
			t.Fatalf("AgendaDeferrals[%d] = %q, want %q", i, AgendaDeferrals[i], d)
		}
	}
}
