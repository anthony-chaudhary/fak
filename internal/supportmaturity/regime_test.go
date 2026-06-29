package supportmaturity

import (
	"fmt"
	"reflect"
	"testing"
)

// goldenRegime pins, per rung, the regime a cell at that rung MUST route to -- the
// issue's witness ("a golden cell at each rung emits the expected regime"). It is
// the authoritative resolution of the doctrine's M1 band overlap: M0->R0, M1-M3->R1,
// M4-M5->R2, M6-M7->R3 (the GPT-NeoX FENCED / M1 -> R1 worked-example cell decides
// M1). If a ninth rung is ever added to the ladder this table must grow with it --
// and TestEveryRungRoutesToOneRegime fails until it does, which is the point.
var goldenRegime = map[Rung]Regime{
	M0None:       R0Explore,
	M1Fenced:     R1Prototype,
	M2Loads:      R1Prototype,
	M3Runs:       R1Prototype,
	M4Correct:    R2Optimize,
	M5Optimized:  R2Optimize,
	M6Parity:     R3Production,
	M7BeyondSOTA: R3Production,
}

// TestEveryRungRoutesToOneRegime is the issue's core witness: the rung->regime map
// is TOTAL (every closed rung routes), DETERMINISTIC (the same rung always yields
// the same regime), and matches the golden cell at each rung. It also asserts the
// emitted regime is Valid and carries a non-empty tooling SET.
func TestEveryRungRoutesToOneRegime(t *testing.T) {
	if len(goldenRegime) != len(Rungs) {
		t.Fatalf("golden table pins %d rungs, ladder has %d -- pin every rung", len(goldenRegime), len(Rungs))
	}
	for _, r := range Rungs {
		want, ok := goldenRegime[r]
		if !ok {
			t.Fatalf("rung %s has no golden regime pinned -- extend goldenRegime", r)
		}
		got := RegimeFor(r)
		if got != want {
			t.Fatalf("RegimeFor(%s) = %s (%s), want %s (%s)", r, got, got.Name(), want, want.Name())
		}
		if !got.Valid() {
			t.Fatalf("RegimeFor(%s) = %v, not a closed R0-R3 regime", r, got)
		}
		if again := RegimeFor(r); again != got { // determinism: a second call is identical
			t.Fatalf("RegimeFor(%s) is non-deterministic: %s then %s", r, got, again)
		}
		if tools := got.Playbook().Tooling; len(tools) == 0 {
			t.Fatalf("regime %s for rung %s carries an empty tooling set", got, r)
		}
	}
}

// TestGoldenToolingPerRung anchors the EXACT tooling set a cell at each rung emits
// -- the second half of the witness ("emits the expected regime + tooling set").
// It pins the doctrine's per-regime tooling so a silent edit to a tooling list
// trips the gate, and confirms PlaybookFor agrees with RegimeFor + Playbook.
func TestGoldenToolingPerRung(t *testing.T) {
	wantTooling := map[Regime][]string{
		R0Explore:    {"idea-scout", "research notes", "scouts"},
		R1Prototype:  {"dispatch worker", "get-to-green", "tests"},
		R2Optimize:   {"rsiloop", "shipgate", "kernel/compiler tooling", "benches"},
		R3Production: {"self-tax #1147 gate", "SLOs", "bgloop", "UX"},
	}
	for _, r := range Rungs {
		g, p := PlaybookFor(r)
		if g != RegimeFor(r) || !reflect.DeepEqual(p, g.Playbook()) {
			t.Fatalf("PlaybookFor(%s) disagrees with RegimeFor+Playbook", r)
		}
		if got := p.Tooling; !reflect.DeepEqual(got, wantTooling[g]) {
			t.Fatalf("rung %s -> regime %s tooling = %v, want %v", r, g, got, wantTooling[g])
		}
	}
}

// TestRegimeMonotoneInRung asserts the router is order-preserving: a higher rung
// never routes to a LOWER regime. This is what makes the regime a faithful handle
// on the ladder -- climbing a rung can only hold or advance the regime, never
// regress it.
func TestRegimeMonotoneInRung(t *testing.T) {
	for i := 1; i < len(Rungs); i++ {
		lo, hi := RegimeFor(Rungs[i-1]), RegimeFor(Rungs[i])
		if hi < lo {
			t.Fatalf("regime regresses %s->%s as rung rises %s->%s", lo, hi, Rungs[i-1], Rungs[i])
		}
	}
}

// TestRegimeBandsTileLadder asserts the four regimes partition the ladder into four
// CONTIGUOUS, non-overlapping bands that together cover all of M0..M7 -- no rung is
// unrouted and no rung routes to two regimes. The bands: R0={M0}, R1={M1,M2,M3},
// R2={M4,M5}, R3={M6,M7}.
func TestRegimeBandsTileLadder(t *testing.T) {
	band := map[Regime][]Rung{}
	for _, r := range Rungs {
		band[RegimeFor(r)] = append(band[RegimeFor(r)], r)
	}
	if len(band) != len(Regimes) {
		t.Fatalf("ladder routes to %d regimes, want all %d to own a band", len(band), len(Regimes))
	}
	prevHi := -1
	for _, g := range Regimes {
		rs := band[g]
		if len(rs) == 0 {
			t.Fatalf("regime %s covers no rung -- every regime must own a band", g)
		}
		for j := 1; j < len(rs); j++ {
			if int(rs[j]) != int(rs[j-1])+1 {
				t.Fatalf("regime %s band not contiguous: %s then %s", g, rs[j-1], rs[j])
			}
		}
		if int(rs[0]) != prevHi+1 {
			t.Fatalf("regime %s band starts at %s -- gap/overlap after rung ordinal %d", g, rs[0], prevHi)
		}
		prevHi = int(rs[len(rs)-1])
	}
	if prevHi != len(Rungs)-1 {
		t.Fatalf("bands cover up to rung ordinal %d, want %d (all of M0..M7)", prevHi, len(Rungs)-1)
	}
}

// TestRegimePlaybookComplete asserts every regime carries a complete, distinct
// playbook: a non-empty expectation, a non-empty tooling SET, a report style, a
// who-operates, ids ("R0".."R3") matching the ordinal and agreeing with String(),
// and distinct one-word names agreeing with Name(). These are the four facets #1250
// asks each regime to carry.
func TestRegimePlaybookComplete(t *testing.T) {
	seenID, seenName := map[string]bool{}, map[string]bool{}
	for i, g := range Regimes {
		p := g.Playbook()
		if p.ID != fmt.Sprintf("R%d", i) {
			t.Fatalf("Regimes[%d].Playbook().ID = %q, want R%d", i, p.ID, i)
		}
		if g.String() != p.ID {
			t.Fatalf("regime %d String()=%q disagrees with Playbook().ID=%q", i, g.String(), p.ID)
		}
		if p.Name == "" || p.Name == "unknown" || g.Name() != p.Name {
			t.Fatalf("regime %s name inconsistent (%q vs %q)", p.ID, p.Name, g.Name())
		}
		if p.Expectation == "" {
			t.Fatalf("regime %s has no expectation", p.ID)
		}
		if len(p.Tooling) == 0 {
			t.Fatalf("regime %s has an empty tooling set", p.ID)
		}
		if p.ReportStyle == "" {
			t.Fatalf("regime %s has no report style", p.ID)
		}
		if p.WhoOperates == "" {
			t.Fatalf("regime %s has no who-operates", p.ID)
		}
		if !p.StepBudget.Continuous && p.StepBudget.MaxSteps <= 0 {
			t.Fatalf("regime %s has no finite step budget", p.ID)
		}
		if seenID[p.ID] {
			t.Fatalf("duplicate regime id %q", p.ID)
		}
		if seenName[p.Name] {
			t.Fatalf("duplicate regime name %q", p.Name)
		}
		seenID[p.ID], seenName[p.Name] = true, true
	}
}

// TestRegimeStepBudgets pins the work/effort horizon axis from #1251: R0 is
// abandon-cheap, R1 is a 10-100 step prototype, R2 is a 1k-10k optimization loop,
// and R3 is continuous production hold-work. The finite budgets are strictly
// increasing, so a cell can be re-regimed instead of silently overrunning.
func TestRegimeStepBudgets(t *testing.T) {
	want := map[Regime]StepBudget{
		R0Explore:    {MinSteps: 1, MaxSteps: 9},
		R1Prototype:  {MinSteps: 10, MaxSteps: 100},
		R2Optimize:   {MinSteps: 1000, MaxSteps: 10000},
		R3Production: {Continuous: true},
	}
	for _, g := range Regimes {
		got := g.StepBudget()
		if got != want[g] {
			t.Fatalf("%s StepBudget = %+v, want %+v", g, got, want[g])
		}
		if !got.Continuous && got.MinSteps > got.MaxSteps {
			t.Fatalf("%s budget min > max: %+v", g, got)
		}
	}
	if !(R0Explore.StepBudget().MaxSteps < R1Prototype.StepBudget().MinSteps &&
		R1Prototype.StepBudget().MaxSteps < R2Optimize.StepBudget().MinSteps) {
		t.Fatalf("finite step budgets are not ordered R0 << R1 << R2")
	}
	if !R3Production.StepBudget().Continuous {
		t.Fatalf("R3 production must be a continuous horizon")
	}
}

// TestRegimeStepBudgetFlagsMisScoped is the #1251 witness: a cell whose observed
// work exceeds its regime budget is flagged for re-regime/escalation, while in-budget
// and continuous production work stays within scope.
func TestRegimeStepBudgetFlagsMisScoped(t *testing.T) {
	cases := []struct {
		name  string
		g     Regime
		steps int
		want  ScopeDecision
	}{
		{"r0 in budget", R0Explore, 9, ScopeWithinBudget},
		{"r0 over budget", R0Explore, 10, ScopeMisScoped},
		{"r1 in budget", R1Prototype, 100, ScopeWithinBudget},
		{"r1 over budget", R1Prototype, 101, ScopeMisScoped},
		{"r2 in budget", R2Optimize, 10000, ScopeWithinBudget},
		{"r2 over budget", R2Optimize, 10001, ScopeMisScoped},
		{"r3 continuous", R3Production, 1_000_000, ScopeWithinBudget},
	}
	for _, tc := range cases {
		if got := tc.g.CheckStepBudget(tc.steps); got.Decision != tc.want {
			t.Fatalf("%s: decision = %s, want %s (check=%+v)", tc.name, got.Decision, tc.want, got)
		}
	}
}

// TestRegimeFloorsClosed asserts the closed-vocabulary guard: an out-of-range rung
// floors to R0Explore (never an invalid regime), and an out-of-range regime is not
// Valid, names "unknown", and its Playbook falls back to R0Explore's rather than a
// zero struct -- the same conservative floor the From* lowerings use.
func TestRegimeFloorsClosed(t *testing.T) {
	badRung := Rung(len(Rungs))
	if got := RegimeFor(badRung); got != R0Explore {
		t.Fatalf("RegimeFor(out-of-range rung) = %s, want R0Explore floor", got)
	}
	badRegime := Regime(len(Regimes))
	if badRegime.Valid() {
		t.Fatalf("out-of-range regime %d reports Valid", uint8(badRegime))
	}
	if got := badRegime.Name(); got != "unknown" {
		t.Fatalf("out-of-range regime Name() = %q, want \"unknown\"", got)
	}
	if got := badRegime.Playbook(); got.ID != "R0" {
		t.Fatalf("out-of-range regime Playbook().ID = %q, want R0 floor", got.ID)
	}
}
