package session

import (
	"math"
	"testing"
)

// TestThrottleRatio pins the (0,1] ratio at every boundary the composition reasons about:
// no-opinion / no-baseline / at-or-above-baseline all read as "no throttle" (1.0), and a
// real sub-baseline cap is the exact quotient.
func TestThrottleRatio(t *testing.T) {
	cases := []struct {
		name     string
		pace     Pace
		baseline int
		want     float64
	}{
		{"no opinion (zero cap)", Pace{MaxTokensPerTurn: 0}, 2048, 1.0},
		{"no opinion (negative cap)", Pace{MaxTokensPerTurn: -5}, 2048, 1.0},
		{"no baseline", Pace{MaxTokensPerTurn: 256}, 0, 1.0},
		{"negative baseline", Pace{MaxTokensPerTurn: 256}, -1, 1.0},
		{"cap above baseline is not a throttle", Pace{MaxTokensPerTurn: 4096}, 2048, 1.0},
		{"cap equal to baseline is not a throttle", Pace{MaxTokensPerTurn: 2048}, 2048, 1.0},
		{"half", Pace{MaxTokensPerTurn: 1024}, 2048, 0.5},
		{"quarter", Pace{MaxTokensPerTurn: 512}, 2048, 0.25},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := c.pace.ThrottleRatio(c.baseline)
			if math.Abs(got-c.want) > 1e-9 {
				t.Fatalf("ThrottleRatio(%+v, %d) = %v, want %v", c.pace, c.baseline, got, c.want)
			}
		})
	}
}

// TestComposePlannerBudgetThrottlesDown is the load-bearing assertion: a throttled Pace
// drives the resident-context window DOWN proportionally, and an unthrottled one leaves it
// byte-for-byte unchanged.
func TestComposePlannerBudgetThrottlesDown(t *testing.T) {
	const base = 4096
	const baseline = 2048

	// Half the per-turn output -> half the window.
	if got := (Pace{MaxTokensPerTurn: 1024}).ComposePlannerBudget(base, baseline); got != 2048 {
		t.Fatalf("half-throttle planner budget = %d, want 2048", got)
	}
	// No opinion -> the base, unchanged (the pre-compose path).
	if got := (Pace{}).ComposePlannerBudget(base, baseline); got != base {
		t.Fatalf("un-paced planner budget = %d, want %d (must be byte-for-byte the base)", got, base)
	}
	// A cap above the baseline is not a throttle -> the base, unchanged.
	if got := (Pace{MaxTokensPerTurn: 9000}).ComposePlannerBudget(base, baseline); got != base {
		t.Fatalf("above-baseline planner budget = %d, want %d", got, base)
	}
}

// TestComposePlannerBudgetFloor proves a deep throttle never starves the window below
// base/MinPlannerBudgetDivisor, and that the floor binds only when the proportional value
// would fall under it.
func TestComposePlannerBudgetFloor(t *testing.T) {
	const base = 4096
	const baseline = 100000
	floor := base / MinPlannerBudgetDivisor // 512

	// MaxTokensPerTurn=1 against a huge baseline -> ratio ~1e-5 -> would round to 0, but the
	// floor must catch it.
	if got := (Pace{MaxTokensPerTurn: 1}).ComposePlannerBudget(base, baseline); got != floor {
		t.Fatalf("deep-throttle planner budget = %d, want the floor %d (a throttle must never zero the window)", got, floor)
	}
	if got := (Pace{MaxTokensPerTurn: 1}).ComposePlannerBudget(base, baseline); got <= 0 {
		t.Fatalf("planner budget %d <= 0: a throttled window must stay usable", got)
	}

	// A throttle that lands ABOVE the floor is honored exactly (the floor does not bind).
	if got := (Pace{MaxTokensPerTurn: 1024}).ComposePlannerBudget(base, 2048); got != 2048 {
		t.Fatalf("above-floor throttle = %d, want 2048 (floor must not clamp a valid value)", got)
	}
}

// TestComposePlannerBudgetNonPositiveBase leaves a non-configured window alone.
func TestComposePlannerBudgetNonPositiveBase(t *testing.T) {
	if got := (Pace{MaxTokensPerTurn: 512}).ComposePlannerBudget(0, 2048); got != 0 {
		t.Fatalf("zero base = %d, want 0 (nothing to scale)", got)
	}
	if got := (Pace{MaxTokensPerTurn: 512}).ComposePlannerBudget(-1, 2048); got != -1 {
		t.Fatalf("negative base = %d, want -1 (returned unchanged)", got)
	}
}

// TestComposeWorkerFraction proves the matmul fraction tracks the throttle and stays a
// valid (0,1] budget model.SetWorkerBudget will accept (any positive fraction floors to >=1
// worker, so even a deep throttle never zeroes compute).
func TestComposeWorkerFraction(t *testing.T) {
	if got := (Pace{MaxTokensPerTurn: 1024}).ComposeWorkerFraction(2048); math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("half-throttle worker fraction = %v, want 0.5", got)
	}
	if got := (Pace{}).ComposeWorkerFraction(2048); got != 1.0 {
		t.Fatalf("un-paced worker fraction = %v, want 1.0 (full machine)", got)
	}
	// A deep throttle stays strictly positive and within (0,1].
	frac := (Pace{MaxTokensPerTurn: 1}).ComposeWorkerFraction(100000)
	if frac <= 0 || frac > 1 {
		t.Fatalf("deep-throttle worker fraction = %v, want a value in (0,1]", frac)
	}
}

// TestComposeFoldsBoth proves the single Compose call agrees with the per-axis methods —
// one knob, both budgets, from the same baseline.
func TestComposeFoldsBoth(t *testing.T) {
	p := Pace{MaxTokensPerTurn: 512}
	const base = 4096
	const baseline = 2048
	got := p.Compose(base, baseline)

	if got.PlannerBudget != p.ComposePlannerBudget(base, baseline) {
		t.Fatalf("Compose.PlannerBudget = %d, disagrees with ComposePlannerBudget", got.PlannerBudget)
	}
	if got.WorkerFraction != p.ComposeWorkerFraction(baseline) {
		t.Fatalf("Compose.WorkerFraction = %v, disagrees with ComposeWorkerFraction", got.WorkerFraction)
	}
	if got.Ratio != p.ThrottleRatio(baseline) {
		t.Fatalf("Compose.Ratio = %v, disagrees with ThrottleRatio", got.Ratio)
	}
	// A quarter-throttle is the same number on both axes (the shared ratio).
	if math.Abs(got.Ratio-0.25) > 1e-9 {
		t.Fatalf("Compose.Ratio = %v, want 0.25", got.Ratio)
	}
}

// TestThroughputRatio pins the observed-throughput ratio at every boundary #1585 reasons
// about: no observation, no expectation, and keeping-pace-or-faster all read as "no
// constraint" (1.0); falling behind is the exact quotient.
func TestThroughputRatio(t *testing.T) {
	cases := []struct {
		name string
		t    Throughput
		want float64
	}{
		{"no observation yet", Throughput{ExpectedTokensPerSec: 100}, 1.0},
		{"no expectation configured", Throughput{ObservedTokensPerSec: 50}, 1.0},
		{"negative observation", Throughput{ObservedTokensPerSec: -5, ExpectedTokensPerSec: 100}, 1.0},
		{"keeping pace exactly", Throughput{ObservedTokensPerSec: 100, ExpectedTokensPerSec: 100}, 1.0},
		{"running ahead", Throughput{ObservedTokensPerSec: 150, ExpectedTokensPerSec: 100}, 1.0},
		{"half pace", Throughput{ObservedTokensPerSec: 50, ExpectedTokensPerSec: 100}, 0.5},
		{"quarter pace", Throughput{ObservedTokensPerSec: 25, ExpectedTokensPerSec: 100}, 0.25},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.t.ThroughputRatio(); math.Abs(got-c.want) > 1e-9 {
				t.Fatalf("ThroughputRatio(%+v) = %v, want %v", c.t, got, c.want)
			}
		})
	}
}

// TestComposePlannerBudgetForThroughputShrinksDown is the #1585 load-bearing assertion: a
// session measurably falling behind its EXPECTED throughput (no configured cap at all) still
// drives the resident-context window down proportionally, and a session keeping pace leaves
// it byte-for-byte unchanged.
func TestComposePlannerBudgetForThroughputShrinksDown(t *testing.T) {
	const base = 4096

	slow := Throughput{ObservedTokensPerSec: 25, ExpectedTokensPerSec: 100} // quarter pace
	if got := slow.ComposePlannerBudgetForThroughput(base); got != 1024 {
		t.Fatalf("quarter-pace planner budget = %d, want 1024", got)
	}

	onPace := Throughput{ObservedTokensPerSec: 100, ExpectedTokensPerSec: 100}
	if got := onPace.ComposePlannerBudgetForThroughput(base); got != base {
		t.Fatalf("on-pace planner budget = %d, want %d (byte-for-byte unchanged)", got, base)
	}

	noSignal := Throughput{}
	if got := noSignal.ComposePlannerBudgetForThroughput(base); got != base {
		t.Fatalf("no-signal planner budget = %d, want %d", got, base)
	}
}

// TestComposePlannerBudgetForThroughputFloor proves an observed-throughput collapse never
// starves the resident window below base/MinPlannerBudgetDivisor — the same floor
// ComposePlannerBudget already guarantees for the configured-cap axis, now proven for the
// OBSERVED axis (the "minimum resident context preserved" done condition).
func TestComposePlannerBudgetForThroughputFloor(t *testing.T) {
	const base = 4096
	floor := base / MinPlannerBudgetDivisor // 512

	crawling := Throughput{ObservedTokensPerSec: 0.001, ExpectedTokensPerSec: 1000}
	if got := crawling.ComposePlannerBudgetForThroughput(base); got != floor {
		t.Fatalf("near-stalled planner budget = %d, want the floor %d", got, floor)
	}
	if got := crawling.ComposePlannerBudgetForThroughput(base); got <= 0 {
		t.Fatalf("planner budget %d <= 0: an observed-throughput collapse must stay usable", got)
	}
}

// TestComposePaceTakesTheTighterConstraint proves ComposePace reconciles the configured cap
// and the observed throughput signal by taking whichever shrinks the window MORE, never
// letting one silently override the other.
func TestComposePaceTakesTheTighterConstraint(t *testing.T) {
	const base = 4096
	const baselineOutput = 2048

	// Configured cap throttles harder (quarter) than the observed throughput (half).
	p := Pace{MaxTokensPerTurn: 512} // quarter of baselineOutput -> 1024
	tp := Throughput{ObservedTokensPerSec: 50, ExpectedTokensPerSec: 100}
	configured := p.ComposePlannerBudget(base, baselineOutput)
	observed := tp.ComposePlannerBudgetForThroughput(base)
	if configured >= observed {
		t.Fatalf("premise broken: configured=%d must be tighter than observed=%d for this test to prove anything", configured, observed)
	}
	if got := p.ComposePace(tp, base, baselineOutput); got != configured {
		t.Fatalf("ComposePace = %d, want the tighter configured budget %d", got, configured)
	}

	// Now flip it: observed throughput is the tighter constraint.
	q := Pace{MaxTokensPerTurn: 1536} // mild throttle -> closer to base
	tq := Throughput{ObservedTokensPerSec: 10, ExpectedTokensPerSec: 100}
	qConfigured := q.ComposePlannerBudget(base, baselineOutput)
	qObserved := tq.ComposePlannerBudgetForThroughput(base)
	if qObserved >= qConfigured {
		t.Fatalf("premise broken: observed=%d must be tighter than configured=%d for this test to prove anything", qObserved, qConfigured)
	}
	if got := q.ComposePace(tq, base, baselineOutput); got != qObserved {
		t.Fatalf("ComposePace = %d, want the tighter observed budget %d", got, qObserved)
	}

	// Neither signal set -> the base, unchanged.
	if got := (Pace{}).ComposePace(Throughput{}, base, baselineOutput); got != base {
		t.Fatalf("no-signal ComposePace = %d, want %d", got, base)
	}
}
