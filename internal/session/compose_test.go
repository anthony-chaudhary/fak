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
