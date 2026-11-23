// fak-test:runtime fast est=10ms lane=default
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

func baseConvergence() ConvergenceConfig {
	return ConvergenceConfig{
		Planner:          PlannerBayesian,
		SearchSpacePaths: []string{"load.concurrency", "load.request_rate"},
		MaxIterations:    30,
	}
}

func TestConvergenceThreeSignals(t *testing.T) {
	nan := math.NaN()
	cases := []struct {
		name          string
		mutate        func(*ConvergenceConfig)
		objectives    []float64
		lowerIsBetter bool
		wantStop      bool
		wantReason    string
	}{
		{
			// Each step improves by 10%, so the last-4 CV (~0.12) is far above
			// 0.01 and the patience counter never leaves zero.
			name:          "still_improving_continues",
			mutate:        func(c *ConvergenceConfig) { c.PlateauWindow = 4; c.ImprovementPatience = 3 },
			objectives:    []float64{100, 90, 81, 72.9, 65.61},
			lowerIsBetter: true,
		},
		{
			// Last 4 are {50, 50.1, 49.9, 50}: mean 50, population std
			// sqrt(0.005) ~ 0.0707, CV ~ 0.0014 < 0.01.
			name:          "plateau_stops",
			mutate:        func(c *ConvergenceConfig) { c.PlateauWindow = 4 },
			objectives:    []float64{100, 70, 50, 50.1, 49.9, 50},
			lowerIsBetter: true,
			wantStop:      true,
			wantReason:    ReasonPlateau,
		},
		{
			// The same flat tail is only 4 long; the default 8-wide window is
			// not yet full, so the plateau cannot fire.
			name:          "plateau_needs_a_full_window",
			objectives:    []float64{100, 70, 50, 50.1, 49.9, 50},
			lowerIsBetter: true,
		},
		{
			// Best is 10 (run 2); runs 3..5 swing 30/12/28 without beating it.
			// Last-4 CV ~ 0.45 rules out a plateau; patience 3 fires.
			name:          "patience_stops_without_plateau",
			mutate:        func(c *ConvergenceConfig) { c.PlateauWindow = 4; c.ImprovementPatience = 3 },
			objectives:    []float64{20, 10, 30, 12, 28},
			lowerIsBetter: true,
			wantStop:      true,
			wantReason:    ReasonNoImprovement,
		},
		{
			// Larger-is-better: 120 is the best; 80/118/60 never beat it.
			name:       "patience_larger_is_better",
			mutate:     func(c *ConvergenceConfig) { c.PlateauWindow = 4; c.ImprovementPatience = 3 },
			objectives: []float64{100, 120, 80, 118, 60},
			wantStop:   true,
			wantReason: ReasonNoImprovement,
		},
		{
			// A flat tail satisfies plateau and patience together and the
			// budget is spent too: the budget is reported first.
			name: "budget_reported_before_plateau_and_patience",
			mutate: func(c *ConvergenceConfig) {
				c.MaxIterations = 5
				c.NInitialPoints = 2
				c.PlateauWindow = 3
				c.ImprovementPatience = 2
			},
			objectives:    []float64{9, 5, 5, 5, 5},
			lowerIsBetter: true,
			wantStop:      true,
			wantReason:    ReasonMaxRuns,
		},
		{
			// Same flat tail with budget left: plateau outranks patience.
			name:          "plateau_reported_before_patience",
			mutate:        func(c *ConvergenceConfig) { c.PlateauWindow = 3; c.ImprovementPatience = 2 },
			objectives:    []float64{9, 5, 5, 5, 5},
			lowerIsBetter: true,
			wantStop:      true,
			wantReason:    ReasonPlateau,
		},
		{
			// A failed iteration inside the window blocks the plateau; it also
			// counts against patience (5, nan, 5 -> three runs without a new best).
			name:          "failed_iteration_blocks_plateau",
			mutate:        func(c *ConvergenceConfig) { c.PlateauWindow = 3; c.ImprovementPatience = 9 },
			objectives:    []float64{9, 5, 5, nan, 5},
			lowerIsBetter: true,
		},
		{
			// {9, 11}: mean 10, population std exactly 1, CV 0.1. A sample
			// std (sqrt 2) would give 0.141 and miss the 0.12 threshold.
			name:          "plateau_uses_population_std",
			mutate:        func(c *ConvergenceConfig) { c.PlateauWindow = 2; c.PlateauThreshold = 0.12 },
			objectives:    []float64{30, 9, 11},
			lowerIsBetter: true,
			wantStop:      true,
			wantReason:    ReasonPlateau,
		},
		{
			// Same window, CV exactly on the threshold: the bound is strict.
			name:          "cv_equal_to_threshold_continues",
			mutate:        func(c *ConvergenceConfig) { c.PlateauWindow = 2; c.PlateauThreshold = 0.1 },
			objectives:    []float64{30, 9, 11},
			lowerIsBetter: true,
		},
		{
			name:          "zero_mean_flat_window_is_a_plateau",
			mutate:        func(c *ConvergenceConfig) { c.PlateauWindow = 3 },
			objectives:    []float64{0, 0, 0},
			lowerIsBetter: true,
			wantStop:      true,
			wantReason:    ReasonPlateau,
		},
		{
			// Mean is exactly zero but the window swings, so CV is undefined
			// and must not read as "flat".
			name:          "zero_mean_swinging_window_is_not",
			mutate:        func(c *ConvergenceConfig) { c.PlateauWindow = 2; c.ImprovementPatience = 9 },
			objectives:    []float64{-1, 1},
			lowerIsBetter: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseConvergence()
			if tc.mutate != nil {
				tc.mutate(&cfg)
			}
			if _, err := cfg.Validate(); err != nil {
				t.Fatalf("Validate: %v", err)
			}
			stop, reason := cfg.Evaluate(tc.objectives, tc.lowerIsBetter)
			if stop != tc.wantStop || reason != tc.wantReason {
				t.Fatalf("Evaluate = (%v, %q), want (%v, %q)", stop, reason, tc.wantStop, tc.wantReason)
			}
		})
	}
}

func TestConvergenceOneDimensionalPlannerHonoursOnlyBudget(t *testing.T) {
	cfg := baseConvergence()
	cfg.Planner = PlannerMonotonicSLA
	cfg.MaxIterations = 6
	cfg.PlateauWindow = 3
	cfg.ImprovementPatience = 2

	flat := []float64{9, 5, 5, 5, 5}
	if stop, reason := cfg.Evaluate(flat, true); stop {
		t.Fatalf("1-D planner stopped on %q; it must ignore plateau and patience", reason)
	}
	if stop, reason := cfg.Evaluate(append(flat, 5), true); !stop || reason != ReasonMaxRuns {
		t.Fatalf("Evaluate at budget = (%v, %q), want (true, %q)", stop, reason, ReasonMaxRuns)
	}
}

func TestConvergenceRejectsDuplicateSearchSpacePaths(t *testing.T) {
	paths := []string{"load.concurrency", "load.rate", "load.concurrency", "model.batch", "load.rate", "load.concurrency"}
	if got, want := DuplicateSearchSpacePaths(paths), []string{"load.concurrency", "load.rate"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("DuplicateSearchSpacePaths = %v, want %v (each once, in order of first repeat)", got, want)
	}
	if got := DuplicateSearchSpacePaths([]string{"a", "b", "c"}); got != nil {
		t.Fatalf("unique paths reported duplicates: %v", got)
	}

	for _, planner := range []string{PlannerBayesian, PlannerMonotonicSLA} {
		cfg := baseConvergence()
		cfg.Planner = planner
		cfg.SearchSpacePaths = paths
		_, err := cfg.Validate()
		if err == nil {
			t.Fatalf("%s: duplicate search-space paths were accepted", planner)
		}
		for _, want := range []string{"unique paths", `"load.concurrency"`, `"load.rate"`} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("%s: error %q does not mention %s", planner, err, want)
			}
		}
		if strings.Contains(err.Error(), "model.batch") {
			t.Fatalf("%s: error %q names a path that is not duplicated", planner, err)
		}
	}
}

func TestConvergenceWarnsOnKnobsOneDimensionalPlannerIgnores(t *testing.T) {
	cfg := baseConvergence()
	cfg.Planner = PlannerSmoothIsotonic
	cfg.ImprovementPatience = 3
	cfg.PlateauThreshold = 0.05

	warnings, err := cfg.Validate()
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings = %q, want exactly one", warnings)
	}
	for _, want := range []string{"improvement_patience", "plateau_threshold", PlannerSmoothIsotonic, "ignored"} {
		if !strings.Contains(warnings[0], want) {
			t.Fatalf("warning %q does not mention %q", warnings[0], want)
		}
	}
	if strings.Contains(warnings[0], "plateau_window") {
		t.Fatalf("warning %q names plateau_window, which was left at its default", warnings[0])
	}

	windowOnly := baseConvergence()
	windowOnly.Planner = PlannerMonotonicSLA
	windowOnly.PlateauWindow = 4
	warnings, err = windowOnly.Validate()
	if err != nil || len(warnings) != 1 || !strings.Contains(warnings[0], "(plateau_window)") {
		t.Fatalf("1-D config with only plateau_window set: warnings=%q err=%v", warnings, err)
	}

	// Nothing set, nothing to warn about; and the same knobs on an N-D
	// planner are consumed, not ignored.
	quiet := baseConvergence()
	quiet.Planner = PlannerMonotonicSLA
	if warnings, err := quiet.Validate(); err != nil || len(warnings) != 0 {
		t.Fatalf("defaulted 1-D config: warnings=%q err=%v, want none", warnings, err)
	}
	consumed := baseConvergence()
	consumed.ImprovementPatience = 3
	consumed.PlateauThreshold = 0.05
	if warnings, err := consumed.Validate(); err != nil || len(warnings) != 0 {
		t.Fatalf("N-D config with knobs set: warnings=%q err=%v, want none", warnings, err)
	}
}

func TestConvergenceInitialPointsGate(t *testing.T) {
	// Default initial points (5) against a budget of 3: fatal for a model-based
	// planner, fine for a 1-D planner that never fits a model.
	cfg := baseConvergence()
	cfg.MaxIterations = 3
	if _, err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "must be < max iterations (3)") {
		t.Fatalf("bayesian with 5 initial points and 3 iterations: err = %v", err)
	}
	cfg.Planner = PlannerMonotonicSLA
	if warnings, err := cfg.Validate(); err != nil || len(warnings) != 0 {
		t.Fatalf("monotonic_sla with 3 iterations: warnings=%q err=%v, want accepted", warnings, err)
	}

	// Equal is refused too: with no iteration left after the initial points
	// the model never fits. One more iteration is enough.
	edge := baseConvergence()
	edge.MaxIterations = 5
	if _, err := edge.Validate(); err == nil || !strings.Contains(err.Error(), "initial points (5) must be < max iterations (5)") {
		t.Fatalf("5 initial points and 5 iterations: err = %v", err)
	}
	edge.MaxIterations = 6
	if _, err := edge.Validate(); err != nil {
		t.Fatalf("5 initial points and 6 iterations refused: %v", err)
	}

	// The default window (8) is larger than the default initial points (5).
	if warnings, err := baseConvergence().Validate(); err != nil || len(warnings) != 0 {
		t.Fatalf("default config: warnings=%q err=%v, want none", warnings, err)
	}
	narrow := baseConvergence()
	narrow.PlateauWindow = 5
	warnings, err := narrow.Validate()
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "plateau window (5) is not larger than initial points (5)") {
		t.Fatalf("window equal to initial points: warnings = %q", warnings)
	}
}

func TestConvergenceRefusesOutOfRangeKnobs(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*ConvergenceConfig)
		wantErr string
	}{
		{"unknown_planner", func(c *ConvergenceConfig) { c.Planner = "grid" }, `unknown planner "grid"`},
		{"empty_search_space", func(c *ConvergenceConfig) { c.SearchSpacePaths = nil }, "at least one dimension"},
		{"budget_too_small", func(c *ConvergenceConfig) { c.MaxIterations = 1 }, "outside [2, 200]"},
		{"budget_too_large", func(c *ConvergenceConfig) { c.MaxIterations = 201 }, "outside [2, 200]"},
		{"negative_initial_points", func(c *ConvergenceConfig) { c.NInitialPoints = -1 }, "must be at least 1"},
		{"window_of_one", func(c *ConvergenceConfig) { c.PlateauWindow = 1 }, "plateau window 1 must be at least 2"},
		{"negative_threshold", func(c *ConvergenceConfig) { c.PlateauThreshold = -0.1 }, "must be positive and finite"},
		{"nan_threshold", func(c *ConvergenceConfig) { c.PlateauThreshold = math.NaN() }, "must be positive and finite"},
		{"infinite_threshold", func(c *ConvergenceConfig) { c.PlateauThreshold = math.Inf(1) }, "must be positive and finite"},
		{"patience_of_one", func(c *ConvergenceConfig) { c.ImprovementPatience = 1 }, "improvement patience 1 must be at least 2"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseConvergence()
			tc.mutate(&cfg)
			_, err := cfg.Validate()
			if err == nil {
				t.Fatal("out-of-range config was accepted")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not contain %q", err, tc.wantErr)
			}
		})
	}
}
