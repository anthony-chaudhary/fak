package gateway

import (
	"net/http/httptest"
	"testing"
)

func TestSelectionObjectives(t *testing.T) {
	t.Run("parse_values_and_cases", func(t *testing.T) {
		cases := []struct {
			input string
			want  SelectionObjective
		}{
			{"default", ObjectiveDefault},
			{"latency", ObjectiveLatency},
			{"throughput", ObjectiveThroughput},
			{"portability", ObjectivePortability},
			{"determinism", ObjectiveDeterminism},
			{"debug", ObjectiveDebug},
			{"LATENCY", ObjectiveLatency},
			{"Throughput", ObjectiveThroughput},
			{"PoRtAbIlItY", ObjectivePortability},
			{"DETERMINISM", ObjectiveDeterminism},
			{"DeBuG", ObjectiveDebug},
			{"DEFAULT", ObjectiveDefault},
			{"", ObjectiveDefault},
			{"   ", ObjectiveDefault},
			{"unknown", ObjectiveDefault},
			{"unrecognized_val", ObjectiveDefault},
		}
		for _, tc := range cases {
			got := ParseSelectionObjective(tc.input)
			if got != tc.want {
				t.Fatalf("ParseSelectionObjective(%q) = %q, want %q", tc.input, got, tc.want)
			}
		}
	})

	t.Run("extract_from_headers", func(t *testing.T) {
		if got := ExtractSelectionObjective(nil); got != ObjectiveDefault {
			t.Fatalf("ExtractSelectionObjective(nil) = %q, want %q", got, ObjectiveDefault)
		}

		reqEmpty := httptest.NewRequest("GET", "/v1/chat/completions", nil)
		if got := ExtractSelectionObjective(reqEmpty); got != ObjectiveDefault {
			t.Fatalf("ExtractSelectionObjective(empty) = %q, want %q", got, ObjectiveDefault)
		}

		reqPrimary := httptest.NewRequest("GET", "/v1/chat/completions", nil)
		reqPrimary.Header.Set(HeaderSelectionObjective, "latency")
		if got := ExtractSelectionObjective(reqPrimary); got != ObjectiveLatency {
			t.Fatalf("ExtractSelectionObjective(primary) = %q, want %q", got, ObjectiveLatency)
		}

		reqFallback := httptest.NewRequest("GET", "/v1/chat/completions", nil)
		reqFallback.Header.Set("x-selection-objective", "throughput")
		if got := ExtractSelectionObjective(reqFallback); got != ObjectiveThroughput {
			t.Fatalf("ExtractSelectionObjective(fallback) = %q, want %q", got, ObjectiveThroughput)
		}

		reqBoth := httptest.NewRequest("GET", "/v1/chat/completions", nil)
		reqBoth.Header.Set(HeaderSelectionObjective, "determinism")
		reqBoth.Header.Set("x-selection-objective", "portability")
		if got := ExtractSelectionObjective(reqBoth); got != ObjectiveDeterminism {
			t.Fatalf("ExtractSelectionObjective(both) = %q, want %q", got, ObjectiveDeterminism)
		}

		reqPrimaryEmpty := httptest.NewRequest("GET", "/v1/chat/completions", nil)
		reqPrimaryEmpty.Header.Set(HeaderSelectionObjective, "   ")
		reqPrimaryEmpty.Header.Set("x-selection-objective", "portability")
		if got := ExtractSelectionObjective(reqPrimaryEmpty); got != ObjectivePortability {
			t.Fatalf("ExtractSelectionObjective(primary empty) = %q, want %q", got, ObjectivePortability)
		}

		reqUnknown := httptest.NewRequest("GET", "/v1/chat/completions", nil)
		reqUnknown.Header.Set(HeaderSelectionObjective, "invalid_choice")
		if got := ExtractSelectionObjective(reqUnknown); got != ObjectiveDefault {
			t.Fatalf("ExtractSelectionObjective(unknown) = %q, want %q", got, ObjectiveDefault)
		}
	})

	fastWorker := RouteEndpoint{
		Name:             "worker-fast",
		LatencyMs:        3.0,
		ThroughputTokS:   50.0,
		PortabilityScore: 0.2,
		DeterminismScore: 0.2,
		BasePriority:     1,
	}
	bulkWorker := RouteEndpoint{
		Name:             "worker-bulk",
		LatencyMs:        200.0,
		ThroughputTokS:   450.0,
		PortabilityScore: 0.2,
		DeterminismScore: 0.2,
		BasePriority:     1,
	}
	portableWorker := RouteEndpoint{
		Name:             "worker-portable",
		LatencyMs:        100.0,
		ThroughputTokS:   50.0,
		PortabilityScore: 0.99,
		DeterminismScore: 0.2,
		BasePriority:     1,
	}
	deterministicWorker := RouteEndpoint{
		Name:             "worker-deterministic",
		LatencyMs:        100.0,
		ThroughputTokS:   50.0,
		PortabilityScore: 0.2,
		DeterminismScore: 0.99,
		BasePriority:     1,
	}
	balancedWorker := RouteEndpoint{
		Name:             "worker-balanced",
		LatencyMs:        10.0,
		ThroughputTokS:   350.0,
		PortabilityScore: 0.95,
		DeterminismScore: 0.95,
		BasePriority:     50,
	}

	cohort := []RouteEndpoint{
		fastWorker,
		bulkWorker,
		portableWorker,
		deterministicWorker,
		balancedWorker,
	}

	t.Run("rank_latency_lowest_first", func(t *testing.T) {
		ranked := RankRouteEndpoints(cohort, ObjectiveLatency)
		if len(ranked) != len(cohort) {
			t.Fatalf("len = %d, want %d", len(ranked), len(cohort))
		}
		if ranked[0].Name != "worker-fast" {
			t.Fatalf("rank #1 = %q, want worker-fast", ranked[0].Name)
		}
	})

	t.Run("rank_throughput_highest_first", func(t *testing.T) {
		ranked := RankRouteEndpoints(cohort, ObjectiveThroughput)
		if len(ranked) != len(cohort) {
			t.Fatalf("len = %d, want %d", len(ranked), len(cohort))
		}
		if ranked[0].Name != "worker-bulk" {
			t.Fatalf("rank #1 = %q, want worker-bulk", ranked[0].Name)
		}
	})

	t.Run("rank_portability_highest_first", func(t *testing.T) {
		ranked := RankRouteEndpoints(cohort, ObjectivePortability)
		if len(ranked) != len(cohort) {
			t.Fatalf("len = %d, want %d", len(ranked), len(cohort))
		}
		if ranked[0].Name != "worker-portable" {
			t.Fatalf("rank #1 = %q, want worker-portable", ranked[0].Name)
		}
	})

	t.Run("rank_determinism_highest_first", func(t *testing.T) {
		ranked := RankRouteEndpoints(cohort, ObjectiveDeterminism)
		if len(ranked) != len(cohort) {
			t.Fatalf("len = %d, want %d", len(ranked), len(cohort))
		}
		if ranked[0].Name != "worker-deterministic" {
			t.Fatalf("rank #1 = %q, want worker-deterministic", ranked[0].Name)
		}
	})

	t.Run("rank_default_balanced", func(t *testing.T) {
		ranked := RankRouteEndpoints(cohort, ObjectiveDefault)
		if len(ranked) != len(cohort) {
			t.Fatalf("len = %d, want %d", len(ranked), len(cohort))
		}
		if ranked[0].Name != "worker-balanced" {
			t.Fatalf("rank #1 = %q, want worker-balanced", ranked[0].Name)
		}
	})

	t.Run("rank_debug_deterministic_order", func(t *testing.T) {
		ranked := RankRouteEndpoints(cohort, ObjectiveDebug)
		if len(ranked) != len(cohort) {
			t.Fatalf("len = %d, want %d", len(ranked), len(cohort))
		}
		if ranked[0].Name != "worker-balanced" {
			t.Fatalf("rank #1 = %q, want worker-balanced (highest BasePriority)", ranked[0].Name)
		}
	})

	t.Run("rank_tie_break_by_name", func(t *testing.T) {
		itemB := RouteEndpoint{Name: "bravo", LatencyMs: 10.0}
		itemA := RouteEndpoint{Name: "alpha", LatencyMs: 10.0}
		ranked := RankRouteEndpoints([]RouteEndpoint{itemB, itemA}, ObjectiveLatency)
		if len(ranked) != 2 {
			t.Fatalf("len = %d, want 2", len(ranked))
		}
		if ranked[0].Name != "alpha" || ranked[1].Name != "bravo" {
			t.Fatalf("tie break order = [%s, %s], want [alpha, bravo]", ranked[0].Name, ranked[1].Name)
		}
	})

	t.Run("rank_empty_and_single_slices", func(t *testing.T) {
		if got := RankRouteEndpoints(nil, ObjectiveDefault); got != nil {
			t.Fatalf("RankRouteEndpoints(nil) = %v, want nil", got)
		}
		empty := RankRouteEndpoints([]RouteEndpoint{}, ObjectiveDefault)
		if empty == nil || len(empty) != 0 {
			t.Fatalf("RankRouteEndpoints(empty) = %v, want empty slice", empty)
		}
		single := []RouteEndpoint{{Name: "lone", LatencyMs: 12.0}}
		rankedSingle := RankRouteEndpoints(single, ObjectiveDefault)
		if len(rankedSingle) != 1 || rankedSingle[0].Name != "lone" {
			t.Fatalf("RankRouteEndpoints(single) = %v, want [lone]", rankedSingle)
		}
	})
}
