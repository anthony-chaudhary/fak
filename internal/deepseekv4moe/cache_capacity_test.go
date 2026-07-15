package deepseekv4moe

import (
	"errors"
	"math"
	"testing"
)

func TestPlanExpertCachePinnedV4ProBudget(t *testing.T) {
	const (
		totalHBM       = int64(480 * 1024 * 1024 * 1024)
		nonRouted      = int64(42_650_568_824)
		runtimeReserve = int64(64 * 1024 * 1024 * 1024)
		groupBytes     = int64(35_094_528)
		selectedGroups = int64(61 * 6)
		selectedStatic = int64(55_495_166_072)
	)
	if got := nonRouted + selectedGroups*groupBytes; got != selectedStatic {
		t.Fatalf("pinned selected static bytes = %d, want %d", got, selectedStatic)
	}

	plan, err := PlanExpertCache(totalHBM, nonRouted, runtimeReserve, groupBytes, 61, 384)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ResidentRoutedGroups != 11_512 || plan.WholeExpertsPerLayer != 188 {
		t.Fatalf("capacity = %d groups / %d experts per layer, want 11512 / 188", plan.ResidentRoutedGroups, plan.WholeExpertsPerLayer)
	}
	if plan.UsedBytes > totalHBM || plan.HeadroomBytes != totalHBM-plan.UsedBytes {
		t.Fatalf("invalid accounting: %+v", plan)
	}
	if selectedGroups > plan.ResidentRoutedGroups {
		t.Fatalf("pinned top-6 working set (%d groups) does not fit plan %+v", selectedGroups, plan)
	}
}

func TestPlanExpertCacheBoundaries(t *testing.T) {
	t.Run("exact fit", func(t *testing.T) {
		plan, err := PlanExpertCache(120, 10, 10, 20, 1, 5)
		if err != nil {
			t.Fatal(err)
		}
		if plan.ResidentRoutedGroups != 5 || plan.UsedBytes != 120 || plan.HeadroomBytes != 0 {
			t.Fatalf("exact-fit plan = %+v", plan)
		}
	})
	t.Run("one byte short", func(t *testing.T) {
		plan, err := PlanExpertCache(119, 10, 10, 20, 1, 5)
		if err != nil {
			t.Fatal(err)
		}
		if plan.ResidentRoutedGroups != 4 || plan.HeadroomBytes != 19 {
			t.Fatalf("one-byte-short plan = %+v", plan)
		}
	})
	t.Run("model cap", func(t *testing.T) {
		plan, err := PlanExpertCache(1_000_000, 1, 1, 1, 61, 384)
		if err != nil {
			t.Fatal(err)
		}
		if plan.ResidentRoutedGroups != 61*384 || plan.WholeExpertsPerLayer != 384 {
			t.Fatalf("capped plan = %+v", plan)
		}
	})
}

func TestPlanExpertCacheFailsClosed(t *testing.T) {
	tests := []struct {
		name                                          string
		total, fixed, reserve, group, layers, experts int64
		want                                          error
	}{
		{"zero total", 0, 0, 0, 1, 1, 1, ErrInvalidCacheBudget},
		{"negative fixed", 10, -1, 0, 1, 1, 1, ErrInvalidCacheBudget},
		{"zero group", 10, 0, 0, 0, 1, 1, ErrInvalidCacheBudget},
		{"fixed exceeds total", 10, 11, 0, 1, 1, 1, ErrInsufficientCacheBudget},
		{"no whole group", 10, 5, 4, 2, 1, 1, ErrInsufficientCacheBudget},
		{"reserve sum overflow", math.MaxInt64, math.MaxInt64, 1, 1, 1, 1, ErrCacheBudgetOverflow},
		{"shape overflow", math.MaxInt64, 0, 0, 1, math.MaxInt64, 2, ErrCacheBudgetOverflow},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := PlanExpertCache(tc.total, tc.fixed, tc.reserve, tc.group, tc.layers, tc.experts)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}
}
