// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestParetoFrontier(t *testing.T) {
	sweep := []ConfigPoint{
		{Config: "fast-low", LatencyMS: 100, Throughput: 100},
		{Config: "slow-high", LatencyMS: 200, Throughput: 300},
		{Config: "mid-worst", LatencyMS: 150, Throughput: 80},
		{Config: "dup", LatencyMS: 100, Throughput: 100},
		{Config: "tiny", LatencyMS: 50, Throughput: 50},
	}
	want := []ConfigPoint{
		{Config: "tiny", LatencyMS: 50, Throughput: 50},
		{Config: "dup", LatencyMS: 100, Throughput: 100},
		{Config: "fast-low", LatencyMS: 100, Throughput: 100},
		{Config: "slow-high", LatencyMS: 200, Throughput: 300},
	}

	t.Run("non_dominated_set", func(t *testing.T) {
		got := ParetoFrontier(sweep)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("frontier = %+v, want %+v", got, want)
		}
	})

	t.Run("order_independent", func(t *testing.T) {
		shuffled := []ConfigPoint{sweep[3], sweep[0], sweep[4], sweep[2], sweep[1]}
		if got := ParetoFrontier(shuffled); !reflect.DeepEqual(got, want) {
			t.Fatalf("shuffled frontier = %+v, want %+v", got, want)
		}
	})

	t.Run("equal_points_both_retained", func(t *testing.T) {
		pts := []ConfigPoint{
			{Config: "a", LatencyMS: 100, Throughput: 100},
			{Config: "b", LatencyMS: 100, Throughput: 100},
		}
		if got := ParetoFrontier(pts); !reflect.DeepEqual(got, pts) {
			t.Fatalf("equal points = %+v, want both retained %+v", got, pts)
		}
	})

	t.Run("same_latency_only_highest_throughput_survives", func(t *testing.T) {
		pts := []ConfigPoint{
			{Config: "low", LatencyMS: 100, Throughput: 50},
			{Config: "high", LatencyMS: 100, Throughput: 80},
		}
		want := []ConfigPoint{{Config: "high", LatencyMS: 100, Throughput: 80}}
		if got := ParetoFrontier(pts); !reflect.DeepEqual(got, want) {
			t.Fatalf("frontier = %+v, want %+v", got, want)
		}
	})

	t.Run("invalid_points_dropped", func(t *testing.T) {
		pts := []ConfigPoint{
			{Config: "nan-latency", LatencyMS: math.NaN(), Throughput: 100},
			{Config: "zero-latency", LatencyMS: 0, Throughput: 100},
			{Config: "negative-throughput", LatencyMS: 100, Throughput: -1},
			{Config: "inf-throughput", LatencyMS: 100, Throughput: math.Inf(+1)},
			{Config: "valid", LatencyMS: 100, Throughput: 100},
		}
		want := []ConfigPoint{{Config: "valid", LatencyMS: 100, Throughput: 100}}
		if got := ParetoFrontier(pts); !reflect.DeepEqual(got, want) {
			t.Fatalf("frontier = %+v, want %+v", got, want)
		}
	})

	t.Run("all_invalid_or_empty", func(t *testing.T) {
		if got := ParetoFrontier(nil); len(got) != 0 {
			t.Fatalf("empty input = %+v, want empty", got)
		}
		invalid := []ConfigPoint{{Config: "x", LatencyMS: math.NaN(), Throughput: 1}}
		if got := ParetoFrontier(invalid); len(got) != 0 {
			t.Fatalf("all-invalid input = %+v, want empty", got)
		}
	})
}

func TestBestConfigUnderConstraints(t *testing.T) {
	f64 := func(v float64) *float64 { return &v }
	sweep := []ConfigPoint{
		{Config: "a", LatencyMS: 100, Throughput: 100},
		{Config: "b", LatencyMS: 150, Throughput: 200},
		{Config: "c", LatencyMS: 80, Throughput: 90},
		{Config: "d", LatencyMS: 300, Throughput: 500},
	}

	t.Run("unconstrained_picks_max_throughput", func(t *testing.T) {
		got, err := BestConfigUnderConstraints(sweep, nil, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.Config != "d" || got.Throughput != 500 {
			t.Fatalf("best = %+v, want d@500", got)
		}
	})

	t.Run("max_latency_bound", func(t *testing.T) {
		got, err := BestConfigUnderConstraints(sweep, f64(120), nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.Config != "a" {
			t.Fatalf("best = %+v, want a (only feasible max-throughput under 120ms)", got)
		}
	})

	t.Run("both_bounds", func(t *testing.T) {
		got, err := BestConfigUnderConstraints(sweep, f64(150), f64(95))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.Config != "b" {
			t.Fatalf("best = %+v, want b (200 tps within 150ms)", got)
		}
	})

	t.Run("bounds_are_inclusive", func(t *testing.T) {
		edge := []ConfigPoint{{Config: "edge", LatencyMS: 100, Throughput: 10}}
		got, err := BestConfigUnderConstraints(edge, f64(100), f64(10))
		if err != nil {
			t.Fatalf("boundary point must be feasible: %v", err)
		}
		if got.Config != "edge" {
			t.Fatalf("best = %+v, want edge", got)
		}
	})

	t.Run("throughput_tie_breaks_lower_latency_then_config", func(t *testing.T) {
		pts := []ConfigPoint{
			{Config: "z", LatencyMS: 100, Throughput: 200},
			{Config: "m", LatencyMS: 90, Throughput: 200},
			{Config: "q", LatencyMS: 120, Throughput: 200},
		}
		got, err := BestConfigUnderConstraints(pts, nil, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.Config != "m" {
			t.Fatalf("best = %+v, want m (lowest latency among ties)", got)
		}
		fullTie := []ConfigPoint{
			{Config: "z", LatencyMS: 100, Throughput: 200},
			{Config: "a", LatencyMS: 100, Throughput: 200},
		}
		got, err = BestConfigUnderConstraints(fullTie, nil, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.Config != "a" {
			t.Fatalf("full tie = %+v, want a (lexicographic)", got)
		}
	})

	t.Run("infeasible_returns_sentinel", func(t *testing.T) {
		if _, err := BestConfigUnderConstraints(sweep, f64(10), nil); !errors.Is(err, ErrNoFeasibleConfig) {
			t.Fatalf("err = %v, want ErrNoFeasibleConfig", err)
		}
		invalid := []ConfigPoint{{Config: "x", LatencyMS: math.NaN(), Throughput: 1}}
		if _, err := BestConfigUnderConstraints(invalid, nil, nil); !errors.Is(err, ErrNoFeasibleConfig) {
			t.Fatalf("err = %v, want ErrNoFeasibleConfig for all-invalid input", err)
		}
	})
}
