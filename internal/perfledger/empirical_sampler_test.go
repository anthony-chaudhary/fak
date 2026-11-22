package perfledger

import (
	"errors"
	"math"
	"slices"
	"testing"
)

func esampMust(t *testing.T, data []float64) *EmpiricalSampler {
	t.Helper()
	s, err := NewEmpiricalSampler(data)
	if err != nil {
		t.Fatalf("NewEmpiricalSampler(%v): %v", data, err)
	}
	return s
}

func esampClose(a, b float64) bool { return math.Abs(a-b) <= 1e-12 }

func TestEmpiricalSamplerStats(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		data []float64
		want EmpiricalSamplerStats
	}{
		// Upstream goldens: np.unique learns values/probs, mean is
		// probability-weighted, median is np.median of the distinct values.
		{"upstream_counts", []float64{3, 1, 2, 3, 1, 3}, EmpiricalSamplerStats{Min: 1, Max: 3, Mean: 13.0 / 6, Median: 2, NumUnique: 3}},
		{"upstream_distinct_median", []float64{1, 1, 1, 1, 10}, EmpiricalSamplerStats{Min: 1, Max: 10, Mean: 2.8, Median: 5.5, NumUnique: 2}},
		{"empty_is_degenerate_zero", nil, EmpiricalSamplerStats{Min: 0, Max: 0, Mean: 0, Median: 0, NumUnique: 1}},
		{"single", []float64{42}, EmpiricalSamplerStats{Min: 42, Max: 42, Mean: 42, Median: 42, NumUnique: 1}},
		{"huge_repeated_no_overflow", []float64{math.MaxFloat64, math.MaxFloat64}, EmpiricalSamplerStats{Min: math.MaxFloat64, Max: math.MaxFloat64, Mean: math.MaxFloat64, Median: math.MaxFloat64, NumUnique: 1}},
		{"negative_and_fractional", []float64{-1.5, 0.5, 0.5, 2}, EmpiricalSamplerStats{Min: -1.5, Max: 2, Mean: 0.375, Median: 0.5, NumUnique: 3}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := esampMust(t, tc.data).Stats()
			if got.Min != tc.want.Min || got.Max != tc.want.Max || got.NumUnique != tc.want.NumUnique ||
				!esampClose(got.Mean, tc.want.Mean) || !esampClose(got.Median, tc.want.Median) {
				t.Fatalf("Stats() = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestEmpiricalSamplerRejectsNonFinite(t *testing.T) {
	t.Parallel()
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := NewEmpiricalSampler([]float64{1, bad}); !errors.Is(err, ErrEmpiricalSamplerNonFinite) {
			t.Fatalf("NewEmpiricalSampler with %v: err = %v, want ErrEmpiricalSamplerNonFinite", bad, err)
		}
	}
}

func TestEmpiricalSamplerDoesNotMutateInput(t *testing.T) {
	t.Parallel()
	data := []float64{3, 1, 2}
	esampMust(t, data)
	if !slices.Equal(data, []float64{3, 1, 2}) {
		t.Fatalf("input mutated: %v", data)
	}
}

func TestEmpiricalSamplerDeterministic(t *testing.T) {
	t.Parallel()
	s := esampMust(t, []float64{10, 20, 20, 30, 30, 30})
	a := s.SampleBatch(NewEmpiricalSamplerRand(7), 64)
	b := s.SampleBatch(NewEmpiricalSamplerRand(7), 64)
	if !slices.Equal(a, b) {
		t.Fatalf("same seed diverged:\n%v\n%v", a, b)
	}
	c := s.SampleBatch(NewEmpiricalSamplerRand(8), 64)
	if slices.Equal(a, c) {
		t.Fatalf("different seeds produced identical 64-draw sequences")
	}
	// Pinned sequence guards against silent changes to the draw algorithm.
	want := esampGoldenSeed7
	if got := s.SampleBatch(NewEmpiricalSamplerRand(7), len(want)); !slices.Equal(got, want) {
		t.Fatalf("seed 7 sequence = %v, want %v", got, want)
	}
}

var esampGoldenSeed7 = []float64{20, 10, 30, 20, 30, 30, 20, 20, 30, 20, 30, 10}

func TestEmpiricalSamplerBatchSizes(t *testing.T) {
	t.Parallel()
	s := esampMust(t, []float64{1, 2})
	for _, tc := range []struct {
		size, want int
	}{{-3, 0}, {0, 0}, {1, 1}, {17, 17}} {
		got := s.SampleBatch(NewEmpiricalSamplerRand(1), tc.size)
		if got == nil || len(got) != tc.want {
			t.Fatalf("SampleBatch(%d) len = %d (nil=%v), want %d", tc.size, len(got), got == nil, tc.want)
		}
	}
}

func TestEmpiricalSamplerFrequencies(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		data []float64
		want map[float64]float64
	}{
		{"single_value", []float64{5, 5, 5}, map[float64]float64{5: 1}},
		{"empty_returns_zero", nil, map[float64]float64{0: 1}},
		{"one_two_three", []float64{1, 2, 2, 3, 3, 3}, map[float64]float64{1: 1.0 / 6, 2: 2.0 / 6, 3: 3.0 / 6}},
		{"skewed", []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 100}, map[float64]float64{0: 0.9, 100: 0.1}},
	}
	const draws = 60000
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			s := esampMust(t, tc.data)
			counts := map[float64]int{}
			for _, v := range s.SampleBatch(NewEmpiricalSamplerRand(12345), draws) {
				if _, ok := tc.want[v]; !ok {
					t.Fatalf("drew unobserved value %v", v)
				}
				counts[v]++
			}
			for v, p := range tc.want {
				if got := float64(counts[v]) / draws; math.Abs(got-p) > 0.01 {
					t.Fatalf("freq(%v) = %.4f, want %.4f +/- 0.01", v, got, p)
				}
			}
		})
	}
}
