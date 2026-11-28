package main

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestSplitBudget(t *testing.T) {
	cases := []struct {
		name     string
		total    int
		weights  []int
		minimums []int
		want     []int
		err      error
	}{
		{name: "even split", total: 9, weights: []int{1, 1, 1}, want: []int{3, 3, 3}},
		{name: "tie goes to lower index", total: 10, weights: []int{1, 1, 1}, want: []int{4, 3, 3}},
		{name: "largest remainder wins", total: 10, weights: []int{1, 2, 3}, want: []int{2, 3, 5}},
		{name: "zero total", total: 0, weights: []int{1, 2}, want: []int{0, 0}},
		{name: "zero total zero weights", total: 0, weights: []int{0, 0}, want: []int{0, 0}},
		{name: "zero weight bucket gets nothing", total: 5, weights: []int{0, 1}, want: []int{0, 5}},
		{name: "no buckets", total: 0, weights: nil, want: []int{}},
		// Upstream worker-pod CPU shape: manager + 2 workers + 1 record processor, 100m floor each.
		{name: "upstream cpu weights with floor", total: 4000, weights: []int{100, 131, 131, 389}, minimums: []int{100, 100, 100, 100},
			want: []int{579, 728, 728, 1965}},
		{name: "minimums exactly cover total", total: 6, weights: []int{5, 1}, minimums: []int{3, 3}, want: []int{3, 3}},
		{name: "minimums without weights", total: 6, weights: []int{0, 0}, minimums: []int{3, 3}, want: []int{3, 3}},
		{name: "per bucket minimums differ", total: 10, weights: []int{1, 1}, minimums: []int{6, 0}, want: []int{8, 2}},
		{name: "no overflow on large products", total: math.MaxInt64, weights: []int{math.MaxInt64, math.MaxInt64}, want: []int{math.MaxInt64/2 + 1, math.MaxInt64 / 2}},
		{name: "weight sum overflow", total: 5, weights: []int{math.MaxInt64, math.MaxInt64, math.MaxInt64}, err: errBudgetWeightSum},
		{name: "weight sum overflow wraps to zero", total: 5, weights: []int{math.MaxInt64, math.MaxInt64, 2}, err: errBudgetWeightSum},
		{name: "negative total", total: -1, weights: []int{1}, err: errBudgetNegative},
		{name: "negative weight", total: 5, weights: []int{1, -1}, err: errBudgetNegativeInput},
		{name: "negative minimum", total: 5, weights: []int{1, 1}, minimums: []int{-1, 0}, err: errBudgetNegativeInput},
		{name: "minimums exceed total", total: 5, weights: []int{1, 1}, minimums: []int{3, 3}, err: errBudgetMinimums},
		{name: "zero weights with budget", total: 5, weights: []int{0, 0}, err: errBudgetZeroWeights},
		{name: "budget with no buckets", total: 5, weights: nil, err: errBudgetZeroWeights},
		{name: "minimums shape mismatch", total: 5, weights: []int{1, 1}, minimums: []int{1}, err: errBudgetShape},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := SplitBudget(tc.total, tc.weights, tc.minimums)
			if tc.err != nil {
				if !errors.Is(err, tc.err) {
					t.Fatalf("err = %v, want %v", err, tc.err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			sum := 0
			for i, v := range got {
				sum += v
				if tc.minimums != nil && v < tc.minimums[i] {
					t.Fatalf("bucket %d = %d below minimum %d", i, v, tc.minimums[i])
				}
			}
			if sum != tc.total {
				t.Fatalf("sum = %d, want exact total %d", sum, tc.total)
			}
		})
	}

	t.Run("exact sum sweep", func(t *testing.T) {
		weights := []int{7, 3, 11, 1, 0}
		mins := []int{1, 0, 2, 0, 1}
		for total := 4; total < 500; total++ {
			got, err := SplitBudget(total, weights, mins)
			if err != nil {
				t.Fatalf("total %d: %v", total, err)
			}
			sum := 0
			for _, v := range got {
				sum += v
			}
			if sum != total || got[4] != 1 {
				t.Fatalf("total %d: got %v", total, got)
			}
		}
	})
}

func TestCPUShares(t *testing.T) {
	specs := []CPUShareSpec{
		{Name: "manager", Pinned: 100, Weight: 100},
		{Name: "worker", Pinned: 100, Weight: 300},
		{Name: "pinned-only", Pinned: 200},
	}
	got, err := CPUShares(1000, specs)
	if err != nil {
		t.Fatal(err)
	}
	if got[2] != 200 || got[0]+got[1]+got[2] != 1000 || got[0] < 100 || got[1] <= got[0] {
		t.Fatalf("shares %v break carve-out/weight/sum contract", got)
	}
	if want := []int{250, 550, 200}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	only, err := CPUShares(1000, []CPUShareSpec{{Name: "a", Pinned: 10}, {Name: "b", Pinned: 20}})
	if err != nil || !reflect.DeepEqual(only, []int{10, 20}) {
		t.Fatalf("no-weight case = %v, %v", only, err)
	}
	_, err = CPUShares(50, specs)
	if !errors.Is(err, errCPUSharesExhausted) {
		t.Fatalf("exhaustion err = %v", err)
	}
	for _, name := range []string{"manager=100", "worker=100", "pinned-only=200"} {
		if !strings.Contains(err.Error(), name) {
			t.Fatalf("error %q does not name term %s", err, name)
		}
	}
}
