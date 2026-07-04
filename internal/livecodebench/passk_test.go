package livecodebench

import (
	"math"
	"testing"
)

// TestPassAtKReferenceFormula is the #2101 parity witness: PassAtK must match the
// unbiased pass@k estimator 1 - C(n-c, k)/C(n, k) (Chen et al. 2021), the same
// estimator lcb_runner uses. Each row is hand-computed from the closed form.
func TestPassAtKReferenceFormula(t *testing.T) {
	cases := []struct {
		name    string
		n, c, k int
		want    float64
	}{
		// c = 0 correct: the estimator is exactly 0 regardless of k.
		{"none-correct-k1", 5, 0, 1, 0.0},
		{"none-correct-k5", 5, 0, 5, 0.0},
		// Every sample correct: any k-subset passes -> 1.
		{"all-correct-k1", 5, 5, 1, 1.0},
		{"all-correct-k5", 5, 5, 5, 1.0},
		// 1 of 5 correct, k=1: pass@1 = c/n = 1/5.
		{"one-of-five-k1", 5, 1, 1, 0.2},
		// n-c < k -> a size-k subset must include a correct sample -> 1.
		{"one-of-five-k5", 5, 1, 5, 1.0},
		// 1 of 10, k=5: 1 - C(9,5)/C(10,5) = 1 - 126/252 = 1/2.
		{"one-of-ten-k5", 10, 1, 5, 0.5},
		// 2 of 6, k=3: 1 - C(4,3)/C(6,3) = 1 - 4/20 = 0.8.
		{"two-of-six-k3", 6, 2, 3, 0.8},
		// 3 of 10, k=2: 1 - C(7,2)/C(10,2) = 1 - 21/45 = 8/15.
		{"three-of-ten-k2", 10, 3, 2, 8.0 / 15.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := PassAtK(tc.n, tc.c, tc.k)
			if err != nil {
				t.Fatalf("PassAtK(%d,%d,%d) error: %v", tc.n, tc.c, tc.k, err)
			}
			if math.Abs(got-tc.want) > 1e-12 {
				t.Fatalf("PassAtK(%d,%d,%d) = %v, want %v", tc.n, tc.c, tc.k, got, tc.want)
			}
			// Cross-check against the direct combinatorial form for coverage.
			if ref := passAtKCombinatorial(tc.n, tc.c, tc.k); math.Abs(got-ref) > 1e-12 {
				t.Fatalf("PassAtK(%d,%d,%d) = %v disagrees with C(n-c,k)/C(n,k) form %v", tc.n, tc.c, tc.k, got, ref)
			}
		})
	}
}

// passAtKCombinatorial is the slow, direct reference the fast product form must match.
func passAtKCombinatorial(n, c, k int) float64 {
	if n-c < k {
		return 1.0
	}
	return 1.0 - float64(choose(n-c, k))/float64(choose(n, k))
}

func choose(n, k int) int64 {
	if k < 0 || k > n {
		return 0
	}
	if k > n-k {
		k = n - k
	}
	var r int64 = 1
	for i := 0; i < k; i++ {
		r = r * int64(n-i) / int64(i+1)
	}
	return r
}

func TestPassAtKRejectsInvalidInputs(t *testing.T) {
	bad := []struct {
		name    string
		n, c, k int
	}{
		{"zero-samples", 0, 0, 1},
		{"k-below-one", 5, 1, 0},
		{"k-exceeds-n", 3, 1, 4},
		{"correct-exceeds-n", 3, 4, 1},
		{"negative-correct", 3, -1, 1},
	}
	for _, tc := range bad {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := PassAtK(tc.n, tc.c, tc.k); err == nil {
				t.Fatalf("PassAtK(%d,%d,%d) = nil error, want a validation error", tc.n, tc.c, tc.k)
			}
		})
	}
}

// TestMeanPassAtKHandComputed is the #2101 end-to-end witness: mean pass@1/pass@5
// over a small per-problem tally set must equal the hand-computed averages.
func TestMeanPassAtKHandComputed(t *testing.T) {
	tallies := []SampleTally{
		{QuestionID: "q1", Samples: 5, Correct: 1}, // pass@1=0.2  pass@5=1.0
		{QuestionID: "q2", Samples: 5, Correct: 5}, // pass@1=1.0  pass@5=1.0
		{QuestionID: "q3", Samples: 5, Correct: 0}, // pass@1=0.0  pass@5=0.0
	}
	// mean pass@1 = (0.2 + 1.0 + 0.0)/3 = 0.4
	got1, err := MeanPassAtK(tallies, 1)
	if err != nil {
		t.Fatalf("MeanPassAtK k=1: %v", err)
	}
	if math.Abs(got1-0.4) > 1e-12 {
		t.Fatalf("MeanPassAtK k=1 = %v, want 0.4", got1)
	}
	// mean pass@5 = (1.0 + 1.0 + 0.0)/3 = 2/3
	got5, err := MeanPassAtK(tallies, 5)
	if err != nil {
		t.Fatalf("MeanPassAtK k=5: %v", err)
	}
	if math.Abs(got5-2.0/3.0) > 1e-12 {
		t.Fatalf("MeanPassAtK k=5 = %v, want %v", got5, 2.0/3.0)
	}
}

func TestMeanPassAtKEmptyIsError(t *testing.T) {
	if _, err := MeanPassAtK(nil, 1); err == nil {
		t.Fatalf("MeanPassAtK(nil,1) = nil error, want an error for an empty tally set")
	}
}
