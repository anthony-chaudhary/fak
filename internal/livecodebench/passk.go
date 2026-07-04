package livecodebench

import "fmt"

// PassAtK computes the unbiased pass@k estimator of Chen et al. (2021, "Evaluating
// Large Language Models Trained on Code"), the same estimator LiveCodeBench's
// lcb_runner uses to score generations: given n samples of which c passed, it
// returns the probability that a uniformly random size-k subset of the samples
// contains at least one passing sample, in expectation over subsets:
//
//	pass@k = 1 - C(n-c, k) / C(n, k)
//
// It is evaluated in the numerically stable product form
// 1 - prod_{i=n-c+1}^{n} (1 - k/i) so it does not overflow on large n. When
// n-c < k every size-k subset must include a passing sample, so the value is 1.
//
// Tolerance vs upstream: this is an exact evaluation of the estimator, so the
// only divergence from lcb_runner is which samples are marked correct. LCB notes
// a timing-induced <0.5pt variance in that upstream correctness labeling (a
// sandbox may time out a borderline-correct sample); the estimator itself carries
// no additional error.
func PassAtK(n, c, k int) (float64, error) {
	if n < 1 {
		return 0, fmt.Errorf("livecodebench pass@k: n must be >= 1, got %d", n)
	}
	if k < 1 {
		return 0, fmt.Errorf("livecodebench pass@k: k must be >= 1, got %d", k)
	}
	if k > n {
		return 0, fmt.Errorf("livecodebench pass@k: k must be <= n (%d), got %d", n, k)
	}
	if c < 0 || c > n {
		return 0, fmt.Errorf("livecodebench pass@k: c must be within [0, n=%d], got %d", n, c)
	}
	if n-c < k {
		return 1.0, nil
	}
	prod := 1.0
	for i := n - c + 1; i <= n; i++ {
		prod *= 1.0 - float64(k)/float64(i)
	}
	return 1.0 - prod, nil
}

// SampleTally is one problem's per-problem sample count: Samples generations of
// which Correct passed the grader. QuestionID is carried only for error context.
type SampleTally struct {
	QuestionID string
	Samples    int // n: total generations for this problem
	Correct    int // c: generations that passed grading
}

// MeanPassAtK is the benchmark-level score: the mean of PassAtK over each
// problem's (Samples, Correct) tally, matching how LiveCodeBench aggregates a
// scenario's pass@k. It errors if the tally set is empty or any tally is invalid.
func MeanPassAtK(tallies []SampleTally, k int) (float64, error) {
	if len(tallies) == 0 {
		return 0, fmt.Errorf("livecodebench pass@k: at least one problem tally is required")
	}
	sum := 0.0
	for _, t := range tallies {
		p, err := PassAtK(t.Samples, t.Correct, k)
		if err != nil {
			return 0, fmt.Errorf("livecodebench pass@k: problem %q: %w", t.QuestionID, err)
		}
		sum += p
	}
	return sum / float64(len(tallies)), nil
}
