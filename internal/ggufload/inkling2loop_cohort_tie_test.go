package ggufload

import (
	"fmt"
	"strings"
	"testing"
)

// inkling2CohortDivergence compares one cohort row's tokens got with the scalar
// reference want, step by step. It returns the first divergent step (-1 when
// none) and whether that step is an admissible near tie: the scalar top-1 leads
// its runner-up by less than inkling2OracleNearTieNats and the cohort picked that
// runner-up. The rows legitimately diverge after a tie, so callers stop there.
func inkling2CohortDivergence(want, runnerUp []int, margin []float64, got []int) (at int, nearTie bool, why string) {
	if len(got) != len(want) || len(runnerUp) != len(want) || len(margin) != len(want) {
		return min(len(got), len(want)), false, fmt.Sprintf("cohort decoded %d tokens, scalar %d, runner-up %d, margin %d",
			len(got), len(want), len(runnerUp), len(margin))
	}
	for i := range got {
		if got[i] == want[i] {
			continue
		}
		why = fmt.Sprintf("token %d = %d, scalar = %d (runner-up %d, margin %.4f nats, near-tie bound %.2f)",
			i, got[i], want[i], runnerUp[i], margin[i], inkling2OracleNearTieNats)
		return i, margin[i] < inkling2OracleNearTieNats && got[i] == runnerUp[i], why
	}
	return -1, false, ""
}

func TestInkling2CohortDivergenceExactMatchHasNoDivergence(t *testing.T) {
	at, nearTie, why := inkling2CohortDivergence([]int{5, 6, 7}, []int{9, 9, 9}, []float64{3, 3, 3}, []int{5, 6, 7})
	if at != -1 || nearTie || why != "" {
		t.Fatalf("at %d nearTie %v why %q, want no divergence", at, nearTie, why)
	}
}

func TestInkling2CohortDivergenceRunnerUpFlipUnderBoundIsNearTie(t *testing.T) {
	at, nearTie, why := inkling2CohortDivergence([]int{5, 6, 7}, []int{9, 8, 9}, []float64{3, 0.02, 3}, []int{5, 8, 0})
	if at != 1 || !nearTie || why == "" {
		t.Fatalf("at %d nearTie %v why %q, want step 1 near tie", at, nearTie, why)
	}
}

func TestInkling2CohortDivergenceWideFlipIsNotNearTie(t *testing.T) {
	at, nearTie, why := inkling2CohortDivergence([]int{5, 6, 7}, []int{9, 8, 9}, []float64{3, 3, 3}, []int{5, 8, 7})
	if at != 1 || nearTie || why == "" {
		t.Fatalf("at %d nearTie %v why %q, want a non-tie divergence at step 1", at, nearTie, why)
	}
}

func TestInkling2CohortDivergenceMarginAtBoundIsNotNearTie(t *testing.T) {
	margin := inkling2OracleNearTieNats + 1e-6
	at, nearTie, why := inkling2CohortDivergence([]int{5, 6}, []int{9, 8}, []float64{margin, margin}, []int{5, 8})
	if at != 1 || nearTie || why == "" {
		t.Fatalf("at %d nearTie %v why %q, want a non-tie divergence just over the bound", at, nearTie, why)
	}
}

func TestInkling2CohortDivergenceNonRunnerUpIsNotNearTie(t *testing.T) {
	at, nearTie, why := inkling2CohortDivergence([]int{5, 6}, []int{9, 8}, []float64{0.02, 0.02}, []int{5, 7})
	if at != 1 || nearTie || why == "" {
		t.Fatalf("at %d nearTie %v why %q, want a non-tie divergence at step 1", at, nearTie, why)
	}
}

func TestInkling2CohortDivergenceLengthMismatchIsNotNearTie(t *testing.T) {
	at, nearTie, why := inkling2CohortDivergence([]int{5, 6, 7}, []int{9, 8, 9}, []float64{0.02, 0.02, 0.02}, []int{5, 8})
	if nearTie || !strings.Contains(why, "cohort decoded 2 tokens, scalar 3") {
		t.Fatalf("at %d nearTie %v why %q, want a length mismatch that is not a near tie", at, nearTie, why)
	}
}
