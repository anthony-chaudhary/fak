package model

import (
	"math"
	"testing"
)

const loglikTol = 1e-5

// TestLogSoftmaxAt pins the numerically-stable log-softmax used per continuation
// token, including the out-of-range -Inf guard.
func TestLogSoftmaxAt(t *testing.T) {
	row := []float32{1, 2, 3}
	// max=3, sumExp = e^-2 + e^-1 + 1 = 1.5032147; ln = 0.4076059.
	const lnSum = 0.4076059
	cases := []struct {
		tok  int
		want float64
	}{
		{0, (1 - 3) - lnSum},
		{1, (2 - 3) - lnSum},
		{2, (3 - 3) - lnSum},
	}
	for _, tc := range cases {
		if got := logSoftmaxAt(row, tc.tok); math.Abs(got-tc.want) > loglikTol {
			t.Errorf("logSoftmaxAt(%v, %d) = %v, want %v", row, tc.tok, got, tc.want)
		}
	}
	if got := logSoftmaxAt(row, 5); !math.IsInf(got, -1) {
		t.Errorf("logSoftmaxAt out-of-range = %v, want -Inf", got)
	}
	if got := logSoftmaxAt(row, -1); !math.IsInf(got, -1) {
		t.Errorf("logSoftmaxAt(-1) = %v, want -Inf", got)
	}
}

// TestScoreContinuation pins the teacher-forcing index alignment (a continuation
// token at position promptLen+i is predicted by logits[promptLen+i-1]), the summed
// log-probability, and the greedy-exact detection — all over hand-crafted logits,
// no model.
func TestScoreContinuation(t *testing.T) {
	const ln3 = 1.0986123 // ln(3)

	// A: single continuation token, greedy-exact. row0 strongly favors token 1.
	sum, n, greedy := scoreContinuation([][]float32{{0, 10, 0}, {0, 0, 0}}, 1, []int{1})
	if n != 1 || !greedy {
		t.Fatalf("A: contLen=%d greedy=%v, want 1,true", n, greedy)
	}
	if math.Abs(sum-(-9.079e-5)) > 1e-4 {
		t.Fatalf("A: sumLogprob=%v, want ~-9.079e-5", sum)
	}

	// B: two continuation tokens over uniform rows; first token (2) is NOT the
	// greedy pick (argmax of a uniform row is index 0), so greedy-exact is false and
	// the summed logprob is 2*ln(1/3).
	sum, n, greedy = scoreContinuation([][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, 1, []int{2, 0})
	if n != 2 || greedy {
		t.Fatalf("B: contLen=%d greedy=%v, want 2,false", n, greedy)
	}
	if math.Abs(sum-(-2*ln3)) > loglikTol {
		t.Fatalf("B: sumLogprob=%v, want %v", sum, -2*ln3)
	}

	// C: promptLen < 1 has no context to predict the first token -> zero score.
	if sum, n, greedy = scoreContinuation([][]float32{{0, 0, 0}}, 0, []int{1}); sum != 0 || n != 1 || greedy {
		t.Fatalf("C: got (%v,%d,%v), want (0,1,false)", sum, n, greedy)
	}

	// D: continuation runs past the available logits -> zero score, not a panic.
	if sum, n, greedy = scoreContinuation([][]float32{{0, 0, 0}}, 1, []int{1, 1}); sum != 0 || n != 2 || greedy {
		t.Fatalf("D: got (%v,%d,%v), want (0,2,false)", sum, n, greedy)
	}
}

// TestAccNorm pins that acc_norm ranks by LENGTH-NORMALIZED log-probability, so a
// longer choice with a lower summed logprob can still win — the whole point of
// acc_norm over raw summed loglikelihood.
func TestAccNorm(t *testing.T) {
	choices := []ChoiceScore{
		{SumLogprob: -2.0, ContLen: 2}, // norm -1.0  (winner under acc_norm)
		{SumLogprob: -1.5, ContLen: 1}, // norm -1.5  (winner under RAW sum)
		{SumLogprob: -3.0, ContLen: 2}, // norm -1.5
	}
	pred, norm := AccNorm(choices)
	if pred != 0 {
		t.Fatalf("AccNorm predicted = %d, want 0 (length-normalized winner)", pred)
	}
	want := []float64{-1.0, -1.5, -1.5}
	for i := range want {
		if math.Abs(norm[i]-want[i]) > loglikTol {
			t.Fatalf("norm[%d] = %v, want %v", i, norm[i], want[i])
		}
	}
	// Sanity: the RAW-sum argmax would be choice 1 (-1.5 > -2.0 > -3.0); acc_norm
	// flips the winner to choice 0. This is the behavioral contrast the scorer adds.
	if choices[1].SumLogprob <= choices[0].SumLogprob {
		t.Fatal("test setup: choice 1 should have the higher RAW sum than choice 0")
	}

	if pred, _ := AccNorm(nil); pred != -1 {
		t.Fatalf("AccNorm(nil) = %d, want -1", pred)
	}
}

// TestChoiceScoreNormLogprob pins the zero-length guard (no divide-by-zero).
func TestChoiceScoreNormLogprob(t *testing.T) {
	if got := (ChoiceScore{SumLogprob: -5, ContLen: 0}).NormLogprob(); got != 0 {
		t.Fatalf("NormLogprob(ContLen 0) = %v, want 0", got)
	}
	if got := (ChoiceScore{SumLogprob: -4, ContLen: 2}).NormLogprob(); math.Abs(got-(-2.0)) > loglikTol {
		t.Fatalf("NormLogprob = %v, want -2.0", got)
	}
}
