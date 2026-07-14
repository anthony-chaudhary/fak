package model

import (
	"math"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// loglik.go is the #4363 behavioral quant-fidelity scorer: grade a low-bit quant
// by whether it still picks the right multiple-choice answer, not only by
// numerical parity vs f32 (that is cmd/q8bench). Each choice is scored with a
// SINGLE teacher-forced prefill — the sum of the continuation tokens' log-
// probabilities, length-normalized (acc_norm) — and argmax across choices is the
// predicted answer. No generation loop: exactly one Forward per option. Clean-room
// borrow of colibri's eval_glm.py acc_norm scorer (epic #4352, verdict inspire;
// Apache-2.0 <-> Apache-2.0, no bytes vendored).
//
// This lands the scoring PRIMITIVES over the existing Forward logits. Grading a
// real Q8/Q4 checkpoint against f32 over a committed MMLU/HellaSwag/ARC fixture is
// the follow-on (it needs loaded quant weights); the pure folds here are the part
// that is deterministic and model-free.

// ChoiceScore is one multiple-choice option's teacher-forced score: the summed
// continuation log-probability and its token length, plus whether greedy decoding
// would have reproduced the continuation exactly.
type ChoiceScore struct {
	SumLogprob  float64
	ContLen     int
	GreedyExact bool
}

// NormLogprob is the length-normalized continuation log-probability — acc_norm's
// per-choice score, SumLogprob / ContLen. A zero-length continuation scores 0.
func (c ChoiceScore) NormLogprob() float64 {
	if c.ContLen == 0 {
		return 0
	}
	return c.SumLogprob / float64(c.ContLen)
}

// ScoreChoice runs ONE teacher-forced prefill over promptIDs++contIDs and folds
// the continuation into a ChoiceScore — the loglikelihood multiple-choice scoring
// primitive (#4363). It is the model-backed wrapper around scoreContinuation:
// exactly one Forward, no generation. promptIDs should carry at least one token (a
// BOS or the shared question stem) so the first continuation token is predicted
// from real context.
func (m *Model) ScoreChoice(promptIDs, contIDs []int) ChoiceScore {
	full := make([]int, 0, len(promptIDs)+len(contIDs))
	full = append(full, promptIDs...)
	full = append(full, contIDs...)
	act := m.Forward(full)
	sum, n, greedy := scoreContinuation(act.Logits, len(promptIDs), contIDs)
	return ChoiceScore{SumLogprob: sum, ContLen: n, GreedyExact: greedy}
}

// scoreContinuation folds a teacher-forced prefill's per-position logits into a
// continuation score. logits[t] is the vocab distribution PREDICTING token t+1
// (standard next-token teacher forcing), so the continuation token at full-
// sequence position promptLen+i is scored by logits[promptLen+i-1]. It returns the
// summed log-probability of the continuation tokens, their count, and whether
// argmax at every continuation position equals the actual token (greedy-exact).
// Pure over the logits matrix, so the index alignment and log-softmax are table-
// testable without a model. promptLen must be >= 1 (the first continuation token
// needs a preceding position to be predicted from); a promptLen < 1, an empty
// continuation, or a continuation running past the logits returns a zero score.
func scoreContinuation(logits [][]float32, promptLen int, contIDs []int) (sumLogprob float64, contLen int, greedyExact bool) {
	if promptLen < 1 || len(contIDs) == 0 {
		return 0, len(contIDs), false
	}
	greedyExact = true
	for i, tok := range contIDs {
		pos := promptLen + i // position of this continuation token in the full sequence
		row := pos - 1       // the logits that predict it
		if row < 0 || row >= len(logits) {
			return 0, len(contIDs), false
		}
		sumLogprob += logSoftmaxAt(logits[row], tok)
		if mathx.ArgmaxF32(logits[row]) != tok {
			greedyExact = false
		}
	}
	return sumLogprob, len(contIDs), greedyExact
}

// logSoftmaxAt returns the log-softmax probability of index tok in a logits row,
// computed with the max-subtraction trick for numerical stability. A tok out of
// range returns -Inf (an impossible token carries no probability mass).
func logSoftmaxAt(row []float32, tok int) float64 {
	if tok < 0 || tok >= len(row) {
		return math.Inf(-1)
	}
	maxv := row[0]
	for _, v := range row[1:] {
		if v > maxv {
			maxv = v
		}
	}
	var sumExp float64
	for _, v := range row {
		sumExp += math.Exp(float64(v - maxv))
	}
	return float64(row[tok]-maxv) - math.Log(sumExp)
}

// AccNorm picks the predicted choice as the argmax of the length-normalized
// continuation log-probabilities (acc_norm), returning the winning index and the
// per-choice normalized scores. An empty slice returns -1. Ties resolve to the
// lowest index — a stable first-best pick, matching mathx.ArgmaxF32 semantics.
func AccNorm(choices []ChoiceScore) (predicted int, norm []float64) {
	if len(choices) == 0 {
		return -1, nil
	}
	norm = make([]float64, len(choices))
	best, bestScore := 0, math.Inf(-1)
	for i, c := range choices {
		norm[i] = c.NormLogprob()
		if norm[i] > bestScore {
			best, bestScore = i, norm[i]
		}
	}
	return best, norm
}
