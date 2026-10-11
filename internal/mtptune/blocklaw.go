package mtptune

import (
	"fmt"
	"math"
)

// BlockLawInput describes one sampled draft chain. Target[i] and Proposal[i]
// are normalized conditional probabilities after Draft[:i], with sampling
// filters already applied. Every live Draft[i] must have Proposal[i][Draft[i]]
// > 0. Rows share a vocabulary; the target has an additional correction row.
//
// LiveLength counts VERIFIED TARGET ROWS, including the correction/bonus row,
// not draft tokens. Nil uses len(Draft)+1; L permits at most L-1 drafts and
// guarantees Accepted+1 <= L. L=1 is target-only; L=0 is refused. Only the live
// prefix is inspected, so unused suffix rows may be absent or invalid.
//
// A cap must be chosen without inspecting the first excluded draft or later
// randomness: the decision to draft position i may depend only on Draft[:i].
// Rows must attend only to earlier positions. These caller obligations cannot
// be checked from probability arrays alone.
//
// AcceptUniform contains independent U[0,1) draws, independent of drafting and
// CorrectionUniform. Supplying randomness makes the helper pure and replayable.
type BlockLawInput struct {
	Target            [][]float64
	Proposal          [][]float64
	Draft             []int
	AcceptUniform     []float64
	CorrectionUniform float64
	LiveLength        *int
}

// BlockLawResult commits Draft[:Accepted], followed by Correction. It never
// commits a row beyond LiveLength. It does not mutate model or KV state.
type BlockLawResult struct {
	Accepted   int
	Correction int
}

// VerifyBlock applies the joint block-verification law described in blocklaw.md.
// It preserves target output probabilities in exact arithmetic under the input
// contract; float64 evaluation is subject to ordinary rounding. This is not a
// bitwise equivalence claim about model execution. No serving path is enabled.
func VerifyBlock(in BlockLawInput) (BlockLawResult, error) {
	live := len(in.Draft) + 1
	if in.LiveLength != nil {
		live = *in.LiveLength
	}
	if live < 1 || live > len(in.Draft)+1 || live > 65 {
		return BlockLawResult{}, fmt.Errorf("mtptune: live target rows must be in [1,min(drafts+1,65)], got %d", live)
	}
	n := live - 1
	if len(in.Target) < live || len(in.Proposal) < n || len(in.AcceptUniform) < n {
		return BlockLawResult{}, fmt.Errorf("mtptune: missing live probability or uniform rows")
	}
	vocab := len(in.Target[0])
	for i := 0; i < live; i++ {
		if err := blockLawRow(in.Target[i], vocab); err != nil {
			return BlockLawResult{}, fmt.Errorf("mtptune: target row %d: %w", i, err)
		}
		if i < n {
			if err := blockLawRow(in.Proposal[i], vocab); err != nil {
				return BlockLawResult{}, fmt.Errorf("mtptune: proposal row %d: %w", i, err)
			}
			x := in.Draft[i]
			if x < 0 || x >= vocab || in.Proposal[i][x] == 0 {
				return BlockLawResult{}, fmt.Errorf("mtptune: draft %d is outside proposal support", i)
			}
			if !blockLawUniform(in.AcceptUniform[i]) {
				return BlockLawResult{}, fmt.Errorf("mtptune: invalid acceptance uniform %d", i)
			}
		}
	}
	if !blockLawUniform(in.CorrectionUniform) {
		return BlockLawResult{}, fmt.Errorf("mtptune: invalid correction uniform")
	}

	weights := make([]float64, live)
	weights[0] = 1
	logWeight := 0.0
	accepted := 0
	for i := 1; i <= n; i++ {
		x := in.Draft[i-1]
		p, q := in.Target[i-1][x], in.Proposal[i-1][x]
		// Keep the recurrence in log space so a small prefix weight can recover
		// after a later large likelihood ratio. Saturate at EVERY position.
		if p == 0 {
			logWeight = math.Inf(-1)
		} else {
			logWeight = math.Min(0, logWeight+math.Log(p)-math.Log(q))
		}
		w := math.Exp(logWeight)
		weights[i] = w
		h := w
		if i < n && w < 1 {
			s := 0.0
			for x, p := range in.Target[i] {
				s += math.Max(w*p-in.Proposal[i][x], 0)
			}
			// Subtract first: (s+1)-w can round away s near w=1 and
			// turn an interior threshold into certainty.
			h = s / (s + (1 - w))
		}
		// w=1 implies h=1, including the otherwise indeterminate 0/0 case.
		// Strict comparison makes a zero threshold impossible even at u=0.
		if in.AcceptUniform[i-1] < h {
			accepted = i
		}
	}

	mass := make([]float64, vocab)
	total := 0.0
	for x, p := range in.Target[accepted] {
		v := p
		if accepted < n {
			v = math.Max(weights[accepted]*p-in.Proposal[accepted][x], 0)
		}
		mass[x] = v
		total += v
	}
	if total <= 0 {
		return BlockLawResult{}, fmt.Errorf("mtptune: selected correction has no probability mass")
	}
	threshold := in.CorrectionUniform * total
	cumulative, last := 0.0, -1
	for x, v := range mass {
		if v > 0 {
			last = x
		}
		cumulative += v
		if threshold < cumulative {
			return BlockLawResult{Accepted: accepted, Correction: x}, nil
		}
	}
	// Only a final cumulative-sum rounding gap can reach here; never choose a
	// zero-support trailing token.
	return BlockLawResult{Accepted: accepted, Correction: last}, nil
}

func blockLawUniform(u float64) bool { return u >= 0 && u < 1 }

func blockLawRow(row []float64, vocab int) error {
	if vocab == 0 || len(row) != vocab {
		return fmt.Errorf("inconsistent or empty vocabulary")
	}
	sum := 0.0
	for _, p := range row {
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			return fmt.Errorf("probability is not finite and in [0,1]")
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		return fmt.Errorf("probabilities must sum to one, got %.17g", sum)
	}
	return nil
}
