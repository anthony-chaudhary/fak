// Copyright contributors to the DeepInfer project.
// Licensed under the Apache License, Version 2.0; see
// ../../third_party/deepinfer-sampling/LICENSE in this repository.
//
// Modified native Go adaptation of DeepInfer's fixed-budget allocation contract:
// https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py
// Float32 sequential products also follow the CUDA implementation at
// deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py in that revision.
// This implementation adds input validation and handles an empty previous batch.
// It does not port the adaptive budget optimizer or enable speculative serving.
package model

import (
	"fmt"
	"math"
	"sort"
)

// AllocateSpeculativeVerificationBudget distributes an externally fixed totalQ
// target-verification budget over the current batch. Each lane receives one
// mandatory target token plus a prefix of at most draftDepth draft tokens.
// previous contains conditional acceptance probabilities from a previous batch;
// remap maps each current lane to a previous row, or -1 for a fresh lane whose
// cumulative scores are all zero. Repeated row mappings are allowed.
//
// Extras are ranked by float32 cumulative acceptance, descending, then by lower
// current-lane-major flat index. This tie rule preserves prefixes even at zero
// confidence. Inputs are not mutated. Empty batches accept only a zero budget.
//
// This is only an allocation primitive, not a serving or quality qualification.
// Callers must choose totalQ independently before invoking it. In particular,
// DeepInfer chooses the budget from step N-2 and allocates using N-1 confidence;
// selecting totalQ from the same confidence here can introduce lookahead bias.
func AllocateSpeculativeVerificationBudget(previous [][]float32, remap []int, draftDepth, totalQ int) ([]int, error) {
	maxInt := int(^uint(0) >> 1)
	if draftDepth < 0 || draftDepth == maxInt || len(remap) > maxInt/(draftDepth+1) {
		return nil, fmt.Errorf("model: invalid speculative verification dimensions")
	}
	n := len(remap)
	if totalQ < n || totalQ > n*(draftDepth+1) {
		return nil, fmt.Errorf("model: speculative verification budget outside [%d,%d]", n, n*(draftDepth+1))
	}
	for row, confidence := range previous {
		if len(confidence) != draftDepth {
			return nil, fmt.Errorf("model: speculative confidence row %d has depth %d, want %d", row, len(confidence), draftDepth)
		}
		for position, p := range confidence {
			if math.IsNaN(float64(p)) || p < 0 || p > 1 {
				return nil, fmt.Errorf("model: invalid speculative confidence at row %d position %d", row, position)
			}
		}
	}
	for lane, row := range remap {
		if row < -1 || row >= len(previous) {
			return nil, fmt.Errorf("model: invalid speculative remap at lane %d", lane)
		}
	}
	allocation := make([]int, n)
	for lane := range allocation {
		allocation[lane] = 1
	}
	if totalQ == n {
		return allocation, nil
	}
	type candidate struct {
		score float32
		flat  int
	}
	candidates := make([]candidate, 0, n*draftDepth)
	for lane, row := range remap {
		cumulative := float32(1)
		for position := 0; position < draftDepth; position++ {
			if row == -1 {
				cumulative = 0
			} else {
				cumulative = float32(cumulative * previous[row][position])
			}
			candidates = append(candidates, candidate{cumulative, lane*draftDepth + position})
		}
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		return candidates[i].flat < candidates[j].flat
	})
	for _, selected := range candidates[:totalQ-n] {
		allocation[selected.flat/draftDepth]++
	}
	return allocation, nil
}
