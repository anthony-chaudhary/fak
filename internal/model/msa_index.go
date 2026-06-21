package model

import (
	"math"
	"sort"
)

// msa_index.go — MiniMax-M3 "MiniMax Sparse Attention" (MSA) block selection.
//
// MiniMax-M3 abandons the M1 lightning (linear) attention and the M2 full-attention
// approaches for MSA: a GQA backbone on the REAL uncompressed K/V (NOT MLA latent
// compression) with a block-level sparse selector. Per HF's MiniMaxM3VL indexer, the
// path is:
//
//  1. a lightning indexer scores every (query, key) pair;
//  2. those per-key scores are MAX-POOLED into blocks of `index_block_size` keys;
//  3. per query, the union of the top-`index_topk_blocks` scored blocks and the
//     always-on `index_local_blocks` most-recent blocks is selected, under
//     block-level causality;
//  4. the per-block 0/-inf choice is broadcast back onto every key in the block, and
//     standard GQA softmax attention runs over only the admitted keys.
//
// This file is the host-tractable WITNESS of steps 2-4: the block max-pool, the
// top-k + always-on-local selection, and the broadcast-to-keys, as deterministic
// hand-references. It is the MSA analogue of dsa_index.go (GLM-5.2's per-KEY Dynamic
// Sparse Attention). The architectural distinction between the two families lives
// exactly in the SELECTION granularity here (DSA picks individual keys; MSA picks
// contiguous blocks of keys with an always-on local window); the sparse softmax over
// the admitted causal key set is family-agnostic, so msaAttention reuses
// dsaSparseAttention for step 4 rather than duplicating it.
//
// What this file does NOT claim: the learned lightning-indexer projections, qk-norm,
// partial RoPE, and the SwiGLU-OAI MoE FFN of a real MiniMax-M3 checkpoint, nor a
// wired hot-path forward or a real-checkpoint HF oracle. Those remain a separate gate
// (GPU/artifact node), exactly as the GLM-DSA forward was staged.

// msaBlockScores max-pools per-key lightning-indexer scores into per-block scores for
// ONE query. keyScores[i] is the indexer score of the key at keyPositions[i]; only
// causal keys (position <= queryPos) contribute. A block's index is pos/blockSize and
// its score is the MAX over its causal member keys — MiniMax-M3 pools per-key scores
// into blocks of blockSize keys before block selection. NaN scores are skipped.
// Returns a map of causal block index -> max-pooled score, or ok=false when the inputs
// are malformed or no causal key has a finite score.
func msaBlockScores(keyScores []float64, keyPositions []int, queryPos, blockSize int) (map[int]float64, bool) {
	if blockSize <= 0 || queryPos < 0 || len(keyScores) != len(keyPositions) || len(keyPositions) == 0 {
		return nil, false
	}
	out := make(map[int]float64, len(keyPositions))
	for i, pos := range keyPositions {
		if pos < 0 {
			return nil, false
		}
		if pos > queryPos {
			continue // future key: block-causal pooling ignores it
		}
		if math.IsNaN(keyScores[i]) {
			continue
		}
		b := pos / blockSize
		if cur, ok := out[b]; !ok || keyScores[i] > cur {
			out[b] = keyScores[i]
		}
	}
	if len(out) == 0 {
		return nil, false
	}
	return out, true
}

// msaSelectBlocks selects, for ONE query, the set of key-blocks MiniMax-M3 MSA attends
// to: the UNION of the top-`topKBlocks` candidate blocks by max-pooled score and the
// always-on `localBlocks` most-recent candidate blocks (the query's own block and the
// window just behind it). Only causal candidate blocks (those present in blockScores)
// are eligible — block-level causality. topKBlocks/localBlocks are clamped to the
// number of candidates, so a query early in the sequence simply attends to all blocks
// it has. Top-k ties break on the smaller block index for determinism. Returns the
// selected block indices in ascending order.
func msaSelectBlocks(blockScores map[int]float64, queryPos, blockSize, topKBlocks, localBlocks int) ([]int, bool) {
	if blockSize <= 0 || queryPos < 0 || topKBlocks < 0 || localBlocks < 0 || len(blockScores) == 0 {
		return nil, false
	}
	cand := make([]int, 0, len(blockScores))
	for b := range blockScores {
		if b < 0 {
			return nil, false
		}
		cand = append(cand, b)
	}
	sort.Ints(cand) // ascending block order

	selected := make(map[int]struct{}, len(cand))

	// Always-on local window: the `localBlocks` most-recent candidate blocks (largest
	// indices, ending at the query's own block). These are selected regardless of score.
	for i := len(cand) - 1; i >= 0 && len(cand)-i <= localBlocks; i-- {
		selected[cand[i]] = struct{}{}
	}

	// Top-k by max-pooled score, tie-broken on the smaller block index.
	byScore := append([]int(nil), cand...)
	sort.SliceStable(byScore, func(i, j int) bool {
		si, sj := blockScores[byScore[i]], blockScores[byScore[j]]
		if si == sj {
			return byScore[i] < byScore[j]
		}
		return si > sj
	})
	for i := 0; i < topKBlocks && i < len(byScore); i++ {
		selected[byScore[i]] = struct{}{}
	}

	if len(selected) == 0 {
		return nil, false
	}
	out := make([]int, 0, len(selected))
	for b := range selected {
		out = append(out, b)
	}
	sort.Ints(out)
	return out, true
}

// msaSelectedKeyPositions expands a per-query block selection onto individual key
// positions: every key whose position is <= queryPos and whose block (pos/blockSize)
// is in selectedBlocks. This is MSA's "broadcast the per-block 0/-inf choice back onto
// every key" step. keyPositions is the actual cache layout; the result is the ascending
// set of admitted causal key positions, suitable as the per-query selection passed to
// the sparse softmax. Duplicate cache positions fail closed.
func msaSelectedKeyPositions(keyPositions []int, queryPos, blockSize int, selectedBlocks []int) ([]int, bool) {
	if blockSize <= 0 || queryPos < 0 || len(keyPositions) == 0 || len(selectedBlocks) == 0 {
		return nil, false
	}
	sel := make(map[int]struct{}, len(selectedBlocks))
	for _, b := range selectedBlocks {
		if b < 0 {
			return nil, false
		}
		sel[b] = struct{}{}
	}
	out := make([]int, 0, len(keyPositions))
	seen := make(map[int]struct{}, len(keyPositions))
	for _, pos := range keyPositions {
		if pos < 0 {
			return nil, false
		}
		if pos > queryPos {
			continue
		}
		if _, dup := seen[pos]; dup {
			return nil, false // duplicate cache position
		}
		seen[pos] = struct{}{}
		if _, ok := sel[pos/blockSize]; ok {
			out = append(out, pos)
		}
	}
	if len(out) == 0 {
		return nil, false
	}
	sort.Ints(out)
	return out, true
}

// msaAttention runs the full MiniMax-M3 MSA path for a batch of queries over a shared
// causal K/V. keyScores[q] is the lightning indexer's per-key score row for query q
// (parallel to keyPositions). For each query it pools scores into blocks, selects the
// top-k blocks plus the always-on local window, broadcasts the selection back to keys,
// and runs GQA softmax over ONLY the admitted keys. The sparse softmax itself is the
// same uncompressed-K/V attention DSA uses (dsaSparseAttention); the MSA-specific part
// is the block-granular selection above. Returns per-query/per-head attention output.
func msaAttention(query, key, value [][][]float64, keyScores [][]float64, queryPositions, keyPositions []int, blockSize, topKBlocks, localBlocks int, scale float64) ([][][]float64, bool) {
	if len(query) == 0 || len(query) != len(queryPositions) || len(keyScores) != len(query) {
		return nil, false
	}
	sel := make([][]int, len(query))
	for qi := range query {
		bs, ok := msaBlockScores(keyScores[qi], keyPositions, queryPositions[qi], blockSize)
		if !ok {
			return nil, false
		}
		blocks, ok := msaSelectBlocks(bs, queryPositions[qi], blockSize, topKBlocks, localBlocks)
		if !ok {
			return nil, false
		}
		keys, ok := msaSelectedKeyPositions(keyPositions, queryPositions[qi], blockSize, blocks)
		if !ok {
			return nil, false
		}
		sel[qi] = keys
	}
	return dsaSparseAttention(query, key, value, queryPositions, keyPositions, sel, scale)
}
