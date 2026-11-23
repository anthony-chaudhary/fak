package modelengine

// Adapted from sgl-project/sglang python/sglang/srt/layers/attention/minimax_sparse_ops/indexer_cp.py
// at commit 2fd45ed82fe07ec64ef7f4d2dbd9071b0eb01763 (Apache-2.0); changes were made: only the
// shard-the-scan and merge-candidates contract is re-expressed as host Go, with no collectives or kernels.

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"slices"
)

// IndexerCandidate is one KV block nominated by the decode-time top-k indexer.
type IndexerCandidate struct {
	BlockID int
	Score   float32
}

// IndexerCPShard is the half-open block range [Start, End) one rank scans.
type IndexerCPShard struct {
	Rank       int
	Start, End int
}

var (
	errIndexerCPRanks     = errors.New("indexer cp: rank count must be at least 1")
	errIndexerCPNegativeK = errors.New("indexer cp: k must not be negative")
	errIndexerCPNaN       = errors.New("indexer cp: NaN score has no total order")
)

// IndexerCPPartition splits numBlocks into ranks contiguous shards whose sizes
// differ by at most one. Ranks beyond numBlocks get empty shards so every rank
// still takes part in the merge with the same collective shape.
func IndexerCPPartition(numBlocks, ranks int) ([]IndexerCPShard, error) {
	if ranks < 1 {
		return nil, errIndexerCPRanks
	}
	if numBlocks < 0 {
		return nil, fmt.Errorf("indexer cp: negative block count %d", numBlocks)
	}
	base, extra := numBlocks/ranks, numBlocks%ranks
	shards := make([]IndexerCPShard, ranks)
	start := 0
	for r := range shards {
		size := base
		if r < extra {
			size++
		}
		shards[r] = IndexerCPShard{Rank: r, Start: start, End: start + size}
		start += size
	}
	return shards, nil
}

// indexerCandidateLess orders by score descending, then block id ascending, so
// equal scores split across ranks resolve identically wherever they meet.
func indexerCandidateLess(a, b IndexerCandidate) int {
	if c := cmp.Compare(b.Score, a.Score); c != 0 {
		return c
	}
	return cmp.Compare(a.BlockID, b.BlockID)
}

// IndexerCPLocalTopK scores one shard of the block scan and keeps its best k.
// scores is indexed by global block id.
func IndexerCPLocalTopK(scores []float32, shard IndexerCPShard, k int) ([]IndexerCandidate, error) {
	if k < 0 {
		return nil, errIndexerCPNegativeK
	}
	if shard.Start < 0 || shard.End < shard.Start || shard.End > len(scores) {
		return nil, fmt.Errorf("indexer cp: shard [%d,%d) outside %d blocks", shard.Start, shard.End, len(scores))
	}
	local := make([]IndexerCandidate, 0, shard.End-shard.Start)
	for id := shard.Start; id < shard.End; id++ {
		if math.IsNaN(float64(scores[id])) {
			return nil, fmt.Errorf("%w: block %d", errIndexerCPNaN, id)
		}
		local = append(local, IndexerCandidate{BlockID: id, Score: scores[id]})
	}
	slices.SortFunc(local, indexerCandidateLess)
	return local[:min(k, len(local))], nil
}

// IndexerCPMerge combines per-rank candidate sets into the global top-k. A
// single participating rank is the non-CP path and is returned untouched
// apart from truncation to k. The result does not depend on rank order or on
// the order of candidates within a rank.
func IndexerCPMerge(perRank [][]IndexerCandidate, k int) ([]IndexerCandidate, error) {
	if len(perRank) < 1 {
		return nil, errIndexerCPRanks
	}
	if k < 0 {
		return nil, errIndexerCPNegativeK
	}
	if len(perRank) == 1 {
		only := perRank[0]
		return only[:min(k, len(only))], nil
	}
	total := 0
	for _, cands := range perRank {
		total += len(cands)
	}
	merged := make([]IndexerCandidate, 0, total)
	seen := make(map[int]int, total)
	for rank, cands := range perRank {
		for _, c := range cands {
			if math.IsNaN(float64(c.Score)) {
				return nil, fmt.Errorf("%w: rank %d block %d", errIndexerCPNaN, rank, c.BlockID)
			}
			// Overlapping shards would double-count a block and silently
			// evict a genuine candidate from the merged set.
			if prev, dup := seen[c.BlockID]; dup {
				return nil, fmt.Errorf("indexer cp: block %d nominated by ranks %d and %d", c.BlockID, prev, rank)
			}
			seen[c.BlockID] = rank
			merged = append(merged, c)
		}
	}
	slices.SortFunc(merged, indexerCandidateLess)
	return merged[:min(k, len(merged))], nil
}

// IndexerCPTopK runs the full shard-scan-merge pipeline over scores. With one
// rank it is exactly the unsharded top-k.
func IndexerCPTopK(scores []float32, ranks, k int) ([]IndexerCandidate, error) {
	shards, err := IndexerCPPartition(len(scores), ranks)
	if err != nil {
		return nil, err
	}
	perRank := make([][]IndexerCandidate, len(shards))
	for i, shard := range shards {
		if perRank[i], err = IndexerCPLocalTopK(scores, shard, k); err != nil {
			return nil, err
		}
	}
	return IndexerCPMerge(perRank, k)
}
