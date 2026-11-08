package model

import (
	"errors"
	"strconv"
)

var ErrSpPpBoundaryShape = errors.New("model: sequence/pipeline boundary shape")

// spPpGeometry computes equal padded token shards without overflowing int.
func spPpGeometry(tokens, width, ranks int) (rows, flat, shardSize int, err error) {
	if tokens < 0 || width <= 0 || ranks <= 0 {
		return 0, 0, 0, ErrSpPpBoundaryShape
	}
	maxInt := int(^uint(0) >> 1)
	if ranks > maxInt/(3*(strconv.IntSize/8)) || width > maxInt/4 || tokens > (maxInt/4)/width {
		return 0, 0, 0, ErrSpPpBoundaryShape
	}
	rows = tokens / ranks
	if tokens%ranks != 0 {
		rows++
	}
	if rows > (maxInt/4)/width || (rows != 0 && ranks > (maxInt/4)/(rows*width)) {
		return 0, 0, 0, ErrSpPpBoundaryShape
	}
	return rows, tokens * width, rows * width, nil
}

// ShardSpPpBoundary is a closed host oracle for token sharding at stage entry,
// adapted from vLLM sequence_parallel.sp_shard at213e64562ba6e82ffce95774fa37fd294f3bc8ee.
// Equal contiguous token shards are zero-padded; outputs own their storage.
// No collective, model route or multi-GPU transport is executed.
func ShardSpPpBoundary(input []float32, tokens, width, ranks int) ([][]float32, error) {
	rows, flat, size, err := spPpGeometry(tokens, width, ranks)
	if err != nil || len(input) != flat {
		return nil, ErrSpPpBoundaryShape
	}
	shards := make([][]float32, ranks)
	for rank := range shards {
		shards[rank] = make([]float32, size)
		startRow := rank * rows
		if startRow >= tokens {
			continue
		}
		endRow := startRow + rows
		if endRow > tokens {
			endRow = tokens
		}
		copy(shards[rank], input[startRow*width:endRow*width])
	}
	return shards, nil
}

// GatherSpPpBoundary restores the replicated token tensor before PP send,
// discarding padded rows as MiniMax-M3 does with sp_all_gather(... )[:num_tokens].
// Each receiving stage re-shards at entry, including the last stage; this is
// not a rule to keep the final model output sharded (upstream gathers it too).
func GatherSpPpBoundary(shards [][]float32, tokens, width int) ([]float32, error) {
	rows, flat, size, err := spPpGeometry(tokens, width, len(shards))
	if err != nil {
		return nil, err
	}
	for _, shard := range shards {
		if len(shard) != size {
			return nil, ErrSpPpBoundaryShape
		}
	}
	out := make([]float32, flat)
	for rank, shard := range shards {
		startRow := rank * rows
		if startRow >= tokens {
			continue
		}
		endRow := startRow + rows
		if endRow > tokens {
			endRow = tokens
		}
		copy(out[startRow*width:endRow*width], shard[:(endRow-startRow)*width])
	}
	return out, nil
}
