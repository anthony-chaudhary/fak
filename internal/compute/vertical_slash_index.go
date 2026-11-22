package compute

// Host mirror of the vertical/slash -> block-sparse attention metadata builder.
// Adapted from microsoft/MInference csrc/vertical_slash_index.cu at commit
// 29ef1974fcb5a1440ccf92d7d7996900230fddcf, MIT.
// Copyright (c) Microsoft Corporation. Full license text:
// cuda/minference_vertical_slash_index.cuh.

import (
	"errors"
	"fmt"
)

// Upstream asserts both tile sizes are 64.
const (
	verticalSlashBlockM = 64
	verticalSlashBlockN = 64
)

// ErrVerticalSlashIndexShape reports inputs the builder kernel has no defined
// result for.
var ErrVerticalSlashIndexShape = errors.New("compute: vertical/slash index shape")

// VerticalSlashIndex is the block-sparse metadata for one [Batch, Heads]
// attention pattern, one entry per 64-row query block. Slices are flat
// row-major over the shapes noted on each field.
type VerticalSlashIndex struct {
	Batch, Heads, Rows int
	NNZV, NNZS         int
	// BlockCount [Batch, Heads, Rows]: dense 64-column key blocks per row block.
	BlockCount []int32
	// BlockOffset [Batch, Heads, Rows, NNZS]: first key column of each block.
	BlockOffset []int32
	// ColumnCount [Batch, Heads, Rows]: stray key columns outside every block.
	ColumnCount []int32
	// ColumnIndex [Batch, Heads, Rows, NNZV]: those key columns.
	ColumnIndex []int32
}

// BuildVerticalSlashIndexHost runs the kernel's per-row-block merge on the CPU.
// vertical is [batch, heads, nnzV] ascending per head; slash is
// [batch, heads, nnzS] descending per head and non-negative; seqlens is [batch].
// Ordering is the caller's contract, as upstream, and is not checked.
func BuildVerticalSlashIndexHost(seqlens, vertical, slash []int32, batch, heads, contextSize int) (*VerticalSlashIndex, error) {
	if batch <= 0 || heads <= 0 || contextSize <= 0 || len(seqlens) != batch {
		return nil, fmt.Errorf("%w: batch=%d heads=%d context=%d seqlens=%d",
			ErrVerticalSlashIndexShape, batch, heads, contextSize, len(seqlens))
	}
	per := batch * heads
	if len(vertical) == 0 || len(slash) == 0 || len(vertical)%per != 0 || len(slash)%per != 0 {
		return nil, fmt.Errorf("%w: vertical=%d slash=%d not a positive multiple of batch*heads=%d",
			ErrVerticalSlashIndexShape, len(vertical), len(slash), per)
	}
	// A negative slash pushes a range past the exhausted-vertical sentinel, where
	// the kernel never terminates.
	for i, s := range slash {
		if s < 0 {
			return nil, fmt.Errorf("%w: slash[%d]=%d is negative", ErrVerticalSlashIndexShape, i, s)
		}
	}
	nnzV, nnzS := len(vertical)/per, len(slash)/per
	rows := (contextSize + verticalSlashBlockM - 1) / verticalSlashBlockM
	out := &VerticalSlashIndex{
		Batch: batch, Heads: heads, Rows: rows, NNZV: nnzV, NNZS: nnzS,
		BlockCount:  make([]int32, per*rows),
		BlockOffset: make([]int32, per*rows*nnzS),
		ColumnCount: make([]int32, per*rows),
		ColumnIndex: make([]int32, per*rows*nnzV),
	}
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			bh := b*heads + h
			for m := 0; m < rows; m++ {
				rowOffset := bh*rows + m
				blk, col := verticalSlashRowBlock(
					seqlens[b], int32(m),
					vertical[bh*nnzV:(bh+1)*nnzV], slash[bh*nnzS:(bh+1)*nnzS],
					out.BlockOffset[rowOffset*nnzS:(rowOffset+1)*nnzS],
					out.ColumnIndex[rowOffset*nnzV:(rowOffset+1)*nnzV])
				out.BlockCount[rowOffset] = blk
				out.ColumnCount[rowOffset] = col
			}
		}
	}
	return out, nil
}

// verticalSlashRowBlock is one kernel thread: statement order and int32
// arithmetic follow convert_vertical_slash_indexes_kernel.
func verticalSlashRowBlock(seqlen, blockIdxM int32, verticalIndexes, slashIndexes, blockOffset, columnIndex []int32) (blkCnt, colCnt int32) {
	const blockSizeM, blockSizeN int32 = verticalSlashBlockM, verticalSlashBlockN
	nnzV, nnzS := len(verticalIndexes), len(slashIndexes)
	startM := blockIdxM * blockSizeM
	if startM >= seqlen {
		return 0, 0
	}
	endM := startM + blockSizeM

	saveBlocks := func(rangeStart, rangeEnd int32) {
		for idx := rangeStart; idx < rangeEnd; idx += blockSizeN {
			blockOffset[blkCnt] = idx
			blkCnt++
		}
	}

	s, v := 0, 0
	vIdx := verticalIndexes[v]
	v++
	sIdx := slashIndexes[s]
	s++
	for sIdx >= endM {
		// Upstream reads past the slash row here; the port leaves the row empty.
		if s >= nnzS {
			return 0, 0
		}
		sIdx = slashIndexes[s]
		s++
	}
	sIdx = max(endM-sIdx, blockSizeM)
	rangeStart, rangeEnd := sIdx-blockSizeM, sIdx
	for {
		if vIdx < rangeEnd {
			if vIdx < rangeStart {
				columnIndex[colCnt] = vIdx
				colCnt++
			}
			if v < nnzV {
				vIdx = verticalIndexes[v]
				v++
			} else {
				vIdx = endM + blockSizeM
			}
		} else {
			if s < nnzS {
				sIdx = max(endM-slashIndexes[s], blockSizeM)
				s++
			} else {
				saveBlocks(rangeStart, rangeEnd)
				break
			}
			if sIdx > rangeEnd+blockSizeM {
				saveBlocks(rangeStart, rangeEnd)
				rangeStart = sIdx - blockSizeM
				rangeEnd = sIdx
			} else if sIdx > rangeEnd {
				rangeEnd += blockSizeM
			}
		}
	}
	return blkCnt, colCnt
}
