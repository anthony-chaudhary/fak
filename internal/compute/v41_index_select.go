package compute

import (
	"fmt"
	"math"
	"slices"
)

// V41IndexSelectMaxRows bounds one selection so every published row index is an
// exact binary32 integer in the output tensor.
const V41IndexSelectMaxRows = 1 << 24

// V41IndexSelectBackend is the optional one-query DeepSeek-V4.1 index-row
// selection after the indexer score. scores is an immutable, unquantized,
// row-major F32 [rows] tensor on this backend; every score position is causally
// visible. With topKBlocks > 0 a candidate-block mask runs first: blocks of
// blockSize rows rank by their maximum score, the newest block is pinned, and
// only the topKBlocks leading blocks whose score is not -Inf stay candidates.
// The final selection ranks masked scores and keeps min(topK, rows) rows.
//
// Every rank orders a higher score first and an equal score by the lower
// position, with -0 equal to +0, exactly like the host selectors. The output is
// a fresh caller-owned F32 [min(topK,rows)] tensor of position-sorted row
// indices encoded as exact integers, with -1 for a masked or -Inf selection.
// A NaN score is an operation error, never a selection. Capability absence
// permits the caller's unchanged host selector; a selected failure must
// propagate without CPU replay.
type V41IndexSelectBackend interface {
	V41IndexSelectAvailable() bool
	V41IndexSelect(scores Tensor, rows, topK, topKBlocks, blockSize int) (Tensor, error)
}

// validateV41IndexSelect checks the geometry before narrowing it to the int32
// push ABI. It neither reads tensor data nor allocates.
func validateV41IndexSelect(scores Tensor, rows, topK, topKBlocks, blockSize int) (selected int, err error) {
	if rows <= 0 || rows > V41IndexSelectMaxRows || topK <= 0 || topK > math.MaxInt32 || topKBlocks < 0 || topKBlocks > math.MaxInt32 ||
		(topKBlocks > 0 && (blockSize <= 0 || blockSize > math.MaxInt32)) {
		return 0, fmt.Errorf("compute: V4.1 index select invalid geometry rows=%d topK=%d topKBlocks=%d blockSize=%d", rows, topK, topKBlocks, blockSize)
	}
	if scores.Dtype != F32 || scores.Layout != RowMajor || scores.Quant != nil || !slices.Equal(scores.Shape, []int{rows}) {
		return 0, fmt.Errorf("compute: V4.1 index select scores must be unquantized row-major F32 with shape [%d]", rows)
	}
	return min(topK, rows), nil
}

// v41IndexSelectScratchWords is the native scratch extent: block keys, block
// keep flags and per-row selection flags.
func v41IndexSelectScratchWords(rows, topKBlocks, blockSize int) int {
	blocks := 0
	if topKBlocks > 0 {
		blocks = (rows-1)/blockSize + 1
	}
	return 2*blocks + rows
}

// V41IndexSelectReference is the portable mirror of the device rank-count
// algorithm. It exists so tests can hold the kernel's integer-key formulation
// against the host sort-based selectors; it is never a runtime fallback.
func V41IndexSelectReference(scores []float32, topK, topKBlocks, blockSize int) ([]int32, error) {
	rows := len(scores)
	if _, err := validateV41IndexSelect(Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{rows}}, rows, topK, topKBlocks, blockSize); err != nil {
		return nil, err
	}
	key := func(v float32) uint32 {
		bits := math.Float32bits(v)
		if bits == 0x80000000 {
			bits = 0
		}
		if bits&0x80000000 != 0 {
			return ^bits
		}
		return bits | 0x80000000
	}
	const keyNegInf, keyPosInf = uint32(0x007fffff), uint32(0xff800000)
	outranks := func(a uint32, ia int, b uint32, ib int) bool { return a > b || (a == b && ia < ib) }
	for i, v := range scores {
		if v != v {
			return nil, fmt.Errorf("compute: V4.1 index select score %d is NaN", i)
		}
	}
	keep := func(int) bool { return true }
	if topKBlocks > 0 {
		blocks := (rows-1)/blockSize + 1
		best := make([]uint32, blocks)
		for b := range best {
			best[b] = keyNegInf
			for p := b * blockSize; p < min((b+1)*blockSize, rows); p++ {
				best[b] = max(best[b], key(scores[p]))
			}
		}
		best[(rows-1)/blockSize] = keyPosInf
		kept := make([]bool, blocks)
		for b, mine := range best {
			rank := 0
			for o, other := range best {
				if outranks(other, o, mine, b) {
					rank++
				}
			}
			kept[b] = rank < topKBlocks && mine != keyNegInf
		}
		keep = func(i int) bool { return kept[i/blockSize] }
	}
	masked := make([]uint32, rows)
	for i, v := range scores {
		masked[i] = keyNegInf
		if keep(i) {
			masked[i] = key(v)
		}
	}
	out := make([]int32, 0, min(topK, rows))
	for i, mine := range masked {
		rank := 0
		for j, other := range masked {
			if outranks(other, j, mine, i) {
				rank++
			}
		}
		if rank >= topK {
			continue
		}
		if mine == keyNegInf {
			out = append(out, -1)
		} else {
			out = append(out, int32(i))
		}
	}
	return out, nil
}
