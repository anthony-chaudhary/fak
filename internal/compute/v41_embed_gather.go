package compute

import (
	"fmt"
	"math"
)

// V41EmbeddingGatherBackend gathers DeepSeek-V4.1 token-embedding rows on the
// device. table is an immutable, unquantized, row-major F32 [vocab, hidden]
// tensor on this backend; every id must be in [0, vocab). The result is a fresh
// caller-owned F32 [len(ids), hidden] panel whose rows are byte copies of the
// table rows, in id order with repetitions. No arithmetic, scaling or BF16
// rounding is applied. Capability absence permits the caller's host gather; a
// selected failure must propagate without CPU replay.
type V41EmbeddingGatherBackend interface {
	V41EmbeddingGatherAvailable() bool
	V41EmbeddingGather(table Tensor, ids []int, hidden int) (Tensor, error)
}

// validateV41EmbeddingGather checks the table shape and every id before any
// allocation. It returns the per-row byte stride.
func validateV41EmbeddingGather(table Tensor, ids []int, hidden int) (rowBytes int, err error) {
	if len(ids) == 0 || hidden <= 0 || hidden > math.MaxInt32/4 {
		return 0, fmt.Errorf("compute: V4.1 embedding gather invalid geometry ids=%d hidden=%d", len(ids), hidden)
	}
	if table.Dtype != F32 || table.Layout != RowMajor || table.Quant != nil || len(table.Shape) != 2 || table.Shape[1] != hidden || table.Shape[0] <= 0 {
		return 0, fmt.Errorf("compute: V4.1 embedding gather table must be unquantized row-major F32 [vocab,%d]", hidden)
	}
	vocab := table.Shape[0]
	if vocab > math.MaxInt/(hidden*4) || len(ids) > math.MaxInt/(hidden*4) {
		return 0, fmt.Errorf("compute: V4.1 embedding gather byte range overflows int")
	}
	for i, id := range ids {
		if id < 0 || id >= vocab {
			return 0, fmt.Errorf("compute: V4.1 embedding gather id[%d]=%d outside [0,%d)", i, id, vocab)
		}
	}
	return hidden * 4, nil
}
