package compute

import (
	"fmt"
	"slices"
)

// V41RouterHashBackend is the optional DeepSeek-V4.1 router for the first
// hash-routed layers, whose experts come from the artifact's token-to-expert
// table (ffn_gate_tid2eid) instead of a top-k. logits is the live row-major F32
// [rows,experts] gate output owned by this backend; expertIDs is the host
// row-major [rows,topK] table lookup for the rows' tokens.
//
// Per row every logit is scored sqrt(softplus(z)) and validated like
// V41RouterTopKBackend (logit, then score, per expert ascending); the weights
// are the table experts' scores, in table order, divided by their in-order F32
// sum (sum+1e-20 when full) times routeScale. Bias never applies. Faults,
// failure classes and the readback are V41RouterTopKBackend's.
type V41RouterHashBackend interface {
	Backend
	V41RouterHashAvailable() bool
	V41RouterHash(logits Tensor, expertIDs []int32, rows, experts, topK int, routeScale float32, full bool) (V41RouterPicks, error)
}

// validateV41RouterHash rejects malformed geometry and table rows (an id out of
// range or repeated within a row) before any device work. It returns the
// logits, picks and status byte extents.
func validateV41RouterHash(logits Tensor, expertIDs []int32, rows, experts, topK int, routeScale float32) (logitBytes, pickBytes, statusBytes int, err error) {
	const maxABIInt = 1<<31 - 1
	if rows <= 0 || experts <= 0 || topK <= 0 || topK > experts || topK > V41RouterTopKMaxK ||
		rows > maxABIInt || experts > maxABIInt || uint64(rows)*uint64(experts) > maxABIInt {
		return 0, 0, 0, fmt.Errorf("compute: V4.1 hash router invalid geometry rows=%d experts=%d topK=%d", rows, experts, topK)
	}
	if !(routeScale > 0) || routeScale > 3.4028234663852886e38 {
		return 0, 0, 0, fmt.Errorf("compute: V4.1 hash router scale must be finite and positive, got %g", routeScale)
	}
	if logits.Dtype != F32 || logits.Layout != RowMajor || logits.Quant != nil || !slices.Equal(logits.Shape, []int{rows, experts}) {
		return 0, 0, 0, fmt.Errorf("compute: V4.1 hash router logits must be unquantized row-major F32 with shape [%d %d]", rows, experts)
	}
	if len(expertIDs) != rows*topK {
		return 0, 0, 0, fmt.Errorf("compute: V4.1 hash router table has %d ids, want %d", len(expertIDs), rows*topK)
	}
	for row := 0; row < rows; row++ {
		ids := expertIDs[row*topK : (row+1)*topK]
		for slot, id := range ids {
			if id < 0 || int(id) >= experts || slices.Contains(ids[:slot], id) {
				return 0, 0, 0, fmt.Errorf("compute: V4.1 hash router row %d slot %d has invalid or repeated expert %d", row, slot, id)
			}
		}
	}
	return rows * experts * 4, rows * topK * 4, rows * 16, nil
}
