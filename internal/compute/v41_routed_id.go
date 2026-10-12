package compute

import (
	"fmt"
	"math"
)

// V41RoutedIDMaxPicks bounds the routed picks one id dispatch covers: the
// slot table binds one ring page per pick.
const V41RoutedIDMaxPicks = 8

// V41RoutedIDBackend records one DeepSeek-V4.1 decode layer's routed experts as
// one dispatch per stage over every pick (fak#13820), instead of one per pick:
// gate and up (V41MatVecID), the routed SwiGLU, down, and the checked BF16
// publication fused with the routed sum. It ports llama.cpp's mul_mat_vec_id
// (grid Y is the selected expert, read from an ids buffer) with the ids
// remapped through a per-layer slot map as llama_moe_cache does; here the slot
// table is the layer's bound ring pages.
//
// Each stage is the per-pick kernel term for term, so every pick's output and
// status record are bitwise the per-pick path's, and the routed sum is the
// ordered F32 accumulation of v41DeviceAccumulate.
type V41RoutedIDBackend interface {
	V41RoutedIDAvailable() bool
	// SupportsV41RoutedIDWeight reports a weight dtype V41MatVecID serves.
	SupportsV41RoutedIDWeight(dt Dtype) bool
	// V41RoutedIDs uploads one layer's ids image: words [0,P) are each pick's
	// slot in the bound table, [P,2P) the route weights' binary32 bits, and
	// [2P,3P) the routed-sum pick order. The tensor is a caller-owned word
	// container shaped [3P]; it is not F32 data.
	V41RoutedIDs(slots []int, weights []float32, order []int) (Tensor, error)
	// V41MatVecID returns the F32 [P*out] rows weights[slot(p)] * x_p, where
	// x_p is x itself (len in) or x's p-th row (len P*in). Every weight is
	// [out,in] of one dtype.
	V41MatVecID(weights []Tensor, ids Tensor, picks int, x Tensor) (Tensor, error)
	// V41RoutedSwiGLUIDDeferred is V41RoutedSwiGLUDeferred over [P*count]
	// gate/up rows, pick p weighted by ids word P+p; its status carries one
	// record per pick.
	V41RoutedSwiGLUIDDeferred(gate, up, ids Tensor, picks int, limit float32) (Tensor, V41DeferredStatus, error)
	// V41RoutedPublishSumDeferred publishes every pick's [count] down row as
	// V41BF16PublishDeferred would (one record per pick) and returns their sum
	// in ids order, valid only when every record is clean.
	V41RoutedPublishSumDeferred(down, ids Tensor, picks int, site uint32) (Tensor, V41DeferredStatus, error)
}

// V41RoutedIDWords lays out the ids image V41RoutedIDs uploads.
func V41RoutedIDWords(slots []int, weights []float32, order []int) ([]uint32, error) {
	picks := len(slots)
	if picks <= 0 || picks > V41RoutedIDMaxPicks || len(weights) != picks || len(order) != picks {
		return nil, fmt.Errorf("compute: V4.1 routed ids need 1..%d picks with one weight and one order entry each", V41RoutedIDMaxPicks)
	}
	words := make([]uint32, 3*picks)
	seen := make([]bool, picks)
	for p := 0; p < picks; p++ {
		if slots[p] < 0 || slots[p] >= picks {
			return nil, fmt.Errorf("compute: V4.1 routed slot %d of pick %d is outside the %d-slot table", slots[p], p, picks)
		}
		if !finite32(weights[p]) {
			return nil, fmt.Errorf("compute: V4.1 routed weight of pick %d is not finite", p)
		}
		if order[p] < 0 || order[p] >= picks || seen[order[p]] {
			return nil, fmt.Errorf("compute: V4.1 routed sum order is not a permutation of the picks")
		}
		seen[order[p]] = true
		words[p] = uint32(slots[p])
		words[picks+p] = math.Float32bits(weights[p])
		words[2*picks+p] = uint32(order[p])
	}
	return words, nil
}

func finite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

// validateV41MatVecID checks geometry before any native call and returns the
// shared [out,in] shape and the per-pick activation stride (0 = shared row).
func validateV41MatVecID(weights []Tensor, ids Tensor, picks int, x Tensor) (out, in, xStride int, err error) {
	if picks <= 0 || picks > V41RoutedIDMaxPicks || len(weights) == 0 || len(weights) > V41RoutedIDMaxPicks {
		return 0, 0, 0, fmt.Errorf("compute: V4.1 matvec id needs 1..%d picks and slots", V41RoutedIDMaxPicks)
	}
	dt := weights[0].Dtype
	for _, w := range weights {
		if w.Dtype != dt || (dt != Q2_K && dt != Q3_K) || len(w.Shape) != 2 {
			return 0, 0, 0, fmt.Errorf("compute: V4.1 matvec id slots must share one Q2_K or Q3_K [out,in] shape")
		}
		if w.Shape[0] != weights[0].Shape[0] || w.Shape[1] != weights[0].Shape[1] {
			return 0, 0, 0, fmt.Errorf("compute: V4.1 matvec id slots must share one [out,in] shape")
		}
	}
	out, in = weights[0].Shape[0], weights[0].Shape[1]
	if out <= 0 || in <= 0 || in%256 != 0 || out > math.MaxInt32/picks {
		return 0, 0, 0, fmt.Errorf("compute: V4.1 matvec id geometry [%d,%d] is not a k-quant row set", out, in)
	}
	if err := validateV41RoutedIDsTensor(ids, picks); err != nil {
		return 0, 0, 0, err
	}
	if x.Dtype != F32 || x.Quant != nil || len(x.Shape) != 1 {
		return 0, 0, 0, fmt.Errorf("compute: V4.1 matvec id activation must be an unquantized F32 vector")
	}
	switch x.Shape[0] {
	case in:
		xStride = 0
	case picks * in:
		xStride = in
	default:
		return 0, 0, 0, fmt.Errorf("compute: V4.1 matvec id activation width %d is neither %d nor %d", x.Shape[0], in, picks*in)
	}
	return out, in, xStride, nil
}

func validateV41RoutedIDsTensor(ids Tensor, picks int) error {
	if ids.Dtype != F32 || ids.Quant != nil || len(ids.Shape) != 1 || ids.Shape[0] != 3*picks {
		return fmt.Errorf("compute: V4.1 routed ids must be the [3*%d] word image V41RoutedIDs uploads", picks)
	}
	return nil
}

// validateV41RoutedRows checks a [P*count] F32 row set.
func validateV41RoutedRows(name string, t Tensor, picks int) (int, error) {
	if picks <= 0 || picks > V41RoutedIDMaxPicks {
		return 0, fmt.Errorf("compute: V4.1 routed id stage needs 1..%d picks", V41RoutedIDMaxPicks)
	}
	if t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || len(t.Shape) != 1 || t.Shape[0] <= 0 || t.Shape[0]%picks != 0 {
		return 0, fmt.Errorf("compute: V4.1 routed id %s must be an unquantized row-major F32 [picks*count] vector", name)
	}
	count := t.Shape[0] / picks
	if count > math.MaxInt32/picks {
		return 0, fmt.Errorf("compute: V4.1 routed id %s exceeds the device count domain", name)
	}
	return count, nil
}

// V41DeferredRecords is how many 16-byte records a deferred status owns.
func V41DeferredRecords(st V41DeferredStatus) int {
	if st.Records > 1 {
		return st.Records
	}
	return 1
}
