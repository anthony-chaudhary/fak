package model

// kQuantInt8PrefillRows bounds activation scratch independently of prompt length.
const kQuantInt8PrefillRows = 16

// kQuantMatRowsIntoBatchInt8 amortizes the ordinary int8 prefill worker barrier
// and reduction scratch across a bounded token panel. It reuses the exact GEMV
// reducers/combines, not the different deferred-reduction Q8 GEMM arithmetic.
func kQuantMatRowsIntoBatchInt8(qt *kQuantTensor, X []float32, P int, Y []float32, workers int) bool {
	if P <= 1 || qt.out <= 0 || qt.in <= 0 || qt.nblk <= 0 || qt.w3MLP ||
		(qt.numaPool != nil && qt.numaPool.Schedule().Eligible) {
		return false
	}
	var reduce func([]byte, int, []int8, []int32, []int32)
	var combine func([]byte, int, []float32, []int32, []int32) float32
	groups := 8
	switch qt.kind {
	case kindQ5K:
		reduce, combine = q5kReduceRow, kQuantCombineRow
	case kindQ6K:
		reduce, combine, groups = q6kReduceRow, q6kCombineRow, q6kGroupsPerBlock
	case kindQ2K:
		reduce, combine, groups = q2kReduceRow, q2kCombineRow, q2kGroupsPerBlock
	default:
		return false
	}
	in, out := qt.in, qt.out
	rowBytes := qt.rowBytes()
	var panel q8Panel
	for first := 0; first < P; first += kQuantInt8PrefillRows {
		rows := min(kQuantInt8PrefillRows, P-first)
		// The panel uses the same quantizeRowQ8 as quantizeVecQ8, without the
		// Q16 view that these byte-code reducers never consume.
		quantizeBatchPanelInto(&panel, X[first*in:(first+rows)*in], rows, in)
		dst := Y[first*out : (first+rows)*out]
		parForRangeWorkers(out, out*in*rows, workers, func(lo, hi int) {
			IS := make([]int32, qt.nblk*groups)
			SS := make([]int32, qt.nblk*groups)
			for o := lo; o < hi; o++ {
				row := qt.raw[o*rowBytes : (o+1)*rowBytes]
				for t := 0; t < rows; t++ {
					qx := panel.q[t*in : (t+1)*in]
					dx := panel.d[t*panel.nblk : (t+1)*panel.nblk]
					reduce(row, qt.nblk, qx, IS, SS)
					dst[t*out+o] = combine(row, qt.nblk, dx, IS, SS)
				}
			}
		})
		// The synchronous row-worker barrier completes before panel reuse.
	}
	return true
}
