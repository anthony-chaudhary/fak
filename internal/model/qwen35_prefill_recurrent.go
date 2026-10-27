package model

import "os"

// qwen35PrefillDecayRead fuses two sweeps of one privately owned recurrent
// head: decay, then the ordered key/state contraction. The caller supplies
// disjoint state, key and scratch storage, with no intervening observer. It
// owns the same head for all prompt positions and joins before publication.
// No work is dispatched here; this leaf is called inside the head worker.
//
// Only exact state and scratch shapes are admitted. The original decay loop
// visits even surplus state cells before clearing scratch and reading any key;
// refusing all other geometry without writes lets the caller retain that
// behavior (including partial mutation before a bounds panic).
func qwen35PrefillDecayRead(st, kn, kvmem []float32, g float32, kHd, vHd int) bool {
	if kHd <= 0 || vHd <= 0 || kHd > int(^uint(0)>>1)/vHd ||
		len(st) != kHd*vHd || len(kn) < kHd || len(kvmem) != vHd {
		return false
	}
	clear(kvmem)
	for i := 0; i < kHd; i++ {
		ki := kn[i]
		base := i * vHd
		for d := 0; d < vHd; d++ {
			// This explicit conversion retains the original float32 decay
			// rounding BEFORE the separately fused retrieval operation. An
			// assignment alone must not authorize contraction across it.
			scaled := float32(st[base+d] * g)
			st[base+d] = scaled
			kvmem[d] = gdnFMA32(scaled, ki, kvmem[d])
		}
	}
	return true
}

// qwen35PrefillVectorizedStep reuses the existing canonical decode transition
// only after the ordinary caller has selected FAK_VECTORIZED_DELTANET=1. This
// is not a new kernel or a default precision-policy promotion. In particular,
// scalar float64-FMA followed by float32 conversion and hardware float32 FMA
// can differ even for finite operands; their physical numeric gate is pending.
// State, normalized inputs, output and scratch are caller-owned disjoint rows.
//
// Every consumed extent and executable capability is admitted before writes.
// The output slice used to be formed after decay/retrieval, so an invalid output
// extent must decline rather than panic early. Once selected, a kernel panic
// propagates; never replay the scalar step over partially advanced state.
func qwen35PrefillVectorizedStep(st, qn, kn, vh, core, kvmem, delta []float32, bt, g float32, token, rowWidth, head int) bool {
	const dim = 128 // exact shape supported by HasVectorizedDeltaNetFor
	if len(st) != dim*dim || len(qn) != dim || len(kn) != dim || len(vh) != dim ||
		len(kvmem) != dim || len(delta) != dim || token < 0 || head < 0 || rowWidth < dim {
		return false
	}
	rowStart, rowOK := checkedMulInt(token, rowWidth)
	headStart, headOK := checkedMulInt(head, dim)
	if !rowOK || !headOK || headStart > rowWidth-dim || len(core) < dim ||
		headStart > len(core)-dim || rowStart > len(core)-dim-headStart {
		return false
	}
	if os.Getenv("FAK_VECTORIZED_DELTANET") != "1" || !HasVectorizedDeltaNetFor(dim, dim) {
		return false
	}
	start := rowStart + headStart
	VectorizedHeadStep(st, qn, kn, vh, bt, g, core[start:start+dim:start+dim], kvmem, delta)
	return true
}
