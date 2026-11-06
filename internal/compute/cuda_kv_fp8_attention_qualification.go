//go:build cuda && cgo && fak_cuda_fp8_kv_qualification

package compute

/*
#include "cuda_kv_fp8_attention_owner.h"
*/
import "C"

// stagedAttentionLocked is uncalled qualification code. Its output exists only
// inside the checked use callback; it is never registered as a public transient
// or handed to generic Free. No public Tensor/Buffer exposure is added here.
func (a *cudaKVFP8Adapter) stagedAttentionLocked(tx *kvFP8AppendTransaction, layer int,
	query Tensor, heads int, scale float32, peakBudget int64, use func(kvFP8AttentionOutput) int) error {
	if tx == nil || !tx.active || !a.owns(tx.owner) || a.be != cudaDev || a.be.capturing ||
		a.identity.stopped.Load() {
		return errKVFP8Attention
	}
	bytes, err := kvFP8AttentionBytes(tx.tokens, heads, tx.owner.heads, tx.owner.headDim)
	if err != nil {
		return tx.uncertain(err, 0)
	}
	b, ok := query.buf.(*cudaBuf)
	width := heads * tx.owner.headDim
	shape := len(query.Shape) == 2 && query.Shape[0] == tx.tokens && query.Shape[1] == width
	if tx.tokens == 1 && len(query.Shape) == 1 && query.Shape[0] == width {
		shape = true
	}
	// This pure typed gate precedes native admission, tracing and allocation.
	if query.be != a.be || query.Dtype != F32 || query.Layout != RowMajor || query.Quant != nil ||
		!shape || !ok || b == nil || b.ptr == nil || b.n != bytes || b.managed || b.device != 0 ||
		b.scales != nil || b.scalesN != 0 || b.graphOncePinned || b.invalidForSubmit() ||
		(b.be != nil && b.be != a.be) {
		return tx.uncertain(errKVFP8Attention, 0)
	}
	if err := a.admitLocked(); err != nil {
		return tx.uncertain(err, 0)
	}
	q := kvFP8AttentionQuery{backend: &a.identity, device: 0, storage: kvFP8Allocation{ptr: b.ptr, bytes: b.n}, root: query}
	return kvFP8AttendEager(tx, layer, q, heads, scale, peakBudget,
		func(k kvFP8AppendLayer, q, out kvFP8Allocation, heads int, scale float32) int {
			var planes C.fcuda_kv_fp8_append_destination
			for i, p := range k.planes {
				planes.plane[i] = p.ptr
				planes.bytes[i] = C.size_t(p.bytes)
			}
			r := C.fcuda_kv_fp8_attention_owned(&planes, (*C.float)(q.ptr), C.size_t(q.bytes),
				(*C.float)(out.ptr), C.size_t(out.bytes), C.int(k.tokens), C.int(k.prefix),
				C.int(heads), C.float(scale), (*C.int)(k.status.ptr))
			a.observe(r)
			return a.selectedStatus(int(r.status), int(C.FCUDA_KV_FP8_PHASE_ATTENTION))
		}, use)
}
