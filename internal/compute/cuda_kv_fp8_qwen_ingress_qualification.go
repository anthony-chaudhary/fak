//go:build cuda && cgo && fak_cuda_fp8_kv_qualification

package compute

/*
#include "cuda_kv_fp8_qwen_ingress.h"
#include "cuda_kv_fp8_attention_owner.h"
*/
import "C"

import "math"

// Pure metadata preflight precedes native admission/allocation/tracing. Actual
// native allocation class, exact extent, disjointness and producer calls are
// checked again by the bounded wrappers. No ordinary public CUDA op is invoked.
func (a *cudaKVFP8Adapter) qwenIngressLocked(tx *kvFP8AppendTransaction, layer int,
	query, key, value, queryGain, keyGain Tensor, epsilon, scale float32,
	peakBudget int64, use func(kvFP8AttentionOutput) int) error {
	if tx == nil || !tx.active || !a.owns(tx.owner) || a.be != cudaDev ||
		a.be.capturing || a.identity.stopped.Load() {
		return errKVFP8QwenIngress
	}
	if layer < 0 || layer >= tx.owner.layers {
		return tx.uncertain(errKVFP8QwenIngress, 0)
	}
	qbytes, err := kvFP8AttentionBytes(tx.tokens, 24, tx.owner.heads, tx.owner.headDim)
	if err != nil {
		return tx.uncertain(err, 0)
	}
	sources := [5]Tensor{query, key, value, queryGain, keyGain}
	input := kvFP8QwenInput{backend: &a.identity, device: 0, heads: 24, root: sources}
	bytes := [5]int{qbytes, qbytes / 6, qbytes / 6, 256 * 4, 256 * 4}
	widths := [5]int{24 * 256, 4 * 256, 4 * 256, 256, 256}
	for i, tensor := range sources {
		b, ok := tensor.buf.(*cudaBuf)
		shape := i < 3 && len(tensor.Shape) == 2 && tensor.Shape[0] == tx.tokens && tensor.Shape[1] == widths[i]
		if (i >= 3 || tx.tokens == 1) && len(tensor.Shape) == 1 && tensor.Shape[0] == widths[i] {
			shape = true
		}
		if tensor.be != a.be || tensor.Dtype != F32 || tensor.Layout != RowMajor || tensor.Quant != nil ||
			!shape || !ok || b == nil || b.ptr == nil || b.n != bytes[i] || b.managed || b.device != 0 ||
			b.scales != nil || b.scalesN != 0 || b.graphOncePinned || b.invalidForSubmit() ||
			(b.be != nil && b.be != a.be) || (i >= 3 && b.class != MemoryWeights) {
			return tx.uncertain(errKVFP8QwenIngress, 0)
		}
		input.operands[i] = kvFP8Allocation{ptr: b.ptr, bytes: b.n}
	}
	if err := a.admitLocked(); err != nil {
		return tx.uncertain(err, 0)
	}
	convert := func(s kvFP8QwenStage) C.fcuda_kv_fp8_qwen_stage {
		var n C.fcuda_kv_fp8_qwen_stage
		for i, b := range s.input.operands {
			n.input[i] = (*C.float)(b.ptr)
			n.input_bytes[i] = C.size_t(b.bytes)
		}
		for i, b := range s.work {
			n.work[i] = (*C.float)(b.ptr)
			n.work_bytes[i] = C.size_t(b.bytes)
		}
		n.status = (*C.int)(s.status.ptr)
		n.tokens, n.first, n.heads = C.int(s.tokens), C.int(s.first), C.int(s.input.heads)
		n.rotary, n.producer = C.int(s.contract.rotaryDim), C.int(s.contract.producer)
		n.epsilon, n.theta = C.float(s.epsilon), C.double(math.Float64frombits(s.contract.thetaBits))
		return n
	}
	observed := func(r C.fcuda_kv_fp8_lifecycle_result) int {
		a.observe(r)
		return a.selectedStatus(int(r.status), int(r.phase))
	}
	ops := kvFP8QwenOps{
		admit:    func() int { return observed(C.fcuda_kv_fp8_qwen_admit_owned()) },
		validate: func(s kvFP8QwenStage) int { n := convert(s); return observed(C.fcuda_kv_fp8_qwen_validate_owned(&n)) },
		norm:     func(s kvFP8QwenStage) int { n := convert(s); return observed(C.fcuda_kv_fp8_qwen_norm_owned(&n)) },
		rope:     func(s kvFP8QwenStage) int { n := convert(s); return observed(C.fcuda_kv_fp8_qwen_rope_owned(&n)) },
		attention: func(k kvFP8AppendLayer, q, out kvFP8Allocation, heads int, scale float32) int {
			var planes C.fcuda_kv_fp8_append_destination
			for i, p := range k.planes {
				planes.plane[i] = p.ptr
				planes.bytes[i] = C.size_t(p.bytes)
			}
			return observed(C.fcuda_kv_fp8_attention_owned(&planes, (*C.float)(q.ptr), C.size_t(q.bytes),
				(*C.float)(out.ptr), C.size_t(out.bytes), C.int(k.tokens), C.int(k.prefix),
				C.int(heads), C.float(scale), (*C.int)(k.status.ptr)))
		},
	}
	return kvFP8QwenIngressEager(tx, layer, input, epsilon, scale, peakBudget, ops, use)
}
