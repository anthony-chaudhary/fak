//go:build cuda && cgo && fak_cuda_fp8_kv_qualification

package compute

/*
#include "cuda_kv_fp8_append.h"
*/
import "C"

// inputLocked is a pure metadata gate, before tracing, allocation or native
// submission. It does NOT prove the producer's normalization/rotary arithmetic.
// The future Qwen binding must supply only the exact named producer at these
// canonical positions, under its live execution owner and cudaMu.
func (a *cudaKVFP8Adapter) inputLocked(o *kvFP8Owner, layer, first, tokens int,
	raw, key, value Tensor) (kvFP8AppendInput, error) {
	var input kvFP8AppendInput
	if !a.owns(o) || a.be != cudaDev || a.be.capturing || a.identity.stopped.Load() ||
		layer < 0 || layer >= o.layers || first < 0 || tokens <= 0 ||
		first != o.image.positions {
		return input, errKVFP8Append
	}
	sizes, _, err := o.planeBytes(tokens)
	if err != nil {
		return input, err
	}
	operands := [3]Tensor{raw, key, value}
	var allocations [3]kvFP8Allocation
	for i, t := range operands {
		b, ok := t.buf.(*cudaBuf)
		shapeOK := len(t.Shape) == 2 && t.Shape[0] == tokens && t.Shape[1] == o.heads*o.headDim
		if tokens == 1 && len(t.Shape) == 1 && t.Shape[0] == o.heads*o.headDim {
			shapeOK = true
		}
		if t.be != a.be || t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil ||
			!shapeOK || !ok || b == nil || b.ptr == nil || b.n != sizes[kvFP8Raw] ||
			b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 ||
			b.graphOncePinned || b.invalidForSubmit() || (b.be != nil && b.be != a.be) {
			return input, errKVFP8Append
		}
		allocations[i] = kvFP8Allocation{ptr: b.ptr, bytes: b.n}
	}
	return kvFP8AppendInput{backend: &a.identity, device: 0, first: first, tokens: tokens,
		contract: o.rotary[layer], raw: allocations[0], key: allocations[1],
		value: allocations[2], root: operands}, nil
}

func cudaKVFP8NativeAppendInput(in kvFP8AppendInput) C.fcuda_kv_fp8_append_input {
	return C.fcuda_kv_fp8_append_input{raw: (*C.float)(in.raw.ptr),
		key: (*C.float)(in.key.ptr), value: (*C.float)(in.value.ptr), bytes: C.size_t(in.raw.bytes)}
}

func (a *cudaKVFP8Adapter) appendOperations() kvFP8AppendOps {
	return kvFP8AppendOps{
		eager: a.operations(),
		validate: func(in kvFP8AppendInput, status kvFP8Allocation) int {
			native := cudaKVFP8NativeAppendInput(in)
			r := C.fcuda_kv_fp8_append_validate_owned(&native, C.int(in.raw.bytes/4), (*C.int)(status.ptr))
			a.observe(r)
			return a.selectedStatus(int(r.status), int(C.FCUDA_KV_FP8_PHASE_APPEND_VALIDATE))
		},
		ingest: func(dst [kvFP8Planes]kvFP8Allocation, prefix int, in kvFP8AppendInput, status kvFP8Allocation) int {
			native := cudaKVFP8NativeAppendInput(in)
			var destination C.fcuda_kv_fp8_append_destination
			for i, p := range dst {
				destination.plane[i] = p.ptr
				destination.bytes[i] = C.size_t(p.bytes)
			}
			r := C.fcuda_kv_fp8_append_ingest_owned(&destination, &native, C.int(prefix),
				C.int(in.tokens), C.int(4), C.int(in.contract.headDim), (*C.int)(status.ptr))
			a.observe(r)
			return a.selectedStatus(int(r.status), int(C.FCUDA_KV_FP8_PHASE_APPEND_INGEST))
		},
		stop: func(code int) { a.selectedStatus(code, int(C.FCUDA_KV_FP8_PHASE_APPEND_INGEST)) },
	}
}

// appendLocked is uncalled and qualification-only. It never acquires the
// non-reentrant ForwardExecutionOwner/cudaMu domain. Public model/serve routes,
// attention Q/output lifetime, complete opaque-view ingress, native fault
// bridging and build/device qualification remain separate activation gates.
func (a *cudaKVFP8Adapter) appendLocked(o *kvFP8Owner, first, tokens int, peakBudget int64,
	run func(*kvFP8AppendTransaction) error) error {
	if !a.owns(o) {
		return errKVFP8Owner
	}
	if err := o.usable(); err != nil {
		return err
	}
	if err := a.admitLocked(); err != nil {
		return err
	}
	return kvFP8AppendEager(o, first, tokens, peakBudget, a.appendOperations(), run)
}
