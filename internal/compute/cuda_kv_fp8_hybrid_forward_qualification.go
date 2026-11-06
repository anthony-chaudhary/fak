//go:build cuda && cgo && fak_cuda_fp8_kv_qualification

package compute

/*
#include "cuda_kv_fp8_qwen_ingress.h"
*/
import "C"

import "math"

// This adapter binds the ACTUAL GDN invalid bits, not copies of their flags.
// It is not called from Qwen35SequencePrefill or model and adds no capability.
// Caller already holds ForwardExecutionOwner/cudaMu and must separately bind
// all model weights, locked callbacks and prospective result lifetimes.
func (a *cudaKVFP8Adapter) hybridForwardLocked(o *kvFP8Owner, req Qwen35SequencePrefillRequest,
	contract kvFP8HybridContract, peakBudget int64,
	steps []func(*kvFP8AppendTransaction, int) error, tail func(*kvFP8AppendTransaction) error) error {
	if !a.owns(o) || a.be != cudaDev || a.be.capturing || a.be.graphOnce != nil || cudaGraphOnceOwner != nil ||
		req.KV != nil || req.CapturePrefixReplay || req.CaptureRawHidden || len(req.FeatureLayers) != 0 {
		return errKVFP8Hybrid
	}
	if err := validateQwen35SequenceGeometry(req); err != nil {
		return err
	}
	if o.layers > int(^uint(0)>>1)/6 || req.NumHeads != 24 || req.NumKVHeads != 4 || req.HeadDim != 256 || req.RotaryDim != 64 ||
		req.ConvKernel < 2 || len(req.Layers)/4 != o.layers || contract.first != req.StartPos ||
		contract.tokens != len(req.TokenIDs) || len(contract.linear) != len(req.Layers) ||
		len(contract.theta) != len(req.Layers) {
		return errKVFP8Hybrid
	}
	max := int64(math.MaxInt32)
	key, ok := qwen35GDNCheckedMul(max, int64(req.NumKeyHeads), int64(req.KeyHeadDim))
	if !ok {
		return errKVFP8Hybrid
	}
	value, ok := qwen35GDNCheckedMul(max, int64(req.NumValueHeads), int64(req.ValueHeadDim))
	if !ok || key > (max-value)/2 {
		return errKVFP8Hybrid
	}
	conv := 2*key + value
	convElements, ok := qwen35GDNCheckedMul(max, int64(req.ConvKernel-1), conv)
	if !ok {
		return errKVFP8Hybrid
	}
	recurrent, ok := qwen35GDNCheckedMul(max, int64(req.NumValueHeads), int64(req.KeyHeadDim), int64(req.ValueHeadDim))
	if !ok {
		return errKVFP8Hybrid
	}
	if convElements > int64(int(^uint(0)>>1)/4) || recurrent > int64(int(^uint(0)>>1)/4) {
		return errKVFP8Hybrid
	}
	states := make([]kvFP8HybridState, 0, 6*o.layers)
	for main, layer := range req.Layers {
		if layer.Linear != contract.linear[main] || math.Float64bits(req.RoPEThetaForLayer[main]) != contract.theta[main] {
			return errKVFP8Hybrid
		}
		if !layer.Linear {
			// The next Qwen producer leaf always normalizes; an absent gain is not an
			// identity-norm mode and must be refused before outer append allocation.
			if layer.QNorm.buf == nil || layer.KNorm.buf == nil || req.States[main].Conv.buf != nil || req.States[main].Recurrent.buf != nil {
				return errKVFP8Hybrid
			}
			continue
		}
		for kind, tensor := range [2]Tensor{req.States[main].Conv, req.States[main].Recurrent} {
			shape := []int{req.ConvKernel - 1, int(conv)}
			bytes := int(convElements) * 4
			if kind == 1 {
				shape = []int{req.NumValueHeads, req.KeyHeadDim, req.ValueHeadDim}
				bytes = int(recurrent) * 4
			}
			b, ok := tensor.buf.(*cudaBuf)
			if tensor.be != a.be || tensor.Dtype != F32 || tensor.Layout != RowMajor || tensor.Quant != nil ||
				!qwen35SequenceSameShape(tensor, shape...) || !ok || b == nil || b.ptr == nil || b.n != bytes ||
				b.class != MemoryKVCache || b.device != 0 || b.managed || b.scales != nil || b.scalesN != 0 ||
				b.graphOncePinned || b.invalidForSubmit() || (b.be != nil && b.be != a.be) {
				return errKVFP8Hybrid
			}
			states = append(states, kvFP8HybridState{backend: &a.identity, device: 0, mainLayer: main, kind: kind,
				storage: kvFP8Allocation{ptr: b.ptr, bytes: b.n}, invalid: &b.invalid, root: tensor})
		}
	}
	// This probe runs BEFORE kvFP8AppendEager reserves its replacement image.
	// Unlike a selected producer failure, an unsupported timing mode here does not
	// poison healthy KV/GDN states. Existing terminal results still stop identity.
	admit := func() error {
		if err := a.admitLocked(); err != nil {
			return err
		}
		r := C.fcuda_kv_fp8_qwen_admit_owned()
		a.observe(r)
		if r.status != 0 {
			return &kvFP8ReceiptError{cause: errKVFP8Hybrid, status: int(r.status)}
		}
		return nil
	}
	return kvFP8HybridForwardEager(o, contract, states, peakBudget, a.appendOperations(), admit, steps, tail)
}
