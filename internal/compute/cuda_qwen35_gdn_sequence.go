//go:build cuda

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"sync/atomic"
	"unsafe"
)

// Sequence algorithm adapted from llama.cpp gated_delta_net.cu at
// 0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd (MIT).
func (c *cudaBackend) Qwen35GDNSequence(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState Tensor, err error) {
	if err := c.faultLatch.Admit("qwen35-gdn-sequence"); err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	if len(normalizedInput.Shape) != 2 || normalizedInput.Shape[0] <= 0 {
		return Tensor{}, Tensor{}, Tensor{}, &Qwen35GDNGeometryError{Operand: "normalized_input", Got: normalizedInput.Shape, Want: "[tokens>0,hidden]"}
	}
	tokens, hidden := normalizedInput.Shape[0], normalizedInput.Shape[1]
	row := normalizedInput
	row.Shape = []int{hidden}
	_, keyDim, valueDim, convDim, err := validateQwen35GDNGeometry(row, inProjQKV, inProjZ, inProjB, inProjA, conv1D, aLog, dtBias, norm, outProj, convState, recurrentState, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon)
	if err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	operands := []struct {
		name string
		t    Tensor
	}{{"normalized_input", normalizedInput}, {"in_proj_qkv", inProjQKV}, {"in_proj_z", inProjZ}, {"in_proj_b", inProjB}, {"in_proj_a", inProjA}, {"conv1d", conv1D}, {"A_log", aLog}, {"dt_bias", dtBias}, {"norm", norm}, {"out_proj", outProj}, {"conv_state", convState}, {"recurrent_state", recurrentState}}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	for _, operand := range operands {
		if err := c.validateQwen35GDNTensor(operand.name, operand.t); err != nil {
			return Tensor{}, Tensor{}, Tensor{}, err
		}
	}
	if err := c.validateQwen35GDNStateOperands(operands, convState, recurrentState); err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	allocations := []struct {
		name  string
		shape []int
	}{{"mixed", []int{convDim}}, {"z", []int{valueDim}}, {"b", []int{numValueHeads}}, {"a", []int{numValueHeads}}, {"conv_out", []int{convDim}}, {"q_norm", []int{keyDim}}, {"k_norm", []int{keyDim}}, {"core", []int{valueDim}}, {"output", []int{tokens, hidden}}}
	tensors := make([]Tensor, 0, len(allocations))
	buffers := make([]*cudaBuf, 0, len(allocations))
	for _, allocation := range allocations {
		tensor, buffer, allocErr := c.devTrDeviceOnly(allocation.shape, F32, "qwen35-gdn-sequence-"+allocation.name)
		if allocErr != nil {
			c.releaseTransientBuffers(buffers)
			c.faultLatch.ObserveError(allocErr, "qwen35-gdn-sequence-"+allocation.name)
			return Tensor{}, Tensor{}, Tensor{}, allocErr
		}
		tensors = append(tensors, tensor)
		buffers = append(buffers, buffer)
	}
	mixed, z, b, a := tensors[0], tensors[1], tensors[2], tensors[3]
	convOut, qNorm, kNorm, core := tensors[4], tensors[5], tensors[6], tensors[7]
	output = tensors[8]
	q8Args := func(t Tensor) (unsafe.Pointer, *C.float, C.int) {
		buf := c.cudaBufForSubmit(t)
		if t.Dtype == Q8_0 {
			return buf.ptr, (*C.float)(buf.scales), 1
		}
		return buf.ptr, nil, 0
	}
	qkvPtr, qkvScale, qkvQ8 := q8Args(inProjQKV)
	zPtr, zScale, zQ8 := q8Args(inProjZ)
	bPtr, bScale, bQ8 := q8Args(inProjB)
	aPtr, aScale, aQ8 := q8Args(inProjA)
	outPtr, outScale, outQ8 := q8Args(outProj)
	status := int(C.fcuda_qwen35_gdn_sequence_f32(c.cf(normalizedInput), C.int(tokens), qkvPtr, qkvScale, qkvQ8, zPtr, zScale, zQ8, bPtr, bScale, bQ8, aPtr, aScale, aQ8, c.cf(conv1D), c.cf(aLog), c.cf(dtBias), c.cf(norm), outPtr, outScale, outQ8, c.cf(convState), c.cf(recurrentState), c.cf(output), c.cf(mixed), c.cf(z), c.cf(b), c.cf(a), c.cf(convOut), c.cf(qNorm), c.cf(kNorm), c.cf(core), C.int(hidden), C.int(numKeyHeads), C.int(numValueHeads), C.int(keyHeadDim), C.int(valueHeadDim), C.int(convKernel), C.float(rmsNormEpsilon)))
	if status != 0 {
		for _, buffer := range buffers {
			atomic.StoreUint32(&buffer.invalid, 1)
		}
		atomic.StoreUint32(&convState.buf.(*cudaBuf).invalid, 1)
		atomic.StoreUint32(&recurrentState.buf.(*cudaBuf).invalid, 1)
		c.releaseTransientBuffers(buffers)
		kernelErr := &Qwen35GDNKernelError{Stage: qwen35GDNKernelStage(status), Code: status}
		c.faultLatch.ObserveError(kernelErr, "qwen35-gdn-sequence")
		return Tensor{}, Tensor{}, Tensor{}, kernelErr
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return output, convState, recurrentState, nil
}
