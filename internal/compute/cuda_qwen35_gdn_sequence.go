//go:build cuda

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import "sync/atomic"

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
	in, _, keyDim, valueDim, convDim, operands, err := qwen35GDNEntry(normalizedInput, row, inProjQKV, inProjZ, inProjB, inProjA, conv1D, aLog, dtBias, norm, outProj, convState, recurrentState, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon)
	if err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if err := c.validateQwen35GDNOperands(operands, convState, recurrentState); err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	allocations := qwen35GDNAllocations("", "_", 0, tokens, hidden, keyDim, valueDim, numValueHeads, convDim)
	tensors, buffers, allocErr := c.allocateQwen35GDN(allocations, "qwen35-gdn-sequence-")
	if allocErr != nil {
		return Tensor{}, Tensor{}, Tensor{}, allocErr
	}
	mixed, z, b, a := tensors[0], tensors[1], tensors[2], tensors[3]
	convOut, qNorm, kNorm, core := tensors[4], tensors[5], tensors[6], tensors[7]
	output = tensors[8]
	qkvPtr, qkvScale, qkvQ8 := c.qwen35GDNQ8Args(inProjQKV)
	zPtr, zScale, zQ8 := c.qwen35GDNQ8Args(inProjZ)
	bPtr, bScale, bQ8 := c.qwen35GDNQ8Args(inProjB)
	aPtr, aScale, aQ8 := c.qwen35GDNQ8Args(inProjA)
	outPtr, outScale, outQ8 := c.qwen35GDNQ8Args(outProj)
	status := int(C.fcuda_qwen35_gdn_sequence_f32(c.cf(normalizedInput), C.int(tokens), qkvPtr, qkvScale, qkvQ8, zPtr, zScale, zQ8, bPtr, bScale, bQ8, aPtr, aScale, aQ8, c.cf(conv1D), c.cf(aLog), c.cf(dtBias), c.cf(norm), outPtr, outScale, outQ8, c.cf(convState), c.cf(recurrentState), c.cf(output), c.cf(mixed), c.cf(z), c.cf(b), c.cf(a), c.cf(convOut), c.cf(qNorm), c.cf(kNorm), c.cf(core), C.int(hidden), C.int(numKeyHeads), C.int(numValueHeads), C.int(keyHeadDim), C.int(valueHeadDim), C.int(convKernel), C.float(rmsNormEpsilon)))
	if status != 0 {
		kernelErr := c.failQwen35GDN(buffers, convState, recurrentState, status, "qwen35-gdn-sequence")
		return Tensor{}, Tensor{}, Tensor{}, kernelErr
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return output, convState, recurrentState, nil
}
