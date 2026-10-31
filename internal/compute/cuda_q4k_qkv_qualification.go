//go:build cuda && cgo && fak_cuda_q4k_qkv_qualification && fak_cuda_v41_index_select && !fak_cuda_q4k_qkv_async

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
)

// Q4KQKVSharedInputQualification is available only in an explicit qualification
// build, with the existing checked-stream archive enabled. It is not an ordinary
// serving capability. False/nil means declined before candidate submission;
// every selected error is terminal for this forward and must never be retried.
// The checked native operation fences before scratch retirement and publication.
func (c *cudaBackend) Q4KQKVSharedInputQualification(wq, wk, wv, x Tensor) (q, k, v Tensor, used bool, err error) {
	const site = "q4k-qkv-qualification"
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var buffers []*cudaBuf
	admitted := false
	defer func() {
		if recovered := recover(); recovered != nil {
			err, _ = ConvertCUDAPanic(recovered, site, "qualification")
			used = true
		}
		if err != nil {
			if admitted {
				if code := int(C.fcuda_q4k_qkv_qualification_finish()); code != 0 {
					err = fmt.Errorf("%s: checked drain %d; native allocations retained: %w", site, code, err)
				}
			}
			for _, b := range buffers {
				atomic.StoreUint32(&b.invalid, 1)
			}
			// A failed finish closes the existing native free/allocator seam;
			// this detaches Go transients but cannot recycle uncertain storage.
			c.releaseTransientBuffers(buffers)
			q, k, v = Tensor{}, Tensor{}, Tensor{}
			if c != nil {
				c.faultLatch.Observe(DeviceFaultExecution, site, 1)
			}
		}
		runtime.KeepAlive([4]Tensor{wq, wk, wv, x})
	}()
	if c == nil || c != cudaDev {
		return q, k, v, true, fmt.Errorf("%s: backend is not the registered CUDA owner", site)
	}
	if err := c.faultLatch.Admit(site); err != nil {
		return q, k, v, true, err
	}
	if c.capturing {
		return q, k, v, false, nil
	}
	// Keep the explicit F32 escape hatch even in a qualification build. The
	// getter follows applyCUDAQ4KDecodeArm's cudaMu -> arm mutex order.
	if CUDAQ4KDecodeArmInUse() != CUDAQ4KDecodeMMVQ {
		return q, k, v, false, nil
	}
	if wq.Dtype != Q4_K || wk.Dtype != Q4_K || wv.Dtype != Q4_K {
		return q, k, v, false, nil
	}
	if x.Dtype != F32 || x.Layout != RowMajor || len(x.Shape) != 1 || x.Shape[0] <= 0 || x.Shape[0] > math.MaxInt32-255 || x.Shape[0]%256 != 0 {
		return q, k, v, false, nil
	}
	in := x.Shape[0]
	weights := [3]Tensor{wq, wk, wv}
	widths := [3]int{}
	operands := [4]*cudaBuf{}
	check := func(t Tensor, bytes int) (*cudaBuf, error) {
		b, ok := t.buf.(*cudaBuf)
		if t.Backend() != c || !ok || b == nil || b.ptr == nil || b.device != 0 || b.managed ||
			b.n != bytes || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
			return nil, fmt.Errorf("%s: operand is not exact live device storage owned by this backend", site)
		}
		if err := b.invalidStateError(site); err != nil {
			return nil, err
		}
		return b, nil
	}
	for i, w := range weights {
		if w.Layout != RowMajor || len(w.Shape) != 2 || w.Shape[1] != in || w.Shape[0] <= 0 || w.Shape[0] > math.MaxInt32/(in/256) ||
			w.Quant == nil || w.Quant.Block != 256 || w.Quant.Bits != 4 || w.Quant.Axis != 2 {
			return q, k, v, false, nil
		}
		widths[i] = w.Shape[0]
		blocks, ok := fp8Block32Product(w.Shape[0], in/256)
		if !ok {
			return q, k, v, false, nil
		}
		bytes, ok := fp8Block32Product(blocks, 144)
		if !ok {
			return q, k, v, false, nil
		}
		if operands[i], err = check(w, bytes); err != nil {
			return q, k, v, true, err
		}
	}
	xBytes, ok := fp8Block32Product(in, 4)
	if !ok {
		return q, k, v, false, nil
	}
	if operands[3], err = check(x, xBytes); err != nil {
		return q, k, v, true, err
	}
	status := int(C.fcuda_q4k_qkv_qualification_state())
	if status == 80006 {
		return q, k, v, false, nil
	}
	if status != 0 {
		return q, k, v, true, fmt.Errorf("%s: native admission status %d", site, status)
	}
	admitted, used = true, true
	outputs := [3]Tensor{}
	for i, width := range widths {
		var b *cudaBuf
		outputs[i], b, err = c.devTrDeviceOnly([]int{width}, F32, site+"-output")
		if err != nil {
			return q, k, v, true, err
		}
		buffers = append(buffers, b)
	}
	// Nine F32 elements have the same byte size and alignment as one q8_1 block;
	// this transient is opaque workspace, never an activation or model output.
	_, scratch, allocErr := c.devTrDeviceOnly([]int{in / 32, 9}, F32, site+"-q8_1-scratch")
	if allocErr != nil {
		return q, k, v, true, allocErr
	}
	buffers = append(buffers, scratch)
	status = int(C.fcuda_q4k_qkv_qualification_f32(
		(*C.uint8_t)(operands[0].ptr), (*C.uint8_t)(operands[1].ptr), (*C.uint8_t)(operands[2].ptr), (*C.float)(operands[3].ptr),
		(*C.float)(buffers[0].ptr), (*C.float)(buffers[1].ptr), (*C.float)(buffers[2].ptr),
		C.int(widths[0]), C.int(widths[1]), C.int(widths[2]), C.int(in), scratch.ptr, C.size_t(scratch.n)))
	if status != 0 {
		// Even a native geometry refusal after selected allocation is an error,
		// never a silent retry after potentially enqueued dependencies.
		return q, k, v, true, fmt.Errorf("%s: selected operation status %d", site, status)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	c.releaseTransientBuffers([]*cudaBuf{scratch})
	return outputs[0], outputs[1], outputs[2], true, nil
}

// Q4KQKVSharedInputQualificationOperations counts checked completed candidate
// calls, not generic MMVQ launches, and makes a device qualification observable.
func (c *cudaBackend) Q4KQKVSharedInputQualificationOperations() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_q4k_qkv_qualification_operations())
}
