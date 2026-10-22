//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

var _ V41CompressorRingBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41CompressorRing() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_compressor_ring() != 0
}

// v41RingSubmissionStatus reads the sticky native submission status for the
// ring witnesses; a refused request must leave it clear.
func v41RingSubmissionStatus() int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return int(C.fvk_submission_status())
}

func (v *vulkanBackend) V41CompressorRingAlloc(ratio, width int) (kv, scores Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var bufs [2]*vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41CompressorRingAlloc")
		}
		if err != nil {
			for _, b := range bufs {
				if b != nil && b.ptr != nil {
					C.fvk_free(b.ptr)
					b.ptr, b.n = nil, 0
				}
			}
			kv, scores = Tensor{}, Tensor{}
		}
	}()
	if v == nil {
		return Tensor{}, Tensor{}, vulkanV41CompressorRingStatusError(3, "Vulkan backend is nil")
	}
	if verr := ValidateV41CompressorRingAlloc(ratio, width); verr != nil {
		return Tensor{}, Tensor{}, vulkanV41CompressorRingStatusError(2, verr.Error())
	}
	bufs[0] = v.dallocForClass(ratio*width*4, MemoryKVCache, "V4.1 compressor ring kv")
	bufs[1] = v.dallocForClass(ratio*width*4, MemoryKVCache, "V4.1 compressor ring scores")
	return makeTensor(v, F32, RowMajor, []int{ratio, width}, nil, bufs[0]),
		makeTensor(v, F32, RowMajor, []int{ratio, width}, nil, bufs[1]), nil
}

// V41CompressorRingStore only records. An emitting store's result is a checked
// read, so inside a V4.1 step batch the caller's one Read submits the dispatch
// and its readback together; V41CompressorPool flushes before its Read instead.
func (v *vulkanBackend) V41CompressorRingStore(kv, scores, kvRow, scoreRow Tensor, slot, ratio, width int, emit bool) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41CompressorRingStore")
		}
		if err != nil {
			if outBuf != nil && outBuf.ptr != nil {
				C.fvk_free(outBuf.ptr)
				outBuf.ptr, outBuf.n = nil, 0
			}
			out = Tensor{}
		}
	}()
	if v == nil {
		return Tensor{}, vulkanV41CompressorRingStatusError(3, "Vulkan backend is nil")
	}
	if verr := ValidateV41CompressorRingStore(kv, scores, kvRow, scoreRow, slot, ratio, width); verr != nil {
		return Tensor{}, vulkanV41CompressorRingStatusError(2, verr.Error())
	}
	var operands [4]*vulkanBuf
	for i, operand := range []Tensor{kv, scores, kvRow, scoreRow} {
		b, ok := operand.buf.(*vulkanBuf)
		if operand.be != v || !ok || b == nil || b.ptr == nil || b.n < operand.Numel()*4 || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, vulkanV41CompressorRingStatusError(2, fmt.Sprintf("operand %d must be a live F32 device buffer owned by this backend", i))
		}
		operands[i] = b
	}
	if C.fvk_have_v41_compressor_ring() == 0 {
		return Tensor{}, vulkanV41CompressorRingStatusError(3, "compressor ring module or binary32 float controls unavailable")
	}
	if code := int(C.fvk_submission_status()); code != 0 {
		return Tensor{}, vulkanV41CompressorRingStatusError(code, "sticky submission fault")
	}
	var outPtr = operands[2].ptr
	emitFlag := 0
	if emit {
		outBuf = v.dallocForClass(2*width*4, MemoryActivation, "V4.1 compressor ring output")
		outPtr, emitFlag = outBuf.ptr, 1
	}
	code := int(C.fvk_v41_compressor_ring_f32(operands[2].ptr, operands[3].ptr, operands[0].ptr, operands[1].ptr, outPtr,
		C.int(ratio), C.int(width), C.int(slot), C.int(emitFlag)))
	if code != 0 {
		return Tensor{}, vulkanV41CompressorRingStatusError(code, "native dispatch refused")
	}
	if !emit {
		return Tensor{}, nil
	}
	outBuf.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{2, width}, nil, outBuf), nil
}
