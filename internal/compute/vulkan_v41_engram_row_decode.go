//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "unsafe"

var _ V41EngramRowDecodeBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41EngramRowDecode() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_engram_row_decode() != 0
}

// V41EngramRowDecode copies the encoded rows to a private device buffer, decodes
// them and fences the dispatch through the checked V4.1 completion path before
// publishing the panel and status. The encoded scratch never enters Go's
// transient pool, and no public Read/Free/Upload is called under vulkanMu.
func (v *vulkanBackend) V41EngramRowDecode(encoded []byte, encoding V41EngramRowEncoding, rows, dim int) (panel, status Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var srcBuf, panelBuf, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41EngramRowDecode")
		}
		for _, b := range []*vulkanBuf{srcBuf, panelBuf, statusBuf} {
			if b != nil && b.ptr != nil && (b == srcBuf || err != nil) {
				C.fvk_free(b.ptr)
				b.ptr = nil
				b.n = 0
			}
		}
		if err != nil {
			panel, status = Tensor{}, Tensor{}
		}
	}()
	if v == nil {
		return Tensor{}, Tensor{}, vulkanV41EngramRowDecodeStatusError(3, "Vulkan backend is nil")
	}
	if validationErr := validateV41EngramRowDecode(encoded, encoding, rows, dim); validationErr != nil {
		return Tensor{}, Tensor{}, vulkanV41EngramRowDecodeStatusError(2, validationErr.Error())
	}
	if C.fvk_have_v41_engram_row_decode() == 0 {
		return Tensor{}, Tensor{}, vulkanV41EngramRowDecodeStatusError(3, "Engram row decode module or binary32 float controls unavailable")
	}
	if code := int(C.fvk_submission_status()); code != 0 {
		return Tensor{}, Tensor{}, vulkanV41EngramRowDecodeStatusError(code, "sticky submission fault")
	}
	srcBuf = v.dallocForClass(len(encoded), MemoryActivation, "V4.1 Engram encoded rows")
	C.fvk_h2d(srcBuf.ptr, unsafe.Pointer(&encoded[0]), C.size_t(len(encoded)))
	panelBuf = v.dallocForClass(rows*dim*4, MemoryActivation, "V4.1 Engram decoded rows")
	statusBuf = v.dallocForClass(rows*4, MemoryActivation, "V4.1 Engram row decode status")
	code := int(C.fvk_v41_engram_row_decode(srcBuf.ptr, panelBuf.ptr, statusBuf.ptr, C.int(encoding), C.int(rows), C.int(dim)))
	if code != 0 {
		return Tensor{}, Tensor{}, vulkanV41EngramRowDecodeStatusError(code, "native dispatch refused")
	}
	// Fence before the encoded scratch is released and before publishing.
	if code = int(C.fvk_batch_flush_status()); code != 0 {
		return Tensor{}, Tensor{}, vulkanV41EngramRowDecodeStatusError(code, "checked completion failed")
	}
	panelBuf.v41CheckedRead = true
	statusBuf.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{rows, dim}, nil, panelBuf),
		makeTensor(v, F32, RowMajor, []int{rows}, nil, statusBuf), nil
}

// V41EngramRowDecodePoison marks target in place; it allocates nothing.
func (v *vulkanBackend) V41EngramRowDecodePoison(target, status Tensor, rows, index int) (err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41EngramRowDecodePoison")
		}
	}()
	if v == nil {
		return vulkanV41EngramRowDecodeStatusError(3, "Vulkan backend is nil")
	}
	if validationErr := validateV41EngramRowPoison(target, status, rows, index); validationErr != nil {
		return vulkanV41EngramRowDecodeStatusError(2, validationErr.Error())
	}
	tb, okTarget := target.buf.(*vulkanBuf)
	sb, okStatus := status.buf.(*vulkanBuf)
	if target.be != v || status.be != v || !okTarget || !okStatus || tb == nil || sb == nil || tb.ptr == nil || sb.ptr == nil ||
		tb.n < target.Numel()*4 || sb.n < rows*4 || tb.scalePtr != nil || len(tb.q8Chunks) != 0 {
		return vulkanV41EngramRowDecodeStatusError(2, "target and status must be live, full-sized F32 device buffers owned by this backend")
	}
	if C.fvk_have_v41_engram_row_decode() == 0 {
		return vulkanV41EngramRowDecodeStatusError(3, "Engram row decode module or binary32 float controls unavailable")
	}
	if code := int(C.fvk_submission_status()); code != 0 {
		return vulkanV41EngramRowDecodeStatusError(code, "sticky submission fault")
	}
	if code := int(C.fvk_v41_engram_row_poison(sb.ptr, tb.ptr, C.int(rows), C.int(target.Numel()), C.int(index))); code != 0 {
		return vulkanV41EngramRowDecodeStatusError(code, "native dispatch refused")
	}
	if code := int(C.fvk_batch_flush_status()); code != 0 {
		return vulkanV41EngramRowDecodeStatusError(code, "checked completion failed")
	}
	return nil
}
