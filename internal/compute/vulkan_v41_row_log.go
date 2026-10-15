//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var _ V41RowLogBackend = (*vulkanBackend)(nil)

// Row-log copies ride the core transfer path, so they need no SPIR-V module.
func (v *vulkanBackend) V41RowLogAvailable() bool { return v != nil }

func (v *vulkanBackend) v41RowLogBuf(t Tensor) (*vulkanBuf, bool) {
	b, ok := t.buf.(*vulkanBuf)
	if t.be != v || !ok || b == nil || b.ptr == nil || b.scalePtr != nil || len(b.q8Chunks) != 0 || b.n < t.Numel()*4 {
		return nil, false
	}
	return b, true
}

func vulkanV41RowLogError(site string, err error) *BackendError {
	return &BackendError{Backend: "vulkan", Site: site, Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: err.Error()}
}

func vulkanV41RowLogStatus(site string, status int) error {
	if status == 0 {
		return nil
	}
	err := vulkanV41EmbeddingGatherStatusError(status)
	err.Site, err.Message = site, fmt.Sprintf("row-log copy failed closed (code %d)", status)
	return err
}

func (v *vulkanBackend) V41RowLogAlloc(rows, width int) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	defer func() {
		if recovered := recover(); recovered != nil {
			out, err = Tensor{}, ClassifyVulkanPanic(recovered, "V41RowLogAlloc")
		}
	}()
	if v == nil {
		return Tensor{}, vulkanV41RowLogStatus("V41RowLogAlloc", 3)
	}
	if err := ValidateV41RowLogAlloc(rows, width); err != nil {
		return Tensor{}, vulkanV41RowLogError("V41RowLogAlloc", err)
	}
	buf := v.dallocForClass(rows*width*4, MemoryKVCache, "V4.1 resident KV row log")
	return makeTensor(v, F32, RowMajor, []int{rows, width}, nil, buf), nil
}

// V41RowLogCopy records one buffer-to-buffer copy. Inside an open batch it is
// ordered after prior work by the batch barrier.
func (v *vulkanBackend) V41RowLogCopy(dst Tensor, dstRow int, src Tensor, srcRow, rows int) (err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41RowLogCopy")
		}
	}()
	if v == nil {
		return vulkanV41RowLogStatus("V41RowLogCopy", 3)
	}
	rowBytes, verr := ValidateV41RowLogCopy(dst, dstRow, src, srcRow, rows)
	if verr != nil {
		return vulkanV41RowLogError("V41RowLogCopy", verr)
	}
	db, dOK := v.v41RowLogBuf(dst)
	sb, sOK := v.v41RowLogBuf(src)
	if !dOK || !sOK {
		return vulkanV41RowLogError("V41RowLogCopy", fmt.Errorf("%w: operands must be live F32 buffers owned by this backend", ErrV41RowLogGeometry))
	}
	if err := vulkanV41RowLogStatus("V41RowLogCopy", int(C.fvk_submission_status())); err != nil {
		return err
	}
	C.fvk_d2d_range(db.ptr, C.size_t(dstRow*rowBytes), sb.ptr, C.size_t(srcRow*rowBytes), C.size_t(rows*rowBytes))
	return vulkanV41RowLogStatus("V41RowLogCopy", int(C.fvk_submission_status()))
}

// V41RowLogGather groups bounded gathers into one native copy per source.
// Larger generic requests retain the ordinary per-span recording path.
func (v *vulkanBackend) V41RowLogGather(spans []V41RowSpan, width int) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41RowLogGather")
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
		return Tensor{}, vulkanV41RowLogStatus("V41RowLogGather", 3)
	}
	total, verr := ValidateV41RowLogGather(spans, width)
	if verr != nil {
		return Tensor{}, vulkanV41RowLogError("V41RowLogGather", verr)
	}
	bufs := make([]*vulkanBuf, len(spans))
	for i, span := range spans {
		b, ok := v.v41RowLogBuf(span.Src)
		if !ok {
			return Tensor{}, vulkanV41RowLogError("V41RowLogGather", fmt.Errorf("%w: span %d source must be a live F32 buffer owned by this backend", ErrV41RowLogGeometry, i))
		}
		bufs[i] = b
	}
	if err := vulkanV41RowLogStatus("V41RowLogGather", int(C.fvk_submission_status())); err != nil {
		return Tensor{}, err
	}
	rowBytes := width * 4
	outBuf = v.dallocForClass(total*rowBytes, MemoryActivation, "V4.1 resident KV panel")
	if len(spans) <= C.FVK_V41_GATHER_MAX_REGIONS {
		ptrs := make([]unsafe.Pointer, len(spans))
		offsets := make([]C.size_t, len(spans))
		bytes := make([]C.size_t, len(spans))
		for i, span := range spans {
			ptrs[i], offsets[i], bytes[i] = bufs[i].ptr, C.size_t(span.Row*rowBytes), C.size_t(span.Rows*rowBytes)
		}
		if err := vulkanV41RowLogStatus("V41RowLogGather", int(C.fvk_v41_d2d_gather(outBuf.ptr, C.int(len(spans)),
			(*unsafe.Pointer)(unsafe.Pointer(&ptrs[0])), &offsets[0], &bytes[0]))); err != nil {
			return Tensor{}, err
		}
	} else {
		at := 0
		for i, span := range spans {
			C.fvk_d2d_range(outBuf.ptr, C.size_t(at*rowBytes), bufs[i].ptr, C.size_t(span.Row*rowBytes), C.size_t(span.Rows*rowBytes))
			at += span.Rows
		}
	}
	if err := vulkanV41RowLogStatus("V41RowLogGather", int(C.fvk_submission_status())); err != nil {
		return Tensor{}, err
	}
	return makeTensor(v, F32, RowMajor, []int{total, width}, nil, outBuf), nil
}
