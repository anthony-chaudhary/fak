//go:build ignore

// SOURCE-ONLY: integration sketch for package compute, not a standalone package.
// Original platform constraint: vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_v41_packed_attention.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// PackedAttention is a private host-result prototype, NOT an optional backend
// interface or model binding. Q/sink/rows must not be mutated during this call.
// Successful output is fresh host-owned F32; it never masquerades as packed KV.
// A selected failure publishes no output and authorizes no F32/CPU replay.
func (p *vulkanV41PackedRows) PackedAttention(source V41PackedRowStorage, rows []int, q, sink []float32, heads int, scale float32, mode V41SharedAttentionMode) (out []float32, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	defer func() {
		if r := recover(); r != nil {
			out = nil
			err = ClassifyVulkanPanic(r, "V41PackedAttention")
		}
	}()
	if err = p.state(); err != nil {
		return nil, err
	}
	h, err := p.handle(source, false)
	if err != nil {
		return nil, err
	}
	if h.extent.LogicalWidth() != 512 || h.extent.RecordBytes() != 288 ||
		heads <= 0 || uint64(heads) > uint64(math.MaxInt32)/512 || heads > math.MaxInt/2048 ||
		uint64(len(rows)) > uint64(math.MaxUint32)/288 || len(rows) > math.MaxInt32-2047 ||
		len(q) != heads*512 || (len(sink) != 0 && len(sink) != heads) ||
		(mode != V41SharedAttentionPlain && mode != V41SharedAttentionCompressed) ||
		math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || scale == 0 {
		return nil, packedStorageError("invalid packed attention geometry, mode or scale")
	}
	for _, values := range [][]float32{q, sink} {
		for _, x := range values {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return nil, packedStorageError("nonfinite Q or sink")
			}
		}
	}
	for _, row := range rows {
		if _, _, e := h.extent.ByteRange(row, 1); e != nil {
			return nil, e
		}
		if uint64(row) > math.MaxUint32 {
			return nil, packedStorageError("selected row index exceeds uint32")
		}
		if !h.attentionValid[row] {
			return nil, packedStorageError("packed row has not passed positive finite scale validation")
		}
	}
	// No allocations/uploads, including an empty output, before actual admission.
	// Currently always absent: no bundle/ABI/float-control proof exists.
	if code := int(C.fvk_v41_packed_attention_admission(C.uint64_t(p.generation))); code != 0 {
		return nil, vulkanV41SharedAttentionStatusError(code)
	}
	if len(rows) == 0 {
		return make([]float32, heads*512), nil
	}
	scratchBytes := uint64(heads)*4096 + uint64(len(rows))*288 + uint64(heads)*16 + 4
	if len(sink) != 0 {
		scratchBytes += uint64(heads)*4 - 4
	}
	if scratchBytes > uint64(math.MaxInt) {
		return nil, packedStorageError("scratch extent overflow")
	}
	p.backend.checkResourceCap(int(scratchBytes), "V4.1 packed attention scratch")
	selection := make([]uint32, len(rows))
	for i, row := range rows {
		if uint64(row) > math.MaxUint32 {
			return nil, packedStorageError("selected row index exceeds uint32")
		}
		selection[i] = uint32(row) // preserve order and duplicates, never sort
	}
	output := make([]float32, heads*512)
	status := make([]uint32, heads*4)
	var sinkPtr *C.float
	var hasSink C.int32_t
	if len(sink) != 0 {
		sinkPtr = (*C.float)(unsafe.Pointer(&sink[0]))
		hasSink = 1
	}
	code := int(C.fvk_v41_packed_attention(h.buf.ptr, C.size_t(h.buf.n), C.size_t(h.extent.Rows()), C.uint64_t(p.generation), (*C.uint32_t)(unsafe.Pointer(&selection[0])), C.uint32_t(len(selection)), (*C.float)(unsafe.Pointer(&q[0])), C.int32_t(heads), sinkPtr, hasSink, C.int32_t(mode), C.float(scale), (*C.float)(unsafe.Pointer(&output[0])), (*C.uint32_t)(unsafe.Pointer(&status[0]))))
	if code != 0 {
		if code == 5 {
			if e := decodeV41SharedAttentionStatus(status, heads, len(rows), 512); e != nil {
				return nil, e
			}
		}
		return nil, vulkanV41SharedAttentionStatusError(code)
	}
	if e := decodeV41SharedAttentionStatus(status, heads, len(rows), 512); e != nil {
		return nil, fmt.Errorf("packed attention: %w", e)
	}
	return output, nil
}
