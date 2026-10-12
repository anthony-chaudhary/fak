//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"encoding/binary"
	"math"
	"unsafe"
)

var _ V41DeferredStatusBackend = (*vulkanBackend)(nil)

// SupportsV41DeferredStatus reports both checked modules the deferred calls
// issue and the selected device's publication count domain.
func (v *vulkanBackend) SupportsV41DeferredStatus(count int) bool {
	if v == nil || count <= 0 || count > math.MaxInt32 || count > math.MaxInt/4 {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_routed_swiglu() != 0 && C.fvk_have_v41_attention_bf16() != 0 &&
		C.fvk_v41_attention_bf16_count_supported(C.int(count)) != 0
}

// V41RoutedSwiGLUDeferred is V41RoutedSwiGLU without its status readback.
func (v *vulkanBackend) V41RoutedSwiGLUDeferred(gate, up Tensor, limit, weight float32) (out Tensor, status V41DeferredStatus, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41RoutedSwiGLU")
		}
		if err != nil {
			v.freeDeferredLocked(statusBuf, output)
			out, status = Tensor{}, V41DeferredStatus{}
		}
	}()
	if v == nil || C.fvk_have_v41_routed_swiglu() == 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(3, "routed SwiGLU module or binary32 float controls unavailable")
	}
	count, invalid := validateV41RoutedSwiGLU(gate, up, limit, weight)
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, invalid.Error())
	}
	bytes := count * 4
	for _, operand := range []struct {
		name string
		t    Tensor
	}{{"gate", gate}, {"up", up}} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n < bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, operand.name+" must be a live, full-sized F32 device buffer owned by this backend")
		}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(status, "sticky submission fault")
	}
	if C.fvk_batch_foreign_active() {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "the routed activation does not join a foreign batch")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 routed SwiGLU output")
	statusBuf = v.dallocForClass(v41RoutedSwiGLUStatusBytes, MemoryActivation, "V4.1 routed SwiGLU status")
	if status := int(C.fvk_v41_routed_swiglu_f32(gate.buf.(*vulkanBuf).ptr, up.buf.(*vulkanBuf).ptr, output.ptr, statusBuf.ptr,
		C.int(count), C.float(limit), C.float(weight))); status != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(status, "initialization or checked dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{count}, nil, output), V41DeferredStatus{
		Kind: V41DeferredRoutedSwiGLU, Count: count,
		Record: makeTensor(v, F32, RowMajor, []int{v41RoutedSwiGLUStatusBytes / 4}, nil, statusBuf),
	}, nil
}

// V41BF16PublishDeferred is V41BF16Publish without its status readback.
func (v *vulkanBackend) V41BF16PublishDeferred(input Tensor, site uint32) (out Tensor, status V41DeferredStatus, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, "V41BF16Publish")
			}
		}
		if err != nil {
			v.freeDeferredLocked(statusBuf, output)
			out, status = Tensor{}, V41DeferredStatus{}
		}
	}()
	fail := func(code int, reason string) error {
		e := vulkanV41AttentionBF16StatusError(code, reason)
		e.Site = "V41BF16Publish"
		return e
	}
	if v == nil {
		return Tensor{}, V41DeferredStatus{}, fail(3, "nil backend")
	}
	if code := int(C.fvk_v41_attention_bf16_state()); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "submission state refuses publication")
	}
	if C.fvk_have_v41_attention_bf16() == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(3, "ordinary checked module unavailable")
	}
	bytes, invalid := validateV41BF16Publish(input, site)
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, fail(2, invalid.Error())
	}
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(2, "input must be live F32 storage owned by this backend")
	}
	count := input.Shape[0]
	if C.fvk_v41_attention_bf16_count_supported(C.int(count)) == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(2, "selected device count limit")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 BF16 publication output")
	statusBuf = v.dallocForClass(v41BF16CheckedStatusBytes, MemoryActivation, "V4.1 BF16 publication status")
	if code := int(C.fvk_v41_bf16_publish_f32(source.ptr, output.ptr, statusBuf.ptr, C.int(count), C.uint32_t(site))); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "initialization or checked dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{count}, nil, output), V41DeferredStatus{
		Kind: V41DeferredBF16Publish, Count: count, Site: site,
		Record: makeTensor(v, F32, RowMajor, []int{v41BF16CheckedStatusBytes / 4}, nil, statusBuf),
	}, nil
}

// V41ReadDeferred reads every status record and row with one fvk_v41_d2h_many
// and frees every record, on success and failure alike.
func (v *vulkanBackend) V41ReadDeferred(statuses []V41DeferredStatus, rows []Tensor) (verdicts []error, values [][]float32, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	records := make([]*vulkanBuf, 0, len(statuses))
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41ReadDeferred")
		}
		v.freeDeferredLocked(records...)
		if err != nil {
			verdicts, values = nil, nil
		}
	}()
	invalid := func(reason string) error {
		return &BackendError{Backend: "vulkan", Site: "V41ReadDeferred", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: reason}
	}
	for _, st := range statuses {
		b, ok := st.Record.buf.(*vulkanBuf)
		if ok && b != nil && b.ptr != nil && st.Record.be == v {
			records = append(records, b)
		}
	}
	if v == nil || len(records) != len(statuses) || len(statuses)+len(rows) == 0 {
		return nil, nil, invalid("every status record must be live storage owned by this backend")
	}
	srcs := make([]*vulkanBuf, 0, len(statuses)+len(rows))
	sizes := make([]int, 0, cap(srcs))
	for _, b := range records {
		if b.n < 16 {
			return nil, nil, invalid("status record is shorter than 16 bytes")
		}
		srcs, sizes = append(srcs, b), append(sizes, 16)
	}
	for _, row := range rows {
		b, ok := row.buf.(*vulkanBuf)
		n := row.Numel()
		if row.be != v || !ok || b == nil || b.ptr == nil || row.Dtype != F32 || row.Quant != nil || n <= 0 || n > math.MaxInt/4 || b.n < n*4 || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return nil, nil, invalid("every row must be live F32 storage owned by this backend")
		}
		srcs, sizes = append(srcs, b), append(sizes, n*4)
	}
	image, readErr := vulkanV41ReadMany("V41ReadDeferred", srcs, sizes)
	if readErr != nil {
		return nil, nil, readErr
	}
	verdicts = make([]error, len(statuses))
	for i, st := range statuses {
		verdicts[i] = v41DeferredVerdict("vulkan", st, image[i])
	}
	values = make([][]float32, len(rows))
	for i := range rows {
		raw := image[len(statuses)+i]
		row := make([]float32, len(raw)/4)
		for j := range row {
			row[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*j:]))
		}
		values[i] = row
	}
	return verdicts, values, nil
}

// vulkanV41ReadMany reads each region behind one checked host fence and
// returns one owned slice per region. Caller holds vulkanMu.
func vulkanV41ReadMany(site string, srcs []*vulkanBuf, sizes []int) ([][]byte, error) {
	if len(srcs) == 0 || len(srcs) != len(sizes) || len(srcs) > math.MaxInt32 {
		return nil, &BackendError{Backend: "vulkan", Site: site, Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: "readback needs one size per region"}
	}
	offsets, total := v41DeferredLayout(sizes)
	host := make([]byte, total)
	ptrs := make([]unsafe.Pointer, len(srcs))
	cOffsets := make([]C.size_t, len(srcs))
	cSizes := make([]C.size_t, len(srcs))
	for i, b := range srcs {
		ptrs[i], cOffsets[i], cSizes[i] = b.ptr, C.size_t(offsets[i]), C.size_t(sizes[i])
	}
	if status := int(C.fvk_v41_d2h_many(unsafe.Pointer(&host[0]), C.size_t(total), C.int(len(srcs)),
		(*unsafe.Pointer)(unsafe.Pointer(&ptrs[0])), &cOffsets[0], &cSizes[0])); status != 0 {
		readErr := vulkanReadError(status)
		readErr.Site = site + " readback"
		return nil, readErr
	}
	out := make([][]byte, len(srcs))
	for i := range srcs {
		out[i] = host[offsets[i] : offsets[i]+sizes[i] : offsets[i]+sizes[i]]
	}
	return out, nil
}

// freeDeferredLocked releases native storage owned by a deferred call. Like
// the synchronous forms it frees directly, never through public Free.
func (v *vulkanBackend) freeDeferredLocked(bufs ...*vulkanBuf) {
	for _, b := range bufs {
		if b != nil && b.ptr != nil {
			C.fvk_free(b.ptr)
			b.ptr, b.n = nil, 0
		}
	}
}
