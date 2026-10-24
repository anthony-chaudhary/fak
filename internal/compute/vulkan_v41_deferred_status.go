//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
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
func (v *vulkanBackend) V41ReadDeferred(statuses []V41DeferredStatus, rows []Tensor) ([]error, [][]float32, error) {
	return v.v41ReadDeferredLayout(statuses, rows, nil, 0, v41DeferredReadOrdinary)
}

// An explicit mode selects the packed-panel layout. Invalid width cannot fall
// back to legacy typed-arena materialization or obtain an independence grant.
func (v *vulkanBackend) v41ReadDeferredLayout(statuses []V41DeferredStatus, rows []Tensor, panelRows []int, width int, mode v41DeferredReadMode) (verdicts []error, values [][]float32, err error) {
	packed := mode != v41DeferredReadOrdinary
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	records := make([]*vulkanBuf, 0, len(statuses))
	bodyComplete := false
	defer func() {
		if !packed {
			// Preserve the legacy reader's cleanup and panic behavior.
			if recovered := recover(); recovered != nil {
				err = ClassifyVulkanPanic(recovered, "V41ReadDeferred")
			}
			v.freeDeferredLocked(records...)
			if err != nil {
				verdicts, values = nil, nil
			}
			return
		}
		recovered := recover()
		failIncomplete := func(p any) {
			if err != nil {
				return
			}
			if primary, ok := p.(error); ok && primary != nil {
				err = primary
				return
			}
			if p == nil {
				p = "incomplete packed deferred read or retirement"
			}
			err = ClassifyVulkanPanic(p, "V41ReadDeferred")
		}
		if !bodyComplete || recovered != nil {
			failIncomplete(recovered)
		}
		cleanupComplete := false
		defer func() {
			if !cleanupComplete {
				failIncomplete(nil)
			}
			if err != nil {
				verdicts, values = nil, nil
			}
		}()
		release := func(b *vulkanBuf) {
			done := false
			defer func() {
				p := recover()
				if !done {
					failIncomplete(p)
				}
			}()
			v.freeDeferredLocked(b)
			done = true
		}
		func() {
			// Register every physical release before invoking any of them.
			// Reverse registration preserves the ordinary release order.
			for i := len(records) - 1; i >= 0; i-- {
				defer release(records[i])
			}
		}()
		cleanupComplete = true
	}()
	invalid := func(reason string) error {
		return &BackendError{Backend: "vulkan", Site: "V41ReadDeferred", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: reason}
	}
	for _, st := range statuses {
		b, ok := st.Record.buf.(*vulkanBuf)
		if ok && b != nil && b.ptr != nil && st.Record.be == v {
			if packed {
				duplicate := false
				for _, accepted := range records {
					if accepted.ptr == b.ptr {
						duplicate = true
						break
					}
				}
				if duplicate {
					continue
				}
			}
			records = append(records, b)
		}
	}
	if v == nil || len(records) != len(statuses) || len(statuses)+len(rows) == 0 {
		return nil, nil, invalid("every status record must be live storage owned by this backend")
	}
	if packed && (len(statuses) > 64 || len(rows) > 32) {
		return nil, nil, invalid("packed deferred descriptor count exceeds cohort bounds")
	}
	srcs := make([]*vulkanBuf, 0, len(statuses)+len(rows))
	sizes := make([]int, 0, cap(srcs))
	verdictCount := 0
	for i, b := range records {
		n := V41DeferredRecords(statuses[i])
		if n > V41RoutedIDMaxPicks || b.n < 16*n {
			return nil, nil, invalid("status record is shorter than its 16-byte records")
		}
		srcs, sizes = append(srcs, b), append(sizes, 16*n)
		verdictCount += n
	}
	for _, row := range rows {
		b, ok := row.buf.(*vulkanBuf)
		n := row.Numel()
		if row.be != v || !ok || b == nil || b.ptr == nil || row.Dtype != F32 || row.Quant != nil || n <= 0 || n > math.MaxInt/4 || b.n < n*4 || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return nil, nil, invalid("every row must be live F32 storage owned by this backend")
		}
		srcs, sizes = append(srcs, b), append(sizes, n*4)
	}
	logicalRows := len(rows)
	if packed {
		var ok bool
		switch mode {
		case v41DeferredReadRoutedPanel:
			logicalRows, ok = v41DeferredPanelRows(rows, panelRows, width)
		case v41DeferredReadGroupedPanel:
			logicalRows, ok = v41GroupedDeferredPanelLayout(statuses, rows, panelRows, width)
		default:
			// Ordinary reads bypass this branch; unknown packed modes fail geometry.
			ok = false
		}
		if !ok || len(statuses) > 64 || verdictCount > 64 {
			return nil, nil, invalid("packed deferred logical row layout exceeds cohort bounds")
		}
		// A physical owner appears at most once, including status/output aliases.
		seen := make(map[unsafe.Pointer]bool, len(srcs))
		for _, b := range srcs {
			if seen[b.ptr] {
				return nil, nil, invalid("packed deferred storage aliases another physical region")
			}
			seen[b.ptr] = true
		}
		// Independently check aggregate transfer alignment/addition before allocation.
		total := 0
		for _, size := range sizes {
			if size <= 0 || total > math.MaxInt-15 {
				return nil, nil, invalid("packed deferred transfer extent overflow")
			}
			total = (total + 15) &^ 15
			if size > math.MaxInt-total {
				return nil, nil, invalid("packed deferred transfer extent overflow")
			}
			total += size
		}
	}
	image, readErr := vulkanV41ReadImage("V41ReadDeferred", srcs, sizes, logicalRows)
	if readErr != nil {
		return nil, nil, readErr
	}
	verdicts = make([]error, 0, verdictCount)
	for i, st := range statuses {
		for r := 0; r < V41DeferredRecords(st); r++ {
			verdicts = append(verdicts, v41DeferredVerdict("vulkan", st, image.region(i, sizes[i])[16*r:16*r+16]))
		}
	}
	if packed {
		values = image.materializePanelRows(sizes, len(statuses), panelRows, width, logicalRows)
	} else {
		values = image.materializeRows(sizes, len(statuses))
	}
	bodyComplete = true
	return verdicts, values, nil
}

// vulkanV41ReadMany reads each region behind one checked host fence and
// returns one owned slice per region. Caller holds vulkanMu.
func vulkanV41ReadMany(site string, srcs []*vulkanBuf, sizes []int) ([][]byte, error) {
	image, err := vulkanV41ReadImage(site, srcs, sizes, 0)
	if err != nil {
		return nil, err
	}
	out := make([][]byte, len(srcs))
	for i := range srcs {
		out[i] = image.region(i, sizes[i])
	}
	return out, nil
}

// The same native transfer writes an owned image. Only the deferred one-row
// consumer can request the bounded typed arena; other readers keep byte storage.
func vulkanV41ReadImage(site string, srcs []*vulkanBuf, sizes []int, rows int) (v41HostReadbackImage, error) {
	if len(srcs) == 0 || len(srcs) != len(sizes) || len(srcs) > math.MaxInt32 {
		return v41HostReadbackImage{}, &BackendError{Backend: "vulkan", Site: site, Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: "readback needs one size per region"}
	}
	image := newV41HostReadbackImage(sizes, rows, v41NativeLittleEndian())
	host, offsets := image.host, image.offsets
	total := len(host)
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
		return v41HostReadbackImage{}, readErr
	}
	return image, nil
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

// Only successful multirow materialization owns independent host allocations.
// The one-row typed image may retain a status prefix and receives no grant.
// V41ReadDeferred remains the sole status owner and the sole native read call;
// its transport/panic cleanup must finish successfully before minting a result.
func (v *vulkanBackend) v41ReadDeferredOwned(statuses []V41DeferredStatus, rows []Tensor) ([]error, V41DeferredReadResult, error) {
	verdicts, values, err := v.V41ReadDeferred(statuses, rows)
	if err != nil {
		return nil, V41DeferredReadResult{}, err
	}
	return verdicts, V41DeferredReadResult{count: len(values), state: &v41DeferredReadState{rows: values, independent: len(rows) >= 2 && len(values) == len(rows)}}, nil
}

// The original interface must contain this exact concrete backend pointer.
// A wrapper's promoted private method does not qualify its overridden reader.
func (v *vulkanBackend) v41BindDeferredPanelReader(original V41DeferredStatusBackend) V41DeferredPanelReader {
	canonical, ok := original.(*vulkanBackend)
	if !ok || canonical != v || v == nil {
		return nil
	}
	return v.v41ReadDeferredPanels
}

func (v *vulkanBackend) v41ReadDeferredPanels(statuses []V41DeferredStatus, panels []Tensor, rows []int, width int) ([]error, V41DeferredReadResult, error) {
	verdicts, values, err := v.v41ReadDeferredLayout(statuses, panels, rows, width, v41DeferredReadRoutedPanel)
	if err != nil {
		return nil, V41DeferredReadResult{}, err
	}
	// Panel materialization always allocates independent rows, including singleton
	// regions mixed with chunks. Mint only after physical status cleanup succeeds.
	return verdicts, V41DeferredReadResult{count: len(values), state: &v41DeferredReadState{rows: values, independent: true}}, nil
}

// The grouped grant is separate from routed cohorts and cannot be inherited by
// a wrapper that overrides deferred reads or publication semantics.
func (v *vulkanBackend) v41BindGroupedDeferredPanelReader(original V41DeferredStatusBackend) V41GroupedDeferredPanelReader {
	canonical, ok := original.(*vulkanBackend)
	if !ok || canonical != v || v == nil {
		return nil
	}
	return v.v41ReadGroupedDeferredPanel
}

func (v *vulkanBackend) v41ReadGroupedDeferredPanel(statuses []V41DeferredStatus, panel Tensor, rows, width int) ([]error, V41DeferredReadResult, error) {
	verdicts, values, err := v.v41ReadDeferredLayout(statuses, []Tensor{panel}, []int{rows}, width, v41DeferredReadGroupedPanel)
	if err != nil {
		return nil, V41DeferredReadResult{}, err
	}
	// The shared packed path has retired every status before this grant.
	return verdicts, V41DeferredReadResult{count: len(values), state: &v41DeferredReadState{rows: values, independent: true}}, nil
}
