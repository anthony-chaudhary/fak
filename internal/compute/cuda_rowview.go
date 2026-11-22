//go:build cuda && cgo

package compute

import "unsafe"

// cudaRowViewAlign is the widest single load (16 bytes) any resident GEMV
// issues. A view base aligned to it gives every in-view offset at least the
// alignment the same offset has in a standalone 256-byte-aligned allocation.
const cudaRowViewAlign = 16

func (b *cudaBuf) rowView(w Tensor, rowStart, rows int) (Tensor, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	refuse := func(reason string) (Tensor, error) {
		return Tensor{}, &RowViewError{Dtype: w.Dtype, Reason: reason}
	}
	rowBytes, scaleRowBytes, reason := rowViewGeometry(w)
	if reason != "" {
		return refuse(reason)
	}
	if b == nil || b.ptr == nil || b.invalidForSubmit() {
		return refuse("parent buffer is not live")
	}
	if (w.Dtype == Q5_K || w.Dtype == Q6_K) && cudaQ56KMMVQQualification {
		// The qualified MMVQ preflight admits only allocation base pointers.
		return refuse("the qualified Q5_K/Q6_K MMVQ arm refuses interior pointers")
	}
	out := w.Shape[0]
	if b.n < out*rowBytes {
		return refuse("parent buffer is shorter than its shape")
	}
	ptr := unsafe.Add(b.ptr, rowStart*rowBytes)
	var scales unsafe.Pointer
	switch w.Dtype {
	case Q8_0:
		if b.scales == nil || b.scalesN != out*scaleRowBytes {
			return refuse("parent Q8_0 scale plane does not match its shape")
		}
		scales = unsafe.Add(b.scales, rowStart*scaleRowBytes)
	case NVFP4:
		base := b.scales
		if !b.view {
			if b.scales != nil || b.n != out*(rowBytes+scaleRowBytes) {
				return refuse("parent NVFP4 payload does not match its shape")
			}
			base = unsafe.Add(b.ptr, out*rowBytes)
		} else if b.scales == nil || b.scalesN != out*scaleRowBytes {
			return refuse("parent NVFP4 scale plane does not match its shape")
		}
		scales = unsafe.Add(base, rowStart*scaleRowBytes)
	default:
		if b.scales != nil {
			return refuse("parent carries an unexpected scale plane")
		}
	}
	if uintptr(ptr)%cudaRowViewAlign != 0 || (scales != nil && uintptr(scales)%cudaRowViewAlign != 0) {
		return refuse("row offset is not 16-byte aligned; kernels assume aligned bases")
	}
	vb := &cudaBuf{ptr: ptr, n: rows * rowBytes, class: b.class, device: b.device, managed: b.managed, view: true}
	if scales != nil {
		vb.scales, vb.scalesN = scales, rows*scaleRowBytes
	}
	q := rowViewQuant(w.Quant, out, rowStart, rows)
	return makeTensor(w.be, w.Dtype, RowMajor, []int{rows, w.Shape[1]}, q, vb), nil
}
