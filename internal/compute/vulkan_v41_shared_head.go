//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

// V41SharedHeadAvailable promises scalar-row composition, not a separately
// optimized panel reduction or measured speedup. BatchedMatMul uses the same
// q6kMatMulLocked(..., 1) as MatMul for every row at this exact Flash shape.
// Copies preserve the F32 bits and one checked window owns all row work.
// Physical end-to-end correctness and throughput remain separate witnesses.
func (v *vulkanBackend) V41SharedHeadAvailable(dtype Dtype, out, in, maxRows int) bool {
	if v == nil || !v41ScalarHeadShape(dtype, out, in, maxRows) {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.v41ScalarHeadAvailableLocked(out, in, maxRows)
}

// The native route is deliberately bounded to the physical AMD UMA device
// class. Device identity is stable while the registered backend owns this lock.
func (v *vulkanBackend) v41ScalarHeadDeviceLocked() bool {
	if v == nil || v != vulkanDev || C.fvk_device_type() != 1 {
		return false
	}
	var name [256]C.char
	var vendor, device, driver, api C.uint32_t
	return C.fvk_device_identity(&name[0], 256, &vendor, &device, &driver, &api) != 0 &&
		uint32(vendor) == vulkanAMDVendorID && device != 0
}

func (v *vulkanBackend) v41ScalarHeadAvailableLocked(out, in, rows int) bool {
	if !v.v41ScalarHeadDeviceLocked() || C.fvk_have_q6k_matmul() == 0 || C.fvk_submission_status() != 0 ||
		C.fvk_batch_active() || C.fvk_v41_step_batch_active() != 0 {
		return false
	}
	// Keep the existing Q6 input validator's conservative panel-grid bound,
	// although each executed scalar launch actually needs only ceil(out/64).
	groups := (uint64(rows)*uint64(out) + 63) / 64
	if maximum := uint64(C.fvk_max_compute_work_group_count_x()); maximum == 0 || groups > maximum {
		return false
	}
	packed := uint64(out) * uint64(in/256) * 210
	for _, limit := range []int64{v.maxBufferBytes, v.maxStorageBufferRange, v.maxMemoryAllocationSize} {
		if limit <= 0 || packed > uint64(limit) || uint64(rows*in*4) > uint64(limit) || uint64(rows*out*4) > uint64(limit) {
			return false
		}
	}
	return true
}

// v41ScalarHeadLocked is selected before the generic BatchedMatMul allocation
// or hazard declaration. The caller holds vulkanMu; the model's cooperative
// ForwardExecutionOwner remains held across all head preparation and readback.
func (v *vulkanBackend) v41ScalarHeadLocked(w, x Tensor, rows int) Tensor {
	const site = "V41ScalarHead"
	out, in := v.validateQ6KMatMulInputs(w, x, rows)
	requireVulkanSubmissionHealthyLocked(site)
	if !v41ScalarHeadShape(w.Dtype, out, in, rows) || !v.v41ScalarHeadAvailableLocked(out, in, rows) {
		panic(vulkanV41RoutedIDError(site, 2, "scalar head shape, resources or exclusive recording window unavailable"))
	}
	wb, wOK := v.ownedVulkanBuf(w)
	xb, xOK := v.ownedVulkanBuf(x)
	if w.Layout != RowMajor || x.Layout != RowMajor || !wOK || !xOK || wb.n < out*(in/256)*210 || xb.n < rows*in*4 {
		panic(vulkanV41RoutedIDError(site, 2, "scalar head requires live full-sized row-major operands"))
	}
	var inputRows, outputRows [4]*vulkanBuf
	var output *vulkanBuf
	completed := false
	defer func() {
		// Native free parks buffers if a submit/wait left completion uncertain.
		// Never recycle into the Go transient pool or publish a partial output.
		v.freeDeferredLocked(inputRows[:]...)
		v.freeDeferredLocked(outputRows[:]...)
		if !completed {
			v.freeDeferredLocked(output)
		}
	}()
	y, output := v.devTr([]int{rows, out}, F32)
	var xs, ys [4]Tensor
	xs[0], ys[0] = x, y
	if rows > 1 {
		// Every scratch buffer is written once. In particular, later copies
		// never overwrite storage read or written by an earlier recorded row.
		for row := 0; row < rows; row++ {
			inputRows[row] = v.dallocForClass(in*4, MemoryActivation, "V4.1 scalar head input row")
			outputRows[row] = v.dallocForClass(out*4, MemoryActivation, "V4.1 scalar head output row")
			xs[row] = makeTensor(v, F32, RowMajor, []int{in}, nil, inputRows[row])
			ys[row] = makeTensor(v, F32, RowMajor, []int{out}, nil, outputRows[row])
		}
	}
	status := func() error {
		if code := int(C.fvk_submission_status()); code != 0 {
			err := vulkanReadError(code)
			err.Site = site
			return err
		}
		return nil
	}
	err := runV41ScalarHeadWindow(rows, func() error {
		code := int(C.fvk_v41_step_batch_begin())
		if err := status(); err != nil {
			return err
		}
		if code != 0 {
			return vulkanV41RoutedIDError(site, 2, "exclusive scalar head recording window refused")
		}
		return nil
	}, func(row int) error {
		if rows > 1 {
			C.fvk_d2d_range(inputRows[row].ptr, 0, xb.ptr, C.size_t(row*in*4), C.size_t(in*4))
			if err := status(); err != nil {
				return err
			}
		}
		// This is exactly MatMul's scalar Q6 entrypoint and push constants.
		v.q6kMatMulLocked(w, xs[row], ys[row], 1)
		if err := status(); err != nil {
			return err
		}
		if rows > 1 {
			C.fvk_d2d_range(output.ptr, C.size_t(row*out*4), outputRows[row].ptr, 0, C.size_t(out*4))
		}
		return status()
	}, func() error {
		code := int(C.fvk_v41_step_batch_end())
		if err := status(); err != nil {
			return err
		}
		if code != 0 {
			return vulkanV41RoutedIDError(site, code, "scalar head recording window failed to close")
		}
		return nil
	})
	if err != nil {
		panic(err) // Selected work is never replayed through another kernel.
	}
	output.v41CheckedRead = true
	completed = true
	return y
}
