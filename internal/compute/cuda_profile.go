package compute

// The CUDA kernel timestamp profile (FAK_CUDA_TIMESTAMP_PROFILE=1, the CUDA peer of
// FAK_VULKAN_TIMESTAMP_PROFILE). The shim (cuda/kernel_profile.cuh) reads the env once at
// load and brackets every g_stream launch, cuBLAS call and async copy/fill with its own
// event pair, so a row is one device kernel keyed by name and grid x/y, the same KernelRow
// shape the Vulkan timestamp fold produces. The pairs resolve in one host wait at the token
// boundary (Recycle), at a reader snapshot, or when the ring fills.
//
// Field mapping: GPUNS sums the timed pairs (disjoint, since g_stream is in order), and the
// rows sum to it. WallNS is host wall from a batch's first timed launch to its token
// boundary; Vulkan's WallNS is host wall per submit, so compare GPUNS against a host window,
// never WallNS across backends. Truncated counts pairs whose events failed, UntimedCompute
// launches inside a graph capture (the profile forces graphs off, so this stays 0 unless a
// capture opened anyway), UntimedTransfer synchronous host copies. BarriersElided stays 0.
//
// An event pair recorded on an idle stream also books the host's launch gap between the
// start event and the kernel, so a host-bound decode overstates short kernels; the CUDA-host
// rung measures that overhead instead of assuming it (docs/cuda-dev.md).
//
// This file is the build-tag-free conversion; the cgo reader lives in cuda.go.

// cudaTimestampProfileEnv opts in with the same exact "1" grammar as the Vulkan control;
// the shim reads it, so no Go env read exists for it.
const cudaTimestampProfileEnv = "FAK_CUDA_TIMESTAMP_PROFILE"

// cudaTSTotals and cudaTSRow are the Go values of fak_cuda_ts_totals and fak_cuda_ts_row.
type cudaTSTotals struct {
	Enabled, Usable, InitTried                                               bool
	Batches, Ops, Truncated, UntimedCompute, UntimedTransfer, BarriersElided uint64
	GPUNS, WallNS                                                            float64
}

type cudaTSRow struct {
	Kid, GX, GY uint32
	Calls       uint64
	NS          float64
}

// cudaKernelProfileFromShim builds the cumulative profile from one shim snapshot. ok is
// false when the profile is off, when the event ring failed, or when the shim refused the
// snapshot (n < 0, a capture is open). A ring that has timed nothing yet is a valid zero
// baseline.
func cudaKernelProfileFromShim(tot cudaTSTotals, n int, rows []cudaTSRow, name func(kid uint32) string) (KernelProfile, bool) {
	if !tot.Enabled || (tot.InitTried && !tot.Usable) || n < 0 || len(rows) != n {
		return KernelProfile{}, false
	}
	p := KernelProfile{
		Batches:         tot.Batches,
		Ops:             tot.Ops,
		Truncated:       tot.Truncated,
		UntimedCompute:  tot.UntimedCompute,
		UntimedTransfer: tot.UntimedTransfer,
		BarriersElided:  tot.BarriersElided,
		GPUNS:           tot.GPUNS,
		WallNS:          tot.WallNS,
		Rows:            make([]KernelRow, 0, n),
	}
	for _, r := range rows {
		p.Rows = append(p.Rows, KernelRow{Kernel: name(r.Kid), GX: r.GX, GY: r.GY, Calls: r.Calls, NS: r.NS})
	}
	sortKernelRows(p.Rows)
	return p, true
}
