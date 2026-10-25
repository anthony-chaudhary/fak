// Package gpubinding is the public seam to the GPU binding doctor: one closed
// verdict per NVIDIA PCI display device, comparing PCI enumeration, driver
// binding, driver device-ID coverage and NVML visibility. Companion
// repositories read the verdict through this package instead of
// re-implementing the per-layer checks.
package gpubinding

import "github.com/anthony-chaudhary/fak/internal/compute"

type (
	// Verdict is one device's closed outcome.
	Verdict = compute.GPUBindingVerdict
	// Report is the host-level result; Report.Ready is true only when every
	// device is GPU_READY and at least ExpectGPUs were enumerated.
	Report = compute.GPUBindingReport
	// Device is one NVIDIA display device with its layer observations.
	Device = compute.GPUBindingDevice
)

// The closed verdict set.
const (
	VerdictReady               = compute.GPUReady
	VerdictNotEnumerated       = compute.GPUNotEnumerated
	VerdictUnbound             = compute.GPUUnbound
	VerdictDriverLacksDeviceID = compute.GPUDriverLacksDeviceID
	VerdictBoundNotCUDAVisible = compute.GPUBoundNotCUDAVisible
	VerdictTunnelUnauthorized  = compute.GPUTunnelUnauthorized
	VerdictUnknown             = compute.GPUBindingUnknown
)

// Options selects what to diagnose; zero values diagnose the live host.
type Options struct {
	// ExpectGPUs adds absent-GPU rows when fewer are enumerated.
	ExpectGPUs int
	// SysfsRoot is a Linux sysfs root; empty selects the live host (sysfs on
	// Linux, PnP on Windows).
	SysfsRoot string
	// PackageINFDir is an extracted Windows driver package to plan against.
	PackageINFDir string
}

// Diagnose runs the binding doctor. It never substitutes a representative
// device: an unobserved layer stays "unknown".
func Diagnose(opts Options) Report {
	return compute.NewCUDAPCIeTopologyDiscovery(opts.SysfsRoot).DiagnoseGPUBinding(compute.GPUBindingOptions{
		ExpectGPUs:    opts.ExpectGPUs,
		PackageINFDir: opts.PackageINFDir,
	})
}
