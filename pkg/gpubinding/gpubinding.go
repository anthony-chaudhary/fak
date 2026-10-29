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
	// Floor is a payload's admission floor on a bound row: minimum driver
	// branch, required module flavor and minimum compute capability.
	Floor = compute.GPUDriverFloor
)

// The closed verdict set.
const (
	VerdictReady                   = compute.GPUReady
	VerdictNotEnumerated           = compute.GPUNotEnumerated
	VerdictUnbound                 = compute.GPUUnbound
	VerdictDriverLacksDeviceID     = compute.GPUDriverLacksDeviceID
	VerdictBoundNotCUDAVisible     = compute.GPUBoundNotCUDAVisible
	VerdictDriverBelowFloor        = compute.GPUDriverBelowFloor
	VerdictModuleFlavorProprietary = compute.GPUModuleFlavorProprietary
	VerdictComputeCapInsufficient  = compute.GPUComputeCapInsufficient
	VerdictTunnelUnauthorized      = compute.GPUTunnelUnauthorized
	VerdictUnknown                 = compute.GPUBindingUnknown

	// Kernel-module flavors a floor can require.
	DriverFlavorOpen        = compute.DriverFlavorOpen
	DriverFlavorProprietary = compute.DriverFlavorProprietary
)

// BlackwellFloor is the sm_120 (Blackwell) payload floor: nvidia-open driver
// branch >= 615.71.09 and compute capability >= 12.0.
var BlackwellFloor = compute.BlackwellDriverFloor

// ParseDriverVersion splits a leading "major.minor.patch" from a driver version
// string such as "615.71.09".
func ParseDriverVersion(v string) (major, minor, patch int, ok bool) {
	return compute.ParseDriverVersion(v)
}

// DriverBranchBelow reports whether an observed driver version is strictly below
// min; an unparseable version is unknown, not below.
func DriverBranchBelow(version string, min [3]int) bool {
	return compute.DriverBranchBelow(version, min)
}

// ParseComputeCapability reads an NVML compute_cap field ("12.0") as
// major*10+minor; an unparseable field stays 0 (unobserved).
func ParseComputeCapability(s string) int {
	return compute.ParseComputeCapability(s)
}

// Options selects what to diagnose; zero values diagnose the live host.
type Options struct {
	// ExpectGPUs adds absent-GPU rows when fewer are enumerated.
	ExpectGPUs int
	// SysfsRoot is a Linux sysfs root; empty selects the live host (sysfs on
	// Linux, PnP on Windows).
	SysfsRoot string
	// PackageINFDir is an extracted Windows driver package to plan against.
	PackageINFDir string
	// Floor demotes rows that would read GPU_READY when they violate the
	// payload's driver-branch / module-flavor / compute-capability floor.
	Floor Floor
}

// Diagnose runs the binding doctor. It never substitutes a representative
// device: an unobserved layer stays "unknown".
func Diagnose(opts Options) Report {
	return compute.NewCUDAPCIeTopologyDiscovery(opts.SysfsRoot).DiagnoseGPUBinding(compute.GPUBindingOptions{
		ExpectGPUs:    opts.ExpectGPUs,
		PackageINFDir: opts.PackageINFDir,
		Floor:         opts.Floor,
	})
}
