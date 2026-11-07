// The GPU binding doctor moved to pkg/gpubinding (the public seam companion
// repositories import). This file keeps the historical compute.* names as thin
// aliases and wrappers INTO that package so every existing caller compiles
// unchanged; the implementation, verdicts and tests live at the seam now.
package compute

import (
	"github.com/anthony-chaudhary/fak/pkg/gpubinding"
)

// GPUBindingVerdict is the closed per-device outcome of the GPU binding doctor.
type GPUBindingVerdict = gpubinding.Verdict

const (
	// GPUReady means enumerated, bound to the started NVIDIA driver, and NVML-visible.
	GPUReady = gpubinding.VerdictReady
	// GPUNotEnumerated means an expected GPU is absent from a complete PCI enumeration.
	GPUNotEnumerated = gpubinding.VerdictNotEnumerated
	// GPUUnbound means enumerated but the NVIDIA driver is not bound or not started.
	GPUUnbound = gpubinding.VerdictUnbound
	// GPUDriverLacksDeviceID means no installed NVIDIA INF/driver lists the device ID.
	GPUDriverLacksDeviceID = gpubinding.VerdictDriverLacksDeviceID
	// GPUBoundNotCUDAVisible means bound to the started NVIDIA driver but absent from NVML.
	GPUBoundNotCUDAVisible = gpubinding.VerdictBoundNotCUDAVisible
	// GPUDriverBelowFloor means a ready row whose driver branch is below the payload floor.
	GPUDriverBelowFloor = gpubinding.VerdictDriverBelowFloor
	// GPUModuleFlavorProprietary means a ready row on the proprietary module while the floor requires open.
	GPUModuleFlavorProprietary = gpubinding.VerdictModuleFlavorProprietary
	// GPUComputeCapInsufficient means a ready row below the floor's minimum compute capability.
	GPUComputeCapInsufficient = gpubinding.VerdictComputeCapInsufficient
	// GPUTunnelUnauthorized means an expected GPU is absent behind an unauthorized Thunderbolt/USB4 device.
	GPUTunnelUnauthorized = gpubinding.VerdictTunnelUnauthorized
	// GPUBindingUnknown means a layer the verdict depends on was not observed.
	GPUBindingUnknown = gpubinding.VerdictUnknown

	// Kernel-module flavors a floor can require.
	DriverFlavorOpen        = gpubinding.DriverFlavorOpen
	DriverFlavorProprietary = gpubinding.DriverFlavorProprietary
)

// GPULayerState is a tri-state observation of one layer; unknown is never collapsed into yes or no.
type GPULayerState = gpubinding.LayerState

const (
	// GPULayerYes means the layer was observed present.
	GPULayerYes = gpubinding.LayerYes
	// GPULayerNo means the layer was observed absent.
	GPULayerNo = gpubinding.LayerNo
	// GPULayerUnknown means the layer was not observed.
	GPULayerUnknown = gpubinding.LayerUnknown
)

// Discovery source labels kept under their historical names.
const (
	// CUDASourceWindowsPnP marks observations from the Windows PnP binding query.
	CUDASourceWindowsPnP = gpubinding.SourceWindowsPnP
	// CUDASourceNvidiaSMI marks observations from nvidia-smi.
	CUDASourceNvidiaSMI = gpubinding.SourceNvidiaSMI
)

// GPUDriverFloor is the admission floor a payload places on a bound GPU row.
type GPUDriverFloor = gpubinding.DriverFloor

// BlackwellDriverFloor is the sm_120 (Blackwell) payload floor.
var BlackwellDriverFloor = gpubinding.BlackwellFloor

// GPUBindingDevice is one NVIDIA PCI display device with its layer observations.
type GPUBindingDevice = gpubinding.Device

// GPUCUDADevice is one GPU as NVML (nvidia-smi) reports it.
type GPUCUDADevice = gpubinding.CUDADevice

// GPUCUDAView is the NVML layer.
type GPUCUDAView = gpubinding.CUDAView

// GPUBindingInputs are the collected observations BuildGPUBindingReport folds.
type GPUBindingInputs = gpubinding.Inputs

// GPUBindingOptions tunes DiagnoseGPUBinding; zero values select host defaults.
type GPUBindingOptions = gpubinding.Options

// GPUBindingReport is the binding doctor's host-level result. Status stays the
// discovery status type so existing callers keep comparing against
// CUDADiscovery*; every other field aliases the seam's shape.
type GPUBindingReport struct {
	Status     CUDADiscoveryStatus       `json:"status"`
	Method     string                    `json:"method,omitempty"`
	Issues     []CUDADiscoveryIssue      `json:"issues,omitempty"`
	ExpectGPUs int                       `json:"expect_gpus,omitempty"`
	CUDA       GPUCUDAView               `json:"cuda"`
	Devices    []GPUBindingDevice        `json:"devices"`
	Verdicts   map[GPUBindingVerdict]int `json:"verdicts"`
	Ready      bool                      `json:"ready"`
	// USB4HostRouters lists the host's USB4 host routers (Windows); a GPU's
	// usb4_host_router names the one its tunnel root port belongs to.
	USB4HostRouters []GPUUSB4HostRouter `json:"usb4_host_routers,omitempty"`
	// Package is the coverage of an extracted driver package (--inf-dir).
	Package *GPUDriverPackage `json:"package,omitempty"`
}

// bindingReportFrom maps the seam report onto the compute-shaped one: the
// status word converts between the two named types (same underlying values),
// everything else is aliased.
func bindingReportFrom(r gpubinding.Report) GPUBindingReport {
	issues := make([]CUDADiscoveryIssue, len(r.Issues))
	for i, is := range r.Issues {
		issues[i] = CUDADiscoveryIssue{Path: is.Path, Error: is.Error}
	}
	return GPUBindingReport{
		Status:          CUDADiscoveryStatus(r.Status),
		Method:          r.Method,
		Issues:          issues,
		ExpectGPUs:      r.ExpectGPUs,
		CUDA:            r.CUDA,
		Devices:         r.Devices,
		Verdicts:        r.Verdicts,
		Ready:           r.Ready,
		USB4HostRouters: r.USB4HostRouters,
		Package:         r.Package,
	}
}

// ParseDriverVersion splits a leading "major.minor.patch" from a driver version string such as "615.71.09".
func ParseDriverVersion(v string) (major, minor, patch int, ok bool) {
	return gpubinding.ParseDriverVersion(v)
}

// DriverBranchBelow reports whether an observed version is strictly below min; an unparseable version is unknown, not below.
func DriverBranchBelow(version string, min [3]int) bool {
	return gpubinding.DriverBranchBelow(version, min)
}

// ParseComputeCapability reads an NVML compute_cap field ("12.0") as major*10+minor; an unparseable field stays 0 (unobserved).
func ParseComputeCapability(s string) int {
	return gpubinding.ParseComputeCapability(s)
}

// ParseNvidiaSMIBusIDs parses `nvidia-smi --query-gpu=index,pci.bus_id,uuid,name --format=csv,noheader` output.
func ParseNvidiaSMIBusIDs(out string) (devs []GPUCUDADevice, unparsed []string) {
	return gpubinding.ParseNvidiaSMIBusIDs(out)
}

// ParseWindowsGPUBindingJSON parses the PnP binding query into enumerated NVIDIA display devices.
func ParseWindowsGPUBindingJSON(raw string) ([]GPUBindingDevice, error) {
	return gpubinding.ParseWindowsBindingJSON(raw)
}

// ClassifyGPUBinding folds one device's layer observations into its verdict.
func ClassifyGPUBinding(dev GPUBindingDevice) GPUBindingVerdict {
	return gpubinding.Classify(dev)
}

// ClassifyGPUBindingWithFloor applies the payload floor on top of the base verdict.
func ClassifyGPUBindingWithFloor(dev GPUBindingDevice, floor GPUDriverFloor) GPUBindingVerdict {
	return gpubinding.ClassifyWithFloor(dev, floor)
}

// BuildGPUBindingReport joins the enumerated devices with the NVML view and classifies every row.
func BuildGPUBindingReport(in GPUBindingInputs) GPUBindingReport {
	return bindingReportFrom(gpubinding.BuildReport(in))
}

// DiagnoseGPUBinding runs the binding doctor with the discovery's platform
// selection: Linux sysfs when a sysfs root is set or the host is Linux, Windows
// PnP on Windows, otherwise an unsupported report. It never substitutes a
// representative device; an unobserved layer stays unknown.
func (d *CUDAPCIeTopologyDiscovery) DiagnoseGPUBinding(opts GPUBindingOptions) GPUBindingReport {
	if opts.SysfsRoot == "" {
		opts.SysfsRoot = d.SysfsRoot
	}
	rep := gpubinding.Diagnose(opts)
	if rep.Status == gpubinding.DiscoveryUnsupported {
		// Preserve the legacy side effect: record the unsupported platform on the orchestrator.
		_ = d.Discover()
	}
	return bindingReportFrom(rep)
}
