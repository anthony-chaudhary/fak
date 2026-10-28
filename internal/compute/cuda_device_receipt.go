package compute

import (
	"fmt"
	"strconv"
	"strings"
)

// cuda_device_receipt.go — the typed device receipt of the CUDA (NVIDIA) backend
// (ticket rtx5090-sm120-detection-receipt). The cgo init in cuda.go captures raw facts
// from the driver (device name, compute capability, total VRAM, driver version); this
// file turns those facts into a CUDADeviceReceipt through PURE functions, so the whole
// mapping is host-testable without a GPU (the fake-injection seam is CUDABuildArches +
// CUDADeviceInitFacts themselves). Classification reuses the same rule ClassifyCUDAArch
// encodes, over an injectable build arch table; every tracked feature arm is admitted or
// explicitly refused-with-reason against the SAME constants the kernels use — nothing is
// silently downgraded.

// CUDADeviceArm names one feature-arm family whose per-arch admission the device receipt
// reports.
type CUDADeviceArm string

const (
	// CUDADeviceArmMxFP8Cast is the native MXFP8 activation-cast kernel
	// (internal/compute/cuda/mxfp8_cast.cuh).
	CUDADeviceArmMxFP8Cast CUDADeviceArm = "mxfp8-cast"
	// CUDADeviceArmQ4KPrefillFP8LT is the experimental FP8 cuBLASLt arm of the Q4_K
	// prefill GEMM (internal/compute/cuda_q4k_prefill_arms.go).
	CUDADeviceArmQ4KPrefillFP8LT CUDADeviceArm = "q4k-prefill-fp8lt"
)

// CUDAArmAdmission is the verdict of one arm at one arch. Enabled arms carry an empty
// Reason; refused arms ALWAYS carry a human-readable reason naming why the arch is outside
// the arm's admitted set — the receipt never reports a silent downgrade.
type CUDAArmAdmission struct {
	Enabled bool   `json:"enabled"`
	Reason  string `json:"reason,omitempty"`
}

// mxfp8 cast arch gate — mirrors fcuda_mxfp8_cast_supported in
// internal/compute/cuda/mxfp8_cast.cuh (`p.major == 10 && (p.minor == 0 || p.minor == 3)`).
// These named constants are the Go-side source of truth for that gate;
// TestCUDAMxFP8CastGateMatchesKernel pins them to the .cuh line so the two cannot drift.
const cudaMxFP8CastMajor = 10

var cudaMxFP8CastMinors = []int{0, 3}

// mxfp8CastArchAdmission reports whether the MXFP8 cast kernel admits a device of compute
// capability (major, minor).
func mxfp8CastArchAdmission(major, minor int) CUDAArmAdmission {
	for _, m := range cudaMxFP8CastMinors {
		if major == cudaMxFP8CastMajor && minor == m {
			return CUDAArmAdmission{Enabled: true}
		}
	}
	admitted := make([]string, len(cudaMxFP8CastMinors))
	for i, m := range cudaMxFP8CastMinors {
		admitted[i] = "sm_" + strconv.Itoa(cudaMxFP8CastMajor*10+m)
	}
	return CUDAArmAdmission{
		Enabled: false,
		Reason:  fmt.Sprintf("mxfp8-cast kernel gate admits %s only; sm_%d not admitted", strings.Join(admitted, "/"), major*10+minor),
	}
}

// q4kPrefillFP8LTArchAdmission reports whether the Q4_K prefill FP8LT arm admits a device
// of compute capability (major, minor). It reuses CheckCUDAQ4KPrefillArm against the arm
// spec map in cuda_q4k_prefill_arms.go, so the min/max SM bounds live in exactly one place.
// An unknown arch (sm <= 0) fails closed instead of inheriting the launcher's skip check.
func q4kPrefillFP8LTArchAdmission(major, minor int) CUDAArmAdmission {
	sm := major*10 + minor
	if sm <= 0 {
		return CUDAArmAdmission{Enabled: false, Reason: "device compute capability unknown; fp8lt arm refuses without an arch"}
	}
	if err := CheckCUDAQ4KPrefillArm(CUDAQ4KPrefillFP8LT, sm); err != nil {
		return CUDAArmAdmission{Enabled: false, Reason: err.Error()}
	}
	return CUDAArmAdmission{Enabled: true}
}

// CUDAArchArmAdmissions returns the arch-level admission of every tracked CUDA feature arm
// on a device of compute capability (major, minor). Pure: no cgo, no GPU.
func CUDAArchArmAdmissions(major, minor int) map[CUDADeviceArm]CUDAArmAdmission {
	return map[CUDADeviceArm]CUDAArmAdmission{
		CUDADeviceArmMxFP8Cast:       mxfp8CastArchAdmission(major, minor),
		CUDADeviceArmQ4KPrefillFP8LT: q4kPrefillFP8LTArchAdmission(major, minor),
	}
}

// CUDABuildArches describes the SASS cubin set and optional PTX floor a CUDA build embeds.
// The zero value is the shipped full build: every cubin in cudaEmbeddedArches plus the
// compute_120 PTX floor. A single-arch build sets Cubins to its one cubin and leaves
// PTXFloor nil (no JIT floor) — the #4184 shape.
type CUDABuildArches struct {
	Cubins   []cudaCC // ascending; nil = the shipped full-build set
	PTXFloor *cudaCC  // nil = no embedded PTX floor
}

// withDefaults fills the zero-value cubin set with the shipped full-build table.
func (b CUDABuildArches) withDefaults() CUDABuildArches {
	if b.Cubins == nil {
		b.Cubins = cudaEmbeddedArches
	}
	return b
}

// classify reports how this build reaches a device of compute capability dev: native if an
// embedded cubin covers it, else JIT-from-PTX when a floor is present and the device sits at
// or above it, else uncovered.
func (b CUDABuildArches) classify(dev cudaCC) CUDACoverage {
	b = b.withDefaults()
	var floor *cudaCC
	if b.PTXFloor != nil {
		floor = b.PTXFloor
	}
	return classifyCUDA(dev, b.Cubins, floor)
}

// CUDADeviceInitFacts are the raw facts the cgo init captures from the driver before any
// mapping happens. Every field is a plain value, so the receipt builder stays pure and can
// be witnessed on any host with captured (or fake-injected) facts.
type CUDADeviceInitFacts struct {
	DeviceName    string // e.g. "NVIDIA GeForce RTX 5090"
	Major         int    // compute capability major (12 for sm_120)
	Minor         int    // compute capability minor (0 for sm_120)
	TotalMemBytes int64  // total device VRAM
	DriverVersion string // formatted driver version ("12.9"); "" when the driver did not report one
}

// CUDADeviceReceipt is the typed device receipt the CUDA backend emits at init: who the
// device is, how much VRAM it has, which driver answers for it, how this build reaches its
// arch (native / jit-from-ptx / uncovered), and the per-arm admission of every tracked
// feature arm — each enabled or explicitly refused-with-reason.
type CUDADeviceReceipt struct {
	Backend       string                             `json:"backend"`
	DeviceName    string                             `json:"device_name"`
	TotalMemBytes int64                              `json:"total_mem_bytes"`
	DriverVersion string                             `json:"driver_version,omitempty"`
	Tier          string                             `json:"tier"`
	Coverage      CUDACoverage                       `json:"coverage"`
	Arms          map[CUDADeviceArm]CUDAArmAdmission `json:"arms"`
}

// BuildCUDADeviceReceipt maps captured init facts into the typed receipt. Pure: no cgo, no
// GPU — the entire mapping is unit-witnessed on any host.
func BuildCUDADeviceReceipt(f CUDADeviceInitFacts, build CUDABuildArches) CUDADeviceReceipt {
	dev := cudaCC{f.Major, f.Minor}
	return CUDADeviceReceipt{
		Backend:       "cuda",
		DeviceName:    f.DeviceName,
		TotalMemBytes: f.TotalMemBytes,
		DriverVersion: f.DriverVersion,
		Tier:          "sm_" + strconv.Itoa(f.Major*10+f.Minor),
		Coverage:      build.classify(dev),
		Arms:          CUDAArchArmAdmissions(f.Major, f.Minor),
	}
}

// formatCUDADriverVersion renders a CUDA version integer (MAJOR*1000 + MINOR*10 + PATCH, the
// CUDA_VERSION layout cuDriverGetVersion returns) as "major.minor[.patch]": 12090 -> "12.9",
// 13000 -> "13.0", 12091 -> "12.9.1". Values below 1000 do not encode a plausible version and
// render as "" (unknown).
func formatCUDADriverVersion(v int) string {
	if v < 1000 {
		return ""
	}
	major, rest := v/1000, v%1000
	minor, patch := rest/10, rest%10
	if patch == 0 {
		return strconv.Itoa(major) + "." + strconv.Itoa(minor)
	}
	return strconv.Itoa(major) + "." + strconv.Itoa(minor) + "." + strconv.Itoa(patch)
}

// CUDADeviceReceiptProvider is the optional extension interface (the RankUploader pattern)
// implemented by backends that emit a typed device receipt. Consumers read it through
// CUDADeviceReceiptOf rather than asserting on the cgo-only backend type.
type CUDADeviceReceiptProvider interface {
	CUDADeviceReceipt() CUDADeviceReceipt
}

// CUDADeviceReceiptOf returns the backend's typed device receipt when it implements
// CUDADeviceReceiptProvider, so consumers (gpucheck parity, ops diagnostics) can read the
// receipt off any Backend without a hard dependency on the CUDA cgo layer.
func CUDADeviceReceiptOf(backend Backend) (CUDADeviceReceipt, bool) {
	provider, ok := backend.(CUDADeviceReceiptProvider)
	if !ok {
		return CUDADeviceReceipt{}, false
	}
	return provider.CUDADeviceReceipt(), true
}
