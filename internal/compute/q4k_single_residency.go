package compute

import (
	"os"
	"strings"
)

// Q4KSingleResidencyEnv is the single operator switch that arms the Q4_K single-residency
// release: if a Q4_K weight's host packed bytes are dropped once an authoritative device copy
// exists, the model holds ~half the host RAM (a 27B Q4_K fits one 64 GB Strix Halo). It is shared
// with the Metal analogue (metal_q4k_on.go, #1067) so operators keep ONE switch; enabling it arms
// BOTH the Metal and the Vulkan release, each behind its own backend-specific safety gate.
//
// Values: "1" arms the unified-memory (integrated:) release only; Q4KSingleResidencyDeviceValue
// ("device") also arms it on a discrete device-memory backend (CUDA, discrete Vulkan), where the
// host copy is separate DRAM that no device-routed matmul reads. Metal reads only "1".
const Q4KSingleResidencyEnv = "FAK_Q4K_FREE_CPU"

// Q4KSingleResidencyDeviceValue is the FAK_Q4K_FREE_CPU value that extends the release to any
// device-memory tier, not only integrated: APUs.
const Q4KSingleResidencyDeviceValue = "device"

// Q4KSingleResidencyMode is the parsed FAK_Q4K_FREE_CPU arm.
type Q4KSingleResidencyMode uint8

const (
	// Q4KSingleResidencyOff keeps every host copy (default).
	Q4KSingleResidencyOff Q4KSingleResidencyMode = iota
	// Q4KSingleResidencyIntegrated releases only on an integrated: (unified-memory) tier.
	Q4KSingleResidencyIntegrated
	// Q4KSingleResidencyDevice releases on any device-memory tier, integrated or discrete.
	Q4KSingleResidencyDevice
)

// ParseQ4KSingleResidencyMode maps a FAK_Q4K_FREE_CPU value to its arm; unknown values are off.
func ParseQ4KSingleResidencyMode(v string) Q4KSingleResidencyMode {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1":
		return Q4KSingleResidencyIntegrated
	case Q4KSingleResidencyDeviceValue:
		return Q4KSingleResidencyDevice
	default:
		return Q4KSingleResidencyOff
	}
}

// Q4KSingleResidencyModeFromEnv reads the release arm from the environment. The env read lives
// here in ONE place so the Metal, Vulkan and CUDA paths cannot drift on spelling.
func Q4KSingleResidencyModeFromEnv() Q4KSingleResidencyMode {
	return ParseQ4KSingleResidencyMode(os.Getenv(Q4KSingleResidencyEnv))
}

// Q4KSingleResidencyEnvEnabled reports whether any release arm is on.
func Q4KSingleResidencyEnvEnabled() bool {
	return Q4KSingleResidencyModeFromEnv() != Q4KSingleResidencyOff
}

// Q4KSingleResidencyRelease decides whether a Q4_K weight's host packed bytes may be released
// after its device upload under the integrated-only arm. It is PURE (tier + the already-read env
// flag) so it compiles and tests without the vulkan build tag or cgo.
//
// The release is safe on a tier that shares physical RAM with the host ("integrated:", the
// unified-memory APU case) because there the device copy and the host copy draw on the SAME pool,
// so dropping the host backing is what makes a 27B Q4_K fit a 64 GB Strix Halo. The tier string
// comes from the backend's physical-device probe (vulkan.go: "integrated:<device>" /
// "discrete:<device>"), never a product-name guess.
func Q4KSingleResidencyRelease(tier string, envEnabled bool) bool {
	return envEnabled && strings.HasPrefix(tier, "integrated:")
}

// Q4KSingleResidencyReleaseMode is the mode-aware form of Q4KSingleResidencyRelease. Under
// Q4KSingleResidencyDevice a backend with its own device memory (deviceMemory) and a probed,
// non-empty tier (CUDA "sm_120", Vulkan "discrete:<gpu>") also releases: on the Raider 5090 the
// retained host copy of a 27B Q4_K is ~15 GiB of Go heap that only a CPU fallback would read.
func Q4KSingleResidencyReleaseMode(tier string, deviceMemory bool, mode Q4KSingleResidencyMode) bool {
	switch mode {
	case Q4KSingleResidencyIntegrated:
		return Q4KSingleResidencyRelease(tier, true)
	case Q4KSingleResidencyDevice:
		return Q4KSingleResidencyRelease(tier, true) || (deviceMemory && strings.TrimSpace(tier) != "")
	default:
		return false
	}
}
