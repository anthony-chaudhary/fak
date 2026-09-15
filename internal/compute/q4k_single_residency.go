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
const Q4KSingleResidencyEnv = "FAK_Q4K_FREE_CPU"

// Q4KSingleResidencyEnvEnabled reads the single-release knob from the environment. The env read
// lives here in ONE place so the Metal and Vulkan paths cannot drift on the spelling or the
// accepted truthy value.
func Q4KSingleResidencyEnvEnabled() bool {
	return os.Getenv(Q4KSingleResidencyEnv) == "1"
}

// Q4KSingleResidencyRelease decides whether a Q4_K weight's host packed bytes may be released
// after its device upload. It is PURE (tier + the already-read env flag) so it compiles and tests
// without the vulkan build tag or cgo.
//
// The release is safe ONLY on a tier that shares physical RAM with the host ("integrated:", the
// unified-memory APU case) because there the device copy and the host copy draw on the SAME pool,
// so dropping the host backing is what makes a 27B Q4_K fit a 64 GB Strix Halo. On a discrete
// device the host copy is separate resident DRAM and keeping it preserves a CPU fallback for free,
// so the release never fires. The tier string comes from the backend's physical-device probe
// (vulkan.go: "integrated:<device>" / "discrete:<device>"), never a product-name guess.
func Q4KSingleResidencyRelease(tier string, envEnabled bool) bool {
	return envEnabled && strings.HasPrefix(tier, "integrated:")
}
