package compute

import (
	"strconv"
	"strings"
)

// gpuBudgetAutoPercent is the conservative fraction of total device-local memory that
// FAK_GPU_BUDGET_MB=auto reserves for resident weights. The remainder is headroom for the KV
// cache, activations, and driver/runtime overhead, so a weight set that would otherwise fill VRAM
// spills its cold tail by choice (in upload order) instead of losing the allocation race.
const gpuBudgetAutoPercent = 90

// resolveGPUBudgetBytes maps FAK_GPU_BUDGET_MB (raw) plus the backend's own capacity into a
// device-local weight-residency budget in bytes. It is the tag-free core that both the CUDA and
// Vulkan backends delegate to, which is what makes the auto-derivation unit-testable without a real
// device — the caller injects the capacity it already queried at init:
//
//   - ""  / invalid / <= 0        -> 0 (unbounded; the prior behavior, byte-identical)
//   - "auto" + known capacity > 0 -> gpuBudgetAutoPercent% of totalDeviceLocal
//   - "auto" + unknown capacity   -> 0 (fail open to unbounded — never worse than today)
//   - a positive integer N        -> N MiB
//
// capacityKnown lets a backend distinguish "capacity queried as 0/overflow" from "capacity not
// probeable"; both resolve auto to unbounded, but keeping the flag explicit documents the fail-open.
func resolveGPUBudgetBytes(raw string, totalDeviceLocal int64, capacityKnown bool) int64 {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0
	}
	if strings.EqualFold(raw, "auto") {
		if !capacityKnown || totalDeviceLocal <= 0 {
			return 0 // capacity unknown -> unbounded, today's behavior
		}
		// Divide before multiply so a near-INT64_MAX capacity cannot overflow the product.
		return totalDeviceLocal / 100 * gpuBudgetAutoPercent
	}
	mb, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || mb <= 0 {
		return 0
	}
	return mb * 1024 * 1024
}
