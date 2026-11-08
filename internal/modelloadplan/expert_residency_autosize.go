package modelloadplan

import "errors"

// ErrExpertResidencyAutosize marks invalid geometry or a budget below the
// router's minimum per-layer expert residency.
var ErrExpertResidencyAutosize = errors.New("modelloadplan: expert residency budget refused")

// ExpertResidencyAutosize is the bounded slot-budget component of SGLang's
// paged-experts auto policy at 8a0edcf799d06bdee8832356d382495e7fb50f22.
// The caller reserves KV and other fixed state first; this function allocates
// equal expert slots per layer from the remaining bytes. It does not choose
// expert identities, observe load, reserve device memory or change a loader.
// bytesPerExpert is the resident allocation per expert in each MoE layer.
func ExpertResidencyAutosize(freeBytes, reservedBytes, bytesPerExpert uint64, layers, experts, routerTopK int) (slots int, residentBytes uint64, err error) {
	if reservedBytes > freeBytes || bytesPerExpert == 0 || layers <= 0 || experts <= 0 || routerTopK <= 0 || routerTopK > experts {
		return 0, 0, ErrExpertResidencyAutosize
	}
	if uint64(layers) > ^uint64(0)/bytesPerExpert {
		return 0, 0, ErrExpertResidencyAutosize
	}
	perSlot := uint64(layers) * bytesPerExpert
	fitting := (freeBytes - reservedBytes) / perSlot
	if fitting < uint64(routerTopK) {
		return 0, 0, ErrExpertResidencyAutosize
	}
	if fitting > uint64(experts) {
		fitting = uint64(experts)
	}
	// fitting is bounded by positive int experts and by the budget quotient.
	return int(fitting), fitting * perSlot, nil
}
