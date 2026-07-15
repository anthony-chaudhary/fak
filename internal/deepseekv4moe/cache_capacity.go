package deepseekv4moe

import (
	"errors"
	"math"
)

var (
	// ErrInvalidCacheBudget means a byte input or model shape is outside the
	// positive domain required by the static admission calculation.
	ErrInvalidCacheBudget = errors.New("deepseekv4moe: invalid expert-cache budget")
	// ErrCacheBudgetOverflow means the declared reserves or model shape cannot
	// be represented without integer overflow.
	ErrCacheBudgetOverflow = errors.New("deepseekv4moe: expert-cache budget overflow")
	// ErrInsufficientCacheBudget means fixed residents and runtime reserve alone
	// exceed HBM, leaving no valid routed-expert cache plan.
	ErrInsufficientCacheBudget = errors.New("deepseekv4moe: insufficient expert-cache budget")
)

// ExpertCachePlan is a static byte-admission estimate for routed (layer,
// expert) groups. It does not model dequantization expansion, KV/activation
// storage, allocator or collective workspaces, cache hit rate, I/O, or
// throughput. Callers must include those unknowns conservatively in
// RuntimeReserveBytes until they have a runtime witness.
type ExpertCachePlan struct {
	TotalHBMBytes          int64
	NonRoutedResidentBytes int64
	RuntimeReserveBytes    int64
	RoutedExpertGroupBytes int64
	ResidentRoutedGroups   int64
	WholeExpertsPerLayer   int64
	UsedBytes              int64
	HeadroomBytes          int64
	ModelRoutedGroups      int64
}

// PlanExpertCache computes the maximum number of whole routed (layer, expert)
// groups that fit after fixed non-routed residents and a caller-declared
// runtime reserve. Capacity is capped at the model's layers*expertsPerLayer.
//
// All byte counts use signed integers deliberately: negative inputs fail
// closed rather than wrapping into a large apparent capacity.
func PlanExpertCache(totalHBMBytes, nonRoutedResidentBytes, runtimeReserveBytes, routedExpertGroupBytes int64, layers, expertsPerLayer int64) (ExpertCachePlan, error) {
	if totalHBMBytes <= 0 || nonRoutedResidentBytes < 0 || runtimeReserveBytes < 0 || routedExpertGroupBytes <= 0 || layers <= 0 || expertsPerLayer <= 0 {
		return ExpertCachePlan{}, ErrInvalidCacheBudget
	}
	if nonRoutedResidentBytes > math.MaxInt64-runtimeReserveBytes || layers > math.MaxInt64/expertsPerLayer {
		return ExpertCachePlan{}, ErrCacheBudgetOverflow
	}
	fixed := nonRoutedResidentBytes + runtimeReserveBytes
	if fixed >= totalHBMBytes {
		return ExpertCachePlan{}, ErrInsufficientCacheBudget
	}

	modelGroups := layers * expertsPerLayer
	groups := (totalHBMBytes - fixed) / routedExpertGroupBytes
	if groups > modelGroups {
		groups = modelGroups
	}
	if groups == 0 {
		return ExpertCachePlan{}, ErrInsufficientCacheBudget
	}
	// groups is bounded by an actual byte quotient, so this multiplication and
	// addition cannot exceed totalHBMBytes.
	used := fixed + groups*routedExpertGroupBytes
	return ExpertCachePlan{
		TotalHBMBytes:          totalHBMBytes,
		NonRoutedResidentBytes: nonRoutedResidentBytes,
		RuntimeReserveBytes:    runtimeReserveBytes,
		RoutedExpertGroupBytes: routedExpertGroupBytes,
		ResidentRoutedGroups:   groups,
		WholeExpertsPerLayer:   groups / layers,
		UsedBytes:              used,
		HeadroomBytes:          totalHBMBytes - used,
		ModelRoutedGroups:      modelGroups,
	}, nil
}
