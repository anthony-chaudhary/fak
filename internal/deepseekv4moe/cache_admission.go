package deepseekv4moe

import "errors"

// ErrRouteWorkingSetExceedsCache means the static byte plan cannot retain one
// complete layer's routed top-k selection. Such a plan would evict a selected
// group before the same layer could consume the full route, so admission fails
// before any simulated page-in.
var ErrRouteWorkingSetExceedsCache = errors.New("deepseekv4moe: route working set exceeds expert cache")

// ExpertCacheAdmission joins static byte admission to deterministic cache-trace
// control evidence. It remains weight-free: it proves neither model I/O nor
// dequant/GPU execution, output parity, latency, cache hit rate, or throughput.
type ExpertCacheAdmission struct {
	Plan  ExpertCachePlan
	Trace ExpertCacheTrace
}

// AdmitExpertCacheTrace computes a byte-bounded routed-group capacity and then
// simulates the supplied trace under that exact capacity. One full top-k route
// must fit concurrently; otherwise the call fails before simulation.
func AdmitExpertCacheTrace(totalHBMBytes, nonRoutedResidentBytes, runtimeReserveBytes, routedExpertGroupBytes int64, layers, expertsPerLayer, topK int, routes []ExpertRoute) (ExpertCacheAdmission, error) {
	plan, err := PlanExpertCache(totalHBMBytes, nonRoutedResidentBytes, runtimeReserveBytes, routedExpertGroupBytes, int64(layers), int64(expertsPerLayer))
	if err != nil {
		return ExpertCacheAdmission{}, err
	}
	if plan.ResidentRoutedGroups < int64(topK) {
		return ExpertCacheAdmission{}, ErrRouteWorkingSetExceedsCache
	}
	trace, err := SimulateExpertCache(routes, int(plan.ResidentRoutedGroups), layers, expertsPerLayer, topK)
	if err != nil {
		return ExpertCacheAdmission{}, err
	}
	if trace.PeakResident > int(plan.ResidentRoutedGroups) {
		return ExpertCacheAdmission{}, ErrRouteWorkingSetExceedsCache
	}
	return ExpertCacheAdmission{Plan: plan, Trace: trace}, nil
}
