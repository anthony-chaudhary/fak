package model

// Q4KExpertStats is the session-local witness surface for the Qwen3.6/q4_k_m MoE
// decode lever: routed experts are the denominator, and MetalFusedQ6KDownExperts
// is the numerator for the Q6_K-down fused expert path.
type Q4KExpertStats struct {
	RoutedExperts              int64 `json:"routed_experts"`
	MetalFusedQ6KDownExperts   int64 `json:"metal_fused_q6k_down_experts"`
	MetalFusedQ6KDownBatches   int64 `json:"metal_fused_q6k_down_batches"`
	MetalFusedQ6KDownFallbacks int64 `json:"metal_fused_q6k_down_fallbacks"`

	// RoutedExpertsDeviceHAL counts routed experts whose gate/up/down projections AND
	// SwiGLU all executed on compute.Backend through expertSwiGLUHAL — the resident
	// device routed path. RoutedExpertsOffHAL counts every routed expert that took any
	// other route (the Metal fused MLP, the host k-quant GEMV, or the generic mulGroup
	// dispatch). They are recorded at the single branch point in expertSwiGLU, so their
	// sum is every expert that function ran on a session-bearing kernel.
	//
	// The pair exists because throughput alone cannot answer "did the device path run".
	// #4843 landed a resident-CUDA routed-expert path that was unreachable dead code on
	// the live glm_moe_dsa decode for three weeks: the DGX evidence showed only tok/s,
	// which is equally consistent with a slow device path and a host scalar batch. A
	// CUDA glm_moe_dsa run reporting RoutedExpertsDeviceHAL == 0 with a nonzero
	// RoutedExpertsOffHAL is running the host path, whatever its tok/s says (#5111).
	RoutedExpertsDeviceHAL int64 `json:"routed_experts_device_hal"`
	RoutedExpertsOffHAL    int64 `json:"routed_experts_off_hal"`
}

// RoutedExpertDeviceHALFraction returns the share of routed experts that executed on
// the backend through expertSwiGLUHAL. A zero denominator reports 0 rather than NaN.
// 1.0 on a CUDA glm_moe_dsa decode is the reachability witness; 0.0 is the #5111
// dead-path condition.
func (s Q4KExpertStats) RoutedExpertDeviceHALFraction() float64 {
	total := s.RoutedExpertsDeviceHAL + s.RoutedExpertsOffHAL
	if total == 0 {
		return 0
	}
	return float64(s.RoutedExpertsDeviceHAL) / float64(total)
}

// MetalFusedQ6KDownFraction returns the fired-expert fraction for the Metal
// Q6_K-down fused path. A zero denominator reports 0 rather than NaN.
func (s Q4KExpertStats) MetalFusedQ6KDownFraction() float64 {
	if s.RoutedExperts == 0 {
		return 0
	}
	return float64(s.MetalFusedQ6KDownExperts) / float64(s.RoutedExperts)
}

// Q4KExpertStats returns a copy of this session's resident-Q4_K expert counters.
func (s *Session) Q4KExpertStats() Q4KExpertStats {
	if s == nil {
		return Q4KExpertStats{}
	}
	return s.q4kExpertStats
}

// ResetQ4KExpertStats clears the session-local counters so a caller can measure one
// prompt/decode window without carrying warm-up traffic into the fraction.
func (s *Session) ResetQ4KExpertStats() {
	if s == nil {
		return
	}
	s.q4kExpertStats = Q4KExpertStats{}
}

func (s *Session) recordQ4KExpertRoute(n int) {
	if s == nil || n <= 0 {
		return
	}
	s.q4kExpertStats.RoutedExperts += int64(n)
}

func (s *Session) recordMetalFusedQ6KDownExperts(n int, batched bool) {
	if s == nil || n <= 0 {
		return
	}
	s.q4kExpertStats.MetalFusedQ6KDownExperts += int64(n)
	if batched {
		s.q4kExpertStats.MetalFusedQ6KDownBatches++
	}
}

// recordRoutedExpertDeviceHAL / recordRoutedExpertOffHAL are the two outcomes of
// expertSwiGLU's device-HAL branch. Both are nil-safe on the receiver: residentKernel
// and splitKernel carry no session (the host-pinned --cpu-offload-experts regime), so
// the call is a no-op there rather than a branch at every call site.
func (s *Session) recordRoutedExpertDeviceHAL(n int) {
	if s == nil || n <= 0 {
		return
	}
	s.q4kExpertStats.RoutedExpertsDeviceHAL += int64(n)
}

func (s *Session) recordRoutedExpertOffHAL(n int) {
	if s == nil || n <= 0 {
		return
	}
	s.q4kExpertStats.RoutedExpertsOffHAL += int64(n)
}

func (s *Session) recordMetalFusedQ6KDownFallback(n int) {
	if s == nil || n <= 0 {
		return
	}
	s.q4kExpertStats.MetalFusedQ6KDownFallbacks += int64(n)
}
