package model

// Q4KExpertStats is the session-local witness surface for the Qwen3.6/q4_k_m MoE
// decode lever: routed experts are the denominator, and MetalFusedQ6KDownExperts
// is the numerator for the Q6_K-down fused expert path.
type Q4KExpertStats struct {
	RoutedExperts              int64 `json:"routed_experts"`
	MetalFusedQ6KDownExperts   int64 `json:"metal_fused_q6k_down_experts"`
	MetalFusedQ6KDownBatches   int64 `json:"metal_fused_q6k_down_batches"`
	MetalFusedQ6KDownFallbacks int64 `json:"metal_fused_q6k_down_fallbacks"`
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

func (s *Session) recordMetalFusedQ6KDownFallback(n int) {
	if s == nil || n <= 0 {
		return
	}
	s.q4kExpertStats.MetalFusedQ6KDownFallbacks += int64(n)
}
