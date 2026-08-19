package gateway

// TrajctlMetrics is a bounded-label projection supplied by the host trajectory ledger.
type TrajctlMetrics struct {
	Objectives map[string]int
	Scores     map[string]float64
	Signals    map[string]int
	Nudges     map[string]int
}
type TrajctlMetricsFunc func() TrajctlMetrics

func (s *Server) trajctlMetricsSnapshot() TrajctlMetrics {
	if s == nil || s.trajctlMetrics == nil {
		return TrajctlMetrics{}
	}
	return s.trajctlMetrics()
}
