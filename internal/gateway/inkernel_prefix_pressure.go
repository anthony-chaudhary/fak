package gateway

import "github.com/anthony-chaudhary/fak/internal/agent"

// InKernelKVPrefixPressureSource returns the physical prefix payload owner only
// for a direct native in-kernel planner. It intentionally does not unwrap proxy,
// dual, or upstream planners: provider KV counters are not fak-owned bytes.
func (s *Server) InKernelKVPrefixPressureSource() agent.KVPrefixPressureSource {
	if s == nil {
		return nil
	}
	planner, ok := s.planner.(*agent.InKernelPlanner)
	if !ok {
		return nil
	}
	return planner
}
