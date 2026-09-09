//go:build wip_metal_mtp

package gateway

import (
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	// HeaderSpeculative is the response header reporting the speculative decoding engine used.
	HeaderSpeculative = "x-fak-speculative"
	// SpeculativeMTPMetal indicates speculative decoding via the resident Metal MTP coordinator.
	SpeculativeMTPMetal = "mtp-metal"
)

// isMetalMTPActive reports whether Metal MTP speculative decoding is configured
// and active for the specified request model on this server.
func (s *Server) isMetalMTPActive(reqModel string) bool {
	if s == nil {
		return false
	}
	p := s.plannerForRequest(reqModel)
	if coord := metalMTPCoordinatorFromPlanner(p); coord != nil {
		return true
	}
	return s.metalMTPCoord != nil
}

// activeMetalMTPCoordinator extracts the active MetalMTPCoordinator from Server or its planner.
func (s *Server) activeMetalMTPCoordinator() *model.MetalMTPCoordinator {
	if s == nil {
		return nil
	}
	if s.metalMTPCoord != nil {
		return s.metalMTPCoord
	}
	return metalMTPCoordinatorFromPlanner(s.planner)
}

// MetalMTPCoordinator returns the active MetalMTPCoordinator for the server, if any.
func (s *Server) MetalMTPCoordinator() *model.MetalMTPCoordinator {
	if s == nil {
		return nil
	}
	if coord := metalMTPCoordinatorFromPlanner(s.planner); coord != nil {
		return coord
	}
	return s.metalMTPCoord
}

// SetMetalMTPCoordinator configures an explicit MetalMTPCoordinator on the server
// and its underlying in-kernel planner.
func (s *Server) SetMetalMTPCoordinator(c *model.MetalMTPCoordinator) {
	if s == nil {
		return
	}
	s.metalMTPCoord = c
	if s.planner != nil {
		if ikp, ok := s.planner.(interface {
			SetMetalMTPCoordinator(*model.MetalMTPCoordinator)
		}); ok {
			ikp.SetMetalMTPCoordinator(c)
		} else if dp, ok := s.planner.(*DualPlanner); ok {
			if ikp, ok := dp.Local().(interface {
				SetMetalMTPCoordinator(*model.MetalMTPCoordinator)
			}); ok {
				ikp.SetMetalMTPCoordinator(c)
			}
		}
	}
}

// plannerForRequest returns the sub-planner serving reqModel, resolving DualPlanner routing if present.
func (s *Server) plannerForRequest(reqModel string) agent.Planner {
	if s == nil || s.planner == nil {
		return nil
	}
	if dp, ok := s.planner.(*DualPlanner); ok {
		if dp.RoutesLocal(reqModel) {
			return dp.Local()
		}
		return dp.Proxy()
	}
	return s.planner
}

// metalMTPCoordinatorFromPlanner extracts the MetalMTPCoordinator from an agent.Planner if supported.
func metalMTPCoordinatorFromPlanner(p agent.Planner) *model.MetalMTPCoordinator {
	if p == nil {
		return nil
	}
	if m, ok := p.(interface {
		MetalMTPCoordinator() *model.MetalMTPCoordinator
	}); ok {
		return m.MetalMTPCoordinator()
	}
	return nil
}
