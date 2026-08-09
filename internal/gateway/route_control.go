package gateway

import "github.com/anthony-chaudhary/fak/internal/modelroute"

// RouteWatcher exposes the installed route-manifest watcher to trusted control-plane
// adapters. A nil result means hot reload was not configured for this server.
func (s *Server) RouteWatcher() *modelroute.Watcher {
	if s == nil {
		return nil
	}
	return s.currentRouteWatcher()
}
