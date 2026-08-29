package model

import "github.com/anthony-chaudhary/fak/internal/compute"

func (s *Session) retireRequestResources() {
	if s == nil || s.Backend == nil {
		return
	}
	compute.RetireBackendRequestResources(s.Backend)
}
