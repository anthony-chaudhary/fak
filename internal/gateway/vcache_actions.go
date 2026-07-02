package gateway

import (
	"net/http"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// handleFakVCacheActions serves the live provider-cache action planner over the
// same rolling turn window that backs /v1/fak/vcache/score. It is read-only:
// heartbeat/explicit-cache rows remain gated until a caller has provider
// capability and byte-identical prefix evidence.
func (s *Server) handleFakVCacheActions(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	writeJSON(w, http.StatusOK, s.vcacheActionPlan())
}

func (s *Server) vcacheActionPlan() vcacheobserve.ProviderActionPlan {
	if s == nil {
		return vcacheobserve.PlanProviderActions(nil, false)
	}
	turns, capped := s.VCacheTurnsSnapshot()
	return vcacheobserve.PlanProviderActions(turns, capped)
}
