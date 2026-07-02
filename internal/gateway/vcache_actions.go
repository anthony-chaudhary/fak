package gateway

import (
	"encoding/json"
	"net/http"
	"strings"

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
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.vcacheActionPlan(vcacheActionOptionsFromRequest(r)))
}

func (s *Server) vcacheActionPlan(opt vcacheobserve.ProviderActionOptions) vcacheobserve.ProviderActionPlan {
	if s == nil {
		return vcacheobserve.PlanProviderActionsWithOptions(nil, false, opt)
	}
	turns, capped := s.VCacheTurnsSnapshot()
	return vcacheobserve.PlanProviderActionsWithOptions(turns, capped, opt)
}

func vcacheActionOptionsFromRequest(r *http.Request) vcacheobserve.ProviderActionOptions {
	if r == nil {
		return vcacheobserve.ProviderActionOptions{}
	}
	q := r.URL.Query()
	return vcacheobserve.ProviderActionOptions{
		Transport: vcacheobserve.ProviderTransportWitness{
			HeartbeatTransport:     queryBool(q.Get("heartbeat_transport")) || queryBool(q.Get("heartbeat")),
			ExplicitCacheTransport: queryBool(q.Get("explicit_cache_transport")) || queryBool(q.Get("explicit_cache")),
			ByteIdenticalPrefix:    queryBool(q.Get("prefix_witness")) || queryBool(q.Get("byte_identical_prefix")),
			DeletionCapable:        queryBool(q.Get("deletion_capable")),
			Source:                 strings.TrimSpace(q.Get("transport_source")),
		},
	}
}

func queryBool(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "t", "yes", "y", "on":
		return true
	default:
		return false
	}
}
