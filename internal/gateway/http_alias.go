package gateway

// http_alias.go — the runtime model-alias management endpoint (#11091):
//
//	GET  /v1/fak/route/aliases  -> {"aliases":[{"name":...,"target":...}, ...]}
//	POST /v1/fak/route/aliases  -> {"name":"a","target":"b"} to set (hot reassign)
//	                               {"remove":"a"}             to remove
//
// The store is concurrency-safe (modelroute.AliasStore), so a POST here takes
// effect on the very next chat/tool resolution without a restart; the load-bearing
// redirect witness is the gateway chat-route test. An unconfigured server (no
// aliases) returns 404, so the surface is opt-in and inert by default.

import (
	"net/http"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// aliasListResponse is the GET body.
type aliasListResponse struct {
	Aliases []modelroute.Alias `json:"aliases"`
}

// aliasMutationRequest is the POST body. Exactly one of (name,target) or remove is
// expected; a request naming neither is a 400, never a silent no-op.
type aliasMutationRequest struct {
	Name   string `json:"name"`
	Target string `json:"target"`
	Remove string `json:"remove"`
}

// handleFakRouteAliases serves the runtime model-alias registry. It mirrors the
// management-handler shape of handleControlConfig (method switch) with the
// requireMethod/writeErr/writeJSON discipline of its siblings.
func (s *Server) handleFakRouteAliases(w http.ResponseWriter, r *http.Request) {
	if s.aliases == nil {
		writeErr(w, http.StatusNotFound, "alias registry is not configured")
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, aliasListResponse{Aliases: s.aliases.List()})
	case http.MethodPost:
		var req aliasMutationRequest
		if !decodeRequestBody(w, r, &req) {
			return
		}
		if req.Remove != "" {
			if !s.aliases.Remove(req.Remove) {
				writeErr(w, http.StatusNotFound, "alias not found: "+req.Remove)
				return
			}
			writeJSON(w, http.StatusOK, aliasListResponse{Aliases: s.aliases.List()})
			return
		}
		if req.Name == "" || req.Target == "" {
			writeErr(w, http.StatusBadRequest, "provide {name,target} to set or {remove} to remove")
			return
		}
		if err := s.aliases.Set(req.Name, req.Target); err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		s.logf("gateway: alias %q -> %q set at runtime", req.Name, req.Target)
		writeJSON(w, http.StatusOK, aliasListResponse{Aliases: s.aliases.List()})
	default:
		writeErr(w, http.StatusMethodNotAllowed, "use GET or POST")
	}
}
