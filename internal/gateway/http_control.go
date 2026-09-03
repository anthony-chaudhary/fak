package gateway

import (
	"encoding/json"
	"net/http"
	"strconv"
)

// handleControlConfig serves GET /v1/control/config and PATCH /v1/control/config (and their /v1/fak/... twins).
// GET returns the active ScalarConfig and current ConfigEpoch with X-Fak-Config-Epoch.
// PATCH validates and applies partial updates atomically, increments ConfigEpoch, and returns
// the updated configuration with X-Fak-Config-Epoch and X-Fak-Witness headers.
func (s *Server) handleControlConfig(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		vc := s.VersionedConfig()
		w.Header().Set("X-Fak-Config-Epoch", strconv.FormatUint(vc.Epoch, 10))
		writeJSON(w, http.StatusOK, ControlConfigResponse{
			ConfigEpoch: vc.Epoch,
			Config:      vc.Config,
		})
	case http.MethodPatch:
		dec := json.NewDecoder(r.Body)
		var patch ScalarConfigPatch
		if err := dec.Decode(&patch); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
			return
		}
		updated, epoch, err := s.PatchScalarConfig(patch)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		w.Header().Set("X-Fak-Config-Epoch", strconv.FormatUint(epoch, 10))
		w.Header().Set("X-Fak-Witness", "verified-atomic-swap")
		writeJSON(w, http.StatusOK, ControlConfigResponse{
			Status:      "applied",
			ConfigEpoch: epoch,
			Config:      *updated,
		})
	default:
		writeErr(w, http.StatusMethodNotAllowed, "use GET or PATCH")
	}
}
