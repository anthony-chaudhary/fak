package gateway

import (
	"encoding/json"
	"net/http"
	"strings"
)

type cachePostureResponse struct {
	Mode string `json:"mode"`
}

func (s *Server) handleFakCachePosture(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/v1/fak/cache/posture" {
		http.NotFound(w, r)
		return
	}
	if r.Method == http.MethodGet {
		writeCachePosture(w, s.cacheTTL1H.Load())
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "GET, POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req cachePostureResponse
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "invalid cache posture body", http.StatusBadRequest)
		return
	}
	var next bool
	switch strings.ToLower(strings.TrimSpace(req.Mode)) {
	case "on":
		next = true
	case "off":
		next = false
	default:
		http.Error(w, `mode must be "on" or "off"; "auto" is launch-resolved`, http.StatusBadRequest)
		return
	}
	prior := s.cacheTTL1H.Swap(next)
	s.logf("fak gateway: managed-cache posture %s->%s", cachePostureMode(prior), cachePostureMode(next))
	writeCachePosture(w, next)
}

func writeCachePosture(w http.ResponseWriter, active bool) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(cachePostureResponse{Mode: cachePostureMode(active)})
}

func cachePostureMode(active bool) string {
	if active {
		return "on"
	}
	return "off"
}
