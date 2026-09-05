package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

type armLeaseCtxKey struct{}

func withArmLease(ctx context.Context, l *researcharm.RequestLease) context.Context {
	if ctx == nil || l == nil {
		return ctx
	}
	return context.WithValue(ctx, armLeaseCtxKey{}, l)
}

func armLeaseFromContext(ctx context.Context) *researcharm.RequestLease {
	if ctx == nil {
		return nil
	}
	l, _ := ctx.Value(armLeaseCtxKey{}).(*researcharm.RequestLease)
	return l
}

// SetResearchArmCoordinator attaches a research project arms coordinator to the server.
func (s *Server) SetResearchArmCoordinator(c *researcharm.Coordinator) {
	if s == nil {
		return
	}
	s.researchArmsMu.Lock()
	defer s.researchArmsMu.Unlock()
	s.researchArms = c
}

// ResearchArmCoordinator returns the attached research arm coordinator, or nil if none.
func (s *Server) ResearchArmCoordinator() *researcharm.Coordinator {
	if s == nil {
		return nil
	}
	s.researchArmsMu.RLock()
	defer s.researchArmsMu.RUnlock()
	return s.researchArms
}

// handleFakArms serves GET /v1/fak/arms, returning the snapshot of all research project arms,
// their current in-flight requests, and active leases.
func (s *Server) handleFakArms(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	c := s.ResearchArmCoordinator()
	if c == nil {
		writeJSON(w, http.StatusOK, researcharm.Snapshot{})
		return
	}
	writeJSON(w, http.StatusOK, c.Snapshot())
}

// handleFakArmsTraffic serves GET /v1/fak/arms/traffic, returning a list of currently active
// in-flight requests and their caller PIDs/endpoints.
func (s *Server) handleFakArmsTraffic(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	c := s.ResearchArmCoordinator()
	if c == nil {
		writeJSON(w, http.StatusOK, []researcharm.InflightRequest{})
		return
	}
	writeJSON(w, http.StatusOK, c.ActiveInflight())
}

// handleFakArmsLease serves GET, POST, and DELETE on /v1/fak/arms/lease.
func (s *Server) handleFakArmsLease(w http.ResponseWriter, r *http.Request) {
	c := s.ResearchArmCoordinator()
	if c == nil {
		writeErr(w, http.StatusServiceUnavailable, "researcharm: coordinator not configured")
		return
	}

	switch r.Method {
	case http.MethodGet:
		snap := c.Snapshot()
		writeJSON(w, http.StatusOK, snap.ActiveLeases)
	case http.MethodPost:
		var req researcharm.LeaseRequest
		if !decodeRequestBody(w, r, &req) {
			return
		}
		lease, err := c.AcquireLease(req)
		if err != nil {
			writeErr(w, http.StatusConflict, err.Error())
			return
		}
		writeJSON(w, http.StatusCreated, lease)
	case http.MethodDelete:
		id := strings.TrimSpace(r.URL.Query().Get("id"))
		token := strings.TrimSpace(r.URL.Query().Get("token"))
		if id == "" {
			var body struct {
				ID    string `json:"id"`
				Token string `json:"token"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			id = strings.TrimSpace(body.ID)
			token = strings.TrimSpace(body.Token)
		}
		if id == "" {
			writeErr(w, http.StatusBadRequest, "researcharm: lease id or arm id required")
			return
		}
		if err := c.ReleaseLease(id, token); err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"released": true, "id": id})
	default:
		writeErr(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

// handleFakArmsLimits serves POST /v1/fak/arms/limits to dynamically update an arm's concurrency limit.
func (s *Server) handleFakArmsLimits(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	c := s.ResearchArmCoordinator()
	if c == nil {
		writeErr(w, http.StatusServiceUnavailable, "researcharm: coordinator not configured")
		return
	}
	var req researcharm.LimitRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.ArmID) == "" {
		writeErr(w, http.StatusBadRequest, "researcharm: arm_id required")
		return
	}
	if err := c.SetLimit(req.ArmID, req.MaxConcurrency); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "arm_id": req.ArmID, "max_concurrency": req.MaxConcurrency})
}
