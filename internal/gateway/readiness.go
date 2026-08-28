package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
)

type readinessCapture struct {
	header http.Header
	body   bytes.Buffer
	status int
}

func (c *readinessCapture) Header() http.Header {
	if c.header == nil {
		c.header = make(http.Header)
	}
	return c.header
}
func (c *readinessCapture) WriteHeader(status int) { c.status = status }
func (c *readinessCapture) Write(p []byte) (int, error) {
	if c.status == 0 {
		c.status = http.StatusOK
	}
	return c.body.Write(p)
}

// handleReady projects the existing health evaluation onto an orchestration
// readiness endpoint. It adds only the listener/startup gate recorded by
// MarkReady; model, warmup, served-failure, and provider gates remain owned by
// handleHealth.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	capture := &readinessCapture{}
	s.handleHealth(capture, r)

	var health map[string]any
	if err := json.Unmarshal(capture.body.Bytes(), &health); err != nil {
		writeErr(w, http.StatusInternalServerError, "readiness evaluation failed")
		return
	}

	startupReady := s != nil && !s.startup.snapshot().ready.IsZero()
	health["startup_ready"] = startupReady
	health["components"] = map[string]bool{
		"gateway":       startupReady,
		"agent_runtime": startupReady,
		"policy":        startupReady,
		"audit":         startupReady,
		"metrics":       startupReady,
	}
	ok, _ := health["ok"].(bool)
	if !startupReady {
		ok = false
		health["ok"] = false
	}

	status := http.StatusOK
	if !ok {
		status = http.StatusServiceUnavailable
	}
	writeJSON(w, status, health)
}
