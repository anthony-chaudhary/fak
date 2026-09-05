package allinone

import (
	"encoding/json"
	"net/http"
	"sync"
)

// Well-known mandatory subsystem names.
const (
	SubsystemHTTP        = "gateway"
	SubsystemInference   = "model_engine"
	SubsystemMCPBroker   = "mcp_broker"
	SubsystemMemoryStore = "memory_store"
)

// HealthAggregator coordinates and aggregates readiness state across all mandatory subsystems.
type HealthAggregator struct {
	mu         sync.RWMutex
	subsystems map[string]SubsystemStatus
}

// NewHealthAggregator initializes a HealthAggregator with all mandatory subsystems registered.
func NewHealthAggregator() *HealthAggregator {
	h := &HealthAggregator{
		subsystems: make(map[string]SubsystemStatus),
	}
	for _, name := range []string{SubsystemHTTP, SubsystemInference, SubsystemMCPBroker, SubsystemMemoryStore} {
		h.subsystems[name] = SubsystemStatus{Name: name, Ready: true}
	}
	return h
}

// SetStatus updates the readiness and error state of an individual subsystem.
func (h *HealthAggregator) SetStatus(name string, ready bool, errStr string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.subsystems[name] = SubsystemStatus{
		Name:  name,
		Ready: ready,
		Error: errStr,
	}
}

// GetStatus returns the current status of a subsystem.
func (h *HealthAggregator) GetStatus(name string) (SubsystemStatus, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	s, ok := h.subsystems[name]
	return s, ok
}

// Snapshot returns the aggregated health response. If all mandatory subsystems are ready,
// status is "ok". If any subsystem is not ready, status is "unavailable".
func (h *HealthAggregator) Snapshot() HealthResponse {
	h.mu.RLock()
	defer h.mu.RUnlock()

	sub := make(map[string]SubsystemStatus, len(h.subsystems))
	allReady := true
	for k, v := range h.subsystems {
		sub[k] = v
		if !v.Ready {
			allReady = false
		}
	}

	status := "ok"
	if !allReady {
		status = "unavailable"
	}

	return HealthResponse{
		Status:     status,
		Subsystems: sub,
	}
}

// Handler returns an HTTP handler for /healthz serving aggregated health status.
// If all mandatory subsystems are ready: returns 200 OK with JSON HealthResponse.
// If any mandatory subsystem fails: returns 503 Service Unavailable with JSON HealthResponse.
func (h *HealthAggregator) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		snap := h.Snapshot()
		w.Header().Set("Content-Type", "application/json")
		if snap.Status == "ok" {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
		_ = json.NewEncoder(w).Encode(snap)
	})
}
