package gateway

import (
	"fmt"
	"net/http"
	"strings"
)

const serveStatusSchema = "fak.serve.status.v1"

// ServeStatusSnapshot is the JSON document returned by GET /v1/serve/status.
// It surfaces the solved Inkling-2 serving envelope and live cohort/autotune
// counters for operator observability and router admission probes.
type ServeStatusSnapshot struct {
	Schema                  string `json:"schema"`
	Model                   string `json:"model,omitempty"`
	Engine                  string `json:"engine,omitempty"`
	MaxCtx                  int    `json:"max_ctx,omitempty"`
	KVDtype                 string `json:"kv_dtype,omitempty"`
	DefaultChunk            int    `json:"default_chunk,omitempty"`
	MinChunk                int    `json:"min_chunk,omitempty"`
	TargetConcurrency       int    `json:"target_concurrency,omitempty"`
	LongCtxC1Threshold      int    `json:"long_ctx_c1_threshold"`
	ChunkStepdownsTotal     uint64 `json:"chunk_stepdowns_total"`
	C1LongCtxFallbacksTotal uint64 `json:"c1_long_ctx_fallbacks_total"`
	LastCohortWidth         int64  `json:"last_cohort_width"`
	LastChunkSize           int64  `json:"last_chunk_size"`
}

// SetServeStatusProvider installs (or clears with nil) the live serve-status
// snapshot provider for /v1/serve/status and /metrics.
func (s *Server) SetServeStatusProvider(fn func() ServeStatusSnapshot) {
	if s == nil {
		return
	}
	s.serveStatusProvider = fn
}

func (s *Server) serveStatusSnapshot() (ServeStatusSnapshot, bool) {
	if s != nil && s.serveStatusProvider != nil {
		snap := s.serveStatusProvider()
		if snap.Schema == "" {
			snap.Schema = serveStatusSchema
		}
		if snap.Model == "" {
			snap.Model = strings.TrimSpace(s.model)
		}
		if snap.Engine == "" {
			snap.Engine = strings.TrimSpace(s.engineID)
		}
		return snap, true
	}
	return ServeStatusSnapshot{
		Schema: serveStatusSchema,
		Model:  strings.TrimSpace(s.model),
		Engine: strings.TrimSpace(s.engineID),
	}, false
}

func (s *Server) handleServeStatus(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	snap, _ := s.serveStatusSnapshot()
	writeJSON(w, http.StatusOK, snap)
}

func (s *Server) writeInkling2AutotuneMetrics(b *strings.Builder) {
	snap, ok := s.serveStatusSnapshot()
	if !ok {
		return
	}
	writeCounter(b, "fak_inkling2_chunk_stepdowns_total", "Total number of Inkling-2 prefill chunk step-downs due to memory pressure.", int64(snap.ChunkStepdownsTotal))
	writeCounter(b, "fak_inkling2_c1_long_ctx_fallbacks_total", "Total number of Inkling-2 C=N to C=1 long-context fallbacks.", int64(snap.C1LongCtxFallbacksTotal))
	writeHelpType(b, "fak_inkling2_last_cohort_width", "Width of the most recently executed Inkling-2 cohort.", "gauge")
	fmt.Fprintf(b, "fak_inkling2_last_cohort_width %d\n", snap.LastCohortWidth)
	writeHelpType(b, "fak_inkling2_last_chunk_size", "Prefill chunk size selected for the most recent Inkling-2 cohort.", "gauge")
	fmt.Fprintf(b, "fak_inkling2_last_chunk_size %d\n", snap.LastChunkSize)
}
