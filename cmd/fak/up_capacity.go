package main

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macfit"
)

// turnkey capacity model (#13075): the turnkey server decodes every concurrent
// session against ONE loaded model, so its real capacity is bounded by the KV
// pool it split out of unified memory at plan time. Without this, an N-session
// fan-out silently inflates every client's TTFT because all sessions share the
// same KV arena with no admission, fairness, or backpressure.
//
// The model derives a per-session KV budget from the admitted context and the
// plan's KV residency envelope (KVPoolBytes / KVBytesPerToken), admits up to
// MaxSessions concurrent requests, and sheds the (N+1)th with a typed 429 +
// Retry-After rather than blocking it unboundedly.
//
// Backward compatibility: a plan that declares no KV pool (mock path, legacy
// harnesses, `&turnkeyServer{...}` in tests) yields MaxSessions == 0, which the
// gate reads as "unbounded" so existing single-session tests are unaffected.

type turnkeyCapacity struct {
	// MaxSessions is how many concurrent requests the KV pool admits. 0 means
	// unbounded (no envelope declared → legacy behavior).
	MaxSessions int
	// PerSessionKVBytes is the KV budget reserved for each admitted session,
	// derived from the admitted context and the KV precision.
	PerSessionKVBytes uint64
	// KVBudgetBytes is the total KV pool the server can hand out.
	KVBudgetBytes uint64
}

// capacity derives the server's session admission model from its execution plan.
// It reads the plan under s.mu because plan is set once at boot but read on the
// request path.
func (s *turnkeyServer) capacity() turnkeyCapacity {
	if s == nil {
		return turnkeyCapacity{}
	}
	s.mu.Lock()
	plan := s.plan
	s.mu.Unlock()
	return turnkeyCapacityFromPlan(plan)
}

// turnkeyCapacityFromPlan is the pure derivation so it can be unit-tested and
// reused without a server.
func turnkeyCapacityFromPlan(plan macfit.TurnkeyProfile) turnkeyCapacity {
	perSession := plan.ContextBudgetTokens * plan.KVBytesPerToken
	if perSession == 0 || plan.KVPoolBytes == 0 {
		// No KV envelope (mock/legacy): unbounded, preserve legacy behavior.
		return turnkeyCapacity{}
	}
	max := int(plan.KVPoolBytes / perSession)
	if max < 1 {
		// The pool cannot even hold one full-context session; admit exactly one
		// rather than collapsing to zero (which would mean "unbounded").
		max = 1
	}
	return turnkeyCapacity{
		MaxSessions:       max,
		PerSessionKVBytes: perSession,
		KVBudgetBytes:     plan.KVPoolBytes,
	}
}

// capacityStats is the read-only telemetry snapshot surfaced on /healthz.
type turnkeyCapacityStats struct {
	MaxSessions       int     `json:"max_sessions"`
	LiveSessions      int     `json:"live_sessions"`
	AdmittedTotal     int64   `json:"admitted_total"`
	ShedTotal         int64   `json:"shed_total"`
	KVBudgetBytes     uint64  `json:"kv_budget_bytes"`
	KVInUseBytes      uint64  `json:"kv_in_use_bytes"`
	KVUtilization     float64 `json:"kv_utilization"`
	PerSessionKVBytes uint64  `json:"per_session_kv_bytes"`
}

// capacityStats reads the live admission accounting. It takes s.mu, so callers
// must not hold it.
func (s *turnkeyServer) capacityStats() turnkeyCapacityStats {
	if s == nil {
		return turnkeyCapacityStats{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.capacityStatsLocked()
}

func (s *turnkeyServer) capacityStatsLocked() turnkeyCapacityStats {
	cap := turnkeyCapacityFromPlan(s.plan)
	inUse := uint64(s.activeRequests) * cap.PerSessionKVBytes
	util := 0.0
	if cap.KVBudgetBytes > 0 {
		util = float64(inUse) / float64(cap.KVBudgetBytes)
		if util > 1 {
			util = 1
		}
	}
	return turnkeyCapacityStats{
		MaxSessions:       cap.MaxSessions,
		LiveSessions:      s.activeRequests,
		AdmittedTotal:     s.admittedTotal,
		ShedTotal:         s.shedTotal,
		KVBudgetBytes:     cap.KVBudgetBytes,
		KVInUseBytes:      inUse,
		KVUtilization:     util,
		PerSessionKVBytes: cap.PerSessionKVBytes,
	}
}

// admissionDecision is the typed outcome of a capacity admission attempt. It is
// a closed enum so the handler can emit a truthful HTTP status + retry hint
// instead of a bare bool that conflates "stopping" with "at capacity".
type admissionDecision int

const (
	admissionGranted admissionDecision = iota
	admissionStopping
	admissionAtCapacity
)

// turnkeyBackpressureRetryAfter is the bounded retry hint (seconds) returned on
// a shed so clients back off instead of hammering a saturated server.
const turnkeyBackpressureRetryAfter = 1

// admitChatRequest is the capacity-aware replacement for beginChatRequest. It
// returns a typed decision: granted (increment live sessions), stopping (shutdown
// in flight), or at-capacity (the KV pool is full → the handler sheds with 429).
func (s *turnkeyServer) admitChatRequest() admissionDecision {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopping {
		return admissionStopping
	}
	cap := turnkeyCapacityFromPlan(s.plan)
	if cap.MaxSessions > 0 && s.activeRequests >= cap.MaxSessions {
		s.shedTotal++
		return admissionAtCapacity
	}
	s.activeRequests++
	s.admittedTotal++
	return admissionGranted
}

// writeTurnkeyBackpressure writes the typed shed decision for a request that
// could not be admitted. It is emitted from the handler prologue, before any
// dispatch or SSE flush, so the status line is honest and the client can retry.
func writeTurnkeyBackpressure(w http.ResponseWriter, code string, maxSessions int) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", strconv.Itoa(turnkeyBackpressureRetryAfter))
	w.WriteHeader(http.StatusTooManyRequests)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
		"message": "server at capacity: " + strconv.Itoa(maxSessions) + " concurrent session(s) admitted; retry after the bounded backoff",
		"type":    "rate_limit_error",
		"code":    code,
	}})
}

// capacityRetryDuration exposes the retry hint as a time.Duration for callers
// that want to advertise it in a header computed elsewhere.
func capacityRetryDuration() time.Duration {
	return time.Duration(turnkeyBackpressureRetryAfter) * time.Second
}
