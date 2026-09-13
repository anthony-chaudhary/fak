package gateway

// unified_memory_admit.go — the LIVE proxy-path enforcement of the real-time Apple
// Silicon unified memory pressure governor (#12305). The macobs package owns the
// mechanism: a Darwin DISPATCH_SOURCE_TYPE_MEMORYPRESSURE subscriber drives a
// UnifiedMemoryPressureGovernor that, on CRITICAL pressure, pauses batch admissions
// and mints a structured 429 CAPACITY_BACKOFF backpressure response. This file is the
// missing half — it attaches that governor to the served request path so the 429 the
// issue specifies actually reaches OpenCode on the wire, instead of living only inside
// an in-package unit test.
//
// The contract mirrors the sibling zero-swap MemoryGovernor refusal (#12509) in
// session_admit.go: evalute at the pre-model session boundary, refuse with a typed
// SessionState whose Reason is the closed token, and let writeSessionRefusal render the
// wire form. A nil governor (the default) is inert, leaving the served request path
// byte-for-byte historical.

import (
	"github.com/anthony-chaudhary/fak/internal/macobs"
)

// UnifiedMemoryPressureGovernor returns the attached real-time unified memory pressure
// governor (#12305), or nil when the host never armed it.
func (s *Server) UnifiedMemoryPressureGovernor() *macobs.UnifiedMemoryPressureGovernor {
	if s == nil {
		return nil
	}
	s.unifiedMemoryGovernorMu.RLock()
	defer s.unifiedMemoryGovernorMu.RUnlock()
	return s.unifiedMemoryGovernor
}

// SetUnifiedMemoryPressureGovernor attaches or replaces the real-time unified memory
// pressure governor (#12305). A nil governor is inert (the default).
func (s *Server) SetUnifiedMemoryPressureGovernor(g *macobs.UnifiedMemoryPressureGovernor) {
	if s == nil {
		return
	}
	s.unifiedMemoryGovernorMu.Lock()
	s.unifiedMemoryGovernor = g
	s.unifiedMemoryGovernorMu.Unlock()
}

// unifiedMemoryGovernorRefusal evaluates the attached real-time unified memory pressure
// governor (#12305) at the pre-model session boundary. Under CRITICAL pressure the
// governor pauses admissions and this returns a SessionState carrying the closed
// CAPACITY_BACKOFF reason, which writeSessionRefusal renders as HTTP 429. Under WARN the
// governor throttles MTP depth and reclaims pages but still admits, so this returns nil.
func (s *Server) unifiedMemoryGovernorRefusal(trace string) *SessionState {
	if s == nil {
		return nil
	}
	gov := s.UnifiedMemoryPressureGovernor()
	if gov == nil {
		return nil
	}
	admission := gov.EvaluateBatchAdmission(macobs.OpenCodeBatchRequest{
		SessionID: trace,
	})
	if admission.Admitted {
		return nil
	}
	retryAfter := macobs.DefaultUnifiedMemoryGovernorConfig().BackpressureRetrySec
	if admission.Backpressure != nil && admission.Backpressure.RetryAfterSec > 0 {
		retryAfter = admission.Backpressure.RetryAfterSec
	}
	return &SessionState{
		TraceID:       trace,
		Run:           "throttled",
		Reason:        macobs.ReasonCapacityBackoff,
		RetryAfterSec: retryAfter,
	}
}
