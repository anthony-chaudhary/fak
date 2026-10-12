package gateway

import (
	"context"
	"errors"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// A warmup that FAILED is terminal for this process: nothing re-runs it, so
// reporting it as warmup_pending told an operator or watchdog "keep waiting" for
// a serve that never becomes ready. The gate records the failure as a distinct
// readiness state with a closed reason token (branch on it, never on prose) and a
// bounded, scrubbed message. Readiness stays held — a failed backend is not warm —
// and a later successful RunWarmup still completes the gate and clears it.
const (
	// WarmupFailureTimeout: the synthetic forward or coherence probe did not answer
	// within the boot/warmup ceiling (ErrWarmupTimeout).
	WarmupFailureTimeout = "warmup_timeout"
	// WarmupFailureCanceled: the serve context was cancelled or expired mid-warmup.
	WarmupFailureCanceled = "canceled"
	// WarmupFailureBackendRefused: the backend forward failed closed with a typed
	// refusal (device-only path refused, unsupported Flash schedule, a V4.1
	// stage-admission or operation error the planner recovers as a typed request
	// error, or a natively unsupported model).
	WarmupFailureBackendRefused = "backend_refused"
	// WarmupFailureBackendError: any other error from the backend warmup forward.
	WarmupFailureBackendError = "backend_error"
	// WarmupFailureCoherenceProbe: the known-answer coherence probe errored.
	WarmupFailureCoherenceProbe = "coherence_probe_error"
	// WarmupFailureKVCapacity: the compressed KV warmup capacity could not derive a
	// token budget.
	WarmupFailureKVCapacity = "kv_capacity_unavailable"
)

// classifyWarmupFailure maps a warmup error onto the closed reason vocabulary.
// Timeout and cancellation win over the call-site hint because they describe why
// the stage stopped, not which stage it was; fallback names the stage otherwise.
func classifyWarmupFailure(err error, fallback string) string {
	switch {
	case errors.Is(err, ErrWarmupTimeout):
		return WarmupFailureTimeout
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return WarmupFailureCanceled
	}
	var capErr *fixedWarmupCapacityError
	if errors.As(err, &capErr) {
		return WarmupFailureKVCapacity
	}
	if fallback != "" {
		return fallback
	}
	if isTypedBackendRefusal(err) {
		return WarmupFailureBackendRefused
	}
	return WarmupFailureBackendError
}

func isTypedBackendRefusal(err error) bool {
	var forward *model.BackendForwardOperationError
	var projection *model.V41ProjectionOperationError
	var expert *model.V41ExpertOperationError
	var stage *model.V41ForwardError
	return errors.As(err, &forward) ||
		errors.As(err, &projection) ||
		errors.As(err, &expert) ||
		errors.As(err, &stage) ||
		errors.Is(err, model.ErrV41PrefillLatency) ||
		errors.Is(err, model.ErrV4FlashAttentionUnimplemented) ||
		errors.Is(err, model.ErrV4FlashAttentionRatioInvalid) ||
		errors.Is(err, model.ErrV41NativeUnsupported)
}

// failWarmup records err as a terminal warmup failure on the gate and returns it
// unchanged, so RunWarmup's callers see exactly the error they did before.
func (s *Server) failWarmup(err error, fallback string) error {
	if err == nil {
		return nil
	}
	s.warmup.markFailed(classifyWarmupFailure(err, fallback), scrubForbiddenDetail(err.Error()))
	return err
}

// markFailed records a terminal failure on an armed, incomplete gate. An unarmed
// gate stays silent (readiness unaffected, as before) and a completed gate never
// regresses. The latest failure wins.
func (g *warmupGate) markFailed(reason, message string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.armed || g.complete {
		return
	}
	g.failed, g.failReason, g.failMessage = true, reason, message
}

// failure reports the recorded terminal failure while the gate is still held.
func (g *warmupGate) failure() (reason, message string, failed bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.armed || g.complete || !g.failed {
		return "", "", false
	}
	return g.failReason, g.failMessage, true
}
