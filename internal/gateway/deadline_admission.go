package gateway

import (
	"context"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// Deadline-aware admission for served chat requests.
//
// A client that declares a time budget (X-Fak-Deadline-Ms, X-Request-Timeout,
// or the X-Stainless-Timeout the OpenAI/Anthropic SDKs send) gets two
// guarantees from this node:
//
//  1. The request's context carries that deadline, so prefill, decode, and
//     any proxied upstream call (managed llama-server included) are canceled
//     when it passes, even if the client keeps its socket open.
//  2. A request whose completion estimate, from the measured per-request
//     prefill and decode throughput rescaled to the live in-flight count,
//     exceeds the remaining budget is refused up front with a typed 503 +
//     Retry-After (pkg/deadlineadmit wire contract), so a router fails over
//     instead of the node computing tokens nobody will read.
//
// A request with no declared budget is admitted unchanged, and a node that
// has not measured any throughput yet admits everything.

// deadlineEstimator returns the node-wide estimator, creating it on first use
// so a directly constructed gatewayMetrics works too.
func (m *gatewayMetrics) deadlineEstimator() *deadlineadmit.Estimator {
	if m == nil {
		return nil
	}
	m.deadlineOnce.Do(func() {
		if m.deadlineEst == nil {
			m.deadlineEst = deadlineadmit.NewEstimator(deadlineadmit.Config{}, nil)
		}
	})
	return m.deadlineEst
}

// observeDeadlineTiming feeds one finished turn into the estimator. ttft <= 0
// (a buffered reply) is recorded as decode over the whole duration.
func (m *gatewayMetrics) observeDeadlineTiming(promptTok, complTok int, dur, ttft time.Duration) {
	est := m.deadlineEstimator()
	if est == nil || dur <= 0 {
		return
	}
	o := deadlineadmit.Observation{PromptTokens: promptTok, CompletionTokens: complTok, Decode: dur}
	if ttft > 0 && ttft < dur {
		o.Prefill = ttft
		o.Decode = dur - ttft
	}
	est.Observe(o)
}

// admitClientDeadline runs deadline-aware admission for one served request.
// On refusal it has already written the typed 503 and returns ok=false. On
// admission it returns the request (carrying the client deadline when one was
// declared) and a release func the caller must defer.
func (s *Server) admitClientDeadline(w http.ResponseWriter, r *http.Request, arrived time.Time, promptTok, maxTokens int) (*http.Request, func(), bool) {
	est := s.metrics.deadlineEstimator()
	if est == nil {
		return r, func() {}, true
	}
	budget, has := deadlineadmit.Budget(r.Header)
	remaining := budget - time.Since(arrived)
	v := est.Admit(promptTok, maxTokens, remaining, has)
	if !v.Admit {
		s.logf("gateway: deadline admission refused: reason=%s estimate=%s remaining=%s in_flight=%d",
			v.Reason, v.Estimate, v.Remaining, est.InFlight())
		deadlineadmit.WriteRefusal(w, v)
		return r, nil, false
	}
	end := est.Begin()
	if !has {
		return r, end, true
	}
	ctx, cancel := context.WithDeadline(r.Context(), arrived.Add(budget))
	return r.WithContext(ctx), func() { end(); cancel() }, true
}
