package gateway

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
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
//
// Observation measures prefill throughput over uncached tokens only. Admission
// can discount tokens when the caller supplies current resident-prefix evidence.
// Direct native execution supplies evidence only after acquiring request-owned
// state. Other chat routes are credited only from the upstream engine's own
// /slots evidence (deadline_residency.go); a previous request's history or
// successful return alone does not prove current compatible residency.

type deadlineAdmissionError struct {
	verdict deadlineadmit.Verdict
}

func (e *deadlineAdmissionError) Error() string {
	return "deadline admission: " + string(e.verdict.Reason)
}

// bindClientDeadline covers routing and request preparation even before an
// estimator exists. WithDeadline preserves an earlier incoming context deadline.
func bindClientDeadline(r *http.Request, arrived time.Time) (*http.Request, context.CancelFunc) {
	budget, has := deadlineadmit.Budget(r.Header)
	if !has {
		return r, func() {}
	}
	ctx, cancel := context.WithDeadline(r.Context(), arrived.Add(budget))
	return r.WithContext(ctx), cancel
}

func (s *Server) admitRoutedClientDeadline(w http.ResponseWriter, r *http.Request, arrived time.Time, messages []agent.Message, maxTokens int) (*http.Request, func(), bool) {
	return s.admitRoutedClientDeadlineModel(w, r, arrived, "", messages, maxTokens)
}

// admitRoutedClientDeadlineModel postpones only a qualified direct native
// verdict. The final prompt and owned cache state are known at execution, after
// all gateway rewrites. Every other route is admitted at ingress, credited only
// by measured residency evidence for the same model; no structural lookup or
// historical cache observation earns credit.
func (s *Server) admitRoutedClientDeadlineModel(w http.ResponseWriter, r *http.Request, arrived time.Time, model string, messages []agent.Message, maxTokens int) (*http.Request, func(), bool) {
	planner, native := s.chatPlanner(r.Context()).(*agent.InKernelPlanner)
	_, hasBudget := deadlineadmit.Budget(r.Header)
	if !native || !hasBudget || !planner.ExecutionDeadlineAdmissionSupported() {
		return s.admitClientDeadlineChat(w, r, arrived, model, messages, maxTokens)
	}
	est := s.metrics.deadlineEstimator()
	if est == nil {
		return r, func() {}, true
	}
	var mu sync.Mutex
	var end func()
	released := false
	requestContext := r.Context()
	check := func(ctx context.Context, prompt, cached, output int) error {
		mu.Lock()
		defer mu.Unlock()
		if released {
			return context.Canceled
		}
		if err := ctx.Err(); err != nil && !errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		// A first-token watchdog owns a shorter child context. Its timeout
		// remains an upstream stall, not exhaustion of the client's deadline.
		if errors.Is(ctx.Err(), context.DeadlineExceeded) && !errors.Is(requestContext.Err(), context.DeadlineExceeded) {
			return context.DeadlineExceeded
		}
		deadline, bound := ctx.Deadline()
		if !bound {
			return context.DeadlineExceeded
		}
		// Re-evaluate every attempt, including OOM retries, with time spent in
		// cloning/restoring/setup already consumed. After Begin, retry estimates
		// conservatively include this request in the existing in-flight count.
		v := est.AdmitCached(prompt, cached, output, time.Until(deadline), true)
		if !v.Admit {
			s.logf("gateway: deadline admission refused: reason=%s estimate=%s remaining=%s in_flight=%d prompt_tokens=%d predicted_cached=%d credit_source=execution-owned-prefix",
				v.Reason, v.Estimate, v.Remaining, est.InFlight(), prompt, cached)
			return &deadlineAdmissionError{verdict: v}
		}
		if end == nil {
			end = est.Begin()
		}
		return nil
	}
	release := func() {
		mu.Lock()
		defer mu.Unlock()
		if !released {
			released = true
			if end != nil {
				end()
			}
		}
	}
	return r.WithContext(planner.WithExecutionDeadlineAdmission(r.Context(), check)), release, true
}

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

// observeDeadlineTiming feeds one finished turn into the estimator. promptTok
// is the full resident prompt and cachedTok the part of it served from the KV
// cache. ttft <= 0 (a buffered reply) is recorded as decode over the whole
// duration.
func (m *gatewayMetrics) observeDeadlineTiming(promptTok, cachedTok, complTok int, dur, ttft time.Duration) {
	est := m.deadlineEstimator()
	if est == nil || dur <= 0 {
		return
	}
	o := deadlineadmit.Observation{PromptTokens: promptTok, CachedTokens: cachedTok, CompletionTokens: complTok, Decode: dur}
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
	return s.admitClientDeadlineCached(w, r, arrived, promptTok, 0, maxTokens)
}

func (s *Server) admitClientDeadlineMessages(w http.ResponseWriter, r *http.Request, arrived time.Time, messages []agent.Message, maxTokens int) (*http.Request, func(), bool) {
	return s.admitClientDeadlineChat(w, r, arrived, "", messages, maxTokens)
}

// admitClientDeadlineChat admits a chat request for model, crediting only
// measured, route-bound residency evidence (deadline_residency.go). On a
// measured, healthy finish it records this request's prefix for the next turn.
func (s *Server) admitClientDeadlineChat(w http.ResponseWriter, r *http.Request, arrived time.Time, model string, messages []agent.Message, maxTokens int) (*http.Request, func(), bool) {
	promptTok := estimateMessageContentTokens(messages)
	chain := deadlinePrefixChain(messages)
	if len(chain) == 0 {
		return s.admitClientDeadlineCached(w, r, arrived, promptTok, 0, maxTokens)
	}
	planner, _ := s.chatPlanner(r.Context()).(*agent.HTTPPlanner)
	ticket := &deadlineResidencyTicket{key: chain[len(chain)-1], estTok: promptTok, model: model, planner: planner}
	ticket.credited = s.deadlineResidencyCredit(r, ticket, chain, arrived, maxTokens)
	r, release, ok := s.admitClientDeadlineCached(w, r, arrived, promptTok, ticket.credited, maxTokens)
	if !ok {
		return r, release, false
	}
	r = r.WithContext(context.WithValue(r.Context(), deadlineResidencyCtxKey{}, ticket))
	ctx := r.Context()
	return r, func() {
		s.releaseDeadlineResidency(ctx, ticket)
		release()
	}, true
}

func (s *Server) admitClientDeadlineCached(w http.ResponseWriter, r *http.Request, arrived time.Time, promptTok, cachedTok, maxTokens int) (*http.Request, func(), bool) {
	est := s.metrics.deadlineEstimator()
	if est == nil {
		return r, func() {}, true
	}
	budget, has := deadlineadmit.Budget(r.Header)
	remaining := budget - time.Since(arrived)
	v := est.AdmitCached(promptTok, cachedTok, maxTokens, remaining, has)
	if !v.Admit {
		s.logf("gateway: deadline admission refused: reason=%s estimate=%s remaining=%s in_flight=%d prompt_tokens=%d predicted_cached=%d",
			v.Reason, v.Estimate, v.Remaining, est.InFlight(), promptTok, cachedTok)
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
