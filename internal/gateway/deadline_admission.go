package gateway

import (
	"context"
	"net/http"
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
// discounts a chat prompt only by the prefix a previous served turn of the same
// conversation reported as cached (deadline_cache_credit.go); a request's
// history alone, or an admitted-then-released request, earns no credit.

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

// admitClientDeadlineMessages is admitClientDeadlineChat for a request with no
// model or tools.
func (s *Server) admitClientDeadlineMessages(w http.ResponseWriter, r *http.Request, arrived time.Time, messages []agent.Message, maxTokens int) (*http.Request, func(), bool) {
	return s.admitClientDeadlineChat(w, r, arrived, "", nil, messages, maxTokens)
}

// admitClientDeadlineChat estimates a chat request, crediting only the prefix a
// served turn of the same conversation verified as cached
// (deadline_cache_credit.go). A request without that evidence is priced cold.
func (s *Server) admitClientDeadlineChat(w http.ResponseWriter, r *http.Request, arrived time.Time, model string, tools []agent.ToolDef, messages []agent.Message, maxTokens int) (*http.Request, func(), bool) {
	promptTok := estimateMessageContentTokens(messages)
	ledger := s.metrics.warmPrefixLedger()
	ticket := newDeadlineCacheTicket(model, tools, messages)
	credit, outcome := ledger.lookup(ticket, time.Now())
	r = r.WithContext(context.WithValue(r.Context(), deadlineCacheTicketKey{}, ticket))
	if credit > 0 {
		if est := s.metrics.deadlineEstimator(); est != nil {
			if budget, has := deadlineadmit.Budget(r.Header); has {
				remaining := budget - time.Since(arrived)
				bound := time.Duration(float64(remaining) * (1 + deadlineCacheCreditMaxColdOverrun))
				if remaining > 0 && !est.AdmitCached(promptTok, 0, maxTokens, bound, true).Admit {
					credit, outcome = 0, cacheCreditOverrun
					ledger.overrunDenied.Add(1)
				}
			}
		}
	}
	ticket.credited = credit
	return s.admitClientDeadlineCredited(w, r, arrived, promptTok, credit, maxTokens, outcome)
}

func (s *Server) admitClientDeadlineCached(w http.ResponseWriter, r *http.Request, arrived time.Time, promptTok, cachedTok, maxTokens int) (*http.Request, func(), bool) {
	return s.admitClientDeadlineCredited(w, r, arrived, promptTok, cachedTok, maxTokens, "")
}

func (s *Server) admitClientDeadlineCredited(w http.ResponseWriter, r *http.Request, arrived time.Time, promptTok, cachedTok, maxTokens int, creditOutcome string) (*http.Request, func(), bool) {
	est := s.metrics.deadlineEstimator()
	if est == nil {
		return r, func() {}, true
	}
	budget, has := deadlineadmit.Budget(r.Header)
	remaining := budget - time.Since(arrived)
	v := est.AdmitCached(promptTok, cachedTok, maxTokens, remaining, has)
	if !v.Admit {
		s.logf("gateway: deadline admission refused: reason=%s estimate=%s remaining=%s in_flight=%d prompt_tokens=%d predicted_cached=%d cache_credit=%s",
			v.Reason, v.Estimate, v.Remaining, est.InFlight(), promptTok, cachedTok, creditOutcome)
		deadlineadmit.WriteRefusal(w, v)
		return r, nil, false
	}
	if creditOutcome == cacheCreditCredited && cachedTok > 0 && has {
		ledger := s.metrics.warmPrefixLedger()
		ledger.creditedAdmits.Add(1)
		ledger.creditedTokens.Add(uint64(cachedTok))
		s.logf("gateway: deadline admission cache credit: credited_tokens=%d prompt_tokens=%d estimate=%s remaining=%s",
			cachedTok, promptTok, v.Estimate, v.Remaining)
	}
	end := est.Begin()
	if !has {
		return r, end, true
	}
	ctx, cancel := context.WithDeadline(r.Context(), arrived.Add(budget))
	return r.WithContext(ctx), func() { end(); cancel() }, true
}
