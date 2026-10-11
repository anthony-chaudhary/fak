package gateway

import (
	"context"
	"crypto/sha256"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// prepareDeadlineSoftSlot defers only a potentially creditable continuation.
// Cold requests retain ingress admission. A deferred request must cross the
// same verdict before every prepared POST, with zero credit whenever shaping,
// routing, reservation or live engine evidence does not match.
func (s *Server) prepareDeadlineSoftSlot(r *http.Request, t *deadlineResidencyTicket, chain [][sha256.Size]byte, arrived time.Time, maxTokens int) (*http.Request, func(), bool) {
	if t.planner == nil {
		return r, nil, false
	}
	est := s.metrics.deadlineEstimator()
	e, found := s.metrics.deadlineResidencyMem().lookup(chain, t.model, t.planner, arrived)
	budget, hasBudget := deadlineadmit.Budget(r.Header)
	remaining := budget - time.Since(arrived)
	deferred := found && e.soft != nil && hasBudget && remaining > 0 && est != nil && est.InFlight() == 0
	if deferred {
		cold := est.AdmitCached(t.estTok, 0, maxTokens, remaining, true)
		deferred = cold.Estimate.Seconds() <= remaining.Seconds()*(1+deadlineCacheCreditMaxColdOverrun)
	}
	var check func(context.Context, []agent.Message, agent.LlamaSlotTask, bool) error
	var previous *agent.LlamaSoftSlotResidency
	var release func()
	ctx := r.Context()
	if deferred {
		previous = e.soft
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, arrived.Add(budget))
		var mu sync.Mutex
		var end func()
		released := false
		check = func(callContext context.Context, prepared []agent.Message, slot agent.LlamaSlotTask, proven bool) error {
			mu.Lock()
			defer mu.Unlock()
			if released {
				return context.Canceled
			}
			if err := callContext.Err(); err != nil && !errors.Is(err, context.DeadlineExceeded) {
				return err
			}
			// A shorter stream watchdog remains a transport timeout. Exhaustion
			// of the client budget still receives the existing typed verdict.
			if errors.Is(callContext.Err(), context.DeadlineExceeded) && !errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return context.DeadlineExceeded
			}
			remaining := budget - time.Since(arrived)
			if deadline, ok := callContext.Deadline(); ok && time.Until(deadline) < remaining {
				remaining = time.Until(deadline)
			}
			// Shaping may preserve the old prefix while growing the tail. Price
			// the actual prepared content with the same existing estimator,
			// never less conservatively than the ingress estimate.
			promptTok := max(t.estTok, estimateMessageContentTokens(prepared))
			credit := 0
			cold := est.AdmitCached(promptTok, 0, maxTokens, remaining, true)
			if proven && est.InFlight() == 0 && time.Since(e.at) <= deadlineResidencyTTL && cold.Estimate.Seconds() <= remaining.Seconds()*(1+deadlineCacheCreditMaxColdOverrun) {
				credit = min(e.tokens, slot.PromptTokens)
			}
			t.mu.Lock()
			t.credited = credit
			t.mu.Unlock()
			v := est.AdmitCached(promptTok, credit, maxTokens, remaining, true)
			if !v.Admit {
				s.logf("gateway: deadline admission refused: reason=%s estimate=%s remaining=%s in_flight=%d prompt_tokens=%d predicted_cached=%d credit_source=prepared-soft-slot",
					v.Reason, v.Estimate, v.Remaining, est.InFlight(), promptTok, credit)
				return &deadlineAdmissionError{verdict: v}
			}
			if end == nil {
				end = est.Begin()
			}
			return nil
		}
		release = func() {
			mu.Lock()
			defer mu.Unlock()
			if !released {
				released = true
				if end != nil {
					end()
				}
				cancel()
			}
		}
	}
	bound, observer := t.planner.WithLlamaSoftSlotResidency(ctx, previous, check)
	if observer == nil {
		if release != nil {
			release()
		}
		return r, nil, false
	}
	t.soft = observer
	return r.WithContext(bound), release, deferred
}
