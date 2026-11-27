package gateway

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/kvbudget"
)

// TierAdvice is the structured 429 downgrade/compaction guidance returned when a
// tiered admission request cannot be admitted in its requested tier.
type TierAdvice struct {
	Code            string                `json:"code"`
	Message         string                `json:"message"`
	RequestedTier   string                `json:"requested_tier"`
	RequestedTokens int                   `json:"requested_tokens"`
	AvailableTiers  []kvbudget.TierStatus `json:"available_tiers,omitempty"`
}

// TieredAdmission pairs a promoted SeqRequest with its acquired KV tier lease
// and admission verdict.
type TieredAdmission struct {
	Request SeqRequest
	Lease   kvbudget.TierLease
	Verdict kvbudget.TierVerdict
}

// SetTierPool attaches an optional multi-level KV tier pool to the admission
// controller. Passing nil detaches it and restores legacy single-pool behavior.
func (c *AdmissionController) SetTierPool(p *kvbudget.TierPool) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tierPool = p
	if p != nil {
		if c.tierLeases == nil {
			c.tierLeases = make(map[string]kvbudget.TierLease)
		}
		if c.leaseToTrace == nil {
			c.leaseToTrace = make(map[uint64]string)
		}
	}
}

// TierPoolSnapshot returns a point-in-time snapshot of the attached tier pool,
// or an inert snapshot with Enabled=false if no tier pool is configured.
func (c *AdmissionController) TierPoolSnapshot() kvbudget.TierPoolSnapshot {
	if c == nil {
		return kvbudget.TierPoolSnapshot{Enabled: false}
	}
	c.mu.Lock()
	p := c.tierPool
	c.mu.Unlock()
	if p == nil {
		return kvbudget.TierPoolSnapshot{Enabled: false}
	}
	return p.Snapshot()
}

// BuildTierAdvice translates a refused/exhausted kvbudget.TierVerdict into
// actionable client/agent downgrade or compaction guidance.
func BuildTierAdvice(verdict kvbudget.TierVerdict, requestedTokens int) TierAdvice {
	var workerAvail, microAvail int
	var workerMaxCtx, microMaxCtx int
	for _, st := range verdict.AvailableTiers {
		switch st.Name {
		case kvbudget.TierWorker:
			workerAvail = st.AvailableNow
			workerMaxCtx = st.MaxContext
		case kvbudget.TierMicro:
			microAvail = st.AvailableNow
			microMaxCtx = st.MaxContext
		}
	}

	code := "WAIT_FOR_SLOT"
	msg := fmt.Sprintf("tier %q has no available capacity; wait for an active slot to release", verdict.Tier)

	switch verdict.Tier {
	case kvbudget.TierMaster:
		if workerAvail > 0 {
			code = "DOWNGRADE_TO_WORKER"
			msg = fmt.Sprintf("master tier exhausted; %d worker slots available (max %d tokens)", workerAvail, workerMaxCtx)
		} else if microAvail > 0 {
			code = "SPLIT_TO_MICRO"
			msg = fmt.Sprintf("master and worker tiers exhausted; %d micro slots available (max %d tokens)", microAvail, microMaxCtx)
		} else if verdict.AdviceCode == "EXCEEDS_TIER_MAX_CONTEXT" {
			code = "COMPACT_CONTEXT"
			msg = fmt.Sprintf("requested %d tokens exceeds master tier context cap (%d); compact context", requestedTokens, verdict.CapTokens)
		}
	case kvbudget.TierWorker:
		if verdict.AdviceCode == "EXCEEDS_TIER_MAX_CONTEXT" && microAvail == 0 {
			code = "COMPACT_CONTEXT"
			msg = fmt.Sprintf("requested %d tokens exceeds worker tier context cap (%d); compact context", requestedTokens, verdict.CapTokens)
		} else if microAvail > 0 {
			code = "SPLIT_TO_MICRO"
			msg = fmt.Sprintf("worker tier exhausted; %d micro slots available (max %d tokens)", microAvail, microMaxCtx)
		} else if verdict.AdviceCode == "EXCEEDS_TIER_MAX_CONTEXT" {
			code = "COMPACT_CONTEXT"
			msg = fmt.Sprintf("requested %d tokens exceeds worker tier context cap (%d); compact context", requestedTokens, verdict.CapTokens)
		}
	case kvbudget.TierMicro:
		if verdict.AdviceCode == "EXCEEDS_TIER_MAX_CONTEXT" {
			code = "COMPACT_CONTEXT"
			msg = fmt.Sprintf("requested %d tokens exceeds micro tier context cap (%d); compact context", requestedTokens, verdict.CapTokens)
		}
	default:
		if workerAvail > 0 {
			code = "DOWNGRADE_TO_WORKER"
			msg = fmt.Sprintf("requested tier %q unavailable; %d worker slots available (max %d tokens)", verdict.Tier, workerAvail, workerMaxCtx)
		} else if microAvail > 0 {
			code = "SPLIT_TO_MICRO"
			msg = fmt.Sprintf("requested tier %q unavailable; %d micro slots available (max %d tokens)", verdict.Tier, microAvail, microMaxCtx)
		} else if verdict.AdviceCode == "EXCEEDS_TIER_MAX_CONTEXT" {
			code = "COMPACT_CONTEXT"
			msg = fmt.Sprintf("requested %d tokens exceeds tier context cap; compact context", requestedTokens)
		}
	}

	return TierAdvice{
		Code:            code,
		Message:         msg,
		RequestedTier:   verdict.Tier,
		RequestedTokens: requestedTokens,
		AvailableTiers:  verdict.AvailableTiers,
	}
}

func (c *AdmissionController) resolveTierLocked(req SeqRequest) (string, bool) {
	if req.ContextTier != "" {
		return req.ContextTier, true
	}
	return c.tierPool.Classify(req.Tokens)
}

func (c *AdmissionController) releaseTierLeaseLocked(traceID string) {
	if c.tierPool == nil || len(c.tierLeases) == 0 {
		return
	}
	if lease, ok := c.tierLeases[traceID]; ok {
		delete(c.tierLeases, traceID)
		delete(c.leaseToTrace, lease.ID)
		c.tierPool.Release(lease)
	}
}

func (c *AdmissionController) effectiveMaxQueuedTokensLocked() int {
	limit := c.policy.MaxQueuedTokens
	if limit <= 0 || c.tierPool == nil || c.policy.TokenBudgetProvenance != "default" || limit != DefaultAdmissionPolicy().MaxQueuedTokens {
		return limit
	}
	snap := c.tierPool.Snapshot()
	if snap.TotalTokens > limit {
		return snap.TotalTokens
	}
	return limit
}

func adviceCodeFromSnapshot(snap kvbudget.TierPoolSnapshot, requestedTier string) string {
	for _, st := range snap.Tiers {
		if st.Name == requestedTier {
			continue
		}
		if st.AvailableNow > 0 {
			return "TIER_EXHAUSTED_DOWNGRADE_AVAILABLE"
		}
	}
	return "POOL_EXHAUSTED"
}

func (c *AdmissionController) hasWaitingForTierLocked(tier string) bool {
	for _, w := range c.waiting {
		wTier, ok := c.resolveTierLocked(w.req)
		if ok && wTier == tier {
			return true
		}
	}
	return false
}

// OfferTiered offers a request to the admission gate with tier-aware,
// non-head-of-line scheduling. When no TierPool is attached, it delegates to
// Offer so legacy behavior is 100% unchanged.
func (c *AdmissionController) OfferTiered(req SeqRequest) (AdmissionVerdict, kvbudget.TierLease, kvbudget.TierVerdict) {
	c.mu.Lock()
	if c.tierPool == nil {
		c.mu.Unlock()
		v := c.Offer(req)
		return v, kvbudget.TierLease{}, kvbudget.TierVerdict{Admitted: v == VerdictAdmitted}
	}
	defer c.mu.Unlock()

	now := c.nowLocked()
	if req.SessionID == "" {
		req.SessionID = baseTraceID(req.TraceID)
	}
	if req.CreatedAt.IsZero() {
		req.CreatedAt = now
	}
	if req.DecodeTTL > 0 && now.Sub(req.CreatedAt) >= req.DecodeTTL {
		c.stats.Expired++
		c.recordDecisionLocked(req, VerdictExpired, "", "")
		return VerdictExpired, kvbudget.TierLease{}, kvbudget.TierVerdict{Reason: "decode ttl expired"}
	}
	if req.Trust.Deny {
		c.stats.Denied++
		c.recordDecisionLocked(req, VerdictDenied, "", req.Trust.Reason)
		return VerdictDenied, kvbudget.TierLease{}, kvbudget.TierVerdict{Reason: req.Trust.Reason}
	}

	tier, ok := c.resolveTierLocked(req)
	if !ok {
		snap := c.tierPool.Snapshot()
		verdict := kvbudget.TierVerdict{
			Admitted:       false,
			Reason:         fmt.Sprintf("requested %d tokens exceeds all tier context caps", req.Tokens),
			AdviceCode:     "EXCEEDS_TIER_MAX_CONTEXT",
			AvailableTiers: snap.Tiers,
		}
		c.stats.Refused++
		c.recordDecisionLocked(req, VerdictRefused, "tier_context_cap", verdict.Reason)
		return VerdictRefused, kvbudget.TierLease{}, verdict
	}
	req.ContextTier = tier

	// Non-HoL fast path: only wait behind waiters of the SAME tier.
	if !c.hasWaitingForTierLocked(tier) {
		lease, verdict := c.tierPool.Acquire(tier, req.Tokens)
		if verdict.Admitted {
			if _, exists := c.running[req.TraceID]; exists {
				c.completeLocked(req.TraceID)
			}
			if c.running == nil {
				c.running = make(map[string]SeqRequest)
			}
			c.running[req.TraceID] = req
			c.tokens += req.Tokens
			if c.tierLeases == nil {
				c.tierLeases = make(map[string]kvbudget.TierLease)
			}
			if c.leaseToTrace == nil {
				c.leaseToTrace = make(map[uint64]string)
			}
			c.tierLeases[req.TraceID] = lease
			c.leaseToTrace[lease.ID] = req.TraceID
			c.stats.Admitted++
			c.recordDecisionLocked(req, VerdictAdmitted, "", "")
			return VerdictAdmitted, lease, verdict
		}
		if verdict.AdviceCode == "EXCEEDS_TIER_MAX_CONTEXT" || verdict.AdviceCode == "UNKNOWN_TIER" || verdict.AdviceCode == "INVALID_TOKENS" {
			c.stats.Refused++
			c.recordDecisionLocked(req, VerdictRefused, "tier_pool", verdict.Reason)
			return VerdictRefused, kvbudget.TierLease{}, verdict
		}
	}

	if c.policy.MaxWaiting > 0 && len(c.waiting) >= c.policy.MaxWaiting {
		snap := c.tierPool.Snapshot()
		verdict := kvbudget.TierVerdict{
			Admitted:       false,
			Reason:         "waiting queue at bound",
			Tier:           tier,
			AdviceCode:     adviceCodeFromSnapshot(snap, tier),
			AvailableTiers: snap.Tiers,
		}
		c.stats.Shed++
		c.recordDecisionLocked(req, VerdictShed, "max_waiting", verdict.Reason)
		return VerdictShed, kvbudget.TierLease{}, verdict
	}
	if maxQueued := c.effectiveMaxQueuedTokensLocked(); maxQueued > 0 && c.queuedTokens+req.Tokens > maxQueued {
		snap := c.tierPool.Snapshot()
		verdict := kvbudget.TierVerdict{
			Admitted:       false,
			Reason:         "queued token volume exceeds cap",
			Tier:           tier,
			AdviceCode:     adviceCodeFromSnapshot(snap, tier),
			AvailableTiers: snap.Tiers,
		}
		c.stats.Shed++
		c.recordDecisionLocked(req, VerdictShed, "max_queued_tokens", verdict.Reason)
		return VerdictShed, kvbudget.TierLease{}, verdict
	}

	c.waiting = append(c.waiting, waitEntry{req: req, enqueuedRound: c.round})
	c.queuedTokens += req.Tokens
	c.stats.Queued++
	c.recordDecisionLocked(req, VerdictQueued, "", "")
	return VerdictQueued, kvbudget.TierLease{}, kvbudget.TierVerdict{
		Admitted:       false,
		Reason:         fmt.Sprintf("queued in tier %q", tier),
		Tier:           tier,
		AvailableTiers: c.tierPool.Snapshot().Tiers,
	}
}

// ScheduleTiered runs one non-HoL admission round across all waiting tiers.
// A blocked L1 (master) waiter never blocks L2 (worker) or L3 (micro) waiters
// when their tier has floor or elastic capacity.
func (c *AdmissionController) ScheduleTiered() []TieredAdmission {
	c.mu.Lock()
	if c.tierPool == nil {
		reqs := c.scheduleLocked()
		c.mu.Unlock()
		out := make([]TieredAdmission, 0, len(reqs))
		for _, r := range reqs {
			out = append(out, TieredAdmission{
				Request: r,
				Verdict: kvbudget.TierVerdict{Admitted: true},
			})
		}
		return out
	}
	defer c.mu.Unlock()

	c.round++
	c.scanReservationsLocked()

	kept := c.waiting[:0]
	for _, e := range c.waiting {
		if e.ctx != nil && e.ctx.Err() != nil {
			c.queuedTokens -= e.req.Tokens
			if e.ready != nil {
				close(e.ready)
			}
			continue
		}
		kept = append(kept, e)
	}
	clear(c.waiting[len(kept):])
	c.waiting = kept
	if len(c.waiting) == 0 {
		return nil
	}

	sort.SliceStable(c.waiting, func(i, j int) bool {
		ei, ej := c.effectivePriorityLocked(c.waiting[i]), c.effectivePriorityLocked(c.waiting[j])
		if ei != ej {
			return ei < ej
		}
		if c.waiting[i].enqueuedRound != c.waiting[j].enqueuedRound {
			return c.waiting[i].enqueuedRound < c.waiting[j].enqueuedRound
		}
		return c.waiting[i].req.TraceID < c.waiting[j].req.TraceID
	})

	var admitted []TieredAdmission
	var remaining []waitEntry
	blockedTiers := make(map[string]bool)

	for _, e := range c.waiting {
		tier, ok := c.resolveTierLocked(e.req)
		if !ok {
			remaining = append(remaining, e)
			continue
		}
		if blockedTiers[tier] {
			remaining = append(remaining, e)
			continue
		}
		lease, verdict := c.tierPool.Acquire(tier, e.req.Tokens)
		if !verdict.Admitted {
			blockedTiers[tier] = true
			remaining = append(remaining, e)
			continue
		}
		if _, exists := c.running[e.req.TraceID]; exists {
			c.completeLocked(e.req.TraceID)
		}
		if c.running == nil {
			c.running = make(map[string]SeqRequest)
		}
		c.running[e.req.TraceID] = e.req
		c.tokens += e.req.Tokens
		if c.tierLeases == nil {
			c.tierLeases = make(map[string]kvbudget.TierLease)
		}
		if c.leaseToTrace == nil {
			c.leaseToTrace = make(map[uint64]string)
		}
		c.tierLeases[e.req.TraceID] = lease
		c.leaseToTrace[lease.ID] = e.req.TraceID
		c.queuedTokens -= e.req.Tokens
		c.stats.Admitted++
		c.recordDecisionLocked(e.req, VerdictAdmitted, "", "")
		if e.ready != nil {
			close(e.ready)
		}
		admitted = append(admitted, TieredAdmission{
			Request: e.req,
			Lease:   lease,
			Verdict: verdict,
		})
	}
	if c.queuedTokens < 0 {
		c.queuedTokens = 0
	}
	c.waiting = remaining
	return admitted
}

// CompleteTiered releases a TierLease back to the attached TierPool and
// removes the associated trace from the running set.
func (c *AdmissionController) CompleteTiered(lease kvbudget.TierLease) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.tierPool != nil {
		c.tierPool.Release(lease)
	}
	if traceID, ok := c.leaseToTrace[lease.ID]; ok {
		delete(c.leaseToTrace, lease.ID)
		delete(c.tierLeases, traceID)
		c.completeLocked(traceID)
	}
}
