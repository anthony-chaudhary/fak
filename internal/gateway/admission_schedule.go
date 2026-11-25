package gateway

import (
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// Schedule runs ONE admission round: it advances the aging clock, orders the waiting queue
// by the configured priority, EDF, or SLO policy, and promotes waiters while the token
// budget and the max-num-seqs cap have headroom — stopping at the first request that does
// not fit so a lower-ranked request cannot bypass the selected queue head
// (head-of-line discipline). It returns the requests admitted this round, in admission
// order. The native loop calls it each iteration; a host with a freed slot calls it after
// Complete. Each call advances priority aging; in EDF mode that aging resolves
// equal deadlines and cannot overtake an earlier deadline; in SLO mode the
// violation-margin order stands on its own (equal margins fall back to arrival
// order then TraceID).
func (c *AdmissionController) Schedule() []SeqRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.scheduleLocked()
}

func (c *AdmissionController) scheduleLocked() []SeqRequest {
	c.round++
	// Drive the scheduler's advisory known-coming scan every round — before the
	// empty-queue early return — so a hint-only round still mints its bounded readback
	// while never touching admission capacity (pub#13388).
	c.scanReservationsLocked()
	// A disconnected caller may not have resumed its Acquire goroutine yet. Prune
	// its waiter before choosing a winner so it cannot consume capacity or block
	// a live request under either priority or weighted-fair ordering.
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
		c.credits = make(map[string]int64)
		return nil
	}

	isEDF := c.policy.Mode == AdmissionModeEDF
	isSLO := c.policy.Mode == AdmissionModeSLO
	isWeightedFair := !isEDF && !isSLO && (c.orderPolicy == session.WeightedFair || (c.scheduler != nil && c.scheduler.Policy() == session.WeightedFair))
	if !isWeightedFair {
		sort.SliceStable(c.waiting, func(i, j int) bool {
			if isSLO {
				if before, decided := c.sloOrderLocked(c.waiting[i], c.waiting[j]); decided {
					return before
				}
			}
			if isEDF {
				if before, decided := deadlineOrder(c.waiting[i].req, c.waiting[j].req); decided {
					return before
				}
			}
			ei, ej := c.effectivePriorityLocked(c.waiting[i]), c.effectivePriorityLocked(c.waiting[j])
			if ei != ej {
				return ei < ej // lower effective priority value served first
			}
			if before, decided := deadlineOrder(c.waiting[i].req, c.waiting[j].req); decided {
				return before
			}
			if c.waiting[i].enqueuedRound != c.waiting[j].enqueuedRound {
				return c.waiting[i].enqueuedRound < c.waiting[j].enqueuedRound // older waiter first
			}
			return c.waiting[i].req.TraceID < c.waiting[j].req.TraceID // deterministic final tiebreak
		})
		var admitted []SeqRequest
		tickPrompt := 0
		for _, e := range c.waiting {
			check := c.batchBudgetStatusLocked(e.req)
			if check.status >= batchBudgetExhausted {
				break // head-of-line: do not let a lower-priority request skip a blocked one
			}
			if !c.prefillFitsLocked(tickPrompt, e.req) {
				break
			}
			if c.admitLocked(e.req) != nil {
				break
			}
			tickPrompt += e.req.promptTokens()
			c.queuedTokens -= e.req.Tokens
			if e.ready != nil {
				close(e.ready)
			}
			admitted = append(admitted, e.req)
			if check.status == batchBudgetReached {
				break
			}
		}
		if c.queuedTokens < 0 {
			c.queuedTokens = 0
		}
		// The admitted set is the sorted prefix; keep the rest. Copy so the released entries
		// are not retained by the backing array across later Offer appends.
		c.waiting = append([]waitEntry(nil), c.waiting[len(admitted):]...)
		return admitted
	}

	var admitted []SeqRequest
	tickPrompt := 0
	for len(c.waiting) > 0 {
		var traceKeys []string
		traceFirstReq := make(map[string]SeqRequest)
		for _, w := range c.waiting {
			tid := w.req.SessionID
			if tid == "" {
				tid = baseTraceID(w.req.TraceID)
			}
			if _, seen := traceFirstReq[tid]; !seen {
				traceKeys = append(traceKeys, tid)
				traceFirstReq[tid] = w.req
			}
		}

		candidates := make([]session.State, 0, len(traceKeys))
		for _, tid := range traceKeys {
			priority := traceFirstReq[tid].Priority
			if c.table != nil {
				st := c.table.Get(tid)
				if st.Rev > 0 {
					priority = st.Priority
				}
			}
			candidates = append(candidates, session.State{
				TraceID:  tid,
				Run:      session.Running,
				Priority: priority,
			})
		}

		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].Priority != candidates[j].Priority {
				return candidates[i].Priority < candidates[j].Priority
			}
			return candidates[i].TraceID < candidates[j].TraceID
		})

		winnerID, ok := c.pickWeightedFairWinnerLocked(candidates)
		if !ok {
			break
		}

		winnerIdx := -1
		for i, w := range c.waiting {
			tid := w.req.SessionID
			if tid == "" {
				tid = baseTraceID(w.req.TraceID)
			}
			if tid == winnerID {
				winnerIdx = i
				break
			}
		}
		if winnerIdx == -1 {
			break
		}

		e := c.waiting[winnerIdx]
		check := c.batchBudgetStatusLocked(e.req)
		if check.status >= batchBudgetExhausted {
			break
		}
		if !c.prefillFitsLocked(tickPrompt, e.req) {
			break
		}
		if c.admitLocked(e.req) != nil {
			break
		}
		tickPrompt += e.req.promptTokens()
		c.queuedTokens -= e.req.Tokens
		if e.ready != nil {
			close(e.ready)
		}
		admitted = append(admitted, e.req)
		copy(c.waiting[winnerIdx:], c.waiting[winnerIdx+1:])
		c.waiting[len(c.waiting)-1] = waitEntry{}
		c.waiting = c.waiting[:len(c.waiting)-1]

		if check.status == batchBudgetReached {
			break
		}
	}
	if c.queuedTokens < 0 {
		c.queuedTokens = 0
	}
	return admitted
}

func (c *AdmissionController) pickWeightedFairWinnerLocked(candidates []session.State) (string, bool) {
	if len(candidates) == 0 {
		return "", false
	}
	if c.scheduler != nil {
		winner, ok := c.scheduler.PickWeightedFair(candidates)
		if ok {
			return winner.TraceID, true
		}
	}
	if c.credits == nil {
		c.credits = make(map[string]int64)
	}
	maxPriority := candidates[0].Priority
	for _, st := range candidates {
		if st.Priority > maxPriority {
			maxPriority = st.Priority
		}
	}
	next := make(map[string]int64, len(candidates))
	var total int64
	bestIdx := -1
	var bestCredit int64
	for i, st := range candidates {
		w := int64(maxPriority-st.Priority) + 1
		total += w
		cred := c.credits[st.TraceID] + w
		next[st.TraceID] = cred
		if bestIdx == -1 || cred > bestCredit {
			bestIdx, bestCredit = i, cred
		}
	}
	winner := candidates[bestIdx]
	next[winner.TraceID] -= total
	c.credits = next
	return winner.TraceID, true
}

// ExpireRequests inspects waiting and running requests against now, pruning any request
// whose per-request DecodeTTL has elapsed without disrupting the healthy scheduler.
func (c *AdmissionController) ExpireRequests(now time.Time) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if now.IsZero() {
		now = c.nowLocked()
	}

	var expired []string

	// 1. Prune expired waiters
	var remaining []waitEntry
	for _, w := range c.waiting {
		if w.req.DecodeTTL > 0 && !w.req.CreatedAt.IsZero() && now.Sub(w.req.CreatedAt) >= w.req.DecodeTTL {
			expired = append(expired, w.req.TraceID)
			c.queuedTokens -= w.req.Tokens
			c.stats.Expired++
			if w.ready != nil {
				close(w.ready)
			}
		} else {
			remaining = append(remaining, w)
		}
	}
	c.waiting = remaining
	if c.queuedTokens < 0 {
		c.queuedTokens = 0
	}

	// 2. Prune expired running requests
	for traceID, req := range c.running {
		if req.DecodeTTL > 0 && !req.CreatedAt.IsZero() && now.Sub(req.CreatedAt) >= req.DecodeTTL {
			expired = append(expired, traceID)
			delete(c.running, traceID)
			c.tokens -= req.Tokens
			c.stats.Expired++
		}
	}

	if len(expired) > 0 {
		c.scheduleLocked()
	}

	return expired
}

// Complete releases a running request's budget when its sequence finishes (the loop's
// per-lane reclaim edge — modelengine.NativeScheduler.finish). It returns true if the
// trace was running. The freed headroom is taken by the next Schedule.
func (c *AdmissionController) Complete(traceID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.completeLocked(traceID)
}

func (c *AdmissionController) completeAndSchedule(traceID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	ok := c.completeLocked(traceID)
	if ok {
		c.scheduleLocked()
	}
	return ok
}

func (c *AdmissionController) completeLocked(traceID string) bool {
	req, ok := c.running[traceID]
	if !ok {
		return false
	}
	delete(c.running, traceID)
	c.tokens -= req.Tokens
	if c.tokens < 0 {
		c.tokens = 0
	}
	return true
}

func (c *AdmissionController) cancelAdmission(traceID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, e := range c.waiting {
		if e.req.TraceID == traceID {
			c.queuedTokens -= e.req.Tokens
			if c.queuedTokens < 0 {
				c.queuedTokens = 0
			}
			copy(c.waiting[i:], c.waiting[i+1:])
			c.waiting[len(c.waiting)-1] = waitEntry{}
			c.waiting = c.waiting[:len(c.waiting)-1]
			// Removing a blocked queue head can make existing capacity usable.
			c.scheduleLocked()
			return
		}
	}
	if req, ok := c.running[traceID]; ok {
		if c.pool != nil {
			c.pool.Return(req.Tokens)
		}
		c.completeLocked(traceID)
		c.scheduleLocked()
	}
}
