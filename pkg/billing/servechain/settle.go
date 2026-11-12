package servechain

import (
	"errors"
	"math"
	"sync"
	"time"

	billing "github.com/anthony-chaudhary/fak/pkg/billing"
	billingledger "github.com/anthony-chaudhary/fak/pkg/billing/ledger"
	billingpricebook "github.com/anthony-chaudhary/fak/pkg/billing/pricebook"
)

// upstreamUsage is the OpenAI-compatible usage object an upstream reports on
// a non-streaming completion. Only the standard fields are mapped; the
// chain's cache-write line stays zero because the OpenAI-compatible shape
// does not report one.
type upstreamUsage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitempty"`
}

// counts maps the reported usage onto the price book's billing lines:
// cached prompt tokens are the cache-read line, the remainder of the prompt
// (floored at zero — a misbehaving upstream must not produce a negative
// count) is fresh input, and completion tokens are output.
func (u *upstreamUsage) counts() billingpricebook.Counts {
	var cached int64
	if u.PromptTokensDetails != nil {
		cached = u.PromptTokensDetails.CachedTokens
	}
	fresh := u.PromptTokens - cached
	if fresh < 0 {
		fresh = 0
	}
	return billingpricebook.Counts{
		FreshInput: fresh,
		CacheRead:  cached,
		Output:     u.CompletionTokens,
	}
}

// estimateUnits converts captured counts into ledger units, valuating by the
// SERVED model name (the name the client asked for — the price book prices
// served names, not upstream ids). A model the book does not price, an
// unpriced rate, or a valuation error yields unknown, never zero: the caller
// leaves the reservation PENDING, the ledger's own convention for an
// unknown outcome.
func (c *Chain) estimateUnits(servedModel string, counts billingpricebook.Counts) (int64, bool) {
	cost, err := c.cfg.Prices.Valuate(servedModel, counts)
	if err != nil || !cost.IsKnown() {
		return 0, false
	}
	return int64(math.Round(cost.Amount * c.cfg.UnitsPerCurrencyUnit)), true
}

// settleJob is one reservation awaiting its known-usage settlement.
type settleJob struct {
	reqID         string
	reservationID string
	servedModel   string
	counts        billingpricebook.Counts
	known         bool // usage parsed from a well-formed upstream response
}

// settleQueue is the bounded asynchronous settle queue. A full queue never
// drops a job: the enqueue falls back to settling inline on the serving
// path, trading that request's tail latency for the ledger entry.
type settleQueue struct {
	ch     chan settleJob
	wg     sync.WaitGroup
	closed bool
	mu     sync.Mutex
}

func newSettleQueue(c *Chain, depth int) *settleQueue {
	q := &settleQueue{ch: make(chan settleJob, depth)}
	q.wg.Add(1)
	go func() {
		defer q.wg.Done()
		for j := range q.ch {
			c.applySettle(j)
		}
	}()
	return q
}

// enqueue hands one job to the worker, or settles it inline when the queue
// is full. Never drops a job.
func (q *settleQueue) enqueue(c *Chain, j settleJob) {
	q.mu.Lock()
	closed := q.closed
	q.mu.Unlock()
	if closed {
		return
	}
	select {
	case q.ch <- j:
	default:
		c.applySettle(j)
	}
}

// stop closes the queue and waits for the worker to drain it. Call only
// after the chain has stopped serving: an enqueue racing the close is
// dropped (the closed flag makes that race a no-op, not a panic).
func (q *settleQueue) stop() {
	q.mu.Lock()
	if q.closed {
		q.mu.Unlock()
		return
	}
	q.closed = true
	close(q.ch)
	q.mu.Unlock()
	q.wg.Wait()
}

// applySettle applies one settlement line to the ledger. The unknown rule is
// structural: usage that was not captured, a model the book does not price,
// or a settle the ledger refuses (including ErrExceedsReserved) leaves the
// reservation PENDING — the ledger's own convention — and counts as pending.
// It is never settled at zero, and never released: the held units stay held
// until an operator reconciles, which is the safe direction for money.
func (c *Chain) applySettle(j settleJob) {
	if !j.known {
		c.stats.Pending.Add(1)
		return
	}
	units, ok := c.estimateUnits(j.servedModel, j.counts)
	if !ok {
		c.stats.Pending.Add(1)
		return
	}
	err := c.cfg.Ledger.Settle(j.reqID+"-settle", j.reservationID, units)
	switch {
	case err == nil:
		c.stats.Settled.Add(1)
	case errors.Is(err, billingledger.ErrAlreadyApplied):
		// A retried settle is the same event: already applied, success.
		c.stats.Settled.Add(1)
	default:
		c.stats.Pending.Add(1)
	}
}

// foldUsage folds captured usage into the synchronous enforcement surfaces —
// the spend caps and the rate window — immediately at capture, so the next
// request's admission sees it deterministically (the settle queue is
// asynchronous; caps must not be). Unknown usage folds nothing.
func (c *Chain) foldUsage(tenantID, sessionID string, counts billingpricebook.Counts, known bool, now time.Time) {
	if !known {
		return
	}
	total := counts.FreshInput + counts.CacheRead + counts.CacheWrite + counts.Output
	if total <= 0 {
		return
	}
	c.caps.Add(billing.Principal{TenantID: tenantID, SessionID: sessionID}, total, now)
	c.window.commitTokens(total, now)
}
