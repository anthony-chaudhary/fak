package gateway

// admission.go — the fak-NATIVE serving node's ADMISSION / PRIORITY / FAIRNESS gate
// (issue #35, under #36). This is the policy layer that sits ABOVE the native
// continuous-batching iteration loop (modelengine.NativeScheduler) — the waiting/running
// queues, the admission gate, request priority, the starvation guard, and the gateway
// backpressure surface — so a fak-native worker SHEDS GRACEFULLY under load instead of
// degrading unboundedly, and so the trust layer has a native admission gate to attach a
// per-tenant verdict to.
//
// WHY IT EXISTS — the gap it closes. modelengine.NativeScheduler.Admit appends EVERY
// offered request to the shared StepBatch loop the instant it arrives ("no admission
// control / fairness" — its own honest fence), and the gateway request path has only a
// body cap + socket timeouts ("no max-concurrency, no 429/backpressure"). So an
// overloaded native node cannot shed: it queues unboundedly behind the planner. This
// file is the missing gate — a pure, deterministic admission POLICY the native loop and
// the gateway consult, expressed in the repo's house form (a value type + a small set of
// total methods, stdlib-only, off the request path, no hidden clock or randomness) so it
// is unit-testable to an exact admission sequence, like batchsched.go.
//
// SHAPE (vLLM V1 / SGLang parity). Requests are OFFERED to the gate, which classifies
// each into the waiting queue (or sheds/denies it); Schedule then promotes waiting
// requests into the running set in EFFECTIVE-PRIORITY order while the token budget and
// the max-num-seqs cap have headroom. Splitting Offer (classify) from Schedule (promote)
// keeps priority+aging the single place admission ORDER is decided — the same posture
// session.Scheduler takes toward the drive-state Table.
//
//   - Admission budget: a request enters the running set only while the running set has
//     token-budget headroom (Σ running tokens + the request's footprint ≤ TokenBudget)
//     AND the running count is below max-num-seqs. Until paged KV lands the budget is the
//     token footprint alone (the issue's "admit on the token budget alone"); the KV-block
//     half attaches to the same Tokens axis once it is exact.
//   - Priority dequeue: the waiting queue is served lowest-Priority-value first (matching
//     the session snapshot's Priority-ascending convention), ties broken by older enqueue
//     round then TraceID — fully deterministic.
//   - Starvation guard (no-starvation guarantee): a waiting request's EFFECTIVE priority
//     improves by one for every AgingRounds rounds it is passed over, so it climbs
//     monotonically toward the head and is admitted within a BOUNDED number of rounds once
//     a slot frees — even under a continuous flood of higher-priority arrivals. With aging
//     off that flood starves it; TestAdmissionNoStarvation asserts both directions.
//   - Backpressure: when a request cannot be admitted now AND the waiting queue is already
//     at its bound, it is SHED (VerdictShed → HTTP 429), replacing the unbounded-queue
//     behavior. A per-tenant trust verdict can DENY admission outright (VerdictDenied).
//
// HONEST FENCE — what this is NOT (yet). This is the admission POLICY plus its L2
// serving-metrics fragment (running/waiting/admitted/rejected counts) and the live
// gateway lease wrapper: a host wires a controller onto the Server with
// SetAdmissionController, served requests acquire a lease before the planner runs,
// VerdictShed maps to HTTP 429, and renderMetrics folds WriteMetrics into the shared
// serving-metrics surface (writeAdmissionMetrics). With no controller attached the
// surface is inert (no fak_sched_* series) and the request path is byte-for-byte
// historical. It runs no model and moves no KV. The KV-block budget, preemption /
// KV swap-out, and the cross-replica router are explicit non-goals here (separate
// sibling seeds).

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/kvbudget"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// DefaultPreallocCeiling is the per-request ceiling on generation tokens preallocated
// at admission (anti-hog preallocation cap, #5268; field-borrow from TGI v3).
// A request with max_tokens above this ceiling preallocates up to this ceiling at
// admission, top-up re-admitting in chunks via Grow as it generates.
const DefaultPreallocCeiling = 1024

// AdmissionPolicy holds the admission knobs. Each cap is disabled by a non-positive value
// (so a test can isolate a single axis); DefaultAdmissionPolicy fills shipping defaults.
type AdmissionPolicy struct {
	// MaxNumSeqs caps the running set (vLLM's max-num-seqs): no request is admitted while
	// the running count is at this bound. ≤0 disables the seq cap.
	MaxNumSeqs int
	// TokenBudget caps the sum of the running set's token footprints (the num-batched-tokens
	// admission budget; until paged KV lands this is the whole budget). ≤0 disables it.
	TokenBudget int
	// TokenBudgetProvenance tracks the source of the token budget ("default", "measured", or "explicit").
	TokenBudgetProvenance string
	// MaxWaiting bounds the waiting queue. A request that cannot be admitted now is shed
	// (429) once the queue is at this bound — the backpressure limit. ≤0 = unbounded
	// (never sheds; the historical "queue forever" behavior).
	MaxWaiting int
	// MaxQueuedTokens caps the cumulative token volume across all currently waiting
	// requests in the queue (vLLM max_num_queued_tokens TTFT-QoS queue admission cap, #10727).
	// A request that would cause queued tokens to exceed this ceiling is shed (429)
	// before long-prompt floods destroy TTFT service levels for concurrent requests.
	// ≤0 disables this bound.
	MaxQueuedTokens int
	// AgingRounds is the starvation guard: a waiting request's effective priority improves
	// by one for every AgingRounds rounds it is passed over. ≤0 disables aging (raw
	// priority stands and a higher-priority flood can starve a low-priority waiter).
	AgingRounds int
	// PreallocCeiling caps the generation tokens preallocated at admission time
	// (anti-hog preallocation cap, #5268). Decouples the preallocated generation
	// budget from requested max_tokens. ≤0 defaults to DefaultPreallocCeiling.
	PreallocCeiling int
}

// DefaultAdmissionPolicy returns the shipping defaults: a 256-sequence running cap (the
// vLLM V1 default), an 8192-token batched-admission budget, a 1024-deep waiting bound
// before shedding, a 65536-token queued volume cap, aging every round so no waiter
// is ever starved, and a 1024-token generation preallocation ceiling (#5268).
func DefaultAdmissionPolicy() AdmissionPolicy {
	return AdmissionPolicy{
		MaxNumSeqs:            256,
		TokenBudget:           8192,
		TokenBudgetProvenance: "default",
		MaxWaiting:            1024,
		MaxQueuedTokens:       65536,
		AgingRounds:           1,
		PreallocCeiling:       DefaultPreallocCeiling,
	}
}

func (p AdmissionPolicy) preallocCeiling() int {
	if p.PreallocCeiling <= 0 {
		return DefaultPreallocCeiling
	}
	return p.PreallocCeiling
}

func (c *AdmissionController) preallocCeiling() int {
	if c == nil {
		return DefaultPreallocCeiling
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.policy.preallocCeiling()
}

// WarmupCapacityReporter is the optional interface a planner implements when it
// can report measured warmup capacity directly.
type WarmupCapacityReporter interface {
	WarmupCapacity() (kvbudget.WarmupCapacity, bool)
}

// AdmissionTrust is the per-tenant trust/SLA verdict hook (the L3 governance seam,
// #504-532). HOOK + SEAM ONLY: the verdict SOURCE is out of scope here; the admission
// decision merely CARRIES it so the governance layer can later gate (or reprioritize)
// admission. A denying verdict rejects admission outright — the one behavior wired now.
type AdmissionTrust struct {
	// Deny, when true, rejects the request regardless of headroom (VerdictDenied).
	Deny bool
	// Reason names the denial (a closed-vocabulary token the governance layer sets),
	// surfaced on the rejection so a client/operator learns why. Empty when Deny is false.
	Reason string
}

// SeqRequest is one request offered to the gate — a sequence the native continuous-
// batching loop would run. Priority is lower-is-higher (Priority-ascending, like the
// session snapshot). Tokens is the footprint charged against the token budget (prompt +
// planned decode; the KV-block estimate folds onto this axis once paged KV is exact).
type SeqRequest struct {
	TraceID   string
	SessionID string
	Priority  int
	Tokens    int
	Trust     AdmissionTrust
	CreatedAt time.Time
	DecodeTTL time.Duration
}

// baseTraceID strips any "#..." suffix from traceID and trims space.
func baseTraceID(traceID string) string {
	traceID = strings.TrimSpace(traceID)
	if idx := strings.Index(traceID, "#"); idx != -1 {
		traceID = traceID[:idx]
	}
	return strings.TrimSpace(traceID)
}

// AdmissionVerdict is the outcome of offering a request to the gate.
type AdmissionVerdict uint8

const (
	// VerdictAdmitted: the request entered the running set immediately (headroom was free
	// and no one was ahead of it in the waiting queue).
	VerdictAdmitted AdmissionVerdict = iota
	// VerdictQueued: no headroom now, but the waiting queue had room — the request waits
	// and a later Schedule promotes it in priority order.
	VerdictQueued
	// VerdictShed: the request could not be admitted and the waiting queue is at its bound —
	// the node sheds rather than queue unboundedly (the gateway maps this to HTTP 429).
	VerdictShed
	// VerdictDenied: a per-tenant trust verdict rejected admission (the governance gate).
	VerdictDenied
	// VerdictExpired: the individual request's DecodeTTL expired while waiting or decoding,
	// shedding only this request without killing the scheduler or other requests.
	VerdictExpired
	// VerdictRefused: the request envelope is statically impossible (e.g. req.Tokens > capacity).
	VerdictRefused
)

// String renders a verdict as its lowercase token; an out-of-range value renders
// "unknown" rather than panicking, matching the package's other enums.
func (v AdmissionVerdict) String() string {
	switch v {
	case VerdictAdmitted:
		return "admitted"
	case VerdictQueued:
		return "queued"
	case VerdictShed:
		return "shed"
	case VerdictDenied:
		return "denied"
	case VerdictExpired:
		return "expired"
	case VerdictRefused:
		return "refused"
	}
	return "unknown"
}

// HTTPStatus maps a verdict to the wire status a gateway returns the client: an overload
// shed is 429 (Too Many Requests, the backpressure signal), a trust denial is 403, and an
// admitted/queued request carries no refusal (0 — the request proceeds or waits).
func (v AdmissionVerdict) HTTPStatus() int {
	switch v {
	case VerdictShed:
		return http.StatusTooManyRequests
	case VerdictDenied:
		return http.StatusForbidden
	case VerdictExpired:
		return http.StatusGatewayTimeout
	case VerdictRefused:
		return http.StatusBadRequest
	default:
		return 0
	}
}

// AdmissionStats is a snapshot of the gate's live gauges plus its cumulative counters —
// the running/waiting counts (and admitted/rejected) the L2 serving-metrics schema exports
// so a fleet router / autoscaler can read per-worker load.
type AdmissionStats struct {
	Running       int   // running-set size right now (gauge)
	Waiting       int   // waiting-queue depth right now (gauge)
	TokensInUse   int   // token budget held by the running set right now (gauge)
	QueuedTokens  int   // token volume across currently waiting requests right now (gauge)
	MaxWaitRounds int64 // oldest current waiter's age in rounds (starvation visibility, gauge)
	Admitted      int64 // cumulative requests promoted into the running set (counter)
	Queued        int64 // cumulative requests placed on the waiting queue (counter)
	Shed          int64 // cumulative requests shed under overload — 429 (counter)
	Denied        int64 // cumulative requests rejected by a trust verdict (counter)
	Expired       int64 // cumulative requests expired by DecodeTTL (counter)
	Refused       int64 // cumulative requests refused for impossible envelopes (counter)
}

// AdmissionController is the admission/priority/fairness gate over the native loop. The
// zero value is not usable — build one with NewAdmissionController. It is safe for
// concurrent use (the gateway request path and the loop both touch it).
type AdmissionController struct {
	mu           sync.Mutex
	policy       AdmissionPolicy
	budgets      []batchBudget         // independent capacity axes folded for every admission
	running      map[string]SeqRequest // admitted, holding budget, keyed by TraceID
	tokens       int                   // Σ running[*].Tokens, maintained incrementally
	waiting      []waitEntry           // waiting queue, re-sorted by effective priority each round
	queuedTokens int                   // Σ waiting[*].req.Tokens, maintained incrementally (#10727)
	round        int64                 // monotone admission-round counter (drives aging)
	stats        AdmissionStats        // cumulative counters (gauges are derived in Stats)
	seq          uint64                // internal per-request suffix for duplicate caller traces
	clock        func() time.Time      // injectable clock, defaults to time.Now
	table        *session.Table
	scheduler    *session.Scheduler
	pool         *session.Pool
	orderPolicy  session.Policy
	credits      map[string]int64
}

// waitEntry is one queued request plus the round it was enqueued, so aging can measure
// how long it has been passed over.
type waitEntry struct {
	req           SeqRequest
	enqueuedRound int64
	ready         chan struct{}
}

// NewAdmissionController builds a gate under the given policy.
func NewAdmissionController(p AdmissionPolicy) *AdmissionController {
	if p.TokenBudgetProvenance == "" {
		p.TokenBudgetProvenance = "default"
	}
	return newAdmissionControllerWithBudgets(p, admissionBatchBudgets(p)...)
}

// Policy returns a snapshot of the configured admission policy.
func (c *AdmissionController) Policy() AdmissionPolicy {
	if c == nil {
		return AdmissionPolicy{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.policy
}

// TokenBudgetProvenance returns the provenance of the admission token budget ("default", "measured", or "explicit").
func (c *AdmissionController) TokenBudgetProvenance() string {
	if c == nil {
		return ""
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	prov := c.policy.TokenBudgetProvenance
	if prov == "" {
		return "default"
	}
	return prov
}

// SetTokenBudgetWithProvenance updates the token budget and records its provenance.
func (c *AdmissionController) SetTokenBudgetWithProvenance(budget int, provenance string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.policy.TokenBudget = budget
	c.policy.TokenBudgetProvenance = provenance
	c.syncTokenBudgetLocked()
}

func (c *AdmissionController) syncTokenBudgetLocked() {
	budgets := make([]batchBudget, 0, len(c.budgets)+1)
	for _, b := range c.budgets {
		if b == nil {
			continue
		}
		check := b(batchBudgetSnapshot{}, SeqRequest{})
		if check.budget == "tokens" {
			continue
		}
		budgets = append(budgets, b)
	}
	if c.policy.TokenBudget > 0 {
		budgets = append(budgets, tokenBatchBudget(c.policy.TokenBudget))
	}
	c.budgets = budgets
}

// SetMaxWaiting dynamically updates the waiting queue admission limit.
func (c *AdmissionController) SetMaxWaiting(n int) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.policy.MaxWaiting = n
}

// MaxWaiting returns the current waiting queue admission limit.
func (c *AdmissionController) MaxWaiting() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.policy.MaxWaiting
}

func newAdmissionControllerWithBudgets(p AdmissionPolicy, budgets ...batchBudget) *AdmissionController {
	return &AdmissionController{
		policy:  p,
		budgets: append([]batchBudget(nil), budgets...),
		running: map[string]SeqRequest{},
		credits: make(map[string]int64),
	}
}

// SetTable attaches or replaces the session table.
func (c *AdmissionController) SetTable(tbl *session.Table) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.table = tbl
}

// Table returns the configured session table.
func (c *AdmissionController) Table() *session.Table {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.table
}

// SetSequencer attaches or replaces the session scheduler.
func (c *AdmissionController) SetSequencer(sched *session.Scheduler) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.scheduler = sched
}

// Scheduler returns the configured session scheduler.
func (c *AdmissionController) Scheduler() *session.Scheduler {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.scheduler
}

// SetFleet attaches or replaces the fleet-wide token pool.
func (c *AdmissionController) SetFleet(pool *session.Pool) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.pool = pool
	c.syncFleetBudgetLocked()
}

// Pool returns the configured fleet-wide token pool.
func (c *AdmissionController) Pool() *session.Pool {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.pool
}

func (c *AdmissionController) syncFleetBudgetLocked() {
	budgets := make([]batchBudget, 0, len(c.budgets)+1)
	for _, b := range c.budgets {
		if b == nil {
			continue
		}
		check := b(batchBudgetSnapshot{}, SeqRequest{})
		if check.budget == "fleet_tokens" {
			continue
		}
		budgets = append(budgets, b)
	}
	if c.pool != nil {
		budgets = append(budgets, fleetBatchBudget(c.pool))
	}
	c.budgets = budgets
}

// SetOrder configures the admission queue scheduling policy.
func (c *AdmissionController) SetOrder(p session.Policy) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.orderPolicy = p
}

// Order returns the admission queue scheduling policy.
func (c *AdmissionController) Order() session.Policy {
	if c == nil {
		return session.StrictPriority
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.orderPolicy
}

// AdmissionLease is the live request's hold on one admitted scheduler slot, plus (when a
// token-rate gate is wired, #2019) its reservation against the provider token window.
// Release is idempotent and schedules the next waiting request after freeing this
// request's budget.
type AdmissionLease struct {
	ctl      *AdmissionController
	traceID  string
	tokenRes *TokenReservation
	once     sync.Once
}

// Grow tops up the lease's token footprint by additionalTokens (issue #5268, TGI anti-hog
// borrow). When a long generation consumes its preallocated chunk and continues, Grow
// re-admits the next chunk against the live scheduler budget (and any token rate gate)
// instead of reserving the whole worst-case upfront. Partial failures roll back cleanly.
func (l *AdmissionLease) Grow(additionalTokens int) error {
	if l == nil || additionalTokens == 0 {
		return nil
	}
	if l.ctl != nil && l.traceID != "" {
		if err := l.ctl.Grow(l.traceID, additionalTokens); err != nil {
			return err
		}
	}
	if l.tokenRes != nil {
		if err := l.tokenRes.Grow(additionalTokens); err != nil {
			if l.ctl != nil && l.traceID != "" {
				l.ctl.refund(l.traceID, additionalTokens)
			}
			return err
		}
	}
	return nil
}

// Release frees the admitted request's token/sequence budget and promotes waiters. A
// token reservation never settled with real usage keeps its conservative estimate
// (TokenReservation.Release).
func (l *AdmissionLease) Release() {
	if l == nil {
		return
	}
	l.once.Do(func() {
		if l.ctl != nil && l.traceID != "" {
			l.ctl.completeAndSchedule(l.traceID)
		}
		l.tokenRes.Release()
	})
}

// SettleUsage feeds the provider's real, normalized usage back into the token-rate
// gate's window (#2019) and the admission controller (#5268) the moment a completion
// reports it, replacing the admission-time estimate. Safe on a nil lease or one with no
// token reservation; the scheduler slot itself is still freed by Release.
func (l *AdmissionLease) SettleUsage(u agent.Usage) {
	if l == nil {
		return
	}
	if l.ctl != nil && l.traceID != "" {
		total := u.TotalTokens
		if total <= 0 {
			total = u.PromptTokens + u.CompletionTokens
		}
		if total > 0 {
			l.ctl.Settle(l.traceID, total)
		}
	}
	l.tokenRes.Settle(tokenUsageFromAgent(u))
}

// AdmissionError is the typed served-path refusal returned when the live admission
// gate sheds overload or a trust verdict denies a request before the planner runs.
type AdmissionError struct {
	Verdict AdmissionVerdict
	Budget  string
	Reason  string
}

func (e *AdmissionError) Error() string {
	if e == nil {
		return "scheduler admission refused"
	}
	if e.Reason != "" {
		return "scheduler admission " + e.Verdict.String() + ": " + e.Reason
	}
	return "scheduler admission " + e.Verdict.String()
}

// SetClock overrides the time source used for CreatedAt defaults and TTL expiry.
func (c *AdmissionController) SetClock(clk func() time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.clock = clk
}

func (c *AdmissionController) nowLocked() time.Time {
	if c.clock != nil {
		return c.clock()
	}
	return time.Now()
}

// Offer presents a new request to the gate and classifies it. It admits straight to the
// running set only on the fast path — an underloaded node with no one already waiting and
// free headroom — so an idle node serves immediately; otherwise the request joins the
// waiting queue (VerdictQueued), is shed if that queue is at its bound (VerdictShed →
// 429), or is rejected by a denying trust verdict (VerdictDenied). Admission ORDER among
// waiters is never decided here — that is Schedule's job, so priority+aging stays the one
// place order is resolved.
func (c *AdmissionController) Offer(req SeqRequest) AdmissionVerdict {
	c.mu.Lock()
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
		return VerdictExpired
	}
	if req.Trust.Deny {
		c.stats.Denied++
		return VerdictDenied
	}
	if check := c.impossibleBudgetLocked(req); check.status == batchBudgetImpossible {
		c.stats.Refused++
		return VerdictRefused
	} else if check.status >= batchBudgetExhausted {
		c.stats.Shed++
		return VerdictShed
	}
	// Fast path: nobody is waiting and there is headroom now — admit immediately so an
	// idle/underloaded node does not pay a Schedule round to serve its first requests.
	if len(c.waiting) == 0 && c.batchBudgetStatusLocked(req).status < batchBudgetExhausted {
		c.admitLocked(req)
		return VerdictAdmitted
	}
	// The request must wait. Shed it if the waiting queue is already at its bound — the
	// backpressure surface that replaces unbounded queueing. Also enforce the queued-token
	// volume cap (#10727) before long-prompt floods destroy TTFT service levels.
	if c.policy.MaxWaiting > 0 && len(c.waiting) >= c.policy.MaxWaiting {
		c.stats.Shed++
		return VerdictShed
	}
	if c.policy.MaxQueuedTokens > 0 && c.queuedTokens+req.Tokens > c.policy.MaxQueuedTokens {
		c.stats.Shed++
		return VerdictShed
	}
	c.waiting = append(c.waiting, waitEntry{req: req, enqueuedRound: c.round})
	c.queuedTokens += req.Tokens
	c.stats.Queued++
	return VerdictQueued
}

// Acquire is the live gateway boundary over Offer/Schedule: it either returns an
// admitted lease, returns an AdmissionError for immediate shed/deny, or waits until a
// queued request is promoted. The controller assigns an internal unique TraceID suffix
// so two concurrent HTTP requests for the same served session do not overwrite each
// other in the running map.
func (c *AdmissionController) Acquire(ctx context.Context, req SeqRequest) (*AdmissionLease, error) {
	if c == nil {
		return nil, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if req.SessionID == "" {
		req.SessionID = baseTraceID(req.TraceID)
	}
	req.TraceID = c.admissionTraceID(req.TraceID)

	c.mu.Lock()
	if req.Trust.Deny {
		c.stats.Denied++
		c.mu.Unlock()
		return nil, &AdmissionError{Verdict: VerdictDenied, Reason: req.Trust.Reason}
	}
	if check := c.impossibleBudgetLocked(req); check.status == batchBudgetImpossible {
		c.stats.Refused++
		c.mu.Unlock()
		return nil, &AdmissionError{Verdict: VerdictRefused, Budget: check.budget, Reason: check.reason}
	} else if check.status >= batchBudgetExhausted {
		c.stats.Shed++
		c.mu.Unlock()
		return nil, &AdmissionError{Verdict: VerdictShed, Budget: check.budget, Reason: check.reason}
	}
	if len(c.waiting) == 0 && c.batchBudgetStatusLocked(req).status < batchBudgetExhausted {
		c.admitLocked(req)
		c.mu.Unlock()
		return &AdmissionLease{ctl: c, traceID: req.TraceID}, nil
	}
	if c.policy.MaxWaiting > 0 && len(c.waiting) >= c.policy.MaxWaiting {
		c.stats.Shed++
		c.mu.Unlock()
		return nil, &AdmissionError{Verdict: VerdictShed}
	}
	if c.policy.MaxQueuedTokens > 0 && c.queuedTokens+req.Tokens > c.policy.MaxQueuedTokens {
		c.stats.Shed++
		c.mu.Unlock()
		return nil, &AdmissionError{Verdict: VerdictShed, Reason: "queued token volume exceeds cap"}
	}
	ready := make(chan struct{})
	c.waiting = append(c.waiting, waitEntry{req: req, enqueuedRound: c.round, ready: ready})
	c.queuedTokens += req.Tokens
	c.stats.Queued++
	c.scheduleLocked()
	c.mu.Unlock()

	select {
	case <-ready:
		return &AdmissionLease{ctl: c, traceID: req.TraceID}, nil
	default:
	}
	select {
	case <-ready:
		return &AdmissionLease{ctl: c, traceID: req.TraceID}, nil
	case <-ctx.Done():
		c.cancelAdmission(req.TraceID)
		return nil, ctx.Err()
	}
}

func (c *AdmissionController) admissionTraceID(traceID string) string {
	traceID = strings.TrimSpace(traceID)
	if traceID == "" {
		traceID = "request"
	}
	return fmt.Sprintf("%s#%d", traceID, atomic.AddUint64(&c.seq, 1))
}

// Schedule runs ONE admission round: it advances the aging clock, orders the waiting queue
// by effective priority, and promotes waiters into the running set while the token budget
// and the max-num-seqs cap have headroom — stopping at the first request that does not fit
// so a lower-priority request can never jump ahead of a blocked higher-priority one
// (head-of-line discipline). It returns the requests admitted this round, in admission
// order. The native loop calls it each iteration; a host with a freed slot calls it after
// Complete. Because the clock advances every call, a request passed over this round ages
// toward the head — the starvation guard.
func (c *AdmissionController) Schedule() []SeqRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.scheduleLocked()
}

func (c *AdmissionController) scheduleLocked() []SeqRequest {
	c.round++
	if len(c.waiting) == 0 {
		c.credits = make(map[string]int64)
		return nil
	}

	isWeightedFair := c.orderPolicy == session.WeightedFair || (c.scheduler != nil && c.scheduler.Policy() == session.WeightedFair)
	if !isWeightedFair {
		sort.SliceStable(c.waiting, func(i, j int) bool {
			ei, ej := c.effectivePriorityLocked(c.waiting[i]), c.effectivePriorityLocked(c.waiting[j])
			if ei != ej {
				return ei < ej // lower effective priority value served first
			}
			if c.waiting[i].enqueuedRound != c.waiting[j].enqueuedRound {
				return c.waiting[i].enqueuedRound < c.waiting[j].enqueuedRound // older waiter first
			}
			return c.waiting[i].req.TraceID < c.waiting[j].req.TraceID // deterministic final tiebreak
		})
		var admitted []SeqRequest
		for _, e := range c.waiting {
			check := c.batchBudgetStatusLocked(e.req)
			if check.status >= batchBudgetExhausted {
				break // head-of-line: do not let a lower-priority request skip a blocked one
			}
			c.admitLocked(e.req)
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

		c.admitLocked(e.req)
		c.queuedTokens -= e.req.Tokens
		if e.ready != nil {
			close(e.ready)
		}
		admitted = append(admitted, e.req)
		c.waiting = append(c.waiting[:winnerIdx], c.waiting[winnerIdx+1:]...)

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
			c.waiting = append(c.waiting[:i], c.waiting[i+1:]...)
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

// Grow tops up the token budget held by a currently running request by additionalTokens
// (issue #5268). It checks whether adding additionalTokens fits within TokenBudget (and
// fleet_tokens if c.pool != nil), updates req.Tokens += additionalTokens and c.tokens += additionalTokens,
// and returns nil or an *AdmissionError with VerdictShed if budget exhausted. Negative tokens
// refunds tokens back to the budget and wakes waiters via scheduleLocked.
func (c *AdmissionController) Grow(traceID string, additionalTokens int) error {
	if c == nil || additionalTokens == 0 {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	req, ok := c.running[traceID]
	if !ok {
		for tid, r := range c.running {
			if baseTraceID(tid) == traceID {
				traceID = tid
				req = r
				ok = true
				break
			}
		}
	}
	if !ok {
		return fmt.Errorf("trace %q not running", traceID)
	}

	if additionalTokens < 0 {
		tokens := -additionalTokens
		if tokens > req.Tokens {
			tokens = req.Tokens
		}
		req.Tokens -= tokens
		c.running[traceID] = req
		c.tokens -= tokens
		if c.tokens < 0 {
			c.tokens = 0
		}
		if c.pool != nil {
			c.pool.Return(tokens)
		}
		c.scheduleLocked()
		return nil
	}

	if c.policy.TokenBudget > 0 && c.tokens+additionalTokens > c.policy.TokenBudget {
		c.stats.Shed++
		return &AdmissionError{
			Verdict: VerdictShed,
			Budget:  "tokens",
			Reason:  fmt.Sprintf("scheduler token budget exhausted (%d in use + %d requested > %d)", c.tokens, additionalTokens, c.policy.TokenBudget),
		}
	}

	if c.pool != nil {
		rem := c.pool.Remaining()
		if rem >= 0 && rem < additionalTokens {
			c.stats.Shed++
			return &AdmissionError{
				Verdict: VerdictShed,
				Budget:  "fleet_tokens",
				Reason:  fmt.Sprintf("session pool budget exhausted (%d remaining < %d requested)", rem, additionalTokens),
			}
		}
		granted, ok := c.pool.Draw(additionalTokens)
		if !ok {
			c.pool.Return(granted)
			c.stats.Shed++
			return &AdmissionError{
				Verdict: VerdictShed,
				Budget:  "fleet_tokens",
				Reason:  fmt.Sprintf("session pool budget exhausted (%d remaining < %d requested)", rem, additionalTokens),
			}
		}
	}

	req.Tokens += additionalTokens
	c.running[traceID] = req
	c.tokens += additionalTokens
	return nil
}

// Settle reconciles req.Tokens and c.tokens with actual usage if over-estimated,
// freeing excess tokens back to the controller budget and promoting queued waiters (#5268).
func (c *AdmissionController) Settle(traceID string, actualTokens int) {
	if c == nil || actualTokens <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	req, ok := c.running[traceID]
	if !ok {
		for tid, r := range c.running {
			if baseTraceID(tid) == traceID {
				traceID = tid
				req = r
				ok = true
				break
			}
		}
	}
	if !ok {
		return
	}

	if actualTokens < req.Tokens {
		diff := req.Tokens - actualTokens
		req.Tokens = actualTokens
		c.running[traceID] = req
		c.tokens -= diff
		if c.tokens < 0 {
			c.tokens = 0
		}
		if c.pool != nil {
			c.pool.Return(diff)
		}
		c.scheduleLocked()
	}
}

func (c *AdmissionController) refund(traceID string, tokens int) {
	_ = c.Grow(traceID, -tokens)
}

// impossibleBudgetLocked reports a capacity constraint that cannot become
// satisfiable by waiting: the composed fold still exhausts on an otherwise
// empty running set. Caller holds c.mu.
func (c *AdmissionController) impossibleBudgetLocked(req SeqRequest) batchBudgetCheck {
	return foldBatchBudgets(batchBudgetSnapshot{}, req, c.budgets)
}

// batchBudgetStatusLocked folds every independent budget against the same
// immutable running-set snapshot. Caller holds c.mu.
func (c *AdmissionController) batchBudgetStatusLocked(req SeqRequest) batchBudgetCheck {
	return foldBatchBudgets(batchBudgetSnapshot{
		running: len(c.running),
		tokens:  c.tokens,
	}, req, c.budgets)
}

// admitLocked moves a request into the running set and charges its tokens. Caller holds c.mu.
func (c *AdmissionController) admitLocked(req SeqRequest) {
	if c.pool != nil {
		c.pool.Draw(req.Tokens)
	}
	c.running[req.TraceID] = req
	c.tokens += req.Tokens
	c.stats.Admitted++
}

// effectivePriorityLocked is a waiting request's priority adjusted for how long it has
// been passed over: it improves (decreases) by one for every AgingRounds rounds waited.
// Lower is served first, so an aged request climbs monotonically toward the head and is
// admitted within a bounded number of rounds once a slot frees — the no-starvation
// guarantee. AgingRounds ≤ 0 disables aging (raw Priority stands). Caller holds c.mu.
func (c *AdmissionController) effectivePriorityLocked(e waitEntry) int {
	if c.policy.AgingRounds <= 0 {
		return e.req.Priority
	}
	waited := c.round - e.enqueuedRound
	if waited < 0 {
		waited = 0
	}
	return e.req.Priority - int(waited/int64(c.policy.AgingRounds))
}

// Stats returns the live gauges plus cumulative counters.
func (c *AdmissionController) Stats() AdmissionStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	st := c.stats
	st.Running = len(c.running)
	st.Waiting = len(c.waiting)
	st.TokensInUse = c.tokens
	st.QueuedTokens = c.queuedTokens
	st.MaxWaitRounds = c.maxWaitRoundsLocked()
	return st
}

// maxWaitRoundsLocked is the oldest current waiter's age in rounds (0 when the queue is
// empty) — the starvation-visibility gauge. Caller holds c.mu.
func (c *AdmissionController) maxWaitRoundsLocked() int64 {
	var oldest int64
	for _, e := range c.waiting {
		if w := c.round - e.enqueuedRound; w > oldest {
			oldest = w
		}
	}
	return oldest
}

// schedMetricPrefix names the native serving scheduler's admission metric family — the L2
// serving-metrics schema's running/waiting/admitted/rejected counts (#35). It is kept
// distinct from fak_gateway_* (HTTP request path) and fak_kernel_* (tool-call adjudication)
// because admission is a third, distinct concern.
const schedMetricPrefix = "fak_sched_"

// WriteMetrics renders the gate's admission counts as Prometheus text into b — the shared
// L2 serving-metrics schema a fleet router / autoscaler reads to see per-worker load. It
// reuses the gateway's own metric writers so the format matches the rest of /metrics
// exactly. The host folds this into renderMetrics once the native scheduler is on the live
// serve path; until then it is the tested schema definition.
func (c *AdmissionController) WriteMetrics(b *strings.Builder) {
	st := c.Stats()
	writeHelpType(b, schedMetricPrefix+"running", "Sequences currently running (admitted) in the native serving scheduler.", "gauge")
	fmt.Fprintf(b, "%srunning %d\n", schedMetricPrefix, st.Running)
	writeHelpType(b, schedMetricPrefix+"waiting", "Sequences waiting for admission in the native serving scheduler.", "gauge")
	fmt.Fprintf(b, "%swaiting %d\n", schedMetricPrefix, st.Waiting)
	writeHelpType(b, schedMetricPrefix+"tokens_in_use", "Token admission budget currently held by the running set.", "gauge")
	fmt.Fprintf(b, "%stokens_in_use %d\n", schedMetricPrefix, st.TokensInUse)
	writeHelpType(b, schedMetricPrefix+"token_budget", "Configured token admission budget (0 = uncapped).", "gauge")
	prov := c.policy.TokenBudgetProvenance
	if prov == "" {
		prov = "default"
	}
	fmt.Fprintf(b, "%stoken_budget{provenance=\"%s\"} %d\n", schedMetricPrefix, prov, c.policy.TokenBudget)
	writeHelpType(b, schedMetricPrefix+"queued_tokens", "Token volume across sequences currently waiting for admission.", "gauge")
	fmt.Fprintf(b, "%squeued_tokens %d\n", schedMetricPrefix, st.QueuedTokens)
	writeHelpType(b, schedMetricPrefix+"max_num_seqs", "Configured max-num-seqs running-set cap (0 = uncapped).", "gauge")
	fmt.Fprintf(b, "%smax_num_seqs %d\n", schedMetricPrefix, c.policy.MaxNumSeqs)
	writeHelpType(b, schedMetricPrefix+"max_wait_rounds", "Oldest current waiter's age in admission rounds (starvation visibility).", "gauge")
	fmt.Fprintf(b, "%smax_wait_rounds %d\n", schedMetricPrefix, st.MaxWaitRounds)
	writeCounter(b, schedMetricPrefix+"admitted_total", "Requests promoted into the running set.", st.Admitted)
	writeCounter(b, schedMetricPrefix+"queued_total", "Requests placed on the waiting queue.", st.Queued)
	writeCounter(b, schedMetricPrefix+"shed_total", "Requests shed under overload (waiting queue at bound; HTTP 429).", st.Shed)
	writeCounter(b, schedMetricPrefix+"denied_total", "Requests rejected by a per-tenant trust verdict.", st.Denied)
	writeCounter(b, schedMetricPrefix+"refused_total", "Requests refused for impossible request envelopes (HTTP 400).", st.Refused)
}

// SetWarmupCapacity records a byte-measured warmup probe.
func (s *Server) SetWarmupCapacity(cap kvbudget.WarmupCapacity) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.warmupCapacity = &cap
	ctl := s.admissionCtl
	fraction := s.warmupReserveFraction
	s.admissionMu.Unlock()

	if ctl != nil && ctl.TokenBudgetProvenance() != "explicit" {
		if fraction <= 0 {
			fraction = kvbudget.DefaultReserveFraction
		}
		derived := cap.DeriveTokenBudget(fraction)
		if derived.Derived() {
			ctl.SetTokenBudgetWithProvenance(int(derived.TokenBudget), "measured")
		}
	}
}

// SetWarmupBlockCapacity records a block-measured warmup probe.
func (s *Server) SetWarmupBlockCapacity(cap kvbudget.WarmupBlockCapacity) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.warmupBlockCapacity = &cap
	ctl := s.admissionCtl
	fraction := s.warmupReserveFraction
	s.admissionMu.Unlock()

	if ctl != nil && ctl.TokenBudgetProvenance() != "explicit" {
		if fraction <= 0 {
			fraction = kvbudget.DefaultReserveFraction
		}
		derived := cap.DeriveTokenBudget(fraction)
		if derived.Derived() {
			ctl.SetTokenBudgetWithProvenance(int(derived.TokenBudget), "measured")
		}
	}
}

// SetWarmupReserveFraction configures the reserve fraction withheld from measured capacity.
func (s *Server) SetWarmupReserveFraction(fraction float64) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	s.warmupReserveFraction = fraction
}

// WarmupCapacity returns the measured warmup capacity and true if available.
func (s *Server) WarmupCapacity() (kvbudget.WarmupCapacity, bool) {
	if s == nil {
		return kvbudget.WarmupCapacity{}, false
	}
	s.admissionMu.RLock()
	defer s.admissionMu.RUnlock()
	if s.warmupCapacity != nil {
		return *s.warmupCapacity, true
	}
	if s.planner != nil {
		if r, ok := s.planner.(WarmupCapacityReporter); ok {
			return r.WarmupCapacity()
		}
		if r, ok := s.planner.(interface {
			WarmupCapacity() kvbudget.WarmupCapacity
		}); ok {
			return r.WarmupCapacity(), true
		}
		if r, ok := s.planner.(agent.KVMemoryReporter); ok {
			st := r.KVMemoryStats()
			if st.BytesPerToken > 0 {
				usable := st.FitBudgetBytes
				if usable <= 0 {
					usable = st.CapacityFreeBytes
				}
				if usable <= 0 {
					usable = st.CapacityTotalBytes
				}
				if usable > 0 {
					return kvbudget.WarmupCapacity{
						UsableBytes:   usable,
						BytesPerToken: st.BytesPerToken,
					}, true
				}
			}
		}
	}
	return kvbudget.WarmupCapacity{}, false
}

// WarmupDerivedBudget calculates the token budget derived from measured warmup capacity.
func (s *Server) WarmupDerivedBudget() (kvbudget.DerivedBudget, bool) {
	if s == nil {
		return kvbudget.DerivedBudget{}, false
	}
	s.admissionMu.RLock()
	cap := s.warmupCapacity
	blockCap := s.warmupBlockCapacity
	fraction := s.warmupReserveFraction
	p := s.planner
	s.admissionMu.RUnlock()

	if fraction <= 0 {
		fraction = kvbudget.DefaultReserveFraction
	}

	if cap != nil {
		derived := cap.DeriveTokenBudget(fraction)
		if derived.Derived() {
			return derived, true
		}
	}
	if blockCap != nil {
		derived := blockCap.DeriveTokenBudget(fraction)
		if derived.Derived() {
			return derived, true
		}
	}
	if p != nil {
		if r, ok := p.(WarmupCapacityReporter); ok {
			if c, okCap := r.WarmupCapacity(); okCap {
				derived := c.DeriveTokenBudget(fraction)
				if derived.Derived() {
					return derived, true
				}
			}
		} else if r, ok := p.(interface {
			WarmupCapacity() kvbudget.WarmupCapacity
		}); ok {
			c := r.WarmupCapacity()
			derived := c.DeriveTokenBudget(fraction)
			if derived.Derived() {
				return derived, true
			}
		}
		if r, ok := p.(interface {
			WarmupBlockCapacity() (kvbudget.WarmupBlockCapacity, bool)
		}); ok {
			if bc, okBC := r.WarmupBlockCapacity(); okBC {
				derived := bc.DeriveTokenBudget(fraction)
				if derived.Derived() {
					return derived, true
				}
			}
		} else if r, ok := p.(interface {
			WarmupBlockCapacity() kvbudget.WarmupBlockCapacity
		}); ok {
			bc := r.WarmupBlockCapacity()
			derived := bc.DeriveTokenBudget(fraction)
			if derived.Derived() {
				return derived, true
			}
		}
		if r, ok := p.(agent.KVMemoryReporter); ok {
			st := r.KVMemoryStats()
			if st.BytesPerToken > 0 {
				usable := st.FitBudgetBytes
				if usable <= 0 {
					usable = st.CapacityFreeBytes
				}
				if usable <= 0 {
					usable = st.CapacityTotalBytes
				}
				if usable > 0 {
					c := kvbudget.WarmupCapacity{UsableBytes: usable, BytesPerToken: st.BytesPerToken}
					derived := c.DeriveTokenBudget(fraction)
					if derived.Derived() {
						return derived, true
					}
				}
			}
		}
	}
	return kvbudget.DerivedBudget{}, false
}

// SetMaxTotalTokens sets the configured upper bound on total tokens.
func (s *Server) SetMaxTotalTokens(n int) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	s.maxTotalTokens = n
}

// CheckMaxTotalTokens validates that n does not exceed the admission controller's token budget.
func (s *Server) CheckMaxTotalTokens(n int) error {
	if s == nil {
		return nil
	}
	s.admissionMu.RLock()
	ctl := s.admissionCtl
	s.admissionMu.RUnlock()
	if n > 0 && ctl != nil {
		budget := ctl.Policy().TokenBudget
		if n > budget {
			return fmt.Errorf("max_total_tokens (%d) exceeds admission token budget (%d); decrease --max-batch-prefill-tokens or --max-total-tokens", n, budget)
		}
	}
	return nil
}

// SetAdmissionController wires the native serving admission gate (#35) onto the Server so its
// L2 serving-metrics fragment (fak_sched_*) renders into the live /metrics surface. The host
// calls this once the native iteration scheduler (modelengine.NativeScheduler) is on the serve
// loop; passing nil detaches it (the surface goes inert — no fak_sched_* series). Settable
// after New, mirroring SetFleetMembership / SetKVResidencyReclaimer. A nil receiver is a no-op.
func (s *Server) SetAdmissionController(c *AdmissionController) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.admissionCtl = c
	s.admissionMu.Unlock()
	if c != nil {
		if s.table != nil {
			c.SetTable(s.table)
		}
		if s.scheduler != nil {
			c.SetSequencer(s.scheduler)
		}
		if s.pool != nil {
			c.SetFleet(s.pool)
		}
		if c.Policy().TokenBudgetProvenance != "explicit" {
			if derived, ok := s.WarmupDerivedBudget(); ok && derived.Derived() {
				c.SetTokenBudgetWithProvenance(int(derived.TokenBudget), "measured")
			}
		}
		s.admissionMu.RLock()
		maxTokens := s.maxTotalTokens
		s.admissionMu.RUnlock()
		if maxTokens > 0 {
			if err := s.CheckMaxTotalTokens(maxTokens); err != nil && s.logf != nil {
				s.logf("gateway: %v", err)
			}
		}
	}
}

// writeAdmissionMetrics folds the wired admission gate's L2 serving-metrics fragment
// (fak_sched_running/waiting/admitted/...) onto the gateway /metrics surface (#35). A Server
// with no controller attached emits nothing — no phantom zero series — the same host-injected,
// inert-by-default posture as writeFleetMembershipMetrics and the KV-residency seams.
func (s *Server) writeAdmissionMetrics(b *strings.Builder) {
	if s == nil || b == nil {
		return
	}
	s.admissionMu.RLock()
	c := s.admissionCtl
	s.admissionMu.RUnlock()
	if c == nil {
		return
	}
	c.WriteMetrics(b)
}

// admitStreamedTurn is beginServedAdmission plus the refusal both streaming surfaces own
// identically: a refused admission is logged with the surface's own `lane` word, answered on w as
// an upstream error, and reported ok=false — the caller's cue to stop and return "response
// owned". On ok=true the caller MUST defer the returned lease's Release, exactly as it would with
// beginServedAdmission directly; the lease is nil-safe when no admission control is attached.
func (s *Server) admitStreamedTurn(ctx context.Context, w http.ResponseWriter, lane string, turn servedSessionTurn, messages []agent.Message, tools []agent.ToolDef, maxTokens int) (*AdmissionLease, bool) {
	lease, err := s.beginServedAdmission(ctx, turn, messages, tools, maxTokens)
	if err != nil {
		s.logf("gateway: scheduler admission refused (%s): %v", lane, err)
		s.writeUpstreamErr(w, err)
		return nil, false
	}
	return lease, true
}

// beginServedAdmission is the served request's admission boundary, narrowest budget last:
// the scheduler slot (admissionCtl, #35) first — a queued request must not hold token
// budget while it waits — then the caller's OWN per-principal allotment
// (principalTokenRates, #5379), then the shared provider token window (tokenRateGate,
// #2019). A shed at any stage frees everything the earlier stages granted. With no gate
// attached it is inert (nil lease, nil error) and the request path is byte-for-byte
// historical.
//
// The per-principal allotment is checked BEFORE the shared provider window on purpose: a
// tenant already over its own cap must be shed without first reserving — and briefly
// holding — a slice of the budget every other tenant is competing for. The reverse order
// would let a tenant that is going to be refused anyway still push the shared window
// toward saturation.
func (s *Server) beginServedAdmission(ctx context.Context, turn servedSessionTurn, messages []agent.Message, tools []agent.ToolDef, maxTokens int) (*AdmissionLease, error) {
	if s == nil {
		return nil, nil
	}
	s.admissionMu.RLock()
	c := s.admissionCtl
	g := s.tokenRateGate
	b := s.principalTokenRates
	s.admissionMu.RUnlock()
	if c == nil && g == nil && b == nil {
		return nil, nil
	}
	var lease *AdmissionLease
	if c != nil {
		var err error
		lease, err = c.Acquire(ctx, SeqRequest{
			TraceID:  turn.traceID,
			Priority: turn.state.Priority,
			Tokens:   estimateServedAdmissionTokens(messages, tools, maxTokens),
		})
		if err != nil {
			return nil, err
		}
	}
	estimate := estimateServedTokenUsage(messages, tools, maxTokens)
	var res *TokenReservation
	if b != nil {
		var err error
		// principalFromContext is "" for an unidentified caller; the book charges those to
		// its single shared allotment rather than to a fresh (bypassable) one.
		res, err = b.Admit(principalFromContext(ctx), estimate)
		if err != nil {
			lease.Release() // nil-safe: free the scheduler slot this tenant's cap refused to feed
			return nil, err
		}
	}
	if g != nil {
		shared, err := g.Admit(estimate)
		if err != nil {
			// The provider window refused, so the provider is never called: cancel (charge
			// nothing) rather than Release (charge the estimate) the allotment this tenant
			// held for one instant — a neighbour's saturation must not spend its budget.
			res.cancel()
			lease.Release()
			return nil, err
		}
		shared.linked = res // settle/release the allotment in lockstep with the shared window
		res = shared
	}
	if res != nil {
		if lease == nil {
			lease = &AdmissionLease{}
		}
		lease.tokenRes = res
	}
	return lease, nil
}

// servedPromptChars is the raw character volume of one served turn's prompt side —
// messages plus tool schemas — shared by the scheduler gate's single-axis estimate and
// the token-rate gate's input/output split.
func servedPromptChars(messages []agent.Message, tools []agent.ToolDef) int {
	chars := 0
	for _, m := range messages {
		chars += len(m.Role) + len(m.Content) + len(m.ToolCallID) + len(m.Name)
		if m.FunctionCall != nil {
			chars += len(m.FunctionCall.Name) + len(m.FunctionCall.Arguments)
		}
		for _, tc := range m.ToolCalls {
			chars += len(tc.ID) + len(tc.Type) + len(tc.Function.Name) + len(tc.Function.Arguments)
		}
	}
	for _, t := range tools {
		chars += len(t.Type) + len(t.Function.Name) + len(t.Function.Description) + len(t.Function.Parameters)
	}
	return chars
}

func estimateServedAdmissionTokens(messages []agent.Message, tools []agent.ToolDef, maxTokens int) int {
	return estimateServedAdmissionTokensWithCap(messages, tools, maxTokens, DefaultPreallocCeiling)
}

func estimateServedAdmissionTokensWithCap(messages []agent.Message, tools []agent.ToolDef, maxTokens int, preallocCeiling int) int {
	chars := servedPromptChars(messages, tools)
	tokens := chars / 4
	if chars > 0 && tokens == 0 {
		tokens = 1
	}
	if preallocCeiling <= 0 {
		preallocCeiling = DefaultPreallocCeiling
	}
	if maxTokens > 0 {
		gen := maxTokens
		if gen > preallocCeiling {
			gen = preallocCeiling
		}
		tokens += gen
	} else {
		tokens++
	}
	if tokens <= 0 {
		return 1
	}
	return tokens
}

func sampleMaxTokens(opts []agent.SampleOpt) int {
	var sp agent.SampleParams
	for _, opt := range opts {
		if opt != nil {
			opt(&sp)
		}
	}
	if sp.MaxTokens == nil {
		return 0
	}
	return *sp.MaxTokens
}

func admissionErrorStatus(err error) (status int, code, msg string, ok bool) {
	var ae *AdmissionError
	if !errors.As(err, &ae) {
		return 0, "", "", false
	}
	switch ae.Verdict {
	case VerdictRefused:
		reason := strings.TrimSpace(ae.Reason)
		if reason == "" {
			reason = "request envelope exceeds capacity"
		}
		return http.StatusBadRequest, "context_length_exceeded", reason, true
	case VerdictShed:
		msg := "scheduler overloaded — back off and retry"
		// A token-rate shed (#2019) names the provider cap that fired so the client's
		// backoff can be informed; the historical slot shed carries no reason.
		if reason := strings.TrimSpace(ae.Reason); reason != "" {
			msg += ": " + reason
		}
		return http.StatusTooManyRequests, "scheduler_overloaded", msg, true
	case VerdictDenied:
		reason := strings.TrimSpace(ae.Reason)
		if reason == "" {
			reason = "trust verdict denied admission"
		}
		return http.StatusForbidden, "scheduler_admission_denied",
			"scheduler admission denied: " + reason, true
	default:
		return http.StatusServiceUnavailable, "scheduler_unavailable",
			"scheduler admission refused", true
	}
}
