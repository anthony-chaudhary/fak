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

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// AdmissionPolicy holds the admission knobs. Each cap is disabled by a non-positive value
// (so a test can isolate a single axis); DefaultAdmissionPolicy fills shipping defaults.
type AdmissionPolicy struct {
	// MaxNumSeqs caps the running set (vLLM's max-num-seqs): no request is admitted while
	// the running count is at this bound. ≤0 disables the seq cap.
	MaxNumSeqs int
	// TokenBudget caps the sum of the running set's token footprints (the num-batched-tokens
	// admission budget; until paged KV lands this is the whole budget). ≤0 disables it.
	TokenBudget int
	// MaxWaiting bounds the waiting queue. A request that cannot be admitted now is shed
	// (429) once the queue is at this bound — the backpressure limit. ≤0 = unbounded
	// (never sheds; the historical "queue forever" behavior).
	MaxWaiting int
	// AgingRounds is the starvation guard: a waiting request's effective priority improves
	// by one for every AgingRounds rounds it is passed over. ≤0 disables aging (raw
	// priority stands and a higher-priority flood can starve a low-priority waiter).
	AgingRounds int
}

// DefaultAdmissionPolicy returns the shipping defaults: a 256-sequence running cap (the
// vLLM V1 default), an 8192-token batched-admission budget, a 1024-deep waiting bound
// before shedding, and aging every round so no waiter is ever starved.
func DefaultAdmissionPolicy() AdmissionPolicy {
	return AdmissionPolicy{
		MaxNumSeqs:  256,
		TokenBudget: 8192,
		MaxWaiting:  1024,
		AgingRounds: 1,
	}
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
	TraceID  string
	Priority int
	Tokens   int
	Trust    AdmissionTrust
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
	MaxWaitRounds int64 // oldest current waiter's age in rounds (starvation visibility, gauge)
	Admitted      int64 // cumulative requests promoted into the running set (counter)
	Queued        int64 // cumulative requests placed on the waiting queue (counter)
	Shed          int64 // cumulative requests shed under overload — 429 (counter)
	Denied        int64 // cumulative requests rejected by a trust verdict (counter)
}

// AdmissionController is the admission/priority/fairness gate over the native loop. The
// zero value is not usable — build one with NewAdmissionController. It is safe for
// concurrent use (the gateway request path and the loop both touch it).
type AdmissionController struct {
	mu      sync.Mutex
	policy  AdmissionPolicy
	running map[string]SeqRequest // admitted, holding budget, keyed by TraceID
	tokens  int                   // Σ running[*].Tokens, maintained incrementally
	waiting []waitEntry           // waiting queue, re-sorted by effective priority each round
	round   int64                 // monotone admission-round counter (drives aging)
	stats   AdmissionStats        // cumulative counters (gauges are derived in Stats)
	seq     uint64                // internal per-request suffix for duplicate caller traces
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
	return &AdmissionController{policy: p, running: map[string]SeqRequest{}}
}

// AdmissionLease is the live request's hold on one admitted scheduler slot. Release is
// idempotent and schedules the next waiting request after freeing this request's budget.
type AdmissionLease struct {
	ctl     *AdmissionController
	traceID string
	once    sync.Once
}

// Release frees the admitted request's token/sequence budget and promotes waiters.
func (l *AdmissionLease) Release() {
	if l == nil || l.ctl == nil || l.traceID == "" {
		return
	}
	l.once.Do(func() { l.ctl.completeAndSchedule(l.traceID) })
}

// AdmissionError is the typed served-path refusal returned when the live admission
// gate sheds overload or a trust verdict denies a request before the planner runs.
type AdmissionError struct {
	Verdict AdmissionVerdict
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
	if req.Trust.Deny {
		c.stats.Denied++
		return VerdictDenied
	}
	// Fast path: nobody is waiting and there is headroom now — admit immediately so an
	// idle/underloaded node does not pay a Schedule round to serve its first requests.
	if len(c.waiting) == 0 && c.hasHeadroomLocked(req.Tokens) {
		c.admitLocked(req)
		return VerdictAdmitted
	}
	// The request must wait. Shed it if the waiting queue is already at its bound — the
	// backpressure surface that replaces unbounded queueing.
	if c.policy.MaxWaiting > 0 && len(c.waiting) >= c.policy.MaxWaiting {
		c.stats.Shed++
		return VerdictShed
	}
	c.waiting = append(c.waiting, waitEntry{req: req, enqueuedRound: c.round})
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
	req.TraceID = c.admissionTraceID(req.TraceID)

	c.mu.Lock()
	if req.Trust.Deny {
		c.stats.Denied++
		c.mu.Unlock()
		return nil, &AdmissionError{Verdict: VerdictDenied, Reason: req.Trust.Reason}
	}
	if len(c.waiting) == 0 && c.hasHeadroomLocked(req.Tokens) {
		c.admitLocked(req)
		c.mu.Unlock()
		return &AdmissionLease{ctl: c, traceID: req.TraceID}, nil
	}
	if c.policy.MaxWaiting > 0 && len(c.waiting) >= c.policy.MaxWaiting {
		c.stats.Shed++
		c.mu.Unlock()
		return nil, &AdmissionError{Verdict: VerdictShed}
	}
	ready := make(chan struct{})
	c.waiting = append(c.waiting, waitEntry{req: req, enqueuedRound: c.round, ready: ready})
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
		return nil
	}
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
		if !c.hasHeadroomLocked(e.req.Tokens) {
			break // head-of-line: do not let a lower-priority request skip a blocked one
		}
		c.admitLocked(e.req)
		if e.ready != nil {
			close(e.ready)
		}
		admitted = append(admitted, e.req)
	}
	// The admitted set is the sorted prefix; keep the rest. Copy so the released entries
	// are not retained by the backing array across later Offer appends.
	c.waiting = append([]waitEntry(nil), c.waiting[len(admitted):]...)
	return admitted
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
			c.waiting = append(c.waiting[:i], c.waiting[i+1:]...)
			return
		}
	}
	if c.completeLocked(traceID) {
		c.scheduleLocked()
	}
}

// hasHeadroomLocked reports whether a request of the given token footprint fits the
// running set right now under both caps. Caller holds c.mu.
func (c *AdmissionController) hasHeadroomLocked(tokens int) bool {
	if c.policy.MaxNumSeqs > 0 && len(c.running) >= c.policy.MaxNumSeqs {
		return false
	}
	if c.policy.TokenBudget > 0 && c.tokens+tokens > c.policy.TokenBudget {
		return false
	}
	return true
}

// admitLocked moves a request into the running set and charges its tokens. Caller holds c.mu.
func (c *AdmissionController) admitLocked(req SeqRequest) {
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
	writeHelpType(b, schedMetricPrefix+"max_num_seqs", "Configured max-num-seqs running-set cap (0 = uncapped).", "gauge")
	fmt.Fprintf(b, "%smax_num_seqs %d\n", schedMetricPrefix, c.policy.MaxNumSeqs)
	writeHelpType(b, schedMetricPrefix+"max_wait_rounds", "Oldest current waiter's age in admission rounds (starvation visibility).", "gauge")
	fmt.Fprintf(b, "%smax_wait_rounds %d\n", schedMetricPrefix, st.MaxWaitRounds)
	writeCounter(b, schedMetricPrefix+"admitted_total", "Requests promoted into the running set.", st.Admitted)
	writeCounter(b, schedMetricPrefix+"queued_total", "Requests placed on the waiting queue.", st.Queued)
	writeCounter(b, schedMetricPrefix+"shed_total", "Requests shed under overload (waiting queue at bound; HTTP 429).", st.Shed)
	writeCounter(b, schedMetricPrefix+"denied_total", "Requests rejected by a per-tenant trust verdict.", st.Denied)
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

func (s *Server) beginServedAdmission(ctx context.Context, turn servedSessionTurn, messages []agent.Message, tools []agent.ToolDef, maxTokens int) (*AdmissionLease, error) {
	if s == nil {
		return nil, nil
	}
	s.admissionMu.RLock()
	c := s.admissionCtl
	s.admissionMu.RUnlock()
	if c == nil {
		return nil, nil
	}
	return c.Acquire(ctx, SeqRequest{
		TraceID:  turn.traceID,
		Priority: turn.state.Priority,
		Tokens:   estimateServedAdmissionTokens(messages, tools, maxTokens),
	})
}

func estimateServedAdmissionTokens(messages []agent.Message, tools []agent.ToolDef, maxTokens int) int {
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
	tokens := chars / 4
	if chars > 0 && tokens == 0 {
		tokens = 1
	}
	if maxTokens > 0 {
		tokens += maxTokens
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
	case VerdictShed:
		return http.StatusTooManyRequests, "scheduler_overloaded",
			"scheduler overloaded — back off and retry", true
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
