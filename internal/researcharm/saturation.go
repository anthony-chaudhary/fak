package researcharm

// saturation.go — the TYPED capacity refusal and the queue observability layered on the
// bounded wait AdmitWithWait already provides.
//
// History, because the layering matters: the arm path originally refused instantly with
// the exception text as the 429 body and a hardcoded `Retry-After: 1` — the same
// free-text drift the closed refusal vocabulary exists to kill, and a backpressure hint
// that was a guess. A bounded wait then landed on the coordinator (AdmitWithWait), which
// fixed the shed-vs-wait shape. What remained is everything a CLIENT or an AUTOSCALER
// needs in order to ACT on the refusal, and a BOUND on how many requests may wait:
//
//  1. A TYPED refusal. A shed returns *SaturationError, which wraps the historical
//     ErrArmConcurrencyExceeded sentinel (so errors.Is keeps working for every existing
//     caller) and carries the closed-vocabulary reason ARM_CONCURRENCY_SATURATED, the
//     arm's live counters, how long the request actually waited, and a Retry-After
//     derived from MEASURED queue wait rather than a constant.
//
//  2. A BOUNDED queue. The wait is capped per arm (QueueDepth), because a queue with no
//     bound converts a capacity signal into unbounded latency — worse than a fast honest
//     refusal. Beyond the bound a new arrival is shed at once, carrying the reason.
//
//  3. Observability that makes the wait legible: per-arm queued/waited/shed counters and
//     the measured wait, so an operator can tell a burst being ABSORBED (waits rising, no
//     sheds) from sustained overload (both rising).
//
// The refusal is rendered as a typed 429 by internal/gateway/arm_admission.go.

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// ReasonArmConcurrencySaturated is the closed-vocabulary reason an arm admission carries
// when it is shed for capacity: its bounded wait for a slot was exhausted, or the wait
// queue was full. Declared in dos.toml as [reasons.ARM_CONCURRENCY_SATURATED].
const ReasonArmConcurrencySaturated = "ARM_CONCURRENCY_SATURATED"

// maxArmAdmissionWait is the hard ceiling on any configured wait, so a typo can never
// park an HTTP handler for minutes. A longer wait belongs to a supervisor that carries
// the turn across it, not to a request-scoped queue.
const maxArmAdmissionWait = 60 * time.Second

// maxArmQueueDepth is the hard ceiling on any configured queue bound.
const maxArmQueueDepth = 4096

// SaturationError is the typed capacity refusal an arm admission returns when a request
// cannot be given a slot within its bounded wait. It wraps ErrArmConcurrencyExceeded, so
// errors.Is(err, ErrArmConcurrencyExceeded) holds for every caller that predates it, and
// adds the closed reason token, the arm's counters, the measured wait, and the
// backpressure hint the client should honor.
type SaturationError struct {
	ArmID      string
	Active     int           // in-flight requests on the arm at refusal
	Limit      int           // the arm's EFFECTIVE concurrency ceiling (0 = unlimited)
	Queued     int           // waiters on the arm's queue at refusal
	Waited     time.Duration // how long THIS request waited for a slot (0 = shed instantly)
	Full       bool          // the wait queue was full, so no wait was attempted
	QueueBound int           // the arm's configured queue bound (set only when Full)
	RetryAfter time.Duration // backpressure hint derived from measured queue wait
	// CtxErr, when set, is the caller-side error that ended an otherwise-valid wait (a
	// canceled context or an expired client deadline). The refusal is STILL a capacity
	// refusal — it carries the closed reason — but it also unwraps to CtxErr, so a caller
	// that branches on context.Canceled/DeadlineExceeded keeps its branch.
	CtxErr error
}

// Error preserves the historical sentence for the instant-shed path (so existing log
// greps and assertions keep matching) and appends what this refusal actually did.
func (e *SaturationError) Error() string {
	if e == nil {
		return ErrArmConcurrencyExceeded.Error()
	}
	base := fmt.Sprintf(": arm %q at %d/%d active requests", e.ArmID, e.Active, e.Limit)
	switch {
	case e.Full:
		return ErrArmConcurrencyExceeded.Error() + base +
			fmt.Sprintf(" and the admission queue is full (%d waiting, bound %d); shed without waiting", e.Queued, e.QueueBound)
	case e.CtxErr != nil:
		return ErrArmConcurrencyExceeded.Error() + base +
			fmt.Sprintf("; waited %s for a slot and the caller gave up (%v)", e.Waited.Round(time.Millisecond), e.CtxErr)
	case e.Waited > 0:
		return ErrArmConcurrencyExceeded.Error() + base +
			fmt.Sprintf("; waited %s for a slot and shed", e.Waited.Round(time.Millisecond))
	default:
		return ErrArmConcurrencyExceeded.Error() + base
	}
}

// Unwrap keeps the historical sentinel reachable through errors.Is and, when the wait
// ended because the caller's context did, keeps the caller's own error reachable too.
func (e *SaturationError) Unwrap() []error {
	if e == nil {
		return nil
	}
	if e.CtxErr != nil {
		return []error{ErrArmConcurrencyExceeded, e.CtxErr}
	}
	return []error{ErrArmConcurrencyExceeded}
}

// Reason is the closed-vocabulary token the refusal carries.
func (e *SaturationError) Reason() string { return ReasonArmConcurrencySaturated }

// QueuedRequests reports the current total number of waiters across all arms.
func (c *Coordinator) QueuedRequests() int {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	total := 0
	for _, arm := range c.arms {
		total += len(arm.waiters)
	}
	return total
}

// armLoadLocked reports (active, effectiveLimit, queued) for an arm. The limit is the
// EFFECTIVE ceiling admission enforces — an active lease's Concurrency overrides the
// arm's own max — so a shed never reports "at 1/16 in flight" while refusing because a
// lease pinned the arm to 1. Must hold c.mu.
func (c *Coordinator) armLoadLocked(armID string) (int, int, int) {
	arm, ok := c.arms[armID]
	if !ok {
		return 0, c.defaultMaxConcurrency, 0
	}
	effective := arm.maxConcurrency
	now := time.Now()
	for _, l := range c.leases {
		if l.ArmID == armID && now.Before(l.ExpiresAt) {
			if l.Concurrency > 0 {
				effective = l.Concurrency
			}
			break
		}
	}
	return arm.activeRequests, effective, len(arm.waiters)
}

// saturationNow builds the typed refusal for a shed, filling the arm's counters and the
// measured Retry-After under the lock. Must hold c.mu.
func (c *Coordinator) saturationNow(armID string, waited time.Duration, ctxErr error) *SaturationError {
	active, limit, queued := c.armLoadLocked(armID)
	return &SaturationError{
		ArmID:      armID,
		Active:     active,
		Limit:      limit,
		Queued:     queued,
		Waited:     waited,
		RetryAfter: c.retryHintLocked(armID),
		CtxErr:     ctxErr,
	}
}

// retryHintLocked reports the backpressure hint for a shed: the measured mean queue wait
// when one exists, otherwise the arm's configured wait budget. Clamped to [1s, 60s] so the
// header is always actionable and never absurd. Must hold c.mu.
func (c *Coordinator) retryHintLocked(armID string) time.Duration {
	hint := c.admitWait
	if arm, ok := c.arms[armID]; ok && arm.waitEMAMicros > 0 {
		hint = time.Duration(arm.waitEMAMicros) * time.Microsecond
	}
	if hint < time.Second {
		hint = time.Second
	}
	if hint > maxArmAdmissionWait {
		hint = maxArmAdmissionWait
	}
	return hint
}

// recordWaitLocked books a measured queue wait for the arm, so the Retry-After a later
// shed carries is derived from what waiters actually experience. Must hold c.mu.
func (c *Coordinator) recordWaitLocked(armID string, waited time.Duration) {
	arm, ok := c.arms[armID]
	if !ok {
		return
	}
	arm.admittedAfterWait++
	// A wait that actually happened must never read as "no measurement": 0 is the sentinel
	// for "never measured", and a fast handoff can complete in well under a microsecond, so
	// truncation to 0 would report a 0s average beside a nonzero admitted_after_wait count
	// and silently discard the measured hint. Floor a real wait at 1µs.
	us := waited.Microseconds()
	if us <= 0 {
		us = 1
	}
	// EMA with a 1/4 gain: responsive to a step change without letting one outlier
	// define the hint.
	if arm.waitEMAMicros == 0 {
		arm.waitEMAMicros = us
		return
	}
	arm.waitEMAMicros = (3*arm.waitEMAMicros + us) / 4
}

// SetAdmitWait records the bounded admission wait the serve boundary arms, so the
// Retry-After a shed carries has a sane base before any wait has been measured. 0 leaves
// the historical instant shed. Clamped to [0, 60s].
func (c *Coordinator) SetAdmitWait(d time.Duration) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if d < 0 {
		d = 0
	}
	if d > maxArmAdmissionWait {
		d = maxArmAdmissionWait
	}
	c.admitWait = d
}

// AdmitWait returns the configured bounded admission wait (0 = instant shed).
func (c *Coordinator) AdmitWait() time.Duration {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.admitWait
}

// SetQueueDepth bounds how many admissions may WAIT per arm. Beyond it a new arrival is
// shed at once rather than joining an unbounded backlog, because a queue with no bound
// converts a capacity signal into unbounded latency. 0 leaves the queue unbounded (the
// behavior before this bound existed). Clamped to [0, 4096].
func (c *Coordinator) SetQueueDepth(n int) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if n < 0 {
		n = 0
	}
	if n > maxArmQueueDepth {
		n = maxArmQueueDepth
	}
	c.queueDepth = n
}

// QueueDepth returns the per-arm bound on waiting admissions (0 = unbounded).
func (c *Coordinator) QueueDepth() int {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.queueDepth
}

// ArmSaturation is the per-arm saturation readout the gateway publishes on /metrics: how
// many requests are waiting for a slot, against what ceiling, and what the queue has seen.
type ArmSaturation struct {
	ArmID             string
	Active            int
	Limit             int
	Queued            int
	WaitedTotal       int64
	AdmittedAfterWait int64
	ShedsTotal        int64
	AvgQueueWait      time.Duration
}

// Saturations returns one readout per registered arm.
func (c *Coordinator) Saturations() []ArmSaturation {
	if c == nil {
		return nil
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]ArmSaturation, 0, len(c.arms))
	for _, arm := range c.arms {
		active, limit, queued := c.armLoadLocked(arm.id)
		out = append(out, ArmSaturation{
			ArmID:             arm.id,
			Active:            active,
			Limit:             limit,
			Queued:            queued,
			WaitedTotal:       arm.waitedTotal,
			AdmittedAfterWait: arm.admittedAfterWait,
			ShedsTotal:        arm.shedsTotal,
			AvgQueueWait:      time.Duration(arm.waitEMAMicros) * time.Microsecond,
		})
	}
	return out
}

// envArmQueueDepth names the operator's per-arm bound on queued waiters. The wait BUDGET
// itself is the --arm-admit-wait flag; this only bounds how many may wait.
const envArmQueueDepth = "FAK_ARM_QUEUE_DEPTH"

// ParseArmQueueDepth folds FAK_ARM_QUEUE_DEPTH into a per-arm bound. An unparseable or
// negative value leaves the queue unbounded (0) and returns a note naming it, so the
// caller can report rather than silently reinterpret.
func ParseArmQueueDepth(getenv func(string) string) (int, string) {
	if getenv == nil {
		return 0, ""
	}
	raw := strings.TrimSpace(getenv(envArmQueueDepth))
	if raw == "" {
		return 0, ""
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Sprintf("%s=%q is not an integer: the arm admission queue stays UNBOUNDED", envArmQueueDepth, raw)
	}
	if n < 0 {
		return 0, fmt.Sprintf("%s=%d is negative: the arm admission queue stays UNBOUNDED", envArmQueueDepth, n)
	}
	if n > maxArmQueueDepth {
		return maxArmQueueDepth, fmt.Sprintf("%s=%d exceeds the %d ceiling: clamped", envArmQueueDepth, n, maxArmQueueDepth)
	}
	return n, ""
}
