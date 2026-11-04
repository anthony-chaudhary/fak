package gateway

// request_rate_bucket.go — per-principal REQUEST-RATE admission for the served path,
// ADAPTED from TensorFold's opt-in model-request admission (Apache-2.0): ashhart/TensorFold
// zig/src/server/request_admission.zig:111 Slot.replenish/available
// @9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/server/request_admission.zig
//
// BORROWED CONTRACT, re-expressed in idiomatic Go (no upstream code copied):
//   - credit is INTEGER ns-tokens on a monotonic clock: one request is one second's
//     worth of credit (rateTokenNs), so a bucket earning r requests/sec gains r
//     ns-tokens per elapsed nanosecond (r × rateTokenNs per second);
//   - capacity = burst × rateTokenNs with burst = rate unless an explicit request_burst
//     is supplied; a fresh bucket starts full;
//   - replenish adds elapsed_ns × rate ns-tokens capped at capacity — credit never
//     exceeds capacity, no matter how far the clock jumped;
//   - credit < token refuses with the rate class and retry_after_s = whole seconds
//     (>= 1) until the deficit is earned, exact against the gate's clock;
//   - a bounded table of distinct principals (default 128 slots) bounds the
//     caller-influenced surface; a NEW principal beyond the bound is refused with its
//     own class until an idle-full bucket can be swept;
//   - refusals are typed by closed classes (rate / capacity), each distinct and both
//     under one parent sentinel, so callers classify with errors.Is — never prose;
//   - a rate <= 0 disables the gate entirely (admit freely, no credit accounting) —
//     the repo's 0 = unlimited posture.
//
// SHAPE — composition over the existing token-admission seam, not a parallel counter
// machine. The gate keeps no liveness state of its own: every admitted hold rides an
// inert TokenRateGate carrier (token_admission.go) whose live-reservation count IS the
// bucket's liveness — the same argument as the inflight gate's inert carrier per
// principal entry (inflight_admission.go). beginServedAdmission (admission_server.go)
// consults this gate FIRST — ahead of the request-count inflight gate, the coarsest
// host-level bound — debits exactly one token on admission, CANCELS it when a later
// stage refuses (the provider was never called), and ties the seat to the request's
// single lease-release path through the TokenReservation linked chain. Inert by
// default: no live gate (SetRequestRateGate(nil)) leaves the request path
// byte-for-byte historical.
//
// HONEST FENCE — what this is NOT. It meters ARRIVAL RATE, not concurrency or tokens:
// the debit stands once admitted (the request consumed rate budget at admission time),
// and nothing here settles usage, charges a window, or claims any throughput or
// hardware property. The request-count inflight dimension (upstream's global/key
// refusal classes) is the sibling borrow-port part-out in inflight_admission.go.

import (
	"errors"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// rateTokenNs is one request expressed in ns-tokens — the upstream 'token' constant
// (std.time.ns_per_s): one request is one second's worth of rate credit.
const rateTokenNs int64 = 1_000_000_000

// DefaultRatePrincipalSlots bounds how many DISTINCT principals may hold a live rate
// bucket at once when RateBucketPolicy.PrincipalSlots is unset — the same memory-bound
// posture as DefaultInflightPrincipalSlots: a principal past the bound is refused until
// an idle-full bucket can be swept, never folded into an unbounded map.
const DefaultRatePrincipalSlots = 128

// ErrRequestRate is the parent sentinel every request-rate refusal satisfies: a caller
// that only needs "the rate gate said no" classifies against it; a caller that needs
// the class uses one of the two specific sentinels below.
var ErrRequestRate = errors.New("gateway request-rate admission refused")

// The two closed refusal classes, mirroring the upstream Reason enum's rate_limit and
// principal_capacity arms (the inflight global/key arms live in inflight_admission.go,
// the sibling borrow). Each is distinct; both satisfy errors.Is(err, ErrRequestRate).
var (
	ErrRequestRateLimited  = errors.New("request rate limit exceeded")
	ErrRequestRateCapacity = errors.New("request rate principal slot bound exhausted")
)

// rateRefusal wraps one class sentinel plus the parent so errors.Is matches both —
// classification by class, never by error prose. retryAfterS (whole seconds, >= 1) is
// set only for the rate class; 0 means "no hint" (the capacity class).
type rateRefusal struct {
	cause       error
	retryAfterS int
}

func (e *rateRefusal) Error() string { return e.cause.Error() }

func (e *rateRefusal) Unwrap() []error {
	return []error{e.cause, ErrRequestRate}
}

// RetryAfterSeconds extracts the retry-after hint (whole seconds) a rate refusal
// carries; ok is false when err is not a rate refusal carrying a hint.
func RetryAfterSeconds(err error) (int, bool) {
	var r *rateRefusal
	if errors.As(err, &r) && r.retryAfterS > 0 {
		return r.retryAfterS, true
	}
	return 0, false
}

// RateBucketPolicy configures the per-principal request-rate gate. RequestsPerSecond
// <= 0 disables the gate entirely (admit freely, no credit accounting) — the repo's
// 0 = unlimited posture. RequestBurst <= 0 takes RequestsPerSecond (burst = rate).
// PrincipalSlots <= 0 takes DefaultRatePrincipalSlots.
type RateBucketPolicy struct {
	RequestsPerSecond int64
	RequestBurst      int64
	PrincipalSlots    int
}

func (p RateBucketPolicy) enabled() bool { return p.RequestsPerSecond > 0 }

// burst is the credit ceiling in requests: an explicit RequestBurst, else the rate.
func (p RateBucketPolicy) burst() int64 {
	if p.RequestBurst > 0 {
		return p.RequestBurst
	}
	return p.RequestsPerSecond
}

// capacityNs is the bucket's credit ceiling in ns-tokens: burst requests, each one
// second's worth of credit.
func (p RateBucketPolicy) capacityNs() int64 {
	return p.burst() * rateTokenNs
}

func (p RateBucketPolicy) slots() int {
	if p.PrincipalSlots > 0 {
		return p.PrincipalSlots
	}
	return DefaultRatePrincipalSlots
}

// rateBucket is one principal's monotonic ns-token credit bucket. Every method runs
// under the owning gate's mutex.
type rateBucket struct {
	principal string
	credit    int64 // ns-tokens held at updated
	updated   time.Time
	// carrier is an INERT TokenRateGate (zero caps): its live-reservation count IS the
	// bucket's liveness, which is what makes sweepIdleLocked a pure memory reclaim —
	// the same argument as the inflight gate's inert carrier per principal entry.
	carrier *TokenRateGate
}

// availableNs reports the credit a Take at now would see WITHOUT mutating — the capped
// top-up (upstream Slot.available): elapsed × rate ns-tokens, clamped so credit never
// exceeds capacity. Capping the elapsed at the fill time BEFORE the multiply keeps the
// product bounded by capacity (+ rate) no matter how far the clock jumped — the
// upstream @min(elapsed, fill_ns) clamp in integer form.
func (b *rateBucket) availableNs(rate, capacity int64, now time.Time) int64 {
	if !now.After(b.updated) {
		return b.credit
	}
	missing := capacity - b.credit
	if missing <= 0 {
		return capacity
	}
	fillNs := ceilDivInt64(missing, rate)
	elapsed := now.Sub(b.updated).Nanoseconds()
	if elapsed > fillNs {
		elapsed = fillNs
	}
	credit := b.credit + elapsed*rate
	if credit > capacity {
		credit = capacity
	}
	return credit
}

// replenishLocked applies the top-up and advances the stamp forward-only (upstream
// Slot.replenish). Caller holds the gate mutex.
func (b *rateBucket) replenishLocked(rate, capacity int64, now time.Time) {
	b.credit = b.availableNs(rate, capacity, now)
	if now.After(b.updated) {
		b.updated = now
	}
}

// RequestRateGate is the per-principal request-rate admission gate. Build with
// NewRequestRateGate; a policy with RequestsPerSecond <= 0 is fully inert: Take admits
// without any credit accounting and returns a nil lease. Safe for concurrent use. A
// nil gate admits freely, so an unwired host is unaffected.
type RequestRateGate struct {
	mu               sync.Mutex
	policy           RateBucketPolicy
	slots            int
	now              func() time.Time // injectable clock (tests); never nil after New
	buckets          map[string]*rateBucket
	admitted         uint64
	rejectedRate     uint64
	rejectedCapacity uint64
}

// NewRequestRateGate builds a gate under the given policy.
func NewRequestRateGate(p RateBucketPolicy) *RequestRateGate {
	return &RequestRateGate{
		policy:  p,
		slots:   p.slots(),
		now:     time.Now,
		buckets: map[string]*rateBucket{},
	}
}

// SetClock overrides the gate's time source (boundary tests).
func (g *RequestRateGate) SetClock(now func() time.Time) {
	if g == nil || now == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.now = now
}

// RateBucketSnapshot is the gate's current accounting — the observability read a host
// or test makes: live distinct buckets and the closed-class refusal counts.
type RateBucketSnapshot struct {
	Buckets          int
	Admitted         uint64
	RejectedRate     uint64
	RejectedCapacity uint64
}

// Snapshot returns the gate's current accounting. A nil gate reports zeros.
func (g *RequestRateGate) Snapshot() RateBucketSnapshot {
	if g == nil {
		return RateBucketSnapshot{}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return RateBucketSnapshot{
		Buckets:          len(g.buckets),
		Admitted:         g.admitted,
		RejectedRate:     g.rejectedRate,
		RejectedCapacity: g.rejectedCapacity,
	}
}

// RateBucketState is one principal's bucket read at the gate's now WITHOUT creating
// one — the observability/test read. The zero state means that principal holds no
// live bucket.
type RateBucketState struct {
	Credit   int64 // ns-tokens available at now (topped up, not mutated)
	Capacity int64
	Updated  time.Time
}

// BucketFor reports the bucket state a principal would be charged to, without
// creating one. A nil gate or a disabled policy reports the zero state.
func (g *RequestRateGate) BucketFor(principal string) RateBucketState {
	if g == nil || !g.policy.enabled() {
		return RateBucketState{}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	b := g.buckets[principal]
	if b == nil {
		return RateBucketState{}
	}
	return RateBucketState{
		Credit:   b.availableNs(g.policy.RequestsPerSecond, g.policy.capacityNs(), g.now()),
		Capacity: g.policy.capacityNs(),
		Updated:  b.updated,
	}
}

// RateLease is one admitted request's hold on a rate bucket's seat. Release frees the
// seat exactly once (the request finished or was abandoned — the credit debit STANDS:
// the request consumed rate budget at admission); Cancel refunds the credit AND frees
// the seat (a later admission stage refused, so the provider was never called). Both
// are idempotent and nil-safe; a refused Take returns a nil lease that holds nothing
// and releases nothing.
type RateLease struct {
	gate      *RequestRateGate
	principal string
	requests  int64
	head      *TokenReservation // the principal entry's inert carrier; always non-nil on an admitted hold
	cancelled sync.Once
}

// Take debits requests × rateTokenNs ns-tokens from the principal's bucket and admits
// while the topped-up credit covers them; otherwise it refuses with the rate class and
// a whole-second retry-after hint, exact against the gate's clock. A disabled gate
// (nil receiver or rate <= 0) admits freely with no credit accounting and returns a
// nil lease; requests <= 0 admit without debiting. On refusal the gate holds nothing.
func (g *RequestRateGate) Take(principal string, requests int64) (*RateLease, error) {
	if g == nil || !g.policy.enabled() || requests <= 0 {
		return nil, nil
	}
	rate := g.policy.RequestsPerSecond
	capacity := g.policy.capacityNs()
	g.mu.Lock()
	defer g.mu.Unlock()
	now := g.now()
	b := g.buckets[principal]
	if b == nil {
		if len(g.buckets) >= g.slots {
			g.sweepIdleLocked(now)
			if len(g.buckets) >= g.slots {
				g.rejectedCapacity++
				return nil, &rateRefusal{cause: ErrRequestRateCapacity}
			}
		}
		b = &rateBucket{
			principal: principal,
			credit:    capacity,
			updated:   now,
			carrier:   NewTokenRateGate(TokenRatePolicy{}),
		}
		g.buckets[principal] = b
	}
	b.replenishLocked(rate, capacity, now)
	need := requests * rateTokenNs
	if b.credit < need {
		g.rejectedRate++
		// Two ceilings, mirroring the upstream divCeil pair: whole ns until the deficit
		// is earned, then whole seconds (>= 1) the client waits.
		waitNs := ceilDivInt64(need-b.credit, rate)
		waitS := ceilDivInt64(waitNs, rateTokenNs)
		if waitS < 1 {
			waitS = 1
		}
		return nil, &rateRefusal{cause: ErrRequestRateLimited, retryAfterS: int(waitS)}
	}
	b.credit -= need
	head, err := b.carrier.Admit(ratelimit.TokenUsage{})
	if err != nil {
		// The carrier is inert (zero caps) and cannot refuse; keep the invariant total.
		b.credit += need
		return nil, err
	}
	g.admitted++
	return &RateLease{gate: g, principal: principal, requests: requests, head: head}, nil
}

// Release frees the seat exactly once: the carrier's settlement drops the bucket's
// live-reservation count, which is what lets sweepIdleLocked reclaim the seat. The
// credit debit stands — the request consumed rate budget at admission. Nil-safe.
func (l *RateLease) Release() {
	if l == nil || l.head == nil {
		return
	}
	l.head.Release()
}

// Cancel refunds the credit (topped up first, capped at capacity) AND frees the seat —
// the cancel semantics for a request that passed the rate gate but was refused by a
// LATER admission stage, so the provider was never called and the tenant's credit is
// restored. Idempotent; nil-safe.
func (l *RateLease) Cancel() {
	if l == nil || l.gate == nil {
		return
	}
	l.cancelled.Do(func() {
		l.gate.refund(l.principal, l.requests)
		l.head.Release()
	})
}

// refund returns requests × rateTokenNs ns-tokens to the principal's bucket, topped up
// first and capped at capacity. A principal with no live bucket refunds nothing.
func (g *RequestRateGate) refund(principal string, requests int64) {
	if g == nil || !g.policy.enabled() || requests <= 0 {
		return
	}
	rate := g.policy.RequestsPerSecond
	capacity := g.policy.capacityNs()
	g.mu.Lock()
	defer g.mu.Unlock()
	b := g.buckets[principal]
	if b == nil {
		return
	}
	b.replenishLocked(rate, capacity, g.now())
	b.credit += requests * rateTokenNs
	if b.credit > capacity {
		b.credit = capacity
	}
}

// sweepIdleLocked drops every bucket whose carrier holds no live reservation AND whose
// topped-up credit is already full. Such a bucket is indistinguishable from a fresh
// one — the next top-up would show full credit regardless — so evicting it forgives no
// budget and cannot launder a spent bucket: the upstream slot-reuse rule ("idle with
// full credit: reusing it refunds nothing") in sweep form. Caller holds g.mu.
func (g *RequestRateGate) sweepIdleLocked(now time.Time) {
	rate := g.policy.RequestsPerSecond
	capacity := g.policy.capacityNs()
	for p, b := range g.buckets {
		if b.carrier.Snapshot().InFlight != 0 {
			continue
		}
		if b.availableNs(rate, capacity, now) < capacity {
			continue
		}
		delete(g.buckets, p)
	}
}

// ceilDivInt64 is integer ceiling division for positive operands — the upstream
// divCeil(u64, ...) in signed form.
func ceilDivInt64(a, b int64) int64 {
	q := a / b
	if a%b > 0 {
		q++
	}
	return q
}

var (
	requestRateGateMu   sync.RWMutex
	liveRequestRateGate *RequestRateGate
)

// SetRequestRateGate installs the live request-rate gate consulted FIRST by
// beginServedAdmission (ahead of the request-count inflight gate); nil detaches it and
// the request path goes byte-for-byte historical. Swaps under one guard alongside
// LiveRequestRateGate.
func SetRequestRateGate(g *RequestRateGate) {
	requestRateGateMu.Lock()
	liveRequestRateGate = g
	requestRateGateMu.Unlock()
}

// LiveRequestRateGate returns the installed gate, or nil when none is attached.
func LiveRequestRateGate() *RequestRateGate {
	requestRateGateMu.RLock()
	defer requestRateGateMu.RUnlock()
	return liveRequestRateGate
}
