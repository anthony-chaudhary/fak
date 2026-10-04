// Package deadlineadmit is the shared mechanism for deadline-aware admission:
// read the time budget a client declared for a request, estimate how long the
// request will take from measured prefill and decode throughput, and refuse
// work that cannot finish before the client gives up.
//
// The failure this exists for: a serving node with six requests in flight,
// each a 1.6k-token prefill plus a ~300-token decode, kept running all six
// past the clients' 300 s deadline. Every token was computed for a client that
// had already left. Refusing the request up front costs the client one fast
// 503 it can fail over on; admitting it costs the node minutes of work that is
// thrown away and slows every other request in flight.
//
// The package is pure mechanism. It holds no policy about which node to try
// next (that is a router's decision) and it performs no I/O. Time comes from
// an injected clock so the estimator is deterministic under test.
//
// Wire contract:
//
//   - A client declares a budget with any of [HeaderFakDeadlineMs] (remaining
//     milliseconds), [HeaderRequestTimeout] (seconds), or
//     [HeaderStainlessTimeout] (seconds; sent by the OpenAI and Anthropic
//     SDKs with the client's own timeout). The smallest positive budget wins.
//     Budgets are relative so no cross-host clock agreement is needed; a hop
//     that forwards the request rewrites [HeaderFakDeadlineMs] with what is
//     left (see [SetRemaining]).
//   - A refusal is HTTP 503 with a Retry-After header, the error code
//     [CodeDeadlineInfeasible], and [HeaderReject] naming the reason. 503 is
//     deliberate: a router reads it as "this target cannot serve, try the
//     next one", where a 429 would ask it to retry the same saturated target.
package deadlineadmit

import (
	"errors"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Wire names. They are part of the contract between clients, routers, and
// serving nodes; do not rename.
const (
	// HeaderFakDeadlineMs is the remaining client budget in milliseconds. A
	// forwarding hop rewrites it with the budget left after its own time.
	HeaderFakDeadlineMs = "X-Fak-Deadline-Ms"
	// HeaderRequestTimeout is a client budget in (possibly fractional) seconds.
	HeaderRequestTimeout = "X-Request-Timeout"
	// HeaderStainlessTimeout is the client timeout in seconds that the
	// Stainless-generated OpenAI and Anthropic SDKs send on every request.
	HeaderStainlessTimeout = "X-Stainless-Timeout"
	// HeaderReject names why a request was refused (a Reason value).
	HeaderReject = "X-Fak-Admission-Reject"
	// HeaderEstimateMs reports the completion estimate behind a refusal.
	HeaderEstimateMs = "X-Fak-Admission-Estimate-Ms"

	// CodeDeadlineInfeasible is the error code of a deadline refusal.
	CodeDeadlineInfeasible = "deadline_infeasible"
)

// Reason is the closed set of admission refusal reasons.
type Reason string

const (
	// ReasonNone means the request was admitted.
	ReasonNone Reason = ""
	// ReasonDeadlineExpired means the budget was already spent on arrival.
	ReasonDeadlineExpired Reason = "deadline_expired"
	// ReasonDeadlineInfeasible means the measured estimate exceeds the budget.
	ReasonDeadlineInfeasible Reason = "deadline_infeasible"
)

// ErrDeadlineInfeasible is the sentinel every refusal wraps, so callers can
// branch with errors.Is instead of matching text.
var ErrDeadlineInfeasible = errors.New(CodeDeadlineInfeasible)

// maxBudget caps a declared budget. A larger value is treated as this cap so
// an absurd header cannot overflow time arithmetic.
const maxBudget = 24 * time.Hour

// Budget returns the smallest positive client budget declared in h. ok is
// false when no budget header is present or none parses to a positive value.
func Budget(h http.Header) (budget time.Duration, ok bool) {
	consider := func(d time.Duration) {
		if d <= 0 {
			return
		}
		if d > maxBudget {
			d = maxBudget
		}
		if !ok || d < budget {
			budget, ok = d, true
		}
	}
	if v := strings.TrimSpace(h.Get(HeaderFakDeadlineMs)); v != "" {
		if ms, err := strconv.ParseFloat(v, 64); err == nil && finite(ms) {
			consider(durationFrom(ms, time.Millisecond))
		}
	}
	for _, name := range [...]string{HeaderRequestTimeout, HeaderStainlessTimeout} {
		if v := strings.TrimSpace(h.Get(name)); v != "" {
			if s, err := strconv.ParseFloat(v, 64); err == nil && finite(s) {
				consider(durationFrom(s, time.Second))
			}
		}
	}
	return budget, ok
}

// SetRemaining writes the budget left into h as [HeaderFakDeadlineMs] for
// the next hop. A non-positive remaining budget is written as 1 ms, so the
// next hop refuses it as expired instead of reading "no deadline".
func SetRemaining(h http.Header, remaining time.Duration) {
	ms := remaining.Milliseconds()
	if ms < 1 {
		ms = 1
	}
	h.Set(HeaderFakDeadlineMs, strconv.FormatInt(ms, 10))
}

// WriteRefusal writes the typed 503 for a refused verdict: Retry-After, the
// reject reason, the estimate, and an OpenAI-shaped error body.
func WriteRefusal(w http.ResponseWriter, v Verdict) {
	h := w.Header()
	h.Set("Retry-After", strconv.Itoa(retryAfterSeconds(v.RetryAfter)))
	h.Set(HeaderReject, string(v.Reason))
	if v.Estimate > 0 {
		h.Set(HeaderEstimateMs, strconv.FormatInt(v.Estimate.Milliseconds(), 10))
	}
	h.Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusServiceUnavailable)
	msg := "request cannot finish before its declared deadline (" + string(v.Reason) + ")"
	_, _ = w.Write([]byte(`{"error":{"message":"` + msg + `","type":"server_overloaded","code":"` + CodeDeadlineInfeasible + `"}}`))
}

func retryAfterSeconds(d time.Duration) int {
	s := int(math.Ceil(d.Seconds()))
	if s < 1 {
		s = 1
	}
	return s
}

// Config tunes an Estimator. The zero value is usable.
type Config struct {
	// HalfLife is the decay half-life of the throughput averages. Zero means
	// DefaultHalfLife.
	HalfLife time.Duration
	// Headroom is the fraction of the remaining budget an estimate may use.
	// Zero means DefaultHeadroom. A request is refused when
	// estimate > Headroom * remaining.
	Headroom float64
	// DefaultDecodeTokens is the decode length assumed before any completion
	// has been observed and the request set no max_tokens. Zero means
	// DefaultDecodeTokens.
	DefaultDecodeTokens int
	// RetryAfterMax caps the Retry-After a refusal advertises. Zero means
	// DefaultRetryAfterMax.
	RetryAfterMax time.Duration
}

// Defaults for Config.
const (
	DefaultHalfLife            = 2 * time.Minute
	DefaultHeadroom            = 0.9
	DefaultDecodeTokens        = 256
	DefaultRetryAfterMax       = 60 * time.Second
	minRateSamples             = 1
	minEstimatorConcurrencyObs = 1.0
)

// Estimator keeps decaying averages of per-request prefill and decode
// throughput, the completion length, and the concurrency those were observed
// at, plus a live in-flight count. All methods are safe for concurrent use
// and take only a short memory lock.
type Estimator struct {
	cfg Config
	now func() time.Time

	mu       sync.Mutex
	inFlight int
	samples  int
	last     time.Time
	prefill  float64 // tokens/second per request
	decode   float64 // tokens/second per request
	complTok float64 // completion tokens per request
	conc     float64 // in-flight count at observation time
	hasPre   bool
	hasDec   bool
}

// NewEstimator returns an Estimator reading time from now (time.Now when nil).
func NewEstimator(cfg Config, now func() time.Time) *Estimator {
	if now == nil {
		now = time.Now
	}
	if cfg.HalfLife <= 0 {
		cfg.HalfLife = DefaultHalfLife
	}
	if cfg.Headroom <= 0 || cfg.Headroom > 1 {
		cfg.Headroom = DefaultHeadroom
	}
	if cfg.DefaultDecodeTokens <= 0 {
		cfg.DefaultDecodeTokens = DefaultDecodeTokens
	}
	if cfg.RetryAfterMax <= 0 {
		cfg.RetryAfterMax = DefaultRetryAfterMax
	}
	return &Estimator{cfg: cfg, now: now}
}

// Begin counts one request in flight and returns the func that ends it. The
// returned func is idempotent.
func (e *Estimator) Begin() (end func()) {
	e.mu.Lock()
	e.inFlight++
	e.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			e.mu.Lock()
			if e.inFlight > 0 {
				e.inFlight--
			}
			e.mu.Unlock()
		})
	}
}

// InFlight returns the live in-flight count.
func (e *Estimator) InFlight() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.inFlight
}

// Observation is one finished request's measured timing.
type Observation struct {
	PromptTokens     int
	Prefill          time.Duration // time to first token
	CompletionTokens int
	Decode           time.Duration // first token to last token
}

// Observe folds one finished request into the averages. The in-flight count
// at observation time (including the observed request when it has not ended
// yet) is recorded with it, so a later estimate can rescale the rates to the
// concurrency it will actually run at. Observations without a usable
// prefill or decode pair update only the half they carry.
func (e *Estimator) Observe(o Observation) {
	pre := 0.0
	if o.PromptTokens > 0 && o.Prefill > 0 {
		pre = float64(o.PromptTokens) / o.Prefill.Seconds()
	}
	dec := 0.0
	if o.CompletionTokens > 1 && o.Decode > 0 {
		dec = float64(o.CompletionTokens) / o.Decode.Seconds()
	}
	if pre <= 0 && dec <= 0 {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	now := e.now()
	alpha := 1.0
	if e.samples > 0 {
		dt := now.Sub(e.last)
		if dt < 0 {
			dt = 0
		}
		// Weight of the old average after dt: 0.5^(dt/halfLife). Even with
		// dt == 0 a new sample moves the average; floor alpha at 0.2.
		keep := math.Pow(0.5, dt.Seconds()/e.cfg.HalfLife.Seconds())
		alpha = 1 - keep
		if alpha < 0.2 {
			alpha = 0.2
		}
	}
	conc := float64(e.inFlight)
	if conc < minEstimatorConcurrencyObs {
		conc = minEstimatorConcurrencyObs
	}
	blend := func(old float64, has bool, v float64) float64 {
		if !has {
			return v
		}
		return old + alpha*(v-old)
	}
	if pre > 0 {
		e.prefill = blend(e.prefill, e.hasPre, pre)
		e.hasPre = true
	}
	if dec > 0 {
		e.decode = blend(e.decode, e.hasDec, dec)
		e.complTok = blend(e.complTok, e.hasDec, float64(o.CompletionTokens))
		e.hasDec = true
	}
	e.conc = blend(e.conc, e.samples > 0, conc)
	e.samples++
	e.last = now
}

// Verdict is the result of an admission check.
type Verdict struct {
	Admit bool
	// Reason is ReasonNone when admitted.
	Reason Reason
	// Estimate is the predicted completion time; zero when unknown.
	Estimate time.Duration
	// Remaining is the budget the check was made against.
	Remaining time.Duration
	// RetryAfter is the advised wait before retrying this target.
	RetryAfter time.Duration
	// Measured is false when no throughput has been observed yet; such a
	// request is admitted (no data is not evidence of overload).
	Measured bool
}

// Err returns nil for an admitted verdict and an error wrapping
// ErrDeadlineInfeasible otherwise.
func (v Verdict) Err() error {
	if v.Admit {
		return nil
	}
	return &RefusalError{Verdict: v}
}

// RefusalError carries a refused Verdict; it wraps ErrDeadlineInfeasible.
type RefusalError struct{ Verdict Verdict }

func (e *RefusalError) Error() string {
	return CodeDeadlineInfeasible + ": " + string(e.Verdict.Reason) +
		" estimate=" + e.Verdict.Estimate.String() + " remaining=" + e.Verdict.Remaining.String()
}

// Unwrap makes errors.Is(err, ErrDeadlineInfeasible) hold.
func (e *RefusalError) Unwrap() error { return ErrDeadlineInfeasible }

// Estimate predicts how long a new request with promptTokens of input and at
// most maxTokens of output (0 = unset) takes if admitted now. ok is false
// until a decode rate has been observed. The prefill term is added once a
// prefill rate has been observed; an observation with no time-to-first-token
// split (a buffered reply) is recorded as decode over the whole duration, so
// its rate already carries its prefill time.
//
// Model: the observed per-request rates already include the contention they
// were measured under (conc requests in flight). Admitting one more request
// makes inFlight+1 share the node, so each per-request rate is scaled by
// conc/(inFlight+1). This holds both for a batching engine (aggregate
// throughput roughly fixed, shared by more streams) and for a serial engine
// (a queue: the wait grows with the number ahead).
func (e *Estimator) Estimate(promptTokens, maxTokens int) (time.Duration, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.estimateLocked(promptTokens, maxTokens)
}

func (e *Estimator) estimateLocked(promptTokens, maxTokens int) (time.Duration, bool) {
	if !e.hasDec || e.decode <= 0 || e.samples < minRateSamples {
		return 0, false
	}
	decodeTok := e.complTok
	if decodeTok <= 0 {
		decodeTok = float64(e.cfg.DefaultDecodeTokens)
	}
	if maxTokens > 0 && float64(maxTokens) < decodeTok {
		decodeTok = float64(maxTokens)
	}
	if promptTokens < 0 {
		promptTokens = 0
	}
	base := decodeTok / e.decode
	if e.hasPre && e.prefill > 0 {
		base += float64(promptTokens) / e.prefill
	}
	conc := e.conc
	if conc < minEstimatorConcurrencyObs {
		conc = minEstimatorConcurrencyObs
	}
	scale := float64(e.inFlight+1) / conc
	if scale < 1 {
		// Never predict faster than the measured per-request rate: an idle
		// node is not faster than the fastest observation.
		scale = 1
	}
	secs := base * scale
	if !finite(secs) || secs > maxBudget.Seconds() {
		return maxBudget, true
	}
	return time.Duration(secs * float64(time.Second)), true
}

// Admit decides whether a request with the given remaining budget should be
// started. A request without a declared budget (hasBudget false) is always
// admitted. The check itself is a memory read; callers that admit should
// then call Begin.
func (e *Estimator) Admit(promptTokens, maxTokens int, remaining time.Duration, hasBudget bool) Verdict {
	if !hasBudget {
		return Verdict{Admit: true}
	}
	if remaining <= 0 {
		return Verdict{Reason: ReasonDeadlineExpired, Remaining: remaining, RetryAfter: time.Second}
	}
	e.mu.Lock()
	est, ok := e.estimateLocked(promptTokens, maxTokens)
	inFlight := e.inFlight
	e.mu.Unlock()
	v := Verdict{Admit: true, Estimate: est, Remaining: remaining, Measured: ok}
	if !ok {
		return v
	}
	if est.Seconds() <= e.cfg.Headroom*remaining.Seconds() {
		return v
	}
	v.Admit = false
	v.Reason = ReasonDeadlineInfeasible
	// One in-flight request's share of the estimate is the expected time
	// until a slot frees up.
	ra := est / time.Duration(inFlight+1)
	if ra < time.Second {
		ra = time.Second
	}
	if ra > e.cfg.RetryAfterMax {
		ra = e.cfg.RetryAfterMax
	}
	v.RetryAfter = ra
	return v
}

func finite(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) }

func durationFrom(v float64, unit time.Duration) time.Duration {
	f := v * float64(unit)
	if f > float64(maxBudget) {
		return maxBudget
	}
	return time.Duration(f)
}
