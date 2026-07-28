// Package loaddebounce is a clean-room Go primitive that publishes a per-worker
// load signal only when it CHANGES (last==current dedup) and coalesces a burst
// of rapid changes behind a short reset-on-every-change debounce window so only
// the LATEST value survives to be emitted.
//
// It is inspired by ai-dynamo's kv_router publisher worker_metrics.rs loop
// (borrow-and-update + a last==current dedup + a 1ms debounce whose deadline is
// reset on every change, over a watch channel). This is an idiomatic Go
// reimplementation, NOT a transliteration: the coalescing logic lives in a pure
// state machine (Coalescer) that the caller drives with an injected clock
// reading, so its behaviour is exercised deterministically without any goroutine,
// real timer, or wall-clock time.Sleep. That is what makes the witness test
// flake-free.
//
// LIVE landing site (wired — the primitive is no longer a shelf part):
//
//	cmd/fak/dispatch_tick_preflight.go
//	  dispatchPublishWorkerLoad wraps the dispatchProbeWorkerCount /
//	  PreflightInput.OSWorkerProcs seam. That probe RECOMPUTES load every tick
//	  (process-tree scan + log-tail classify + scraped-metric fold); admission now
//	  consumes the PUBLISHED value from a Publisher instead of the raw sample, so
//	  an unchanged sample moves nothing and a burst of changes inside the window
//	  settles to its latest member before it reaches the cap arithmetic.
//
// Structurally-closest analog:
//
//	internal/gateway/serving_autoscaler.go:64
//	  ServingSignalsFromMetricRows — folds normalized serving telemetry rows into
//	  the per-worker signal shape an autoscaler plans over.
package loaddebounce

import "time"

// DefaultDebounce mirrors the 1ms reset-on-change window from the source pattern.
// It is short by design: long enough to absorb a same-tick burst, short enough
// that a settled value is published almost immediately.
const DefaultDebounce = time.Millisecond

// Coalescer is a deterministic, single-value load-signal debouncer for one worker.
//
// It is a PURE state machine: the caller feeds it observed samples and a
// monotonic clock reading, and it reports when a coalesced value is ready to
// publish. It owns no goroutine, channel, or timer, so a test advances "time"
// simply by passing later timestamps — there is no real sleeping and no data
// race to reason about.
//
// Semantics (both are the DoD witnesses):
//
//   - Dedup: a sample equal to the last PUBLISHED value is dropped, and it also
//     cancels any in-flight burst that has drifted back to the published value
//     (the watch channel in the source coalesces to the latest value before its
//     last==current comparison — a Rust A->B->A within the window publishes
//     nothing, and so does this).
//   - Coalesce: a distinct sample (re)arms the debounce deadline to now+debounce;
//     a rapid burst of distinct values keeps pushing the deadline out, so only
//     the LAST value in the burst is what Emit returns once the window elapses.
//
// The zero value is not usable; construct with New.
type Coalescer[T comparable] struct {
	debounce time.Duration

	published     bool // whether any value has ever been emitted
	lastPublished T    // the most recently emitted value — the dedup reference

	pending    bool      // whether a change is waiting out the debounce window
	pendingVal T         // the latest observed value inside the current burst
	deadline   time.Time // when the current debounce window elapses
}

// New returns a Coalescer with the given reset-on-change debounce window. A
// non-positive window is treated as DefaultDebounce.
func New[T comparable](debounce time.Duration) *Coalescer[T] {
	if debounce <= 0 {
		debounce = DefaultDebounce
	}
	return &Coalescer[T]{debounce: debounce}
}

// Observe folds one sampled value observed at time now and reports whether a
// pending value is now armed behind the debounce window.
//
// If the sample equals the last published value it is a no-op that also cancels
// any in-flight burst (dedup / no-net-change): the primitive goes quiet and
// armed is false. Otherwise the sample becomes the pending value and the
// debounce deadline is reset to now+debounce (reset-on-every-change), so a burst
// of distinct values keeps only its final member.
func (c *Coalescer[T]) Observe(value T, now time.Time) (armed bool) {
	if c.published && value == c.lastPublished {
		// No net change from what we last published: drop it and collapse any
		// burst that has circled back to the published value.
		c.disarm()
		return false
	}
	if c.pending && value == c.pendingVal {
		// Already waiting out the window for this exact value: keep the existing
		// deadline rather than extending it, so a stuck-repeating sample cannot
		// starve emission forever.
		return true
	}
	c.pending = true
	c.pendingVal = value
	c.deadline = now.Add(c.debounce)
	return true
}

// Due reports whether a pending value's debounce window has elapsed at now.
func (c *Coalescer[T]) Due(now time.Time) bool {
	return c.pending && !now.Before(c.deadline)
}

// Deadline returns the current debounce deadline and whether one is armed. A live
// runner reads it to schedule (or reset) the real timer it waits on — this is the
// injectable-timer seam: the Coalescer decides WHEN, the runner owns the clock.
func (c *Coalescer[T]) Deadline() (deadline time.Time, armed bool) {
	return c.deadline, c.pending
}

// Emit publishes the pending value if its debounce window has elapsed at now. On
// emission it records the value as the new dedup reference, disarms, and returns
// (value, true); otherwise it returns the zero value and false.
func (c *Coalescer[T]) Emit(now time.Time) (T, bool) {
	if !c.Due(now) {
		var zero T
		return zero, false
	}
	v := c.pendingVal
	c.published = true
	c.lastPublished = v
	c.disarm()
	return v, true
}

// Prime records value as the published dedup reference WITHOUT waiting out a
// debounce window, and cancels any in-flight burst.
//
// It exists for the cold start of a POLLING caller: a debounce has nothing to
// debounce against until something has been published, so a caller that must hand
// a value to a consumer on its very first sample (the dispatch preflight must put
// some OSWorkerProcs in front of admission on tick 1) primes with that first
// sample instead of publishing a zero it never observed. Every LATER sample then
// goes through the normal Observe/Emit path, so the dedup and the coalescing
// window apply from the second sample on.
func (c *Coalescer[T]) Prime(value T) {
	c.published = true
	c.lastPublished = value
	c.disarm()
}

func (c *Coalescer[T]) disarm() {
	c.pending = false
	var zero T
	c.pendingVal = zero
}

// Publisher wraps a Coalescer with an injected now-func and an emit callback,
// giving the intended live seam a single Sample/Flush surface while keeping the
// clock fully injectable: production passes time.Now, tests pass a fake clock's
// reader so the debounce is exercised without any wall-clock sleep.
//
// The zero value is not usable; construct with NewPublisher.
type Publisher[T comparable] struct {
	core *Coalescer[T]
	now  func() time.Time
	emit func(T)
}

// NewPublisher builds a Publisher over a fresh Coalescer. now supplies the clock
// (nil defaults to time.Now); emit is invoked once per coalesced publication.
func NewPublisher[T comparable](debounce time.Duration, now func() time.Time, emit func(T)) *Publisher[T] {
	if now == nil {
		now = time.Now
	}
	return &Publisher[T]{core: New[T](debounce), now: now, emit: emit}
}

// Sample folds one observed value at the current clock reading and then publishes
// any value whose debounce window has already elapsed. It never emits the value
// it was just handed on the same call (the window has not passed yet); coalescing
// is what makes a same-tick burst collapse to one Flush later.
func (p *Publisher[T]) Sample(value T) {
	p.core.Observe(value, p.now())
	p.drain()
}

// Flush publishes the pending value if its debounce window has elapsed at the
// current clock reading. A live runner calls Flush from the select arm that wakes
// on the debounce deadline (see Deadline); a caller with only a sample stream can
// call it opportunistically.
func (p *Publisher[T]) Flush() {
	p.drain()
}

// Prime seeds the dedup reference with value at cold start and invokes emit once
// for it, so a polling caller's first sample reaches its consumer immediately
// (there is nothing to debounce against yet) while every later sample is gated by
// the normal dedup + coalescing window. See Coalescer.Prime.
func (p *Publisher[T]) Prime(value T) {
	p.core.Prime(value)
	p.emit(value)
}

// NextWake returns the debounce deadline and whether one is armed, so a live
// runner can (re)arm the real timer it blocks on.
func (p *Publisher[T]) NextWake() (deadline time.Time, armed bool) {
	return p.core.Deadline()
}

func (p *Publisher[T]) drain() {
	if v, ok := p.core.Emit(p.now()); ok {
		p.emit(v)
	}
}
