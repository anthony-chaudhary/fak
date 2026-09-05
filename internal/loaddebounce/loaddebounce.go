// Package loaddebounce provides a load-signal debouncer that publishes changes
// with dedup and burst coalescing over an injectable clock.
package loaddebounce

import "time"

// DefaultDebounce mirrors the reset-on-change window (1ms).
const DefaultDebounce = time.Millisecond

// Coalescer is a deterministic single-value load-signal debouncer for one worker.
// The zero value is not usable; construct with New.
type Coalescer[T comparable] struct {
	debounce time.Duration

	published     bool
	lastPublished T

	pending    bool
	pendingVal T
	deadline   time.Time
}

// New returns a Coalescer with the given reset-on-change debounce window.
func New[T comparable](debounce time.Duration) *Coalescer[T] {
	if debounce <= 0 {
		debounce = DefaultDebounce
	}
	return &Coalescer[T]{debounce: debounce}
}

// Observe folds one sampled value observed at time now and reports if armed.
func (c *Coalescer[T]) Observe(value T, now time.Time) (armed bool) {
	if c.published && value == c.lastPublished {
		c.disarm()
		return false
	}
	if c.pending && value == c.pendingVal {
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

// Deadline returns the current debounce deadline and whether one is armed.
func (c *Coalescer[T]) Deadline() (deadline time.Time, armed bool) {
	return c.deadline, c.pending
}

// Emit publishes the pending value if its debounce window has elapsed at now.
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

// Prime records value as the published dedup reference without waiting.
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

// Publisher wraps a Coalescer with an injected clock function and emit callback.
type Publisher[T comparable] struct {
	core *Coalescer[T]
	now  func() time.Time
	emit func(T)
}

// NewPublisher builds a Publisher over a fresh Coalescer.
func NewPublisher[T comparable](debounce time.Duration, now func() time.Time, emit func(T)) *Publisher[T] {
	if now == nil {
		now = time.Now
	}
	return &Publisher[T]{core: New[T](debounce), now: now, emit: emit}
}

// Sample folds one observed value at current clock reading and publishes if ready.
func (p *Publisher[T]) Sample(value T) {
	p.core.Observe(value, p.now())
	p.drain()
}

// Flush publishes the pending value if its debounce window has elapsed.
func (p *Publisher[T]) Flush() {
	p.drain()
}

// Prime seeds the dedup reference with value at cold start and invokes emit once.
func (p *Publisher[T]) Prime(value T) {
	p.core.Prime(value)
	p.emit(value)
}

// NextWake returns the debounce deadline and whether one is armed.
func (p *Publisher[T]) NextWake() (deadline time.Time, armed bool) {
	return p.core.Deadline()
}

func (p *Publisher[T]) drain() {
	if v, ok := p.core.Emit(p.now()); ok {
		p.emit(v)
	}
}
