// Package turntaxmeter holds the observer-effect fence for the self-tax plane's
// per-turn meter (epic #1147, ticket T4 / issue #1156).
//
// The problem this package exists to solve is the literal observer effect: a meter
// fak puts on a hot path is not free, and a meter that records on EVERY event
// full-instruments that path — the 10–53% variable overhead the profiling
// literature warns about. The honesty contract in docs/standards/observer-effect.md
// states the fix in one line: the meter must SAMPLE (rate-bounded), never
// full-instrument, and its own per-event cost must stay under a declared cap proven
// by a green test. This package is that rate bound. The shipped decode
// AcceptanceMeter (internal/spec) already pins the PER-SAMPLE cost at zero
// allocations; Sampler is its companion — it bounds HOW OFTEN the meter samples, the
// fence the AcceptanceMeter's own doc note flags as the missing piece.
//
// Sampler is a 1-in-N admission gate. A per-turn meter calls ShouldSample once per
// hot-path event; it returns true on exactly every Nth call, so a meter that would
// record every event instead records a 1/N fraction of them. The decision is a
// deterministic atomic counter, not a random draw — so the realized rate is EXACT,
// not a statistical approximation that would only converge over many samples. That is
// what lets "the sampling rate is honored under load" be a witnessed equality (a test
// reads back exactly total/N admissions) rather than a noisy tolerance band.
package turntaxmeter

import "sync/atomic"

// Sampler is a lock-free 1-in-N admission gate for a hot-path meter. The zero value
// is NOT ready to use — construct one with NewSampler so the rate is explicit and
// validated. A Sampler is safe for concurrent use by many hot-path callers: the
// admission decision is a single atomic increment, so it adds no lock and allocates
// nothing per call (the observer-effect cost cap, witnessed in sampler_cost_test.go).
type Sampler struct {
	// n is the sampling period: ShouldSample admits one event in every n. n >= 1 is
	// an invariant established by NewSampler (n == 1 means "sample every event", the
	// full-instrument degenerate case a caller can still ask for explicitly).
	n uint64
	// count is the running event counter, advanced once per ShouldSample call. It is
	// touched only by atomic ops, so the gate needs no mutex.
	count atomic.Uint64
	// admitted tallies the events ShouldSample returned true for — the realized
	// sample count, readable via Admitted() without re-deriving it from count. It lets
	// the under-load witness read back EXACTLY ceil(seen/n) admissions, proving the
	// rate held under concurrency rather than only on a single-threaded run.
	admitted atomic.Uint64
}

// NewSampler returns a 1-in-n sampler: ShouldSample admits one event in every n. A
// non-positive n is clamped up to 1 (sample everything) rather than rejected, so a
// miscomputed rate degrades to the safe full-instrument behavior instead of dividing
// by zero or silently sampling nothing — a meter that records too much is a cost
// problem the cost cap catches, but a meter that records NOTHING is a correctness
// problem that would hide a regression, so the clamp fails toward observability.
func NewSampler(n int) *Sampler {
	if n < 1 {
		n = 1
	}
	return &Sampler{n: uint64(n)}
}

// Rate reports the sampling period n — the meter admits one event in every Rate().
// A Rate of 1 means the sampler full-instruments (every event admitted); larger
// values bound the meter's frequency proportionally.
func (s *Sampler) Rate() int { return int(s.n) }

// ShouldSample advances the event counter and reports whether THIS event is the one
// admitted in the current window of n. It returns true on exactly every nth call —
// the 1st, the (1+n)th, the (1+2n)th, … — so over any run of T calls it admits
// exactly ceil(T/n) events. The decision is one atomic add and one modulo: no lock,
// no allocation, no I/O, so the gate's own cost stays under the observer-effect cap
// on every hot-path call.
//
// The 1-in-n contract is the load-bearing property: it lets a caller wrap a
// full-instrument meter in a Sampler and KNOW the meter now fires 1/n as often, under
// any concurrency, without measuring — the rate is structural, not statistical.
// Admitting on the FIRST call (counter == 1) means a short-lived turn still produces
// one sample rather than none.
func (s *Sampler) ShouldSample() bool {
	// One atomic increment is the whole decision; the modulo selects the first event
	// of each n-window. With n == 1 every call admits (the full-instrument case).
	if s.count.Add(1)%s.n == 1%s.n {
		s.admitted.Add(1)
		return true
	}
	return false
}

// Seen reports the total events offered to ShouldSample so far.
func (s *Sampler) Seen() uint64 { return s.count.Load() }

// Admitted reports the events ShouldSample admitted so far — the realized sample
// count. After T calls of a 1-in-n sampler this is exactly ceil(T/n), the equality
// the under-load witness checks.
func (s *Sampler) Admitted() uint64 { return s.admitted.Load() }
