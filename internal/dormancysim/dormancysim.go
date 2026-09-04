// Package dormancysim is the deterministic time-travel harness for fak's dormancy
// organs (epic #1178, Phase 3, #1192): the acceptance substrate every other child is
// exercised through. It threads one injectable clock seam — the `now func() time.Time`
// the issue names — through the already-pure measure and rehydrate leaves and lets a
// single test fast-forward an agent's dormancy hours -> months with ZERO real waits and
// no wall-clock read, driving every rehydration rung across each horizon band.
//
// # Why a harness and not new organs
//
// The time organs are already injectable, they just had no shared driver:
//
//   - internal/dormancy is PURE — Stamp.HorizonAt(now time.Time) buckets the elapsed
//     gap onto {warm, cool, cold, frozen, ancient} with the caller's `now`, reading no
//     clock of its own.
//   - internal/rehydrate is PURE — Gate.Admit(ctx, horizon) runs exactly the staged
//     rungs the band reaches, in ladder order, recording each in Admission.Ran.
//   - internal/bgloop is already fast-forwardable deterministically under
//     testing/synctest (its loops tick across a simulated window with no real sleep).
//
// What was missing is the single clock those seams share and the harness that advances
// it across a day/week/month gap while driving the rungs. This package is that clock and
// that harness — it re-implements NONE of the dormancy clock, the horizon bucketer, or
// any rehydration rung (those are #1179 / #1181-#1186, already on main); it injects a
// clock into them and asserts them.
//
// # Honest fences
//
//   - Pure and stdlib-plus-two-internal. Imports only internal/dormancy(1),
//     internal/rehydrate(1), and stdlib. Reads no wall clock: the virtual Clock's only
//     time source is Advance/Set, which is what makes a 90-day simulation take
//     microseconds and stay byte-for-byte deterministic.
//   - Monotonic rigor. A longer gap reaches a colder band and therefore runs at least as
//     many rungs as a shorter one (the epic's monotonic-in-the-gap rule); the harness
//     surfaces the ordered per-rung witness so a test can assert exactly that.
//   - Drives, never fakes. Advance moves the real Clock and calls the real Gate.Admit at
//     the real dormancy.Horizon — the admission it returns is the same verdict the
//     production composition produces, only at a simulated instant.
//
// Invariant: dormancy simulation is fail-closed and monotonic. A longer gap never runs fewer rungs than a shorter one, and uninitialized sessions default conservatively to Ancient.
// Guard: non-positive advance durations are ignored to guarantee virtual time never regresses.
package dormancysim

import (
	"context"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

// Clock is the injectable clock seam — the `now func() time.Time` the dormancy clock,
// bgloop, and the rehydrate leaves consume — realized as a mutable virtual clock a test
// fast-forwards hours -> months. It reads NO wall clock: Advance and Set are its only
// time source, so a simulation is deterministic and takes no real time. Bind an organ's
// `now` to NowFunc to drive it from this one clock.
type Clock struct{ t time.Time }

// NewClock returns a Clock reading start.
func NewClock(start time.Time) *Clock { return &Clock{t: start} }

// Now returns the current virtual instant.
func (c *Clock) Now() time.Time { return c.t }

// Advance fast-forwards the clock by d and returns the new instant. A non-positive d is
// ignored (the clock never runs backwards) — the same monotonic discipline dormancy.Stamp
// applies to its durable mark.
func (c *Clock) Advance(d time.Duration) time.Time {
	if d > 0 {
		c.t = c.t.Add(d)
	}
	return c.t
}

// Set moves the clock to t (a jump to an absolute instant, e.g. to place the gap exactly
// on a band boundary). Unlike Advance it may move to any instant.
func (c *Clock) Set(t time.Time) { c.t = t }

// NowFunc returns the `now func() time.Time` seam bound to this Clock — the value an organ
// that takes an injectable clock is wired to so the harness drives one clock, not several.
// The returned func reads the live instant at call time (it tracks later Advance/Set).
func (c *Clock) NowFunc() func() time.Time { return c.Now }

// Simulator drives a staged rehydrate.Gate across a simulated dormancy: it holds a virtual
// Clock, the LastActiveAt Stamp of the dormant session/loop/lease, and the Gate. Advance
// fast-forwards the clock and runs the gate at the dormancy band the elapsed gap now falls
// in — the deterministic substitute for waiting out a real day/week/month. It is the
// acceptance substrate: a sibling (#1188 durable wake, #1191 wake-on-event) constructs one,
// advances it across a horizon, and asserts the ordered rung set that fired.
type Simulator struct {
	clock *Clock
	stamp dormancy.Stamp
	gate  *rehydrate.Gate
}

// New builds a Simulator whose clock starts at start, whose dormant session was last active
// at stamp, and whose re-entry gate is gate. Advance the returned Simulator to simulate the
// session sleeping and waking at a later instant.
func New(start time.Time, stamp dormancy.Stamp, gate *rehydrate.Gate) *Simulator {
	return &Simulator{clock: NewClock(start), stamp: stamp, gate: gate}
}

// Now is the current simulated instant.
func (s *Simulator) Now() time.Time { return s.clock.Now() }

// Clock exposes the underlying virtual clock so a caller can bind other organs' `now`
// seam to it (Clock().NowFunc()) and drive them from the same simulated time.
func (s *Simulator) Clock() *Clock { return s.clock }

// Horizon is the dormancy band the elapsed gap (now - stamp) currently falls in. A
// never-stamped session buckets to Ancient (revalidate everything), per dormancy's
// conservative-on-unknown rule.
func (s *Simulator) Horizon() dormancy.Horizon { return s.stamp.HorizonAt(s.clock.Now()) }

// Advance fast-forwards the virtual clock by d and runs the staged gate at the new horizon,
// returning the admission — the ordered per-rung witness of which rehydration rungs fired,
// in ladder order. No real time passes. This is the harness's core move: "the session slept
// d longer, wake it and see which rungs it must now clear."
func (s *Simulator) Advance(ctx context.Context, d time.Duration) rehydrate.Admission {
	s.clock.Advance(d)
	return s.gate.Admit(ctx, s.Horizon())
}

// AdmitNow runs the gate at the current horizon WITHOUT advancing the clock — a re-entry
// at the present simulated instant (e.g. immediately after a simulated restart that
// reconstructed the Simulator from a persisted Stamp).
func (s *Simulator) AdmitNow(ctx context.Context) rehydrate.Admission {
	return s.gate.Admit(ctx, s.Horizon())
}
