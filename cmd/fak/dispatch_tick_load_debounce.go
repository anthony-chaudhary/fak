package main

// dispatch_tick_load_debounce.go — the per-worker load publish seam for the dispatch tick
// (#3376). It lives beside dispatch_tick_preflight.go rather than inside it because the
// preflight file is already at the god-file ceiling: the publish state, its clock, its env
// knob and its fold are one self-contained concern with one exported-to-the-tick entry point
// (dispatchPublishWorkerLoad), which is exactly the seam the split gate asks for.

import (
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loaddebounce"
)

// Change-gated + debounced per-worker load publish (#3376, parent #3365, scale-to-100
// #1333). dispatchProbeWorkerCount RECOMPUTES load every tick -- lease pidfile scan +
// goal breadcrumbs + a full Win32_Process cmdline classify -- and every tick handed its
// raw answer straight to admission, so a single-tick blip in that scan (a spawning child
// that never leases, a scan racing a teardown, an ambient codex session flickering in and
// out of the marker match) moved the cap arithmetic immediately and symmetrically.
//
// The probe still runs every tick (you cannot know a value changed without sampling it),
// but what admission CONSUMES is now the published value of a loaddebounce.Publisher:
//
//   - dedup: a sample equal to the last published load publishes nothing at all, so a
//     steady fleet holds one load value across arbitrarily many ticks;
//   - coalesce: a CHANGED sample arms a reset-on-every-change window and only reaches
//     admission once that window elapses with the value still standing, so a burst of
//     samples inside one window settles to its LAST member and an A->B->A round trip
//     publishes nothing.
//
// With the default sub-millisecond window and a multi-second tick cadence that is a
// one-tick confirmation: a change takes effect on the next tick that still observes it.
// The lag cuts both ways -- a blip UP no longer suppresses admits, a blip DOWN no longer
// releases them -- and a lagging load is bounded on the conservative side anyway, because
// EvaluatePreflight takes live = max(kernel alive, OSWorkerProcs) and the kernel lease
// count is not debounced. Operators who want a wider confirmation window (a flappy host,
// a fast tick loop) widen it with dispatchWorkerLoadDebounceEnv.
const dispatchWorkerLoadDebounceEnv = "FAK_DISPATCH_LOAD_DEBOUNCE_MS"

// dispatchWorkerLoadNow is the clock the coalescing window is measured on. It is a var so
// a test can hand-crank the window instead of sleeping through a real one.
var dispatchWorkerLoadNow = time.Now

// dispatchWorkerLoadSignal is the per-process publish state for the worker-load signal:
// the coalescer, the value it last published to admission, and the raw sample behind it.
type dispatchWorkerLoadSignal struct {
	mu         sync.Mutex
	pub        *loaddebounce.Publisher[int]
	lastSample int // the newest RAW probe answer, published or not
	published  int // the value admission consumed on the last tick
	publishes  int // how many times a value has reached admission (the dedup witness)
	samples    int // how many raw probes have been folded in
}

var dispatchWorkerLoad = &dispatchWorkerLoadSignal{}

// dispatchWorkerLoadDebounce resolves the reset-on-change window from
// FAK_DISPATCH_LOAD_DEBOUNCE_MS (a positive integer count of milliseconds), falling back
// to loaddebounce.DefaultDebounce on empty/zero/negative/unparseable input.
func dispatchWorkerLoadDebounce() time.Duration {
	if ms, err := strconv.Atoi(strings.TrimSpace(os.Getenv(dispatchWorkerLoadDebounceEnv))); err == nil && ms > 0 {
		return time.Duration(ms) * time.Millisecond
	}
	return loaddebounce.DefaultDebounce
}

// publish folds one raw worker-count sample through the change gate plus the coalescing
// window and returns the load admission should consume on THIS tick -- the newly settled
// value if the window has elapsed, otherwise the value still standing from before.
func (s *dispatchWorkerLoadSignal) publish(sample int) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.samples++
	s.lastSample = sample
	if s.pub == nil {
		// Cold start: with nothing published there is nothing to debounce against, and
		// admission still needs a load for this tick -- prime with the first sample
		// rather than hand the cap arithmetic a zero nobody observed.
		s.pub = loaddebounce.NewPublisher[int](dispatchWorkerLoadDebounce(),
			func() time.Time { return dispatchWorkerLoadNow() }, s.onPublish)
		s.pub.Prime(sample)
		return s.published
	}
	s.pub.Sample(sample)
	return s.published
}

// onPublish is the Publisher's emit sink; it runs under the caller's lock.
func (s *dispatchWorkerLoadSignal) onPublish(v int) {
	s.published = v
	s.publishes++
}

// pending reports the newest raw sample and whether it currently disagrees with the
// published load -- i.e. whether a change is still waiting out its window.
func (s *dispatchWorkerLoadSignal) pending() (int, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastSample, s.samples > 0 && s.lastSample != s.published
}

// dispatchPublishWorkerLoad probes the live worker count and returns the debounced,
// change-gated load for this tick. This is the live seam: PreflightInput.OSWorkerProcs.
func dispatchPublishWorkerLoad(root, product string) int {
	return dispatchWorkerLoad.publish(dispatchProbeWorkerCount(root, product))
}
