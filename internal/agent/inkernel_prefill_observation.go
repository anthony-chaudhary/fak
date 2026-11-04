package agent

import (
	"context"
	"strings"
)

type ownedPrefillObservationKey struct{}

type ownedPrefillObservation struct {
	planner *InKernelPlanner
	start   func(promptTokens, cachedTokens int) func(completed bool)
}

// WithOwnedPrefillObservation observes this planner's ordinary CPU or CUDA text
// attempts after owned prefix acquisition and deadline admission. The callback
// receives exact usable persistent-cache counts, not estimates or fleet-wide
// candidate evidence. It returns an optional completion callback, called once:
// true when prefill finishes (including an exact hit), false when the attempt
// unwinds first. Completion occurs before decode, not at first emitted text.
//
// Each retry starts a separate observation. No observation is emitted before
// the owned-state seam, for another planner, or for unsupported routes: Metal,
// non-CUDA devices, speculative/MTP, multimodal, cache population and cold CPU
// requests with prefix reuse enabled (which may later enter flight coalescing).
// CPU requests without reuse and already-owned warm CPU prefixes are supported.
// Absence MUST NOT be interpreted as known zero work.
//
// Callbacks run synchronously on the execution goroutine and must not panic,
// block or reenter the planner. A shared observer must synchronize its own state.
// This API is observation only: it does not reserve fleet load or alter routing.
func (p *InKernelPlanner) WithOwnedPrefillObservation(ctx context.Context, start func(int, int) func(bool)) context.Context {
	return context.WithValue(ctx, ownedPrefillObservationKey{}, ownedPrefillObservation{p, start})
}

func (p *InKernelPlanner) beginOwnedPrefillObservation(ctx context.Context, prompt, cached int) func(bool) {
	o, ok := ctx.Value(ownedPrefillObservationKey{}).(ownedPrefillObservation)
	if !ok || o.planner != p || o.start == nil || p == nil || p.metal || !p.ExecutionDeadlineAdmissionSupported() {
		return nil
	}
	if p.backend != nil && !strings.EqualFold(strings.TrimSpace(p.backend.Name()), "cuda") {
		return nil
	}
	if prompt < 0 || cached < 0 || cached > prompt {
		return nil
	}
	finish := o.start(prompt, cached)
	if finish == nil {
		return nil
	}
	done := false
	return func(completed bool) {
		if !done {
			done = true
			finish(completed)
		}
	}
}

// Cold reusable CPU requests may acquire flight reuse after this seam. Decline
// regardless of current GOMAXPROCS, which can change before the flight branch.
func ownedPrefillRouteEligible(vision, populate, cpu, reuse bool, matched int) bool {
	return !vision && !populate && !(cpu && reuse && matched == 0)
}

// TryWithOwnedPrefillObservation installs an observer only when this context
// has no existing binding. Automatic instrumentation must not replace a caller's
// observer, including one bound to a different planner. A false result leaves
// ctx unchanged; callers must treat omitted observations as unknown, not zero.
func (p *InKernelPlanner) TryWithOwnedPrefillObservation(ctx context.Context, start func(int, int) func(bool)) (context.Context, bool) {
	if p == nil || start == nil || ctx.Value(ownedPrefillObservationKey{}) != nil {
		return ctx, false
	}
	return p.WithOwnedPrefillObservation(ctx, start), true
}
