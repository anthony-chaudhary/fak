package main

import "time"

type guardChildWaitKind uint8

const (
	guardChildCompleted guardChildWaitKind = iota + 1
	guardChildRestart
	guardChildTimeBudget
)

type guardChildWaitEvent struct {
	Kind    guardChildWaitKind
	RunErr  error
	Restart guardBudgetRestartEvent
	Reason  string
}

// waitGuardChild keeps polling one already-started child. A non-terminal wall-clock tick is
// deliberately consumed here instead of returning to the caller's launch loop; returning on
// every tick used to spawn a second concurrent child every 15 seconds while the first was
// still running.
func waitGuardChild(wait <-chan error, restarts <-chan guardBudgetRestartEvent, ticks <-chan time.Time, exhausted func(time.Time) (bool, string)) guardChildWaitEvent {
	// handleTick returns the terminal budget event (done=true) when a wall-clock tick
	// exhausts the budget, or done=false for a non-terminal tick that is consumed in place.
	handleTick := func(now time.Time) (guardChildWaitEvent, bool) {
		if exhausted == nil {
			return guardChildWaitEvent{}, false
		}
		if stop, reason := exhausted(now); stop {
			return guardChildWaitEvent{Kind: guardChildTimeBudget, Reason: reason}, true
		}
		return guardChildWaitEvent{}, false
	}
	for {
		// Drain a ready tick first. A non-terminal tick MUST be consumed here; a plain
		// select would let a simultaneously-ready child-exit win at random, dropping the
		// pending tick and returning to the launch loop — which used to spawn a second
		// concurrent child every 15s while the first still ran (see the doc above).
		select {
		case now := <-ticks:
			if ev, done := handleTick(now); done {
				return ev
			}
			continue
		default:
		}
		select {
		case runErr := <-wait:
			return guardChildWaitEvent{Kind: guardChildCompleted, RunErr: runErr}
		case ev := <-restarts:
			return guardChildWaitEvent{Kind: guardChildRestart, Restart: ev}
		case now := <-ticks:
			if ev, done := handleTick(now); done {
				return ev
			}
		}
	}
}
