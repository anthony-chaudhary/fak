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
	for {
		select {
		case runErr := <-wait:
			return guardChildWaitEvent{Kind: guardChildCompleted, RunErr: runErr}
		case ev := <-restarts:
			return guardChildWaitEvent{Kind: guardChildRestart, Restart: ev}
		case now := <-ticks:
			if exhausted == nil {
				continue
			}
			if stop, reason := exhausted(now); stop {
				return guardChildWaitEvent{Kind: guardChildTimeBudget, Reason: reason}
			}
		}
	}
}
