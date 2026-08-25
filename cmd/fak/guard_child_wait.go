package main

import "time"

type guardChildWaitKind uint8

const (
	guardChildCompleted guardChildWaitKind = iota + 1
	guardChildRestart
	guardChildTimeBudget
	guardChildResourceLimit
)

type guardChildWaitEvent struct {
	Kind     guardChildWaitKind
	RunErr   error
	Restart  guardBudgetRestartEvent
	Reason   string
	Resource *guardResourceDecision
}

func waitGuardChild(wait <-chan error, restarts <-chan guardBudgetRestartEvent, ticks <-chan time.Time, exhausted func(time.Time) (bool, string), resources ...<-chan guardChildWaitEvent) guardChildWaitEvent {
	var resource <-chan guardChildWaitEvent
	if len(resources) > 0 {
		resource = resources[0]
	}
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
		select {
		case ev := <-resource:
			return ev
		case now := <-ticks:
			if ev, done := handleTick(now); done {
				return ev
			}
			continue
		default:
		}
		select {
		case ev := <-resource:
			return ev
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
