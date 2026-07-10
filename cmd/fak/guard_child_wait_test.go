package main

import (
	"errors"
	"testing"
	"time"
)

func TestWaitGuardChildDoesNotReturnOnNonterminalBudgetTick(t *testing.T) {
	wait := make(chan error, 1)
	restarts := make(chan guardBudgetRestartEvent, 1)
	ticks := make(chan time.Time, 1)
	ticks <- time.Unix(1, 0)
	wantErr := errors.New("child finished")
	wait <- wantErr
	checks := 0
	ev := waitGuardChild(wait, restarts, ticks, func(time.Time) (bool, string) {
		checks++
		return false, ""
	})
	if checks != 1 || ev.Kind != guardChildCompleted || !errors.Is(ev.RunErr, wantErr) {
		t.Fatalf("event=%+v checks=%d, want the same child completion after one nonterminal tick", ev, checks)
	}
}

func TestWaitGuardChildReturnsTerminalBudgetTick(t *testing.T) {
	ticks := make(chan time.Time, 1)
	ticks <- time.Unix(2, 0)
	ev := waitGuardChild(make(chan error), make(chan guardBudgetRestartEvent), ticks, func(time.Time) (bool, string) {
		return true, "TIME_BUDGET_EXHAUSTED"
	})
	if ev.Kind != guardChildTimeBudget || ev.Reason != "TIME_BUDGET_EXHAUSTED" {
		t.Fatalf("event=%+v", ev)
	}
}
