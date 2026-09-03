package adjudicator

import (
	"errors"
	"testing"
	"time"
)

func TestLifecycleFSM(t *testing.T) {
	t.Run("HappyPaths", func(t *testing.T) {
		t.Run("Allowed", func(t *testing.T) {
			fsm := NewLifecycleFSM()
			if fsm.State != StateSubmitted {
				t.Fatalf("initial state = %q, want %q", fsm.State, StateSubmitted)
			}
			if fsm.IsTerminal() {
				t.Fatalf("expected non-terminal initial state")
			}

			if !fsm.CanTransition(EventEvaluate) {
				t.Fatalf("expected CanTransition(EventEvaluate) == true")
			}
			res, err := fsm.Transition(EventEvaluate)
			if err != nil {
				t.Fatalf("unexpected error on evaluate: %v", err)
			}
			if res != StateEvaluating || fsm.State != StateEvaluating {
				t.Fatalf("state = %q, want %q", fsm.State, StateEvaluating)
			}
			if fsm.IsTerminal() {
				t.Fatalf("expected non-terminal state evaluating")
			}

			if !fsm.CanTransition(EventAllow) {
				t.Fatalf("expected CanTransition(EventAllow) == true")
			}
			res, err = fsm.Transition(EventAllow)
			if err != nil {
				t.Fatalf("unexpected error on allow: %v", err)
			}
			if res != StateAllowed || fsm.State != StateAllowed {
				t.Fatalf("state = %q, want %q", fsm.State, StateAllowed)
			}
			if fsm.IsTerminal() {
				t.Fatalf("expected non-terminal state allowed")
			}

			if !fsm.CanTransition(EventExecute) {
				t.Fatalf("expected CanTransition(EventExecute) == true")
			}
			res, err = fsm.Transition(EventExecute)
			if err != nil {
				t.Fatalf("unexpected error on execute: %v", err)
			}
			if res != StateExecuted || fsm.State != StateExecuted {
				t.Fatalf("state = %q, want %q", fsm.State, StateExecuted)
			}
			if fsm.IsTerminal() {
				t.Fatalf("expected non-terminal state executed")
			}

			if !fsm.CanTransition(EventFinish) {
				t.Fatalf("expected CanTransition(EventFinish) == true")
			}
			res, err = fsm.Transition(EventFinish)
			if err != nil {
				t.Fatalf("unexpected error on finish: %v", err)
			}
			if res != StateFinished || fsm.State != StateFinished {
				t.Fatalf("state = %q, want %q", fsm.State, StateFinished)
			}
			if !fsm.IsTerminal() {
				t.Fatalf("expected IsTerminal() == true for finished")
			}
		})

		t.Run("Denied", func(t *testing.T) {
			fsm := NewLifecycleFSM()
			if _, err := fsm.Transition(EventEvaluate); err != nil {
				t.Fatalf("unexpected error on evaluate: %v", err)
			}
			if !fsm.CanTransition(EventDeny) {
				t.Fatalf("expected CanTransition(EventDeny) == true")
			}
			res, err := fsm.Transition(EventDeny)
			if err != nil {
				t.Fatalf("unexpected error on deny: %v", err)
			}
			if res != StateDenied || fsm.State != StateDenied {
				t.Fatalf("state = %q, want %q", fsm.State, StateDenied)
			}
			if fsm.IsTerminal() {
				t.Fatalf("expected non-terminal state denied")
			}

			if !fsm.CanTransition(EventFinish) {
				t.Fatalf("expected CanTransition(EventFinish) == true")
			}
			res, err = fsm.Transition(EventFinish)
			if err != nil {
				t.Fatalf("unexpected error on finish: %v", err)
			}
			if res != StateFinished || fsm.State != StateFinished {
				t.Fatalf("state = %q, want %q", fsm.State, StateFinished)
			}
			if !fsm.IsTerminal() {
				t.Fatalf("expected IsTerminal() == true for finished")
			}
		})

		t.Run("Quarantined", func(t *testing.T) {
			fsm := NewLifecycleFSM()
			if _, err := fsm.Transition(EventEvaluate); err != nil {
				t.Fatalf("unexpected error on evaluate: %v", err)
			}
			if !fsm.CanTransition(EventQuarantine) {
				t.Fatalf("expected CanTransition(EventQuarantine) == true")
			}
			res, err := fsm.Transition(EventQuarantine)
			if err != nil {
				t.Fatalf("unexpected error on quarantine: %v", err)
			}
			if res != StateQuarantined || fsm.State != StateQuarantined {
				t.Fatalf("state = %q, want %q", fsm.State, StateQuarantined)
			}
			if fsm.IsTerminal() {
				t.Fatalf("expected non-terminal state quarantined")
			}

			if !fsm.CanTransition(EventFinish) {
				t.Fatalf("expected CanTransition(EventFinish) == true")
			}
			res, err = fsm.Transition(EventFinish)
			if err != nil {
				t.Fatalf("unexpected error on finish: %v", err)
			}
			if res != StateFinished || fsm.State != StateFinished {
				t.Fatalf("state = %q, want %q", fsm.State, StateFinished)
			}
			if !fsm.IsTerminal() {
				t.Fatalf("expected IsTerminal() == true for finished")
			}
		})
	})

	t.Run("ExhaustiveInvalidMatrix", func(t *testing.T) {
		allStates := []LifecycleState{
			StateSubmitted,
			StateEvaluating,
			StateAllowed,
			StateDenied,
			StateQuarantined,
			StateExecuted,
			StateFinished,
		}
		allEvents := []LifecycleEvent{
			EventEvaluate,
			EventAllow,
			EventDeny,
			EventQuarantine,
			EventExecute,
			EventFinish,
		}

		type validStep struct {
			state LifecycleState
			event LifecycleEvent
		}
		validSet := map[validStep]bool{
			{state: StateSubmitted, event: EventEvaluate}:    true,
			{state: StateEvaluating, event: EventAllow}:      true,
			{state: StateEvaluating, event: EventDeny}:       true,
			{state: StateEvaluating, event: EventQuarantine}: true,
			{state: StateAllowed, event: EventExecute}:       true,
			{state: StateExecuted, event: EventFinish}:       true,
			{state: StateDenied, event: EventFinish}:         true,
			{state: StateQuarantined, event: EventFinish}:    true,
		}

		for _, st := range allStates {
			for _, ev := range allEvents {
				if validSet[validStep{state: st, event: ev}] {
					continue
				}
				name := string(st) + "_" + string(ev)
				t.Run(name, func(t *testing.T) {
					fsm := &LifecycleFSM{State: st}
					if fsm.CanTransition(ev) {
						t.Errorf("CanTransition(%s) from state %s should be false", ev, st)
					}
					got, err := fsm.Transition(ev)
					if err == nil {
						t.Fatalf("expected error transitioning from %s via %s, got nil", st, ev)
					}
					if !errors.Is(err, ErrInvalidTransition) {
						t.Errorf("expected ErrInvalidTransition, got %v", err)
					}
					if got != "" {
						t.Errorf("expected empty state on invalid transition, got %q", got)
					}
					if fsm.State != st {
						t.Errorf("state mutated on invalid transition: got %s, want %s", fsm.State, st)
					}
					if len(fsm.History) != 0 {
						t.Errorf("history mutated on invalid transition: %v", fsm.History)
					}
				})
			}
		}
	})

	t.Run("CanTransition", func(t *testing.T) {
		allStates := []LifecycleState{
			StateSubmitted,
			StateEvaluating,
			StateAllowed,
			StateDenied,
			StateQuarantined,
			StateExecuted,
			StateFinished,
		}
		allEvents := []LifecycleEvent{
			EventEvaluate,
			EventAllow,
			EventDeny,
			EventQuarantine,
			EventExecute,
			EventFinish,
		}

		type validStep struct {
			state LifecycleState
			event LifecycleEvent
		}
		validSet := map[validStep]bool{
			{state: StateSubmitted, event: EventEvaluate}:    true,
			{state: StateEvaluating, event: EventAllow}:      true,
			{state: StateEvaluating, event: EventDeny}:       true,
			{state: StateEvaluating, event: EventQuarantine}: true,
			{state: StateAllowed, event: EventExecute}:       true,
			{state: StateExecuted, event: EventFinish}:       true,
			{state: StateDenied, event: EventFinish}:         true,
			{state: StateQuarantined, event: EventFinish}:    true,
		}

		for _, st := range allStates {
			for _, ev := range allEvents {
				fsm := &LifecycleFSM{State: st}
				expected := validSet[validStep{state: st, event: ev}]
				got := fsm.CanTransition(ev)
				if got != expected {
					t.Errorf("CanTransition(%s) at state %s = %v, want %v", ev, st, got, expected)
				}
				if fsm.State != st {
					t.Errorf("CanTransition mutated state: got %s, want %s", fsm.State, st)
				}
				if len(fsm.History) != 0 {
					t.Errorf("CanTransition mutated history: %v", fsm.History)
				}
			}
		}
	})

	t.Run("IsTerminal", func(t *testing.T) {
		allStates := []LifecycleState{
			StateSubmitted,
			StateEvaluating,
			StateAllowed,
			StateDenied,
			StateQuarantined,
			StateExecuted,
			StateFinished,
		}

		for _, st := range allStates {
			fsm := &LifecycleFSM{State: st}
			expected := (st == StateFinished)
			if got := fsm.IsTerminal(); got != expected {
				t.Errorf("IsTerminal() for state %s = %v, want %v", st, got, expected)
			}
		}
	})

	t.Run("HistoryRecording", func(t *testing.T) {
		fsm := NewLifecycleFSM()
		startTime := time.Now()

		steps := []struct {
			event LifecycleEvent
			from  LifecycleState
			to    LifecycleState
		}{
			{event: EventEvaluate, from: StateSubmitted, to: StateEvaluating},
			{event: EventAllow, from: StateEvaluating, to: StateAllowed},
			{event: EventExecute, from: StateAllowed, to: StateExecuted},
			{event: EventFinish, from: StateExecuted, to: StateFinished},
		}

		for _, s := range steps {
			if _, err := fsm.Transition(s.event); err != nil {
				t.Fatalf("unexpected error during transition %s: %v", s.event, err)
			}
		}

		if len(fsm.History) != len(steps) {
			t.Fatalf("history length = %d, want %d", len(fsm.History), len(steps))
		}

		prevTime := startTime
		for idx, rec := range fsm.History {
			step := steps[idx]
			if rec.From != step.from {
				t.Errorf("record %d: From = %q, want %q", idx, rec.From, step.from)
			}
			if rec.To != step.to {
				t.Errorf("record %d: To = %q, want %q", idx, rec.To, step.to)
			}
			if rec.Event != step.event {
				t.Errorf("record %d: Event = %q, want %q", idx, rec.Event, step.event)
			}
			if rec.At.IsZero() {
				t.Errorf("record %d: At is zero", idx)
			}
			if rec.At.Before(prevTime) {
				t.Errorf("record %d: At (%v) before previous (%v)", idx, rec.At, prevTime)
			}
			prevTime = rec.At
		}
	})
}
