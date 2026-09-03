package adjudicator

import (
	"errors"
	"fmt"
	"time"
)

// LifecycleState represents the current state in the lifecycle FSM.
type LifecycleState string

const (
	StateSubmitted   LifecycleState = "submitted"
	StateEvaluating  LifecycleState = "evaluating"
	StateAllowed     LifecycleState = "allowed"
	StateDenied      LifecycleState = "denied"
	StateQuarantined LifecycleState = "quarantined"
	StateExecuted    LifecycleState = "executed"
	StateFinished    LifecycleState = "finished"
)

// LifecycleEvent represents an event triggering a transition in the lifecycle FSM.
type LifecycleEvent string

const (
	EventEvaluate   LifecycleEvent = "evaluate"
	EventAllow      LifecycleEvent = "allow"
	EventDeny       LifecycleEvent = "deny"
	EventQuarantine LifecycleEvent = "quarantine"
	EventExecute    LifecycleEvent = "execute"
	EventFinish     LifecycleEvent = "finish"
)

// ErrInvalidTransition is returned when a requested state transition is invalid.
var ErrInvalidTransition = errors.New("lifecycle fsm: invalid state transition")

// TransitionRecord logs a completed transition in the lifecycle history.
type TransitionRecord struct {
	From  LifecycleState
	To    LifecycleState
	Event LifecycleEvent
	At    time.Time
}

type transitionKey struct {
	from  LifecycleState
	event LifecycleEvent
}

var validTransitions = map[transitionKey]LifecycleState{
	{from: StateSubmitted, event: EventEvaluate}:    StateEvaluating,
	{from: StateEvaluating, event: EventAllow}:      StateAllowed,
	{from: StateEvaluating, event: EventDeny}:       StateDenied,
	{from: StateEvaluating, event: EventQuarantine}: StateQuarantined,
	{from: StateAllowed, event: EventExecute}:       StateExecuted,
	{from: StateExecuted, event: EventFinish}:       StateFinished,
	{from: StateDenied, event: EventFinish}:         StateFinished,
	{from: StateQuarantined, event: EventFinish}:    StateFinished,
}

// LifecycleFSM tracks the state and transition history of a request or tool call.
type LifecycleFSM struct {
	State   LifecycleState
	History []TransitionRecord
}

// NewLifecycleFSM initializes an FSM in StateSubmitted.
func NewLifecycleFSM() *LifecycleFSM {
	return &LifecycleFSM{
		State: StateSubmitted,
	}
}

// Transition advances the FSM to the target state if the transition is valid.
// Returns an error wrapping ErrInvalidTransition if the transition is invalid.
func (f *LifecycleFSM) Transition(event LifecycleEvent) (LifecycleState, error) {
	next, ok := validTransitions[transitionKey{from: f.State, event: event}]
	if !ok {
		return "", fmt.Errorf("%w: cannot transition from %s via %s", ErrInvalidTransition, f.State, event)
	}
	f.History = append(f.History, TransitionRecord{
		From:  f.State,
		To:    next,
		Event: event,
		At:    time.Now(),
	})
	f.State = next
	return f.State, nil
}

// CanTransition returns whether the given event would succeed from the current state.
func (f *LifecycleFSM) CanTransition(event LifecycleEvent) bool {
	_, ok := validTransitions[transitionKey{from: f.State, event: event}]
	return ok
}

// IsTerminal returns true only for StateFinished.
func (f *LifecycleFSM) IsTerminal() bool {
	return f.State == StateFinished
}
