package session

import (
	"strings"
	"time"
)

// ProviderSessionBoundarySchema is the wire token for a provider-originated
// conversation boundary. It is deliberately distinct from ResetTransactionSchema:
// /clear starts a new user-visible conversation, while Recontinue preserves one
// logical conversation across an internal context-window reset.
const ProviderSessionBoundarySchema = "fak.session.provider_boundary.v1"

// ProviderSessionBoundary records why a fresh fak trace replaced the previous one.
// ProviderSessionID is the provider's new conversation/thread id, not transcript
// content or a credential.
type ProviderSessionBoundary struct {
	Schema            string `json:"schema,omitempty"`
	Provider          string `json:"provider,omitempty"`
	Source            string `json:"source,omitempty"`
	PreviousTrace     string `json:"previous_trace,omitempty"`
	ProviderSessionID string `json:"provider_session_id,omitempty"`
}

// IsZero supports json omitzero on State and Descriptor.
func (b ProviderSessionBoundary) IsZero() bool {
	return b.Schema == "" && b.Provider == "" && b.Source == "" &&
		b.PreviousTrace == "" && b.ProviderSessionID == ""
}

func normalizeProviderSessionBoundary(b ProviderSessionBoundary, previous string) ProviderSessionBoundary {
	b.Schema = ProviderSessionBoundarySchema
	b.Provider = strings.ToLower(strings.TrimSpace(b.Provider))
	b.Source = strings.ToLower(strings.TrimSpace(b.Source))
	b.PreviousTrace = strings.TrimSpace(previous)
	b.ProviderSessionID = strings.TrimSpace(b.ProviderSessionID)
	return b
}

// BeginProviderSessionAt closes parent and creates child for a provider /clear or
// /new event. The context window is fresh, so its context-token axis is re-armed to
// the configured cap. Cumulative envelopes carry their remaining values: turns,
// output, queries, spend, tool calls, wall clock, and throughput cannot be reset by
// clicking a provider UI command. Conversation-local roots, assumptions, cache
// affinity, pending-turn state, and cost history start empty on the child.
//
// The child keeps Priority/Pace and QualityEnvelope because those are guard/operator
// policy, not conversation content. A duplicate event for the same child is
// idempotent and returns applied=false.
func (t *Table) BeginProviderSessionAt(parent, child string, boundary ProviderSessionBoundary, now time.Time) (State, bool) {
	parent = strings.TrimSpace(parent)
	child = strings.TrimSpace(child)
	boundary = normalizeProviderSessionBoundary(boundary, parent)
	if child == "" || child == parent {
		if t == nil {
			return DefaultState(child), false
		}
		return t.Get(child), false
	}
	if t == nil {
		st := DefaultState(child)
		st.ProviderBoundary = boundary
		st.LastActive = st.LastActive.Refresh(now)
		return st, true
	}

	t.mu.Lock()
	t.ensureLocked()
	if existing, ok := t.state[child]; ok {
		if existing.ProviderBoundary == boundary {
			t.mu.Unlock()
			return existing, false
		}
		// A child id already owned by another lineage is a collision, never an
		// invitation to overwrite that session.
		t.mu.Unlock()
		return existing, false
	}

	parentState := t.getLocked(parent)
	from := parentState.Run
	parentState.Time = parentState.Time.Pause(now)
	parentState.Run = Stopped
	parentState.Reason = ReasonProviderSessionClear
	parentState.LastActive = parentState.LastActive.Refresh(now)
	parentState = t.putLocked(parentState)
	if from == Paused {
		t.signalResumeLocked(parent)
	}

	budget := parentState.Budget.withContextCap()
	if budget.ContextTokensCap > 0 {
		budget.ContextTokensLeft = budget.ContextTokensCap
	}
	carriedTime := parentState.Time
	if carriedTime.Bounded() || carriedTime.ElapsedNanos > 0 {
		carriedTime = carriedTime.Start(now)
	}
	childState := State{
		TraceID:          child,
		Run:              Running,
		Budget:           budget,
		Priority:         parentState.Priority,
		Pace:             parentState.Pace,
		LastActive:       parentState.LastActive.Refresh(now),
		Time:             carriedTime,
		Throughput:       parentState.Throughput,
		QualityEnvelope:  parentState.QualityEnvelope,
		ProviderBoundary: boundary,
	}
	childState = t.putLocked(childState)
	obs := t.transObs
	fire := obs != nil && notableTransition(from, Stopped)
	t.mu.Unlock()
	if fire {
		obs(transitionEvent(parentState, from, Stopped))
	}
	return childState, true
}
