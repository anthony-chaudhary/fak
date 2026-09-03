package superstream

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ContextSafetyStatus represents the context health of the stream.
type ContextSafetyStatus string

const (
	// StatusContextSafe means turn and token usage are well within healthy bounds.
	StatusContextSafe ContextSafetyStatus = "SAFE"
	// StatusContextPressureWarn means the active item has reached >= 70% of its turn
	// or token budget. The coordinator should prepare to checkpoint or wrap up.
	StatusContextPressureWarn ContextSafetyStatus = "PRESSURE_WARN"
	// StatusContextResetRequired means the active item has reached its turn limit,
	// or the item completed and a clean boundary reset is required before the next item.
	// Emitting a StreamCarryoverSeed and clearing the raw transcript is necessary.
	StatusContextResetRequired ContextSafetyStatus = "RESET_REQUIRED"
	// StatusContextExhausted means the entire stream's cumulative turn or token limit is reached.
	StatusContextExhausted ContextSafetyStatus = "EXHAUSTED"
)

// ContextSafetyVerdict encapsulates the assessment of context health and recommended action.
type ContextSafetyVerdict struct {
	Status             ContextSafetyStatus `json:"status"`
	Reason             string              `json:"reason"`
	TurnsRemainingItem int                 `json:"turns_remaining_item"`
	TurnsRemainingAll  int                 `json:"turns_remaining_all"`
	TokensRemainingAll int                 `json:"tokens_remaining_all"`
	RecommendReset     bool                `json:"recommend_reset"`
}

// EvaluateContextSafety inspects current stream usage against limits.
func EvaluateContextSafety(spec StreamSpec, state StreamState) ContextSafetyVerdict {
	norm := spec.NormalizedSpec()

	totalTurnsLeft := norm.MaxTurnsTotal - state.TotalTurnsSpent
	totalTokensLeft := norm.MaxTokensTotal - state.TotalTokensSpent

	if totalTurnsLeft <= 0 {
		return ContextSafetyVerdict{
			Status:             StatusContextExhausted,
			Reason:             fmt.Sprintf("stream cumulative turn budget exhausted (%d/%d spent)", state.TotalTurnsSpent, norm.MaxTurnsTotal),
			TurnsRemainingItem: 0,
			TurnsRemainingAll:  0,
			TokensRemainingAll: totalTokensLeft,
			RecommendReset:     false,
		}
	}

	if totalTokensLeft <= 0 {
		return ContextSafetyVerdict{
			Status:             StatusContextExhausted,
			Reason:             fmt.Sprintf("stream cumulative token budget exhausted (%d/%d spent)", state.TotalTokensSpent, norm.MaxTokensTotal),
			TurnsRemainingItem: 0,
			TurnsRemainingAll:  totalTurnsLeft,
			TokensRemainingAll: 0,
			RecommendReset:     false,
		}
	}

	active := state.ActiveItem()
	if active == nil {
		return ContextSafetyVerdict{
			Status:             StatusContextSafe,
			Reason:             "no active item in queue",
			TurnsRemainingItem: 0,
			TurnsRemainingAll:  totalTurnsLeft,
			TokensRemainingAll: totalTokensLeft,
			RecommendReset:     false,
		}
	}

	maxItemTurns := active.MaxTurns
	if maxItemTurns <= 0 {
		maxItemTurns = norm.MaxTurnsPerItem
	}
	itemTurnsLeft := maxItemTurns - state.CurrentItemTurns

	if itemTurnsLeft <= 0 {
		return ContextSafetyVerdict{
			Status:             StatusContextResetRequired,
			Reason:             fmt.Sprintf("item %q reached turn ceiling (%d/%d turns)", active.ID, state.CurrentItemTurns, maxItemTurns),
			TurnsRemainingItem: 0,
			TurnsRemainingAll:  totalTurnsLeft,
			TokensRemainingAll: totalTokensLeft,
			RecommendReset:     true,
		}
	}

	if active.MaxTokens > 0 && state.CurrentItemTokens >= active.MaxTokens {
		return ContextSafetyVerdict{
			Status:             StatusContextResetRequired,
			Reason:             fmt.Sprintf("item %q reached token ceiling (%d/%d tokens)", active.ID, state.CurrentItemTokens, active.MaxTokens),
			TurnsRemainingItem: itemTurnsLeft,
			TurnsRemainingAll:  totalTurnsLeft,
			TokensRemainingAll: totalTokensLeft,
			RecommendReset:     true,
		}
	}

	// Warn if >= 70% of item budget consumed.
	if float64(state.CurrentItemTurns)/float64(maxItemTurns) >= 0.70 {
		return ContextSafetyVerdict{
			Status:             StatusContextPressureWarn,
			Reason:             fmt.Sprintf("item %q under context pressure (%d/%d turns spent)", active.ID, state.CurrentItemTurns, maxItemTurns),
			TurnsRemainingItem: itemTurnsLeft,
			TurnsRemainingAll:  totalTurnsLeft,
			TokensRemainingAll: totalTokensLeft,
			RecommendReset:     false,
		}
	}

	return ContextSafetyVerdict{
		Status:             StatusContextSafe,
		Reason:             fmt.Sprintf("item %q has %d turns remaining", active.ID, itemTurnsLeft),
		TurnsRemainingItem: itemTurnsLeft,
		TurnsRemainingAll:  totalTurnsLeft,
		TokensRemainingAll: totalTokensLeft,
		RecommendReset:     false,
	}
}

// ItemSummary is a compact representation of a completed work item for O(1) carryover.
type ItemSummary struct {
	ID            string `json:"id"`
	Title         string `json:"title"`
	Lane          string `json:"lane"`
	Status        string `json:"status"`
	CommitSHA     string `json:"commit_sha,omitempty"`
	WitnessResult string `json:"witness_result,omitempty"`
}

// StreamCarryoverSeed is the compact, structured O(1) state handoff object emitted
// when transitioning across item boundaries or resetting context.
// It carries only verified receipts, active task pins, and remaining queue items,
// discarding the bloated linear transcript of past tool outputs.
type StreamCarryoverSeed struct {
	Schema         string          `json:"schema"`
	StreamID       string          `json:"stream_id"`
	Intent         string          `json:"intent"`
	ActiveIndex    int             `json:"active_index"`
	TotalItems     int             `json:"total_items"`
	CompletedItems []ItemSummary   `json:"completed_items"`
	CurrentItem    *WorkItem       `json:"current_item,omitempty"`
	NextItem       *WorkItem       `json:"next_item,omitempty"`
	StreamPins     []string        `json:"stream_pins,omitempty"`
	TurnsRemaining int             `json:"turns_remaining"`
	TokensRemain   int             `json:"tokens_remaining"`
	Layout         *ctxplan.Layout `json:"layout,omitempty"`
}

// BuildStreamLayout returns a recommended ctxplan.Layout configured specifically
// for the Super Workstream's O(1) resident view.
func BuildStreamLayout(pinCount int) ctxplan.Layout {
	spans := pinCount
	if spans < 4 {
		spans = 4
	}
	return ctxplan.Layout{
		Base: ctxplan.AreaPolicy{
			MaxSpans:  spans,
			Precision: ctxplan.PrecisionExact,
		},
		Current: ctxplan.AreaPolicy{
			MaxSpans:  2,
			Precision: ctxplan.PrecisionExact,
		},
		Recent: ctxplan.AreaPolicy{
			MaxSpans:  4,
			Precision: ctxplan.PrecisionPlanned,
		},
		Deep: ctxplan.AreaPolicy{
			MaxSpans:  8,
			Precision: ctxplan.PrecisionPointer,
		},
		MaxCandidates: 32,
	}
}

// BuildCarryoverSeed synthesizes a fresh StreamCarryoverSeed from spec and state.
func BuildCarryoverSeed(spec StreamSpec, state StreamState) StreamCarryoverSeed {
	norm := spec.NormalizedSpec()

	completed := make([]ItemSummary, 0, len(state.Queue))
	for _, it := range state.Queue {
		if it.Status == ItemCompleted {
			completed = append(completed, ItemSummary{
				ID:            it.ID,
				Title:         it.Title,
				Lane:          it.Lane,
				Status:        string(it.Status),
				CommitSHA:     it.CommitSHA,
				WitnessResult: it.WitnessResult,
			})
		}
	}

	var cur *WorkItem
	var next *WorkItem
	if state.ActiveIndex >= 0 && state.ActiveIndex < len(state.Queue) {
		itemCopy := state.Queue[state.ActiveIndex]
		cur = &itemCopy
	}
	if state.ActiveIndex+1 < len(state.Queue) {
		itemCopy := state.Queue[state.ActiveIndex+1]
		next = &itemCopy
	}

	turnsLeft := norm.MaxTurnsTotal - state.TotalTurnsSpent
	tokensLeft := norm.MaxTokensTotal - state.TotalTokensSpent
	layout := BuildStreamLayout(len(norm.BasePins))

	return StreamCarryoverSeed{
		Schema:         CarryoverSchema,
		StreamID:       state.StreamID,
		Intent:         state.Intent,
		ActiveIndex:    state.ActiveIndex,
		TotalItems:     len(state.Queue),
		CompletedItems: completed,
		CurrentItem:    cur,
		NextItem:       next,
		StreamPins:     append([]string(nil), norm.BasePins...),
		TurnsRemaining: turnsLeft,
		TokensRemain:   tokensLeft,
		Layout:         &layout,
	}
}
