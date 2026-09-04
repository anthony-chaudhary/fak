// Package superstream provides the functional top-level coordinator for "Super Workstreams":
// an organized, queue-ordered execution stream that bridges high-level operator intents,
// dynamic per-item lane leases, and rigorous context safety across long multi-turn sessions.
//
// # Why Super Workstreams Exist
//
// Existing coordination primitives in fak address specific granular layers:
//   - Single-issue dispatch (fak dispatch tick / dos-dispatch) executes one task under one lease,
//     shutting down on exit with no multi-item queue continuity.
//   - Super loops (fak superloop walk/drive) operate as read-first interior nodes selecting the
//     single worst-first member loop to enter, but do not manage an ordered stream queue of tasks
//     with structured state handoffs across long turns.
//   - Bulk waves (/super-loop / fleet-wave) spawn N detached concurrent processes, trading away
//     sequential context continuity for breadth.
//   - Unconstrained agent loops attempt multiple sequential tasks in a single raw session, which
//     inevitably hits two fatal failure modes:
//     1. Context Bloat and Attention Dilution: linear accumulation of turns degrades prompt
//     coherence, exhausts token budgets, and causes hallucination across tasks.
//     2. Coarse Lease Starvation / Collisions: holding one coarse lane lease for an entire 50-turn
//     session starves peers on the shared trunk, while failing to lease per-item causes collisions.
//
// # The Three Pillars of a Super Workstream
//
//  1. Queue-Ordered Progression: A workstream maintains an explicit, prioritized queue of WorkItems.
//     Each item has its own bounded turn budget, token limit, target lane, and acceptance witness.
//  2. Dynamic Per-Item Lane Leasing: The stream does not hold an over-broad lease for its lifetime.
//     Instead, it acquires the specific lane lease just-in-time for the active item, enforces
//     collision safety (laneadmit.COLLISION_RISK), optionally skips to disjoint queue items if the
//     head is contended, and releases the lease as soon as the item completes or yields.
//  3. Long-Turn Context Safety: The stream enforces O(1) coordinator context safety between queue
//     items. Rather than accumulating unbounded linear history, it evaluates context pressure,
//     bounds per-item turns, and produces a compact StreamCarryoverSeed containing only structured
//     receipts and active pins. This integrates seamlessly with fak's natural context control
//     architecture (ctxplan 4-area layouts, session budget resets, and ctxmmu write-time admission).
package superstream

import (
	"time"
)

// Schema identifiers for versioned JSON serialization.
const (
	StreamSchema    = "fak.superstream.v1"
	CarryoverSchema = "fak.superstream.carryover.v1"
	ReportSchema    = "fak.superstream.report.v1"
)

// ItemStatus represents the lifecycle stage of an individual work item within a stream.
type ItemStatus string

const (
	ItemPending       ItemStatus = "pending"
	ItemLeaseAcquired ItemStatus = "lease_acquired"
	ItemExecuting     ItemStatus = "executing"
	ItemWitnessed     ItemStatus = "witnessed"
	ItemCommitted     ItemStatus = "committed"
	ItemCompleted     ItemStatus = "completed"
	ItemFailed        ItemStatus = "failed"
	ItemYielded       ItemStatus = "yielded"
)

// Terminal returns true if the item has reached a final state for this iteration.
func (s ItemStatus) Terminal() bool {
	return s == ItemCompleted || s == ItemFailed || s == ItemYielded
}

// WorkItem is a single bounded unit of work in the stream queue (e.g. an issue, task, or leaf).
type WorkItem struct {
	ID            string     `json:"id"`
	Title         string     `json:"title"`
	Lane          string     `json:"lane"`
	Tree          []string   `json:"tree,omitempty"`
	Status        ItemStatus `json:"status"`
	MaxTurns      int        `json:"max_turns,omitempty"`
	MaxTokens     int        `json:"max_tokens,omitempty"`
	Witness       string     `json:"witness,omitempty"`
	WitnessResult string     `json:"witness_result,omitempty"`
	CommitSHA     string     `json:"commit_sha,omitempty"`
	Error         string     `json:"error,omitempty"`
	TurnsSpent    int        `json:"turns_spent,omitempty"`
	TokensSpent   int        `json:"tokens_spent,omitempty"`
}

// StreamSpec defines the static configuration and queue for a Super Workstream.
type StreamSpec struct {
	ID              string     `json:"id"`
	Intent          string     `json:"intent"`
	Queue           []WorkItem `json:"queue"`
	BasePins        []string   `json:"base_pins,omitempty"`
	MaxTurnsTotal   int        `json:"max_turns_total,omitempty"`
	MaxTokensTotal  int        `json:"max_tokens_total,omitempty"`
	MaxTurnsPerItem int        `json:"max_turns_per_item,omitempty"`
}

// Default limits if unspecified.
const (
	DefaultMaxTurnsPerItem = 8
	DefaultMaxTurnsTotal   = 40
	DefaultMaxTokensTotal  = 250000
)

// NormalizedSpec returns a copy of spec with default values populated where zero.
func (s StreamSpec) NormalizedSpec() StreamSpec {
	out := s
	if out.MaxTurnsPerItem <= 0 {
		out.MaxTurnsPerItem = DefaultMaxTurnsPerItem
	}
	if out.MaxTurnsTotal <= 0 {
		out.MaxTurnsTotal = DefaultMaxTurnsTotal
	}
	if out.MaxTokensTotal <= 0 {
		out.MaxTokensTotal = DefaultMaxTokensTotal
	}
	out.Queue = make([]WorkItem, len(s.Queue))
	for i, item := range s.Queue {
		it := item
		if it.Status == "" {
			it.Status = ItemPending
		}
		if it.MaxTurns <= 0 {
			it.MaxTurns = out.MaxTurnsPerItem
		}
		out.Queue[i] = it
	}
	return out
}

// HeldLease tracks an active lane lease currently held by the stream.
type HeldLease struct {
	LeaseID      string    `json:"lease_id"`
	Lane         string    `json:"lane"`
	Tree         []string  `json:"tree,omitempty"`
	Holder       string    `json:"holder"`
	AcquiredAt   time.Time `json:"acquired_at"`
	FencingToken string    `json:"fencing_token,omitempty"`
}

// StreamState holds the mutable operational state of an executing stream.
type StreamState struct {
	StreamID          string     `json:"stream_id"`
	Intent            string     `json:"intent"`
	ActiveIndex       int        `json:"active_index"`
	Queue             []WorkItem `json:"queue"`
	CurrentLease      *HeldLease `json:"current_lease,omitempty"`
	CompletedCount    int        `json:"completed_count"`
	FailedCount       int        `json:"failed_count"`
	YieldedCount      int        `json:"yielded_count"`
	TotalTurnsSpent   int        `json:"total_turns_spent"`
	TotalTokensSpent  int        `json:"total_tokens_spent"`
	CurrentItemTurns  int        `json:"current_item_turns"`
	CurrentItemTokens int        `json:"current_item_tokens"`
	Closed            bool       `json:"closed"`
	CloseReason       string     `json:"close_reason,omitempty"`
}

// NewStreamState initializes execution state from a normalized specification.
func NewStreamState(spec StreamSpec) StreamState {
	norm := spec.NormalizedSpec()
	return StreamState{
		StreamID: norm.ID,
		Intent:   norm.Intent,
		Queue:    norm.Queue,
	}
}

// ActiveItem returns a pointer to the current work item in the queue, or nil if none.
func (s *StreamState) ActiveItem() *WorkItem {
	if s.ActiveIndex < 0 || s.ActiveIndex >= len(s.Queue) {
		return nil
	}
	return &s.Queue[s.ActiveIndex]
}

// AllTerminal reports whether every item in the queue has completed, failed, or yielded.
func (s *StreamState) AllTerminal() bool {
	for _, it := range s.Queue {
		if !it.Status.Terminal() {
			return false
		}
	}
	return true
}
