// Copyright 2026 The fak Authors
// Licensed under the Apache License, Version 2.0.

package compactcohere

import "sort"

// ToolResult is the producer-authored portion of a tool result that matters to
// compaction. Useless is opt-in: its zero value is false, and omitting the field
// preserves the pre-flag eviction behavior.
//
// A producer may set Useless only when it can mechanically prove that retaining
// the result adds no task context. The bit is a preference, never permission to
// prune through a cache-unsafe prefix event.
type ToolResult struct {
	Useless bool `json:"useless,omitempty"`
}

// ProduceSearchResult marks a completed zero-match search as useless. An
// incomplete search or an invalid negative match count is never marked.
func ProduceSearchResult(matches int, complete bool) ToolResult {
	return ToolResult{
		Useless: complete && matches == 0,
	}
}

// ProduceStatusResult marks a completed clean status result as useless. An
// incomplete status read or an invalid negative change count is never marked.
func ProduceStatusResult(changedEntries int, complete bool) ToolResult {
	return ToolResult{
		Useless: complete && changedEntries == 0,
	}
}

// ProduceReadResult marks a completed, byte-empty read as useless. Both the
// producer's read count and emitted payload count must be empty; disagreement
// fails closed and preserves the result.
func ProduceReadResult(bytesRead, payloadBytes int, complete bool) ToolResult {
	return ToolResult{
		Useless: complete && bytesRead == 0 && payloadBytes == 0,
	}
}

// ToolResultEntry is one tool result considered for eviction. Entries
// arrive in the caller's existing age/size heuristic order. AgeTurns and Tokens
// are carried for content-free observability; compactcohere does not reinterpret
// them, so unflagged callers retain exactly their prior order.
type ToolResultEntry struct {
	ID       string     `json:"id,omitempty"`
	Result   ToolResult `json:"result"`
	AgeTurns uint64     `json:"age_turns,omitempty"`
	Tokens   int        `json:"tokens,omitempty"`
}

// ToolResultSelection is a cache-gated eviction decision. Ordered is a copy of the
// candidate sequence after applying the producer preference. Evicted is the
// bounded prefix selected from Ordered when Allowed is true.
type ToolResultSelection struct {
	Event           PrefixEvent       `json:"event"`
	Allowed         bool              `json:"allowed"`
	Ordered         []ToolResultEntry `json:"ordered,omitempty"`
	Evicted         []ToolResultEntry `json:"evicted,omitempty"`
	FlaggedSelected uint64            `json:"flagged_selected"`
}

// PrefixAllowsToolResultPrune translates compactcohere's existing prefix-event
// classification into the cache-safety gate for individual result eviction.
// Stable prefixes and fak's prefix-preserving cut are safe. World breaks,
// harness rewrites, cold caches, and unknown events fail closed.
func PrefixAllowsToolResultPrune(event PrefixEvent) bool {
	switch event {
	case EventStable, EventFakCut:
		return true
	default:
		return false
	}
}

// SelectToolResults stably promotes producer-flagged results ahead of the caller's
// existing heuristic order, then selects at most limit results. The preference
// is applied only when the prefix event is cache-safe; a Useless flag can never
// override Classify's event decision. The input slice is never mutated.
func SelectToolResults(event PrefixEvent, candidates []ToolResultEntry, limit int) ToolResultSelection {
	ordered := append([]ToolResultEntry(nil), candidates...)
	selection := ToolResultSelection{
		Event:   event,
		Allowed: PrefixAllowsToolResultPrune(event),
		Ordered: ordered,
	}
	if !selection.Allowed || limit <= 0 {
		return selection
	}

	sort.SliceStable(selection.Ordered, func(i, j int) bool {
		return selection.Ordered[i].Result.Useless && !selection.Ordered[j].Result.Useless
	})

	if limit > len(selection.Ordered) {
		limit = len(selection.Ordered)
	}
	selection.Evicted = append([]ToolResultEntry(nil), selection.Ordered[:limit]...)
	for _, candidate := range selection.Evicted {
		if candidate.Result.Useless {
			selection.FlaggedSelected++
		}
	}
	return selection
}

// ToolResultPruneCounters is the content-free observable tally for tool-result
// eviction. FlaggedResultsPruned is the producer-signal value witness; the
// total keeps that number interpretable.
type ToolResultPruneCounters struct {
	ResultsPruned        uint64 `json:"results_pruned"`
	FlaggedResultsPruned uint64 `json:"flagged_results_pruned"`
}

// CountToolResultPrunes folds a completed selection into value counters.
func CountToolResultPrunes(c ToolResultPruneCounters, selection ToolResultSelection) ToolResultPruneCounters {
	c.ResultsPruned += uint64(len(selection.Evicted))
	for _, candidate := range selection.Evicted {
		if candidate.Result.Useless {
			c.FlaggedResultsPruned++
		}
	}
	return c
}
