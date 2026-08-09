// Copyright 2026 The fak Authors
// Licensed under the Apache License, Version 2.0.

package compactcohere

import (
	"strings"
	"unicode/utf8"
)

// PrunePlaceholder is fak's smallest real compaction-stub content: the one-turn,
// no-tombstone form emitted by agent.compactStubContent. compactcohere stays a
// stdlib-only foundation leaf, so the value is repeated here instead of importing
// agent upward; the arithmetic witness below pins the two implementations together.
const PrunePlaceholder = "[fak] compacted 1 earlier turn(s) to stay within the context budget; their detail is omitted from this request."

// prunePlaceholderEnvelopeBytes is the JSON message framing charged by
// agent.compactStubTokenCost. The user and assistant forms have the same length.
const prunePlaceholderEnvelopeBytes = len(`{"role":"assistant","content":""}`)

// PlaceholderTokenCost measures fak's actual minimum placeholder in the same
// approximate four-bytes-per-token currency as CompactAnthropicHistory:
// 111 content bytes + 33 message-envelope bytes = 144 bytes = 36 tokens.
const PlaceholderTokenCost = (len(PrunePlaceholder) + prunePlaceholderEnvelopeBytes) / 4

// MinimumPruneTokens is the first span size that can reclaim a token after the
// minimum placeholder is inserted: 36 + 1 = 37 tokens. It is an arithmetic
// correctness bound, not a tuning knob: spans below it cost at least as much to
// replace as to retain. Callers with a larger dynamic tombstone must still use
// their exact net-shed calculation in addition to this minimum floor.
const MinimumPruneTokens = PlaceholderTokenCost + 1

// TranscriptKind identifies the structural role of a transcript item at a
// prospective prefix cut.
type TranscriptKind string

const (
	TranscriptOther      TranscriptKind = "other"
	TranscriptToolCall   TranscriptKind = "tool_call"
	TranscriptToolResult TranscriptKind = "tool_result"
)

// TranscriptItem is the minimum structure needed to preserve tool call/result
// pairing across a prefix prune. CallID joins a result to its call.
type TranscriptItem struct {
	Kind   TranscriptKind
	CallID string
}

// PruneRefusal is a stable reason for refusing an individual prune.
type PruneRefusal string

const (
	PruneAllowed          PruneRefusal = ""
	PruneBelowMinimum     PruneRefusal = "below_minimum"
	PruneStructuralOrphan PruneRefusal = "structural_orphan"
	PruneInvalidCut       PruneRefusal = "invalid_cut"
)

// PruneResult is the pure result of checking one proposed prefix prune.
type PruneResult struct {
	Allowed           bool
	Reason            PruneRefusal
	SpanTokens        int
	PlaceholderTokens int
	Cut               int
}

// CheckPrune checks the arithmetic and transcript structure of a proposed
// prefix prune. It deliberately has no clock or coordinator state; cache timing
// remains the responsibility of Classify and is unchanged by this guard.
func CheckPrune(spanTokens int, transcript []TranscriptItem, cut int) PruneResult {
	d := PruneResult{
		Allowed:           false,
		SpanTokens:        spanTokens,
		PlaceholderTokens: PlaceholderTokenCost,
		Cut:               cut,
	}
	if cut < 0 || cut > len(transcript) {
		d.Reason = PruneInvalidCut
		return d
	}
	if spanTokens < MinimumPruneTokens {
		d.Reason = PruneBelowMinimum
		return d
	}
	if orphansToolResult(transcript, cut) {
		d.Reason = PruneStructuralOrphan
		return d
	}
	d.Allowed = true
	d.Reason = PruneAllowed
	return d
}

func orphansToolResult(transcript []TranscriptItem, cut int) bool {
	removedCalls := make(map[string]struct{})
	for _, item := range transcript[:cut] {
		if item.Kind == TranscriptToolCall && strings.TrimSpace(item.CallID) != "" {
			removedCalls[item.CallID] = struct{}{}
		}
	}
	for _, item := range transcript[cut:] {
		if item.Kind != TranscriptToolResult || strings.TrimSpace(item.CallID) == "" {
			continue
		}
		if _, removed := removedCalls[item.CallID]; removed {
			return true
		}
	}
	return false
}

// RefusalCounters is the observable tally of prune refusals by stable reason.
// CountRefusal returns an updated value, keeping the guard itself pure and
// allowing each prune-path owner to publish the counters through its own sink.
type RefusalCounters struct {
	BelowMinimum     uint64 `json:"below_minimum"`
	StructuralOrphan uint64 `json:"structural_orphan"`
	InvalidCut       uint64 `json:"invalid_cut"`
}

// CountRefusal records a refused decision in a value counter.
func CountRefusal(c RefusalCounters, d PruneResult) RefusalCounters {
	if d.Allowed {
		return c
	}
	switch d.Reason {
	case PruneBelowMinimum:
		c.BelowMinimum++
	case PruneStructuralOrphan:
		c.StructuralOrphan++
	case PruneInvalidCut:
		c.InvalidCut++
	}
	return c
}

// PlaceholderRuneCount exposes the measured unit for proof and diagnostics.
func PlaceholderRuneCount() int { return utf8.RuneCountInString(PrunePlaceholder) }
