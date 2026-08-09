// Copyright 2026 The fak Authors
// Licensed under the Apache License, Version 2.0.

package compactcohere

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

func TestToolResultUselessDefaultsFalseAndOmitsFromLegacyJSON(t *testing.T) {
	var got ToolResult
	if err := json.Unmarshal([]byte(`{}`), &got); err != nil {
		t.Fatal(err)
	}
	if got.Useless {
		t.Fatal("absent useless flag defaulted true")
	}

	encoded, err := json.Marshal(ToolResult{})
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != "{}" {
		t.Fatalf("zero-value flag should be omitted, got %s", encoded)
	}

	encoded, err = json.Marshal(ToolResult{Useless: true})
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{"useless":true}` {
		t.Fatalf("true flag missing from result JSON, got %s", encoded)
	}
}

func TestProducersSetUselessOnlyForMechanicalEmptyResults(t *testing.T) {
	tests := []struct {
		name string
		got  ToolResult
		want bool
	}{
		{name: "search zero matches", got: ProduceSearchResult(0, true), want: true},
		{name: "search has match", got: ProduceSearchResult(1, true), want: false},
		{name: "search incomplete zero", got: ProduceSearchResult(0, false), want: false},
		{name: "search invalid count", got: ProduceSearchResult(-1, true), want: false},

		{name: "status clean", got: ProduceStatusResult(0, true), want: true},
		{name: "status changed", got: ProduceStatusResult(1, true), want: false},
		{name: "status incomplete clean", got: ProduceStatusResult(0, false), want: false},
		{name: "status invalid count", got: ProduceStatusResult(-1, true), want: false},

		{name: "read empty", got: ProduceReadResult(0, 0, true), want: true},
		{name: "read has bytes", got: ProduceReadResult(1, 1, true), want: false},
		{name: "read incomplete empty", got: ProduceReadResult(0, 0, false), want: false},
		{name: "read count payload disagreement", got: ProduceReadResult(0, 5, true), want: false},
		{name: "read invalid count", got: ProduceReadResult(-1, 0, true), want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.Useless != tt.want {
				t.Fatalf("Useless = %v, want %v", tt.got.Useless, tt.want)
			}
		})
	}
}

func TestSelectToolResultsPrefersFlaggedResultAtEqualAgeAndSize(t *testing.T) {
	candidates := []ToolResultEntry{
		{ID: "unflagged", Result: ToolResult{}, AgeTurns: 7, Tokens: 90},
		{ID: "flagged", Result: ToolResult{Useless: true}, AgeTurns: 7, Tokens: 90},
	}
	before := append([]ToolResultEntry(nil), candidates...)

	selection := SelectToolResults(EventStable, candidates, 1)
	if !selection.Allowed {
		t.Fatal("stable prefix should permit eviction")
	}
	if got, want := candidateIDs(selection.Ordered), []string{"flagged", "unflagged"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("ordered IDs = %v, want %v", got, want)
	}
	if got, want := candidateIDs(selection.Evicted), []string{"flagged"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("evicted IDs = %v, want %v", got, want)
	}
	if selection.FlaggedSelected != 1 {
		t.Fatalf("FlaggedSelected = %d, want 1", selection.FlaggedSelected)
	}
	if !reflect.DeepEqual(candidates, before) {
		t.Fatalf("input mutated: got %#v, want %#v", candidates, before)
	}
}

func TestSelectToolResultsCannotOverrideClassifiedUnsafePrefix(t *testing.T) {
	prev := TurnObservation{InboundPrefixDigest: "before"}
	cur := TurnObservation{InboundPrefixDigest: "after"}
	event := Classify(prev, cur, time.Minute)
	if event != EventHarnessRewrite {
		t.Fatalf("Classify event = %q, want %q", event, EventHarnessRewrite)
	}

	candidates := []ToolResultEntry{
		{ID: "unflagged", Result: ToolResult{}},
		{ID: "flagged", Result: ToolResult{Useless: true}},
	}
	selection := SelectToolResults(event, candidates, 2)
	if selection.Allowed {
		t.Fatal("harness rewrite unexpectedly permitted eviction")
	}
	if len(selection.Evicted) != 0 || selection.FlaggedSelected != 0 {
		t.Fatalf("unsafe event evicted results: %#v", selection)
	}
	if got, want := candidateIDs(selection.Ordered), []string{"unflagged", "flagged"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("unsafe event reordered candidates: got %v, want %v", got, want)
	}
}

func TestPrefixAllowsToolResultPruneFailsClosed(t *testing.T) {
	tests := []struct {
		event PrefixEvent
		want  bool
	}{
		{event: EventStable, want: true},
		{event: EventFakCut, want: true},
		{event: EventFakWorldBreak, want: false},
		{event: EventHarnessRewrite, want: false},
		{event: EventColdTTL, want: false},
		{event: PrefixEvent("future_event"), want: false},
	}
	for _, tt := range tests {
		t.Run(string(tt.event), func(t *testing.T) {
			if got := PrefixAllowsToolResultPrune(tt.event); got != tt.want {
				t.Fatalf("PrefixAllowsToolResultPrune(%q) = %v, want %v", tt.event, got, tt.want)
			}
		})
	}
}

func TestCoordinatorCountsFlaggedResultsPruned(t *testing.T) {
	c := New(time.Minute)
	candidates := []ToolResultEntry{
		{ID: "ordinary", Result: ToolResult{}},
		{ID: "empty-search", Result: ProduceSearchResult(0, true)},
	}

	selection := c.SelectToolResults(EventStable, candidates, 2)
	if len(selection.Evicted) != 2 {
		t.Fatalf("evicted = %d, want 2", len(selection.Evicted))
	}
	if got, want := c.ToolResultPrunes(), (ToolResultPruneCounters{
		ResultsPruned:        2,
		FlaggedResultsPruned: 1,
	}); got != want {
		t.Fatalf("counters = %#v, want %#v", got, want)
	}

	c.SelectToolResults(EventColdTTL, []ToolResultEntry{
		{ID: "blocked", Result: ToolResult{Useless: true}},
	}, 1)
	if got, want := c.ToolResultPrunes(), (ToolResultPruneCounters{
		ResultsPruned:        2,
		FlaggedResultsPruned: 1,
	}); got != want {
		t.Fatalf("unsafe plan changed counters: got %#v, want %#v", got, want)
	}
}

func TestUnflaggedProducerSurveyPreservesExistingOrder(t *testing.T) {
	candidates := []ToolResultEntry{
		{ID: "search", Result: ToolResult{}, AgeTurns: 9, Tokens: 120},
		{ID: "status", Result: ToolResult{}, AgeTurns: 7, Tokens: 80},
		{ID: "read", Result: ToolResult{}, AgeTurns: 4, Tokens: 40},
		{ID: "other", Result: ToolResult{}, AgeTurns: 1, Tokens: 10},
	}

	selection := SelectToolResults(EventFakCut, candidates, len(candidates))
	if got, want := candidateIDs(selection.Ordered), candidateIDs(candidates); !reflect.DeepEqual(got, want) {
		t.Fatalf("unflagged order changed: got %v, want %v", got, want)
	}
	if got, want := candidateIDs(selection.Evicted), candidateIDs(candidates); !reflect.DeepEqual(got, want) {
		t.Fatalf("unflagged eviction changed: got %v, want %v", got, want)
	}
	if selection.FlaggedSelected != 0 {
		t.Fatalf("FlaggedSelected = %d, want 0", selection.FlaggedSelected)
	}
}

func candidateIDs(candidates []ToolResultEntry) []string {
	ids := make([]string, len(candidates))
	for i, candidate := range candidates {
		ids[i] = candidate.ID
	}
	return ids
}
