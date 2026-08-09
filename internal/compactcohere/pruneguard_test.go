// Copyright 2026 The fak Authors
// Licensed under the Apache License, Version 2.0.

package compactcohere

import "testing"

func TestCheckPruneRefusesSpanThatPlaceholderWouldGrow(t *testing.T) {
	spanTokens := PlaceholderTokenCost - 1
	got := CheckPrune(spanTokens, nil, 0)
	if got.Allowed || got.Reason != PruneBelowMinimum {
		t.Fatalf("CheckPrune(%d) = %+v, want below-minimum refusal", spanTokens, got)
	}
	if got.PlaceholderTokens <= got.SpanTokens {
		t.Fatalf("proof precondition lost: placeholder=%d must exceed pruned span=%d", got.PlaceholderTokens, got.SpanTokens)
	}
	// Recompute from the bytes fak emits rather than merely echoing the constants.
	wireBytes := len(PrunePlaceholder) + len(`{"role":"assistant","content":""}`)
	measuredCost := wireBytes / 4
	if wireBytes != 144 || measuredCost != 36 {
		t.Fatalf("placeholder measurement changed: bytes=%d measured_cost=%d declared_cost=%d floor=%d", wireBytes, measuredCost, PlaceholderTokenCost, MinimumPruneTokens)
	}
	if PlaceholderTokenCost != measuredCost {
		t.Fatalf("declared placeholder cost=%d, measured=%d", PlaceholderTokenCost, measuredCost)
	}
	if MinimumPruneTokens != measuredCost+1 {
		t.Fatalf("minimum prune floor=%d, want first profitable span=%d", MinimumPruneTokens, measuredCost+1)
	}
	if netGrowth := got.PlaceholderTokens - got.SpanTokens; netGrowth <= 0 {
		t.Fatalf("refused prune must have grown context: placeholder=%d span=%d growth=%d", got.PlaceholderTokens, got.SpanTokens, netGrowth)
	}
}

func TestCheckPruneFloorAndUnchangedScore(t *testing.T) {
	for _, tc := range []struct {
		name string
		span int
		want bool
	}{
		{name: "equal placeholder is not profitable", span: PlaceholderTokenCost, want: false},
		{name: "first net-positive span", span: MinimumPruneTokens, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := CheckPrune(tc.span, nil, 0)
			if got.Allowed != tc.want {
				t.Fatalf("CheckPrune(%d).Allowed = %v, want %v (%+v)", tc.span, got.Allowed, tc.want, got)
			}
		})
	}
}

func TestCheckPruneRefusesOrphanedToolResult(t *testing.T) {
	transcript := []TranscriptItem{
		{Kind: TranscriptOther},
		{Kind: TranscriptToolCall, CallID: "call-1"},
		{Kind: TranscriptToolResult, CallID: "call-1"},
		{Kind: TranscriptOther},
	}
	got := CheckPrune(MinimumPruneTokens, transcript, 2)
	if got.Allowed || got.Reason != PruneStructuralOrphan {
		t.Fatalf("cut between tool call/result = %+v, want structural refusal", got)
	}
}

func TestCheckPruneRefusesOrphanedLastToolResult(t *testing.T) {
	transcript := []TranscriptItem{
		{Kind: TranscriptToolCall, CallID: "last"},
		{Kind: TranscriptOther},
		{Kind: TranscriptToolResult, CallID: "last"},
	}
	got := CheckPrune(MinimumPruneTokens, transcript, len(transcript)-1)
	if got.Allowed || got.Reason != PruneStructuralOrphan {
		t.Fatalf("cut before final tool result = %+v, want structural refusal", got)
	}
}

func TestCheckPruneAllowsCompleteToolPairRemoval(t *testing.T) {
	transcript := []TranscriptItem{
		{Kind: TranscriptToolCall, CallID: "done"},
		{Kind: TranscriptToolResult, CallID: "done"},
		{Kind: TranscriptOther},
	}
	got := CheckPrune(MinimumPruneTokens, transcript, 2)
	if !got.Allowed || got.Reason != PruneAllowed {
		t.Fatalf("complete pair removal = %+v, want allowed", got)
	}
}

func TestCountRefusalReportsReasons(t *testing.T) {
	var got RefusalCounters
	got = CountRefusal(got, CheckPrune(PlaceholderTokenCost, nil, 0))
	got = CountRefusal(got, CheckPrune(MinimumPruneTokens, []TranscriptItem{
		{Kind: TranscriptToolCall, CallID: "x"},
		{Kind: TranscriptToolResult, CallID: "x"},
	}, 1))
	got = CountRefusal(got, CheckPrune(MinimumPruneTokens, nil, 1))
	got = CountRefusal(got, CheckPrune(MinimumPruneTokens, nil, 0)) // allowed: no increment
	want := RefusalCounters{BelowMinimum: 1, StructuralOrphan: 1, InvalidCut: 1}
	if got != want {
		t.Fatalf("CountRefusal = %+v, want %+v", got, want)
	}
}

func TestCoordinatorPrunePathRecordsRefusal(t *testing.T) {
	c := New(DefaultProviderCacheTTL)
	got := c.CheckPrune(PlaceholderTokenCost, nil, 0)
	if got.Allowed || got.Reason != PruneBelowMinimum {
		t.Fatalf("coordinator guard = %+v, want below-minimum refusal", got)
	}
	if counters := c.PruneRefusals(); counters.BelowMinimum != 1 || counters.StructuralOrphan != 0 {
		t.Fatalf("coordinator counters = %+v, want one below-minimum refusal", counters)
	}
}
