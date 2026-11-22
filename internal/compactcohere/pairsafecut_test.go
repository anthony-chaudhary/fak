// Copyright 2026 The fak Authors
// Licensed under the Apache License, Version 2.0.

package compactcohere

import "testing"

func TestPairSafeCut(t *testing.T) {
	call := func(id string) TranscriptItem { return TranscriptItem{Kind: TranscriptToolCall, CallID: id} }
	result := func(id string) TranscriptItem { return TranscriptItem{Kind: TranscriptToolResult, CallID: id} }
	other := TranscriptItem{Kind: TranscriptOther}

	cases := []struct {
		name       string
		transcript []TranscriptItem
		cut        int
		want       int
	}{
		// Upstream's case: the prefix ends on an AI message whose tool_calls
		// have results immediately after it; those results join the prefix.
		{"last call pulls its result", []TranscriptItem{other, call("a"), result("a"), other}, 2, 3},
		// Upstream appends every matching ToolMessage for the last message's
		// tool_calls, so a multi-call turn pulls all of its results.
		{"multi-call turn pulls every result", []TranscriptItem{other, call("a"), call("b"), result("a"), result("b"), other}, 3, 5},
		{"results out of call order", []TranscriptItem{call("a"), call("b"), result("b"), other, result("a"), other}, 2, 5},
		{"result past an intervening item", []TranscriptItem{call("a"), other, result("a"), other}, 1, 3},
		// Upstream skips a tool_call id that has no ToolMessage.
		{"call without result keeps cut", []TranscriptItem{other, call("a"), other}, 2, 2},
		{"clean cut unchanged", []TranscriptItem{call("a"), result("a"), other, other}, 2, 2},
		{"result of a kept call ignored", []TranscriptItem{other, other, call("b"), result("b")}, 2, 2},
		{"blank call id ignored", []TranscriptItem{call(""), result(""), other}, 1, 1},
		{"swept-in call chains forward", []TranscriptItem{call("a"), call("b"), result("a"), result("b"), other}, 1, 4},
		{"chain through a second turn", []TranscriptItem{call("a"), other, call("b"), result("a"), other, result("b"), other}, 1, 6},
		{"zero cut unchanged", []TranscriptItem{call("a"), result("a")}, 0, 0},
		{"full cut unchanged", []TranscriptItem{call("a"), result("a")}, 2, 2},
		{"negative cut unchanged", []TranscriptItem{call("a"), result("a")}, -1, -1},
		{"overlong cut unchanged", []TranscriptItem{call("a"), result("a")}, 3, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := PairSafeCut(tc.transcript, tc.cut)
			if got != tc.want {
				t.Fatalf("PairSafeCut(cut=%d) = %d, want %d", tc.cut, got, tc.want)
			}
			if tc.cut < 0 || tc.cut > len(tc.transcript) {
				if r := CheckPrune(MinimumPruneTokens, tc.transcript, got); r.Reason != PruneInvalidCut {
					t.Fatalf("CheckPrune(invalid cut %d) = %+v, want invalid-cut refusal", got, r)
				}
				return
			}
			if r := CheckPrune(MinimumPruneTokens, tc.transcript, got); !r.Allowed {
				t.Fatalf("CheckPrune(adjusted cut %d) = %+v, want allowed", got, r)
			}
		})
	}
}

func TestPairSafeCutRepairsRefusedCut(t *testing.T) {
	transcript := []TranscriptItem{
		{Kind: TranscriptOther},
		{Kind: TranscriptToolCall, CallID: "toolu_1"},
		{Kind: TranscriptToolResult, CallID: "toolu_1"},
		{Kind: TranscriptOther},
	}
	if r := CheckPrune(MinimumPruneTokens, transcript, 2); r.Reason != PruneStructuralOrphan {
		t.Fatalf("precondition: CheckPrune(cut=2) = %+v, want structural-orphan refusal", r)
	}
	adjusted := PairSafeCut(transcript, 2)
	if r := CheckPrune(MinimumPruneTokens, transcript, adjusted); !r.Allowed || r.Cut != 3 {
		t.Fatalf("CheckPrune(PairSafeCut) = %+v, want allowed at cut 3", r)
	}
}
