package main

import "testing"

// Tests for the aiperf-style ISL budget compensation port
// (cmd/loadgen/isl_budget.go), derived from the upstream spec
// (ai-dynamo/aiperf src/aiperf/dataset/composer/base.py @
// rather than from implementation internals.
//
// Upstream accounting (BaseDatasetComposer):
//
//	first_turn_isl_adjustment     = per_request_fixed + per_msg_wrap
//	                              + first_turn_cache_bust_marker_tokens
//	subsequent_turn_isl_adjustment = per_msg_wrap
//	compensated_shared_sys_len    = max(1, configured - marker)  [marker on system]
//
// The compensated per-turn budget is target ISL minus the adjustment, so the
// composed wire ISL (synthetic body + wrapping + marker) matches the target.

// TestISLBudgetCompensation verifies the ISL budget compensation correctly
// adjusts the budget so the composed workload respects the ISL constraint:
// each expected value is hand-computed from the source's accounting.
func TestISLBudgetCompensation(t *testing.T) {
	tests := []struct {
		name   string
		target int
		comp   ISLCompensation
		turns  int
		want   []int
	}{
		{
			// Component (a) on the first user turn + component (b) on every
			// turn: first = 1000 - (5 + 3 + 7) = 985, later = 1000 - 3 = 997.
			// Wire check: 985 + 5 + 3 + 7 = 1000; 997 + 3 = 1000.
			name:   "first-turn marker with chat template",
			target: 1000,
			comp:   NewISLCompensation(CacheBustFirstTurnPrefix, false, true, 7, 5, 3),
			turns:  3,
			want:   []int{985, 997, 997},
		},
		{
			// SYSTEM_* target with a system prompt: marker routes to the
			// system message, so the first user turn is charged only the
			// chat-template costs: 500 - (4 + 2) = 494, later 500 - 2 = 498.
			name:   "system marker leaves first turn marker-free",
			target: 500,
			comp:   NewISLCompensation(CacheBustSystemSuffix, true, true, 6, 4, 2),
			turns:  2,
			want:   []int{494, 498},
		},
		{
			// SYSTEM_* target with NO system prompt: the marker falls through
			// to the first user turn (upstream worker._apply_cache_bust
			// fallback): 300 - 5 = 295, later 300 - 0 = 300.
			name:   "system target without system prompt falls through",
			target: 300,
			comp:   NewISLCompensation(CacheBustSystemPrefix, false, false, 5, 0, 0),
			turns:  2,
			want:   []int{295, 300},
		},
		{
			// NONE target: no marker cost at all; only the chat-template
			// costs apply: 200 - (2 + 1) = 197, later 200 - 1 = 199.
			name:   "none target charges chat template only",
			target: 200,
			comp:   NewISLCompensation(CacheBustNone, true, true, 9, 2, 1),
			turns:  2,
			want:   []int{197, 199},
		},
		{
			// WARMUP_ISOLATION_* targets carry no profiling-phase token cost
			// (upstream estimate_marker_token_cost returns 0): the measured
			// cost of 9 must not leak into the budget: 100 - 0 = 100.
			name:   "warmup isolation ignores measured marker cost",
			target: 100,
			comp:   NewISLCompensation(CacheBustWarmupIsolationFirstTurn, false, false, 9, 0, 0),
			turns:  2,
			want:   []int{100, 100},
		},
		{
			// --apply-chat-template opt-out: measured overheads are zeroed;
			// only the first-turn marker applies: 100 - 2 = 98, later 100.
			name:   "chat template opt-out zeroes measured overheads",
			target: 100,
			comp:   NewISLCompensation(CacheBustFirstTurnSuffix, false, false, 2, 5, 3),
			turns:  2,
			want:   []int{98, 100},
		},
		{
			// Over-compensation floors at 1 (upstream's sampler floors draws
			// at 1): 5 - (3 + 2 + 4) = -4 -> 1; later 5 - 2 = 3.
			name:   "budget floors at one token",
			target: 5,
			comp:   NewISLCompensation(CacheBustFirstTurnSuffix, false, true, 4, 3, 2),
			turns:  2,
			want:   []int{1, 3},
		},
		{
			// Single-turn request: only the first-turn adjustment applies.
			name:   "single turn",
			target: 64,
			comp:   NewISLCompensation(CacheBustFirstTurnPrefix, false, true, 1, 2, 1),
			turns:  1,
			want:   []int{60},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CompensateISLBudget(tc.target, tc.comp, tc.turns)
			if len(got) != len(tc.want) {
				t.Fatalf("turns=%d: got %d budgets %v, want %d %v", tc.turns, len(got), got, len(tc.want), tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("turn %d: budget=%d, want %d (target=%d, adjustment first=%d subsequent=%d)",
						i, got[i], tc.want[i], tc.target, tc.comp.FirstTurnAdjustment(), tc.comp.SubsequentTurnAdjustment())
				}
			}
			// The composed workload must respect the ISL constraint: when the
			// adjustment fits, the compensated budget plus its own
			// wrapping/marker costs must exactly match the target; when it
			// does not (degenerate config), the budget floors at 1, mirroring
			// the source's downstream sampler.
			for i, b := range got {
				adj := tc.comp.SubsequentTurnAdjustment()
				if i == 0 {
					adj = tc.comp.FirstTurnAdjustment()
				}
				if tc.target-adj >= 1 {
					if b+adj != tc.target {
						t.Fatalf("turn %d: wire ISL %d, want exactly target %d", i, b+adj, tc.target)
					}
				} else if b != 1 {
					t.Fatalf("turn %d: degenerate config (adjustment %d > target %d): budget=%d, want floor 1", i, adj, tc.target, b)
				}
			}
		})
	}
}

// TestISLBudgetCompensation_Workload verifies the set-of-requests path:
// per-turn budgets and the component (c) shared-system-prompt shrink.
func TestISLBudgetCompensation_Workload(t *testing.T) {
	// SYSTEM_SUFFIX marker with a system prompt: marker on the system
	// message (cost 6), chat template fixed 4 / wrap 2.
	comp := NewISLCompensation(CacheBustSystemSuffix, true, true, 6, 4, 2)
	if !comp.MarkerOnSystemPrompt || comp.MarkerOnFirstUserTurn {
		t.Fatalf("routing: MarkerOnSystemPrompt=%v MarkerOnFirstUserTurn=%v, want (true,false)",
			comp.MarkerOnSystemPrompt, comp.MarkerOnFirstUserTurn)
	}

	requests := []ISLRequest{
		{Turns: 2, SharedSystemLength: 100},
		{Turns: 1},
	}
	got := CompensateWorkload(500, comp, requests)

	// Request 0: first turn 500 - (4 + 2) = 494, second 500 - 2 = 498;
	// shared system prompt shrinks 100 - 6 = 94 so its wire length (94 + 6)
	// still matches the configured 100.
	if len(got) != 2 {
		t.Fatalf("got %d compensated requests, want 2", len(got))
	}
	if want := []int{494, 498}; len(got[0].TurnBudgets) != 2 || got[0].TurnBudgets[0] != want[0] || got[0].TurnBudgets[1] != want[1] {
		t.Fatalf("request 0 turn budgets = %v, want %v", got[0].TurnBudgets, want)
	}
	if got[0].SharedSystemLength != 94 {
		t.Fatalf("request 0 shared system length = %d, want 94 (100 - 6 marker)", got[0].SharedSystemLength)
	}
	// Request 1: single turn 500 - 6 = 494; no shared system prompt.
	if len(got[1].TurnBudgets) != 1 || got[1].TurnBudgets[0] != 494 {
		t.Fatalf("request 1 turn budgets = %v, want [494]", got[1].TurnBudgets)
	}
	if got[1].SharedSystemLength != 0 {
		t.Fatalf("request 1 shared system length = %d, want 0 (none configured)", got[1].SharedSystemLength)
	}

	// Component (c) floor: a configured length smaller than the marker cost
	// shrinks to 1, never 0 or negative.
	if v, ok := comp.CompensatedSharedSystemLength(3); !ok || v != 1 {
		t.Fatalf("CompensatedSharedSystemLength(3) = (%d,%v), want (1,true)", v, ok)
	}
	// No compensation when the marker is not on the system prompt.
	compFirstTurn := NewISLCompensation(CacheBustFirstTurnPrefix, true, true, 6, 4, 2)
	if v, ok := compFirstTurn.CompensatedSharedSystemLength(100); ok || v != 100 {
		t.Fatalf("first-turn marker: CompensatedSharedSystemLength(100) = (%d,%v), want (100,false)", v, ok)
	}
	// No configured length (0 stands in for None): passes through unchanged.
	if v, ok := comp.CompensatedSharedSystemLength(0); ok || v != 0 {
		t.Fatalf("CompensatedSharedSystemLength(0) = (%d,%v), want (0,false)", v, ok)
	}
}

// TestEstimateChatTemplateOverheads pins the probe formula against
// hand-computed values from _estimate_chat_template_overheads:
//
//	avg_wrap = (triple - single - 2*bare) / 2
//	per_request_fixed = single - bare - avg_wrap - bos
//
// and the defensive (0, 0) result on an implausible probe.
func TestEstimateChatTemplateOverheads(t *testing.T) {
	// Probe 1: wrap = (380 - 130 - 100)/2 = 75; fixed = 130 - 50 - 75 - 1 = 4.
	// Probe 2: wrap = (388 - 134 - 104)/2 = 75; fixed = 134 - 52 - 75 - 1 = 6.
	// Means: fixed = (4 + 6)/2 = 5; wrap = 75.
	fixed, wrap := EstimateChatTemplateOverheads([]ChatTemplateProbe{
		{SingleLen: 130, TripleLen: 380, BareLen: 50},
		{SingleLen: 134, TripleLen: 388, BareLen: 52},
	}, 1)
	if fixed != 5 || wrap != 75 {
		t.Fatalf("overheads = (%d,%d), want (5,75)", fixed, wrap)
	}

	// Implausible probe: negative per-request fixed cost (single - bare -
	// wrap - bos = 100 - 50 - 50 - 1 = -1) -> skip compensation entirely.
	if fixed, wrap := EstimateChatTemplateOverheads([]ChatTemplateProbe{
		{SingleLen: 100, TripleLen: 300, BareLen: 50},
	}, 1); fixed != 0 || wrap != 0 {
		t.Fatalf("negative-fixed probe = (%d,%d), want (0,0)", fixed, wrap)
	}

	// Implausible probe: negative wrap ((12 - 10 - 100)/2 = -49) -> (0, 0).
	if fixed, wrap := EstimateChatTemplateOverheads([]ChatTemplateProbe{
		{SingleLen: 10, TripleLen: 12, BareLen: 50},
	}, 0); fixed != 0 || wrap != 0 {
		t.Fatalf("negative-wrap probe = (%d,%d), want (0,0)", fixed, wrap)
	}

	// No probes (no usable chat template) -> (0, 0).
	if fixed, wrap := EstimateChatTemplateOverheads(nil, 0); fixed != 0 || wrap != 0 {
		t.Fatalf("no probes = (%d,%d), want (0,0)", fixed, wrap)
	}
}
