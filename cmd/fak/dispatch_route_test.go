package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestRenderDispatchRouteSummarizesSkippedReasons(t *testing.T) {
	out := renderDispatchRoute(dispatchtick.RouterPayload{
		OK:      true,
		Verdict: "OK",
		Reason:  "routed",
		Coverage: dispatchtick.RouterCoverage{
			Complete: true,
		},
		Counts: dispatchtick.RouterCounts{
			Routed:              2,
			RoutedStepBudget:    7,
			SkippedHumanBlocked: 2,
			SkippedByReason: map[string]int{
				"ISSUE_NOT_DISPATCH_LEAF": 1,
				"BLOCKED_BY_HUMAN":        1,
			},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"gateway": {
				Count:      2,
				StepBudget: 7,
				Issues:     []int{10, 11},
				SubLanes: []dispatchtick.RouterSubLane{
					{Prefix: "cmd/fak", Count: 1, StepBudget: 3, Issues: []int{10}},
					{Prefix: "internal/dispatchtick", Count: 1, StepBudget: 4, Issues: []int{11}},
				},
			},
		},
		RepairQueues: []dispatchtick.RouterRepairQueue{
			{
				Kind:       "dispatch",
				Count:      2,
				StepBudget: 7,
				NextAction: "launch scoped leaf issues through their routed lanes",
				Issues:     []int{10, 11},
			},
			{
				Kind:             "split",
				Count:            1,
				StepBudget:       1,
				ChildIssueBudget: 1,
				NextAction:       "decompose non-leaves or oversized rows into child issues",
				ByReason:         map[string]int{"ISSUE_NOT_DISPATCH_LEAF": 1},
				Issues:           []int{3},
			},
		},
		SkippedHumanBlocked: []dispatchtick.SkippedIssue{
			{Number: 2, Title: "blocked", Reason: "BLOCKED_BY_HUMAN"},
			{Number: 3, Title: "epic", Reason: "ISSUE_NOT_DISPATCH_LEAF"},
		},
	})
	for _, want := range []string{
		"routed=2 steps=7",
		"gateway            2 issue(s)   7 step(s): 10,11",
		"split cmd/fak                    1 issue(s)   3 step(s): 10",
		"split internal/dispatchtick      1 issue(s)   4 step(s): 11",
		"skipped=2",
		"skipped: 2 (BLOCKED_BY_HUMAN=1, ISSUE_NOT_DISPATCH_LEAF=1)",
		"repair_queue[dispatch]: 2 issue(s) 7 step(s) issues=10,11 next=launch scoped leaf issues through their routed lanes",
		"repair_queue[split]: 1 issue(s) 1 step(s) child_issues=1 issues=3 next=decompose non-leaves or oversized rows into child issues",
		"reasons: ISSUE_NOT_DISPATCH_LEAF=1",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered route missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "skipped-human-blocked") {
		t.Fatalf("rendered route kept legacy skipped-human wording:\n%s", out)
	}
}
