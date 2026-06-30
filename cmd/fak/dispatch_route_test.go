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
		"skipped=2",
		"skipped: 2 (BLOCKED_BY_HUMAN=1, ISSUE_NOT_DISPATCH_LEAF=1)",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered route missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "skipped-human-blocked") {
		t.Fatalf("rendered route kept legacy skipped-human wording:\n%s", out)
	}
}
