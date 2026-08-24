package devcmd

import (
	"slices"
	"testing"
	"time"
)

func healthyGateFixture() (hookCensusReport, hookProfileReport) {
	now := time.Now()
	c := hookCensusReport{GeneratedAt: now, Window: "15m", ProfileMatch: true, DispatchedCalls: 2, TelemetryFresh: true, PreToolUse: lifecycleCounts{Denominator: 2, Attempted: 2, Succeeded: 2}, PostToolUseRequirement: "optional", PostToolUseStatus: "intentionally_disabled", StopSource: "hook-notifications.jsonl"}
	p := hookProfileReport{Hooks: mandatoryEffectiveHooks()}
	applyCodexHookContract(&p)
	return c, p
}
func TestEvaluateHookGatePassesZeroPostToolUseWithMandatoryEnforcementHooks(t *testing.T) {
	c, p := healthyGateFixture()
	g := evaluateHookRecurrence(c, p, codexToolErrorSummary{Categories: map[string]int{}}, 0, 0)
	if g.Verdict != "PASS" || g.UnexpectedDenominator != 2 || g.Census.PostToolUseStatus != "intentionally_disabled" {
		t.Fatalf("gate=%+v", g)
	}
}
func TestEvaluateHookGateFailsEveryClosureHazard(t *testing.T) {
	tests := map[string]func(*hookCensusReport, *hookProfileReport, *codexToolErrorSummary){"unknown": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) { c.PreToolUse.Unknown = 1 }, "disabled": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) {
		c.PostToolUseStatus = "disabled"
		c.PostToolUse.Disabled = 1
	}, "stale": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) { c.TelemetryFresh = false }, "profile mismatch": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) { c.ProfileMatch = false }, "hook failure": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) {
		c.PostToolUseStatus = "failing"
		c.PostToolUse.Failed = 1
	}, "unexpected": func(_ *hookCensusReport, _ *hookProfileReport, o *codexToolErrorSummary) { o.OutcomeErrors = 1 }}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			c, p := healthyGateFixture()
			o := codexToolErrorSummary{Categories: map[string]int{}}
			mutate(&c, &p, &o)
			if got := evaluateHookRecurrence(c, p, o, 0, 0); got.Verdict != "FAIL" {
				t.Fatalf("gate=%+v", got)
			}
		})
	}
}

func TestEvaluateHookGateRejectsPresentStalePostToolUse(t *testing.T) {
	c, p := healthyGateFixture()
	p.Hooks = append(p.Hooks, effectiveHook{EventName: "post_tool_use", Enabled: true, State: "stale_hash"})
	applyCodexHookContract(&p)
	g := evaluateHookRecurrence(c, p, codexToolErrorSummary{}, 0, 0)
	if g.Verdict != "FAIL" || !slices.Contains(g.Reasons, "EFFECTIVE_PROFILE_UNHEALTHY") {
		t.Fatalf("gate=%+v", g)
	}
}

func TestEvaluateHookGateRejectsPresentPostToolUseWithoutReceipt(t *testing.T) {
	c, _ := healthyGateFixture()
	p := effectivePostToolProfile()
	c.PostToolUseStatus = "incomplete"
	c.PostToolUse = lifecycleCounts{Denominator: 2, Unknown: 2}
	g := evaluateHookRecurrence(c, p, codexToolErrorSummary{}, 0, 0)
	if g.Verdict != "FAIL" || !slices.Contains(g.Reasons, "UNKNOWN_LIFECYCLE_ROWS") {
		t.Fatalf("gate=%+v", g)
	}
}

func TestEvaluateHookGateRejectsFailingPresentPostToolUseReceipt(t *testing.T) {
	c, p := healthyGateFixture()
	c.PostToolUseStatus = "failing"
	c.PostToolUse = lifecycleCounts{Denominator: 1, Attempted: 1, Failed: 1}
	g := evaluateHookRecurrence(c, p, codexToolErrorSummary{}, 0, 0)
	if g.Verdict != "FAIL" || !slices.Contains(g.Reasons, "HOOK_FAILURE_BUDGET_EXCEEDED") {
		t.Fatalf("gate=%+v", g)
	}
}

func TestEvaluateHookRecurrenceFailsClosedOnStopStates(t *testing.T) {
	base := hookCensusReport{Window: "15m", ProfileMatch: true, TelemetryFresh: true, DispatchedCalls: 1, PreToolUse: lifecycleCounts{Denominator: 1, Succeeded: 1}, PostToolUse: lifecycleCounts{Denominator: 1, Succeeded: 1}}
	profile := hookProfileReport{Verdict: "HEALTHY"}
	cases := []struct {
		name   string
		mutate func(*hookCensusReport)
		reason string
	}{
		{"invalid-json", func(c *hookCensusReport) { c.Stop.InvalidJSON = 1; c.Stop.Denominator = 1; c.Stop.Attempted = 1 }, "STOP_INVALID_JSON"},
		{"failed", func(c *hookCensusReport) {
			c.StopFailure.Failed = 1
			c.StopFailure.Denominator = 1
			c.StopFailure.Attempted = 1
		}, "HOOK_FAILURE_BUDGET_EXCEEDED"},
		{"unknown", func(c *hookCensusReport) {
			c.SubagentStop.Unknown = 1
			c.SubagentStop.Denominator = 1
			c.SubagentStop.Attempted = 1
		}, "UNKNOWN_LIFECYCLE_ROWS"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := base
			tc.mutate(&c)
			g := evaluateHookRecurrence(c, profile, codexToolErrorSummary{}, 0, 0)
			if g.Verdict != "FAIL" || !slices.Contains(g.Reasons, tc.reason) {
				t.Fatalf("gate=%+v", g)
			}
		})
	}
}

func TestEvaluateHookRecurrenceAllowsIntentionalStopBlock(t *testing.T) {
	c := hookCensusReport{Window: "15m", ProfileMatch: true, TelemetryFresh: true, DispatchedCalls: 1, PreToolUse: lifecycleCounts{Denominator: 1, Succeeded: 1}, PostToolUse: lifecycleCounts{Denominator: 1, Succeeded: 1}, Stop: lifecycleCounts{Denominator: 1, Attempted: 1, Blocked: 1}}
	g := evaluateHookRecurrence(c, hookProfileReport{Verdict: "HEALTHY"}, codexToolErrorSummary{}, 0, 0)
	if g.Verdict != "PASS" {
		t.Fatalf("gate=%+v", g)
	}
}
