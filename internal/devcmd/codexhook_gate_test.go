package devcmd

import (
	"testing"
	"time"
)

func healthyGateFixture() (hookCensusReport, hookProfileReport) {
	now := time.Now()
	c := hookCensusReport{GeneratedAt: now, Window: "15m", ProfileMatch: true, DispatchedCalls: 2, TelemetryFresh: true, PreToolUse: lifecycleCounts{Denominator: 2, Attempted: 2, Succeeded: 2}, PostToolUse: lifecycleCounts{Denominator: 2, Attempted: 2, Succeeded: 2}}
	return c, hookProfileReport{Verdict: "HEALTHY"}
}
func TestEvaluateHookGatePassesZeroUnknown(t *testing.T) {
	c, p := healthyGateFixture()
	g := evaluateHookRecurrence(c, p, codexToolErrorSummary{Categories: map[string]int{}}, 0, 0)
	if g.Verdict != "PASS" || g.UnexpectedDenominator != 2 {
		t.Fatalf("gate=%+v", g)
	}
}
func TestEvaluateHookGateFailsEveryClosureHazard(t *testing.T) {
	tests := map[string]func(*hookCensusReport, *hookProfileReport, *codexToolErrorSummary){"unknown": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) { c.PreToolUse.Unknown = 1 }, "disabled": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) { c.PostToolUse.Disabled = 1 }, "stale": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) { c.TelemetryFresh = false }, "profile mismatch": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) { c.ProfileMatch = false }, "hook failure": func(c *hookCensusReport, _ *hookProfileReport, _ *codexToolErrorSummary) { c.PostToolUse.Failed = 1 }, "unexpected": func(_ *hookCensusReport, _ *hookProfileReport, o *codexToolErrorSummary) { o.OutcomeErrors = 1 }}
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
