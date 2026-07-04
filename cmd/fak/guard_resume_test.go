package main

import (
	"strings"
	"testing"
)

// formatGuardResumeGuidance is printed when the wrapped agent exits abnormally. It must name
// the agent and exit code, point at the same-command resume, surface the replayable decision
// journal, and carry the bare-resume recovery for the "upstream model error" failure mode —
// so an operator whose `fak guard -- claude` session crashed knows exactly how to get back in.
func TestFormatGuardResumeGuidance(t *testing.T) {
	out := formatGuardResumeGuidance("claude", 1)
	for _, want := range []string{
		"claude",               // the wrapped agent is named
		"code 1",               // the abnormal exit code is surfaced
		"fak guard --",         // the resume re-run command
		"--continue",           // the agent's own resume/continue flag
		"fak audit verify",     // the journal is replayable
		"WITHOUT fak guard",    // the bare-resume recovery
		"upstream model error", // the specific failure it recovers
	} {
		if !strings.Contains(out, want) {
			t.Errorf("guidance missing %q\n--- guidance ---\n%s", want, out)
		}
	}
}

func TestFormatGuardResumeGuidanceSurfacesGuardActivity(t *testing.T) {
	out := formatGuardResumeGuidanceWithRefusals("claude", 1, []guardRefusalCarry{{
		Reason: "OFF_TRUNK",
		Count:  2,
		Fix:    "commit directly to main",
	}})
	for _, want := range []string{
		"guard activity",
		"recovery/debugging",
		"OFF_TRUNK x2",
		"commit directly to main",
		"do not retry the same refused call unchanged",
		"WITHOUT fak guard",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("guidance missing %q\n--- guidance ---\n%s", want, out)
		}
	}
}
