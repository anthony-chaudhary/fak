package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestGuardInfoDefaultViewCapturesFleetStatus(t *testing.T) {
	v := guardInfoVars{Adjudication: &gateway.AdjudicationSummary{}, Fleet: &gateway.SessionFleet{
		Verdict: "ACTION", Machines: 3, Sessions: 8,
		HealthySeats: 3, SeatCapacity: 4, Stale: 1, Action: 1,
		AuthBlocked: 1, ThrottledSeats: 2, ResumeBacklog: 3,
		VersionMismatches: 1, HostLoad: 2.5,
	}}

	// The identity row is pinned in the default overview, so fleet posture stays visible even
	// when the panel body scrolls. This is the captured render witness for the default surface.
	got := renderGuardInfoVisualBlock(v, newGuardInfoTrend(4), 120, 8)
	for _, want := range []string{"fleet ACTION", "machines 3", "sessions 8", "seats 3/4 healthy", "attention 2"} {
		if !strings.Contains(got, want) {
			t.Fatalf("default guard info frame missing %q:\n%s", want, got)
		}
	}
}

func TestRenderInfoAgentsViewIncludesFleetMachinesAndSessions(t *testing.T) {
	v := guardInfoVars{
		Fleet: &gateway.SessionFleet{
			Verdict: "ACTION", Machines: 2, Sessions: 5,
			Rows: []gateway.SessionFleetMachine{
				{ID: "gpu-1", State: "OK", Sessions: 3, Version: "r42+gabc"},
				{ID: "control", State: "ACTION", AgeMin: 1.5, Sessions: 2, Version: "r43+gdef"},
			},
		},
		Sessions: []guardInfoSession{{TraceID: "agent-1", Run: "working"}},
	}
	got := strings.Join(renderInfoAgentsView(v), "\n")
	for _, want := range []string{"fleet ACTION", "machines 2", "sessions 5", "control", "ACTION", "age 1.5m", "r43+gdef", "gpu-1", "sessions 3", "r42+gabc", "agents:", "agent-1"} {
		if !strings.Contains(got, want) {
			t.Fatalf("expanded agents frame missing %q:\n%s", want, got)
		}
	}
	if strings.Index(got, "control") > strings.Index(got, "gpu-1") {
		t.Fatalf("attention machine should sort first:\n%s", got)
	}
}

func TestRenderInfoAgentsViewShowsFleetWithoutLocalSessions(t *testing.T) {
	v := guardInfoVars{Adjudication: &gateway.AdjudicationSummary{}, Fleet: &gateway.SessionFleet{Verdict: "OK", Machines: 1, Rows: []gateway.SessionFleetMachine{{ID: "control", State: "OK"}}}}
	got := strings.Join(renderInfoAgentsView(v), "\n")
	if !strings.Contains(got, "fleet OK") || !strings.Contains(got, "agents: none running") {
		t.Fatalf("fleet-only frame lost fleet or empty-session state:\n%s", got)
	}
}
