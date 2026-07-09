package trendreport

import "testing"

// TestTriageSelfcheck runs the packaged no-I/O proof.
func TestTriageSelfcheck(t *testing.T) {
	if err := TriageSelfcheck(); err != nil {
		t.Fatalf("TriageSelfcheck: %v", err)
	}
}

// TestAdvisoryGateTriagedSoak proves the soak switch: the same incomplete report
// pages under warn and routes to the fleet under enforce when its NextAction is a
// runnable rerun.
func TestAdvisoryGateTriagedSoak(t *testing.T) {
	warn := AdvisoryGateTriaged("MILESTONE", "milestone_unmeasured", "roadmap unreadable", "regenerate `fak milestone report --json`", "milestone_unmeasured", false)
	if warn.Exit != 1 {
		t.Fatalf("warn: an incomplete report should page, got exit %d", warn.Exit)
	}
	enforce := AdvisoryGateTriaged("MILESTONE", "milestone_unmeasured", "roadmap unreadable", "regenerate `fak milestone report --json`", "milestone_unmeasured", true)
	if enforce.Exit != 0 {
		t.Fatalf("enforce: a runnable rerun should route to the fleet, got exit %d (%s)", enforce.Exit, enforce.Message)
	}
	if enforce.Message == "" || enforce.Message == warn.Message {
		t.Fatalf("enforce should name the route, got %q", enforce.Message)
	}
}

// TestTriageEnvelopeDisposition proves the source-level disposition: a runnable
// envelope is not human-residual; an authority-bearing one is.
func TestTriageEnvelopeDisposition(t *testing.T) {
	routable := TriageEnvelope("CADENCE", Envelope{Finding: "cadence_unmeasured", Reason: "incomplete", NextAction: "rerun `fak cadence`"})
	if routable.NeedsHuman {
		t.Fatalf("a runnable rerun must not need a person, got %+v", routable)
	}
	authority := TriageEnvelope("RELEASE", Envelope{Finding: "release_pending", Reason: "publish decision pending", NextAction: "approve the release before publish"})
	if !authority.NeedsHuman {
		t.Fatalf("an authority decision must need a person, got %+v", authority)
	}
}
