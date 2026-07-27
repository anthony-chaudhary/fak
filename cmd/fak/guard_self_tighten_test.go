package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// TestAdmitSelfTightenAllowsNoOp: an identical proposed floor is a no-op and is
// self-admissible.
func TestAdmitSelfTightenAllowsNoOp(t *testing.T) {
	cur := adjudicator.Policy{Allow: map[string]bool{"read_file": true}}
	admit, class, _ := admitSelfTightenOverlay(cur, cur)
	if !admit || class != policy.AmendmentNone {
		t.Fatalf("no-op must admit as AmendmentNone, got admit=%v class=%q", admit, class)
	}
}

// TestAdmitSelfTightenAllowsTightenOnly: adding a Deny entry only narrows the
// floor, so a self-authored channel may apply it.
func TestAdmitSelfTightenAllowsTightenOnly(t *testing.T) {
	cur := adjudicator.Policy{}
	proposed := adjudicator.Policy{Deny: map[string]abi.ReasonCode{"dangerous_tool": abi.ReasonPolicyBlock}}
	admit, class, reason := admitSelfTightenOverlay(cur, proposed)
	if !admit || class != policy.AmendmentTighten {
		t.Fatalf("added deny must admit as AmendmentTighten, got admit=%v class=%q (%s)", admit, class, reason)
	}
}

// TestAdmitSelfTightenRefusesWiden: adding an Allow entry widens the floor and
// must be refused on the agent's own authority.
func TestAdmitSelfTightenRefusesWiden(t *testing.T) {
	cur := adjudicator.Policy{Allow: map[string]bool{"read_file": true}}
	proposed := adjudicator.Policy{Allow: map[string]bool{"read_file": true, "write_file": true}}
	admit, class, reason := admitSelfTightenOverlay(cur, proposed)
	if admit {
		t.Fatalf("added allow (widening) must be refused, got admit=true (%s)", reason)
	}
	if class != policy.AmendmentWiden {
		t.Fatalf("want class AmendmentWiden, got %q", class)
	}
}

// TestAdmitSelfTightenRefusesLoosenedPosture: fail_closed -> admit_and_log is a
// widening of the default posture and must be refused.
func TestAdmitSelfTightenRefusesLoosenedPosture(t *testing.T) {
	cur := adjudicator.Policy{Posture: adjudicator.PostureFailClosed}
	proposed := adjudicator.Policy{Posture: adjudicator.PostureAdmitAndLog}
	admit, class, reason := admitSelfTightenOverlay(cur, proposed)
	if admit {
		t.Fatalf("loosened posture must be refused, got admit=true (%s)", reason)
	}
	if class != policy.AmendmentWiden {
		t.Fatalf("want class AmendmentWiden for loosened posture, got %q", class)
	}
}
