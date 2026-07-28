package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// TestPolicyDiffReportIdenticalIsNoDrift: an effective floor byte-equal to the
// shipped floor reports no drift and is not gate-able.
func TestPolicyDiffReportIdenticalIsNoDrift(t *testing.T) {
	floor := adjudicator.Policy{Allow: map[string]bool{"read_file": true}}
	lines, gateable := policyDiffReport(floor, floor)
	if gateable {
		t.Fatal("identical floors must not be gate-able")
	}
	if len(lines) != 1 || !strings.Contains(lines[0], "none") {
		t.Fatalf("want single 'none' line, got %v", lines)
	}
}

// TestPolicyDiffReportAddedAllowIsWidenDrift: adding an Allow entry to the
// effective floor is a WIDENING drift and must trip the gate-able signal.
func TestPolicyDiffReportAddedAllowIsWidenDrift(t *testing.T) {
	floor := adjudicator.Policy{Allow: map[string]bool{"read_file": true}}
	effective := adjudicator.Policy{Allow: map[string]bool{"read_file": true, "write_file": true}}
	lines, gateable := policyDiffReport(floor, effective)
	if !gateable {
		t.Fatal("added_allow must be gate-able widen-drift")
	}
	joined := strings.Join(lines, "\n")
	if !strings.Contains(joined, "WIDENED") || !strings.Contains(joined, "write_file") {
		t.Fatalf("want a WIDENED line naming write_file, got:\n%s", joined)
	}
}

// TestPolicyDiffReportAddedDenyIsTightenOnly: adding a Deny entry only tightens,
// so it is reported as drift but NOT gate-able (it never loosens the guard).
func TestPolicyDiffReportAddedDenyIsTightenOnly(t *testing.T) {
	floor := adjudicator.Policy{}
	effective := adjudicator.Policy{Deny: map[string]abi.ReasonCode{"dangerous_tool": abi.ReasonPolicyBlock}}
	lines, gateable := policyDiffReport(floor, effective)
	if gateable {
		t.Fatal("a tighten-only drift (added deny) must not be gate-able")
	}
	if !strings.Contains(strings.Join(lines, "\n"), "TIGHTENED") {
		t.Fatalf("want a TIGHTENED line, got %v", lines)
	}
}
