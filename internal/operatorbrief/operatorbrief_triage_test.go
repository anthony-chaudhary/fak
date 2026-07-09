package operatorbrief

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// TestTriageHumanBucketKeepsAuthorityRoutesRest drives the real Report object:
// a genuine authority decision stays in the human bucket and keeps paging, while
// a "cadence incomplete" page — whose action is a runnable `fak cadence` rerun —
// is folded off to the fleet as TAKE_OBVIOUS and stops contributing to the gate.
func TestTriageHumanBucketKeepsAuthorityRoutesRest(t *testing.T) {
	r := Report{Schema: Schema}
	r.addHuman("release", "decision", "release decision needed", "approve the tagged build before publish", "approve the release")
	r.addHuman("cadence", "page", "cadence incomplete", "cadence report incomplete", "repair scores, then rerun `fak cadence`")

	if code, _ := CheckGate(r); code != 1 {
		t.Fatalf("pre-triage gate should page on 2 human items, got exit %d", code)
	}

	triaged, moved := TriageHumanBucket(r)

	if len(triaged.Human) != 1 || triaged.Human[0].Source != "release" {
		t.Fatalf("want only the release authority decision to remain human, got %+v", triaged.Human)
	}
	if triaged.Counts.Human != 1 || triaged.Counts.Agent != 1 {
		t.Fatalf("counts should follow the reconciled buckets, got %+v", triaged.Counts)
	}
	if len(moved) != 1 || moved[0].Source != "cadence" {
		t.Fatalf("want the cadence page routed to the fleet, got %+v", moved)
	}
	if moved[0].Disposition != choicetriage.TakeObvious {
		t.Fatalf("a runnable rerun is TAKE_OBVIOUS, got %s", moved[0].Disposition)
	}
	// The routed item lands in the agent bucket carrying its runnable action.
	if len(triaged.Agent) != 1 || triaged.Agent[0].Bucket != "agent" || triaged.Agent[0].Action != "repair scores, then rerun `fak cadence`" {
		t.Fatalf("routed item should carry its runnable action into the agent bucket, got %+v", triaged.Agent)
	}
	if code, _ := CheckGate(triaged); code != 1 {
		t.Fatalf("gate should still page on the residual authority decision, got exit %d", code)
	}
}

// TestFoldTriageGateWarnVsEnforce proves the soak switch end-to-end through Fold:
// a missing/unmeasured source pages the operator under the default (warn), but
// under enforce the same page routes to the fleet and the brief gates clean.
func TestFoldTriageGateWarnVsEnforce(t *testing.T) {
	c := cleanCadence()
	c.OK, c.Verdict, c.Finding = false, "ACTION", "cadence_unmeasured"
	c.Reason = "cadence report incomplete - could not measure scores"
	c.NextAction = "repair scores, then rerun `fak cadence`"
	p := cleanProgram()
	m := cleanMilestone()

	warn := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m})
	if warn.OK || warn.Pace != "intervene" || len(warn.Human) != 1 {
		t.Fatalf("warn (default) should page on the unmeasured source, got OK=%v pace=%q human=%+v", warn.OK, warn.Pace, warn.Human)
	}
	if code, _ := CheckGate(warn); code != 1 {
		t.Fatalf("warn gate should page, got exit %d", code)
	}

	enforce := Fold(Inputs{Cadence: &c, Program: &p, Milestone: &m, TriageGate: "enforce"})
	if len(enforce.Human) != 0 {
		t.Fatalf("enforce should route the runnable page off the human bucket, got %+v", enforce.Human)
	}
	if !enforce.OK || enforce.Pace != "delegate" {
		t.Fatalf("enforce should gate clean and delegate, got OK=%v pace=%q", enforce.OK, enforce.Pace)
	}
	if code, _ := CheckGate(enforce); code != 0 {
		t.Fatalf("enforce gate should not page a runnable rerun, got exit %d", code)
	}
}

// TestTriageSelfcheckPasses runs the package's own no-I/O invariant proof.
func TestTriageSelfcheckPasses(t *testing.T) {
	if err := TriageSelfcheck(); err != nil {
		t.Fatalf("TriageSelfcheck: %v", err)
	}
}
