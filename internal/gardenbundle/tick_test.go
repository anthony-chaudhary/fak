package gardenbundle

import "testing"

// mkResult is a tiny MemberResult fixture for the tick planner tests.
func mkResult(key, state string) MemberResult {
	return MemberResult{Key: key, Label: key, State: state, Detail: key + " detail"}
}

// decisionFor returns the decision for a member key, or a zero decision if absent.
func decisionFor(p TickPlan, key string) ActDecision {
	for _, d := range p.Decisions {
		if d.Key == key {
			return d
		}
	}
	return ActDecision{}
}

// TestPlanTickActsOnPastThresholdLeaseAndOrphan proves that a stale_lease and an
// orphaned_runs member that surfaced an ACTION condition are ACTED on: reap for the
// lease, surface for the orphans.
func TestPlanTickActsOnPastThresholdLeaseAndOrphan(t *testing.T) {
	results := []MemberResult{
		mkResult("stale_leases", "action"),
		mkResult("orphaned_runs", "action"),
	}
	plan := PlanTick(results, false)

	lease := decisionFor(plan, "stale_leases")
	if lease.Act != ActReap || !lease.Perform {
		t.Fatalf("stale_leases: want reap+perform, got act=%s perform=%v", lease.Act, lease.Perform)
	}
	orphan := decisionFor(plan, "orphaned_runs")
	if orphan.Act != ActSurface || !orphan.Perform {
		t.Fatalf("orphaned_runs: want surface+perform, got act=%s perform=%v", orphan.Act, orphan.Perform)
	}
	if plan.ToReap != 1 || plan.ToSurface != 1 {
		t.Fatalf("want ToReap=1 ToSurface=1, got %d/%d", plan.ToReap, plan.ToSurface)
	}
	if !plan.Acted() {
		t.Fatalf("Acted() should be true when the tick reaps + surfaces")
	}
}

// TestPlanTickLeavesFreshLiveUntouched proves the safety invariant: a member whose
// state is "ok" (a fresh/live lease, a run with nothing to recover) is never acted
// on, even though its key has a registered remediation.
func TestPlanTickLeavesFreshLiveUntouched(t *testing.T) {
	results := []MemberResult{
		mkResult("stale_leases", "ok"),
		mkResult("orphaned_runs", "ok"),
	}
	plan := PlanTick(results, false)

	for _, key := range []string{"stale_leases", "orphaned_runs"} {
		d := decisionFor(plan, key)
		if d.Perform {
			t.Fatalf("%s: an ok member must not be acted on, got perform=%v act=%s", key, d.Perform, d.Act)
		}
		if d.Act != ActNone {
			t.Fatalf("%s: an ok member must resolve to ActNone, got %s", key, d.Act)
		}
	}
	if plan.ToReap != 0 || plan.ToSurface != 0 || plan.Acted() {
		t.Fatalf("an all-ok tick must act on nothing, got ToReap=%d ToSurface=%d acted=%v", plan.ToReap, plan.ToSurface, plan.Acted())
	}
}

// TestPlanTickDryRunActsOnNothing proves that --dry-run on the SAME surfaced
// conditions performs no side effect: every decision is Perform=false / Mode="dry-run".
func TestPlanTickDryRunActsOnNothing(t *testing.T) {
	results := []MemberResult{
		mkResult("stale_leases", "action"),
		mkResult("orphaned_runs", "action"),
	}
	plan := PlanTick(results, true)

	if !plan.DryRun {
		t.Fatalf("plan.DryRun should be true")
	}
	for _, d := range plan.Decisions {
		if d.Perform {
			t.Fatalf("%s: dry-run must perform nothing, got perform=true", d.Key)
		}
		if d.Mode != "dry-run" {
			t.Fatalf("%s: dry-run decision mode = %q, want dry-run", d.Key, d.Mode)
		}
	}
	if plan.ToReap != 0 || plan.ToSurface != 0 || plan.Acted() {
		t.Fatalf("dry-run must act on nothing, got ToReap=%d ToSurface=%d acted=%v", plan.ToReap, plan.ToSurface, plan.Acted())
	}
}

// TestPlanTickReleaseStalenessAdvisoryOnly proves release_staleness, even when RED,
// is reported as advisory and never auto-acted (its remediation is the release path, #1367).
func TestPlanTickReleaseStalenessAdvisoryOnly(t *testing.T) {
	plan := PlanTick([]MemberResult{mkResult("release_staleness", "red")}, false)
	d := decisionFor(plan, "release_staleness")
	if d.Perform {
		t.Fatalf("release_staleness must never be auto-acted, got perform=true")
	}
	if d.Act != ActAdvisory {
		t.Fatalf("release_staleness: want advisory, got %s", d.Act)
	}
	if plan.Advisory != 1 {
		t.Fatalf("want Advisory=1, got %d", plan.Advisory)
	}
}

// TestPlanTickErroredMemberNotActed proves an errored member (could not measure) is
// never acted on, even if its key carries a remediation — acting on an unmeasured
// condition would be unsafe.
func TestPlanTickErroredMemberNotActed(t *testing.T) {
	plan := PlanTick([]MemberResult{mkResult("stale_leases", "errored")}, false)
	d := decisionFor(plan, "stale_leases")
	if d.Perform || d.Act != ActNone {
		t.Fatalf("errored member must not be acted on, got perform=%v act=%s", d.Perform, d.Act)
	}
}

// TestPlanTickUnknownMemberReportedNotActed proves a surfaced member with no
// registered remediation is reported but never acted on.
func TestPlanTickUnknownMemberReportedNotActed(t *testing.T) {
	plan := PlanTick([]MemberResult{mkResult("scorecard", "action")}, false)
	d := decisionFor(plan, "scorecard")
	if d.Perform || d.Act != ActNone {
		t.Fatalf("a member with no remediation must not be acted on, got perform=%v act=%s", d.Perform, d.Act)
	}
}
