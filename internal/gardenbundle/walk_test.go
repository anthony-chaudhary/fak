package gardenbundle

import "testing"

// sampleItems builds a small mixed item set: two acts, one review, one healthy,
// one fresh-but-tagged (idle 1d). Scores chosen so worst-first ordering is testable.
func sampleItems() []WalkItem {
	return []WalkItem{
		{ID: 1, Title: "stale A", Score: 400, IdleDays: 90, Disposition: DispAct, Action: "mark-stale", Command: "gh issue edit 1 --add-label stale"},
		{ID: 2, Title: "dormant Q", Score: 120, IdleDays: 45, Disposition: DispAct, Action: "close-dormant-question", Command: "gh issue close 2"},
		{ID: 3, Title: "needs area", Score: 200, IdleDays: 30, Disposition: DispReview, Action: "review", Reason: "needs-area"},
		{ID: 4, Title: "healthy", Score: 60, IdleDays: 10, Disposition: DispSkip},
		{ID: 5, Title: "fresh tagged", Score: 999, IdleDays: 1, Disposition: DispAct, Action: "mark-stale"},
	}
}

// TestPlanWalkSkipFreshDropsLiveItems proves the cheap freshness pre-filter drops a
// just-touched item BEFORE the worklist, regardless of its (high) score.
func TestPlanWalkSkipFreshDropsLiveItems(t *testing.T) {
	plan := PlanWalk("issue", sampleItems(), WalkPolicy{SkipFreshDays: 3, DryRun: true})
	if plan.Fresh != 1 {
		t.Fatalf("Fresh = %d, want 1 (item 5 idle 1d)", plan.Fresh)
	}
	for _, d := range plan.Decisions {
		if d.ID == 5 {
			t.Fatalf("fresh item 5 (idle 1d) must not be on the worklist, score %d notwithstanding", d.Score)
		}
	}
	if plan.Healthy != 1 {
		t.Fatalf("Healthy = %d, want 1 (item 4)", plan.Healthy)
	}
	if plan.Attention != 3 {
		t.Fatalf("Attention = %d, want 3 (items 1,2,3 after fresh+healthy dropped)", plan.Attention)
	}
}

// TestPlanWalkWorstFirstOrder proves attention items are emitted worst-first by score.
func TestPlanWalkWorstFirstOrder(t *testing.T) {
	plan := PlanWalk("issue", sampleItems(), WalkPolicy{SkipFreshDays: 3, DryRun: true})
	if len(plan.Decisions) != 3 {
		t.Fatalf("want 3 decisions, got %d", len(plan.Decisions))
	}
	wantOrder := []int{1, 3, 2} // scores 400, 200, 120
	for i, want := range wantOrder {
		if plan.Decisions[i].ID != want {
			t.Fatalf("decision[%d].ID = %d, want %d (worst-first by score)", i, plan.Decisions[i].ID, want)
		}
	}
}

// TestPlanWalkBudgetBoundsWorklist proves the budget caps the emitted worklist and
// defers the rest — the resource bound that keeps a 100s-item set tractable.
func TestPlanWalkBudgetBoundsWorklist(t *testing.T) {
	plan := PlanWalk("issue", sampleItems(), WalkPolicy{Budget: 2, SkipFreshDays: 3, DryRun: true})
	if len(plan.Decisions) != 2 {
		t.Fatalf("want 2 decisions under budget 2, got %d", len(plan.Decisions))
	}
	if plan.Deferred != 1 {
		t.Fatalf("Deferred = %d, want 1 (3 attention - budget 2)", plan.Deferred)
	}
	// The two highest scores survive; the lowest (item 2, score 120) is deferred.
	if plan.Decisions[0].ID != 1 || plan.Decisions[1].ID != 3 {
		t.Fatalf("budgeted = [%d,%d], want [1,3] (the two worst)", plan.Decisions[0].ID, plan.Decisions[1].ID)
	}
}

// TestPlanWalkDryRunNeverPerforms proves dry-run forces every Perform=false even for
// act dispositions (propose-don't-execute).
func TestPlanWalkDryRunNeverPerforms(t *testing.T) {
	plan := PlanWalk("issue", sampleItems(), WalkPolicy{SkipFreshDays: 3, DryRun: true})
	for _, d := range plan.Decisions {
		if d.Perform {
			t.Fatalf("decision %d Perform=true under dry-run", d.ID)
		}
	}
	if !plan.DryRun {
		t.Fatalf("plan.DryRun = false, want true")
	}
}

// TestPlanWalkActPerformsWhenNotDryRun proves an act decision is Perform=true when the
// caller opts out of dry-run, while a review is never auto-performed.
func TestPlanWalkActPerformsWhenNotDryRun(t *testing.T) {
	plan := PlanWalk("issue", sampleItems(), WalkPolicy{SkipFreshDays: 3, DryRun: false})
	var sawActPerform, sawReview bool
	for _, d := range plan.Decisions {
		if d.Disposition == DispAct && d.Perform {
			sawActPerform = true
		}
		if d.Disposition == DispReview {
			sawReview = true
			if d.Perform {
				t.Fatalf("review decision %d Perform=true; review is never auto-performed", d.ID)
			}
		}
	}
	if !sawActPerform {
		t.Fatalf("no act decision had Perform=true under !dry-run")
	}
	if !sawReview {
		t.Fatalf("expected a review decision in the sample")
	}
}

// TestPlanWalkClearWhenNoAttention proves an all-healthy/fresh set folds to OK/clear.
func TestPlanWalkClearWhenNoAttention(t *testing.T) {
	items := []WalkItem{
		{ID: 1, Disposition: DispSkip, IdleDays: 100},
		{ID: 2, Disposition: DispAct, IdleDays: 1}, // fresh -> skipped
	}
	plan := PlanWalk("issue", items, WalkPolicy{SkipFreshDays: 3, DryRun: true})
	if plan.Attention != 0 {
		t.Fatalf("Attention = %d, want 0", plan.Attention)
	}
	if !plan.OK || plan.Verdict != "OK" || plan.Finding != "garden_walk_clear" {
		t.Fatalf("clear walk folded to ok=%v verdict=%q finding=%q", plan.OK, plan.Verdict, plan.Finding)
	}
}

// TestPlanWalkWorklistIsOKVerdict proves a surfaced worklist is OK (the pass working),
// not a red garden — only the verdict signals the advisory condition.
func TestPlanWalkWorklistIsOKVerdict(t *testing.T) {
	plan := PlanWalk("issue", sampleItems(), WalkPolicy{SkipFreshDays: 3, DryRun: true})
	if !plan.OK {
		t.Fatalf("walk with a worklist must stay OK=true (a backlog is not a broken garden)")
	}
	if plan.Verdict != "ACTION" || plan.Finding != "garden_walk_worklist" {
		t.Fatalf("worklist folded to verdict=%q finding=%q, want ACTION/garden_walk_worklist", plan.Verdict, plan.Finding)
	}
}

// TestPlanWalkUnboundedBudget proves Budget<=0 emits the whole attention set.
func TestPlanWalkUnboundedBudget(t *testing.T) {
	plan := PlanWalk("issue", sampleItems(), WalkPolicy{SkipFreshDays: 3, DryRun: true, Budget: 0})
	if plan.Deferred != 0 {
		t.Fatalf("Deferred = %d, want 0 under unbounded budget", plan.Deferred)
	}
	if len(plan.Decisions) != plan.Attention {
		t.Fatalf("emitted %d != attention %d under unbounded budget", len(plan.Decisions), plan.Attention)
	}
}

// TestPlanWalkSkipInProgressDropsActiveItems proves the in-progress pre-filter drops
// an actively-worked item BEFORE the worklist, even with skip-fresh off and a high
// score — the resource-saver that fires when timestamps are unreliable.
func TestPlanWalkSkipInProgressDropsActiveItems(t *testing.T) {
	items := []WalkItem{
		{ID: 1, Title: "active hot", Score: 999, IdleDays: 90, InProgress: true, Disposition: DispReview},
		{ID: 2, Title: "needs work", Score: 100, IdleDays: 90, Disposition: DispReview},
	}
	plan := PlanWalk("issue", items, WalkPolicy{SkipInProgress: true, DryRun: true})
	if plan.Active != 1 {
		t.Fatalf("Active = %d, want 1 (item 1 in-progress)", plan.Active)
	}
	if plan.Attention != 1 {
		t.Fatalf("Attention = %d, want 1 (item 2 only)", plan.Attention)
	}
	for _, d := range plan.Decisions {
		if d.ID == 1 {
			t.Fatalf("in-progress item 1 must not be on the worklist, score %d notwithstanding", d.Score)
		}
	}
	// With the filter off, the active item rejoins the worklist.
	off := PlanWalk("issue", items, WalkPolicy{SkipInProgress: false, DryRun: true})
	if off.Active != 0 || off.Attention != 2 {
		t.Fatalf("filter off: Active=%d Attention=%d, want 0/2", off.Active, off.Attention)
	}
}
