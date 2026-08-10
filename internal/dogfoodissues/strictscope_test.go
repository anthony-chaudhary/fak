package dogfoodissues

import (
	"strings"
	"testing"
)

// liveStrictOptions is the armed --live shape: dedupe checked and capped, with
// the project-work contract satisfied, so a held row can only be held by the
// strict root-scope control under test.
func liveStrictOptions() BuildOptions {
	return BuildOptions{
		Live: true, DedupeChecked: true, DedupeCap: 300,
		ParentIssue: 36, ParentBaseline: 20, CompletionStandard: "development",
	}
}

func dryRunStrictOptions() BuildOptions {
	return BuildOptions{ParentIssue: 36, ParentBaseline: 20, CompletionStandard: "development"}
}

// A fully scoped row — root-point change, witness, lane AND path hints — is the
// only shape a live sync plans. This is the positive control for the three
// negative cases below.
func TestStrictScopeAcceptsFullyScopedLiveRow(t *testing.T) {
	plan, skipped := BuildPlanWithOptions([]ActionItem{scopedGuardActionItem()}, nil, liveStrictOptions())
	if len(skipped) != 0 {
		t.Fatalf("skipped = %+v, want none for a fully scoped live row", skipped)
	}
	if len(plan) != 1 {
		t.Fatalf("plan len = %d, want 1", len(plan))
	}
	if !plan[0].Review.OK || plan[0].Review.Dispatchability != "dispatchable" {
		t.Fatalf("review = %+v, want dispatchable", plan[0].Review)
	}
}

// The shared contract routes on lane OR paths. A live dogfood sync needs BOTH: a
// lane alone cannot tell a worker which files to touch, and path hints alone
// leave the lane lease unnameable. Each half missing must hold the row.
func TestStrictScopeHoldsLiveRowMissingLaneOrPathHints(t *testing.T) {
	for _, tc := range []struct {
		name       string
		mutate     func(*ActionItem)
		wantReason string
	}{
		{"lane missing", func(i *ActionItem) { i.Lane = "" }, ReasonLaneMissing},
		{"path hints missing", func(i *ActionItem) { i.Paths = nil }, ReasonPathHintsMissing},
		{"blank path hints", func(i *ActionItem) { i.Paths = []string{"  "} }, ReasonPathHintsMissing},
	} {
		t.Run(tc.name, func(t *testing.T) {
			item := scopedGuardActionItem()
			tc.mutate(&item)
			plan, skipped := BuildPlanWithOptions([]ActionItem{item}, nil, liveStrictOptions())
			if len(plan) != 0 {
				t.Fatalf("plan = %+v, want no dispatchable row", plan)
			}
			if len(skipped) != 1 {
				t.Fatalf("skipped len = %d, want 1", len(skipped))
			}
			if !strings.Contains(skipped[0].Reason, tc.wantReason) {
				t.Fatalf("skip reason = %q, want %s", skipped[0].Reason, tc.wantReason)
			}
			if skipped[0].Dispatchability != "triage_only" {
				t.Fatalf("dispatchability = %q, want triage_only", skipped[0].Dispatchability)
			}
			if skipped[0].Review.OK {
				t.Fatalf("held review must not stay OK: %+v", skipped[0].Review)
			}
		})
	}
}

// A live row's done condition has to be provable by an independent oracle. A
// self-reported witness is advisory on a dry-run and a hold on --live, so the
// same row flips verdict purely on the live opt-in.
func TestStrictScopeHoldsForgeableWitnessOnLiveOnly(t *testing.T) {
	item := scopedGuardActionItem()
	item.Witness = "the worker reports the honesty hole is closed"
	item.AcceptanceGate = item.Witness

	plan, skipped := BuildPlanWithOptions([]ActionItem{item}, nil, liveStrictOptions())
	if len(plan) != 0 || len(skipped) != 1 {
		t.Fatalf("live: plan=%+v skipped=%+v, want the forgeable witness held", plan, skipped)
	}
	if !strings.Contains(skipped[0].Reason, "ISSUE_WITNESS_FORGEABLE") {
		t.Fatalf("live skip reason = %q, want ISSUE_WITNESS_FORGEABLE", skipped[0].Reason)
	}

	dryPlan, drySkipped := BuildPlanWithOptions([]ActionItem{item}, nil, dryRunStrictOptions())
	if len(dryPlan) != 1 || len(drySkipped) != 0 {
		t.Fatalf("dry-run: plan=%+v skipped=%+v, want the witness grade advisory only", dryPlan, drySkipped)
	}
}

// The strict hold is a live control, not a new always-on contract: a dry-run
// still plans a lane-only row so operators can see what a scoped row would
// become before arming --live.
func TestStrictScopeStaysAdvisoryOnDryRun(t *testing.T) {
	item := scopedGuardActionItem()
	item.Paths = nil
	plan, skipped := BuildPlanWithOptions([]ActionItem{item}, nil, dryRunStrictOptions())
	if len(skipped) != 0 || len(plan) != 1 {
		t.Fatalf("plan=%+v skipped=%+v, want one advisory dry-run row", plan, skipped)
	}
}

// The cohort fold re-reviews candidates through issuecontract alone, which knows
// nothing about the lane-AND-paths hold. A row the plan refused must not come
// back as a dispatchable cohort leaf.
func TestStrictScopeHeldRowIsNotACohortLeaf(t *testing.T) {
	item := scopedGuardActionItem()
	item.Lane = "" // still routed by paths, so issuecontract alone would pass it
	opt := liveStrictOptions()
	plan, skipped := BuildPlanWithOptions([]ActionItem{item}, nil, opt)
	if len(plan) != 0 || len(skipped) != 1 {
		t.Fatalf("plan=%+v skipped=%+v, want the lane-less row held", plan, skipped)
	}
	if cohort := CohortPlan([]ActionItem{item}, opt); cohort.Dispatchable != 0 {
		t.Fatalf("cohort dispatchable = %d, want 0 to match the refused plan", cohort.Dispatchable)
	}
}

// The refusal message has to name the held keys and both ways forward, or the
// operator gets a nonzero exit with nothing to act on.
func TestStrictScopeRefusalMessageNamesHeldRowsAndRemedies(t *testing.T) {
	msg := StrictScopeRefusalMessage([]SkippedRow{{
		Key:    "recent-feature-dogfood/code-slop-scorecard/code_slop",
		Reason: "ISSUE_SCOPE_INCOMPLETE,ISSUE_UNROUTED",
	}})
	for _, want := range []string{
		"recent-feature-dogfood/code-slop-scorecard/code_slop",
		"ISSUE_UNROUTED",
		"root-point change",
		"path hints",
		"--live",
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("refusal message missing %q:\n%s", want, msg)
		}
	}
}
