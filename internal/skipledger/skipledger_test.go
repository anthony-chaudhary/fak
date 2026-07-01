package skipledger

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

func TestRecord_EmitsOneRowPerCandidate_SkipAndSelected(t *testing.T) {
	// The #1776 witness: "a test fixture emits at least one skip row and one
	// selected row."
	res := dispatchorder.Plan(dispatchorder.Input{
		Candidates: []dispatchorder.Candidate{
			{ID: "1", Key: "a", UpdatedUnix: 200},
			{ID: "2", Live: true, UpdatedUnix: 100},
		},
		NowUnix: 1000,
	})
	rep := Record(res, 1000)

	if len(rep.Rows) != 2 {
		t.Fatalf("want 2 rows, got %d: %+v", len(rep.Rows), rep.Rows)
	}
	if rep.SelectedCount != 1 || rep.SkippedCount != 1 {
		t.Fatalf("want 1 selected, 1 skipped, got selected=%d skipped=%d", rep.SelectedCount, rep.SkippedCount)
	}
	var sawSelected, sawSkipped bool
	for _, row := range rep.Rows {
		if row.TimestampUnix != 1000 {
			t.Errorf("row %q: want timestamp 1000, got %d", row.Issue, row.TimestampUnix)
		}
		if row.Skipped {
			sawSkipped = true
			if row.Category == CategoryNone {
				t.Errorf("skipped row %q: want a non-empty category", row.Issue)
			}
		} else {
			sawSelected = true
			if row.Category != CategoryNone {
				t.Errorf("selected row %q: want no category, got %q", row.Issue, row.Category)
			}
		}
	}
	if !sawSelected || !sawSkipped {
		t.Fatalf("want at least one selected and one skipped row, got %+v", rep.Rows)
	}
}

func TestRecord_CollisionRiskIsSafety_EverythingElseIsCapacity(t *testing.T) {
	cases := []struct {
		disposition dispatchorder.Disposition
		want        Category
	}{
		{dispatchorder.DispLive, CategoryCapacity},
		{dispatchorder.DispCooling, CategoryCapacity},
		{dispatchorder.DispSuperseded, CategoryCapacity},
		{dispatchorder.DispGenerationHeld, CategoryCapacity},
		{dispatchorder.DispCollisionRisk, CategorySafety},
	}
	for _, c := range cases {
		got := categorize(c.disposition)
		if got != c.want {
			t.Errorf("categorize(%s) = %q, want %q", c.disposition, got, c.want)
		}
	}
}

func TestRecord_EmptyResult_ZeroRows(t *testing.T) {
	rep := Record(dispatchorder.Result{}, 1000)
	if len(rep.Rows) != 0 || rep.SkippedCount != 0 || rep.SelectedCount != 0 {
		t.Fatalf("want an empty report for an empty result, got %+v", rep)
	}
}

func TestRecord_CarriesLaneAndReason(t *testing.T) {
	res := dispatchorder.Plan(dispatchorder.Input{
		Candidates: []dispatchorder.Candidate{
			{ID: "9", Lane: "gateway", LastAttemptUnix: 990, UpdatedUnix: 900},
		},
		NowUnix:         1000,
		CooldownSeconds: 120,
	})
	rep := Record(res, 1000)
	if len(rep.Rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rep.Rows))
	}
	row := rep.Rows[0]
	if row.Lane != "gateway" {
		t.Errorf("want lane %q, got %q", "gateway", row.Lane)
	}
	if row.Reason != dispatchorder.ReasonCooldown {
		t.Errorf("want reason %q, got %q", dispatchorder.ReasonCooldown, row.Reason)
	}
	if row.Category != CategoryCapacity {
		t.Errorf("want cooldown categorized as capacity, got %q", row.Category)
	}
}
