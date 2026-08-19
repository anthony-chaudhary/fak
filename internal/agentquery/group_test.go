package agentquery

import (
	"math"
	"testing"
	"time"
)

func TestGroupLaneStateRangeUnknownAndOrder(t *testing.T) {
	now := time.Date(2026, 8, 18, 0, 0, 0, 0, time.UTC)
	since := now.Add(-7 * 24 * time.Hour)
	laneA, laneB := "a", "b"
	in := []Row{{State: "LIVE", Lane: &laneB, StartedAt: timeString(now.Add(-time.Hour)), ElapsedMS: int64Ptr(20)}, {State: "CLOSED", Lane: &laneA, StartedAt: timeString(since), ElapsedMS: int64Ptr(40)}, {State: "LIVE", Lane: nil, StartedAt: timeString(now.Add(-2 * time.Hour)), ElapsedMS: int64Ptr(30)}, {State: "OLD", Lane: &laneA, StartedAt: timeString(since.Add(-time.Nanosecond)), ElapsedMS: int64Ptr(99)}}
	got := GroupLaneState(in, since, now, "history", nil)
	if got.Metadata.MatchedRows != 3 || len(got.Rows) != 3 {
		t.Fatalf("got=%+v", got)
	}
	if got.Rows[0].Lane == nil || *got.Rows[0].Lane != "a" || got.Rows[2].Lane != nil {
		t.Fatalf("order=%+v", got.Rows)
	}
}
func timeString(v time.Time) *string { s := v.Format(time.RFC3339Nano); return &s }
func int64Ptr(v int64) *int64        { return &v }

func TestGroupLaneStateFullAggregatesNullsAndBoundary(t *testing.T) {
	observed := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	since := observed.Add(-24 * time.Hour)
	lane := "cmd"
	started := since.Format(time.RFC3339)
	v10, v20 := int64(10), int64(20)
	plan, err := GroupedPlan(24 * time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	plan.Aggregates = append([]string(nil), fullAggregates...)
	got := GroupLaneStatePlan([]Row{
		{Lane: &lane, State: "closed", StartedAt: &started, ElapsedMS: &v10},
		{Lane: &lane, State: "closed", StartedAt: &started, ElapsedMS: &v20},
		{Lane: &lane, State: "closed", StartedAt: &started},
	}, plan, observed, "history", nil)
	if len(got.Rows) != 1 {
		t.Fatalf("rows=%+v", got.Rows)
	}
	r := got.Rows[0]
	if r.Count != 3 || r.MinElapsedMS == nil || *r.MinElapsedMS != 10 || r.MaxElapsedMS == nil || *r.MaxElapsedMS != 20 || r.SumElapsedMS == nil || *r.SumElapsedMS != 30 || r.AvgElapsedMS == nil || *r.AvgElapsedMS != 15 {
		t.Fatalf("row=%+v", r)
	}
}

func TestGroupLaneStateSumOverflowIsTypedNull(t *testing.T) {
	observed := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	started := observed.Format(time.RFC3339)
	lane := "cmd"
	max, one := int64(math.MaxInt64), int64(1)
	plan, _ := GroupedPlan(time.Hour)
	plan.Aggregates = append([]string(nil), fullAggregates...)
	got := GroupLaneStatePlan([]Row{{Lane: &lane, State: "live", StartedAt: &started, ElapsedMS: &max}, {Lane: &lane, State: "live", StartedAt: &started, ElapsedMS: &one}}, plan, observed, "history", nil)
	if got.Rows[0].SumElapsedMS != nil || got.Rows[0].AvgElapsedMS != nil {
		t.Fatalf("overflow row=%+v", got.Rows[0])
	}
}
