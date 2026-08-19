package agentquery

import (
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
