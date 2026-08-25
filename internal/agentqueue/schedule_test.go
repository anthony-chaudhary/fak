package agentqueue

import (
	"testing"
	"time"
)

func mustTime(t *testing.T, value string) time.Time {
	t.Helper()
	got, err := time.Parse(time.RFC3339, value)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func TestDailyWindowDSTFoldStableOccurrence(t *testing.T) {
	w := DailyWindow{ID: "night", Timezone: "America/Los_Angeles", Start: "01:15", Stop: "02:15", Misfire: MisfireCatchUp}
	now := mustTime(t, "2026-11-01T09:45:00Z")
	first, ok, err := w.OccurrenceAt(now)
	if err != nil || !ok {
		t.Fatalf("OccurrenceAt = %#v, %v, %v", first, ok, err)
	}
	second, ok, err := w.OccurrenceAt(now)
	if err != nil || !ok || second.ID != first.ID {
		t.Fatalf("unstable fold occurrence: %#v %#v %v", first, second, err)
	}
	if !first.StartsAt.Equal(mustTime(t, "2026-11-01T08:15:00Z")) {
		t.Fatalf("fold start = %s", first.StartsAt)
	}
}

func TestDailyWindowRejectsDSTGapBoundary(t *testing.T) {
	w := DailyWindow{ID: "gap", Timezone: "America/Los_Angeles", Start: "02:30", Stop: "04:00", Misfire: MisfireCatchUp}
	if _, _, err := w.OccurrenceAt(mustTime(t, "2026-03-08T11:00:00Z")); err == nil {
		t.Fatal("accepted nonexistent DST-gap boundary")
	}
}

func TestDailyWindowCatchUpAndSkipMisfire(t *testing.T) {
	previous := mustTime(t, "2026-08-25T16:00:00Z")
	now := mustTime(t, "2026-08-25T16:45:00Z")
	catch := DailyWindow{ID: "work", Timezone: "America/Los_Angeles", Start: "09:30", Stop: "10:00", Misfire: MisfireCatchUp}
	due, err := catch.Due(previous, now)
	if err != nil || len(due) != 1 {
		t.Fatalf("catch-up due = %#v, %v", due, err)
	}
	skip := catch
	skip.Misfire = MisfireSkip
	due, err = skip.Due(previous, now)
	if err != nil || len(due) != 0 {
		t.Fatalf("skip due = %#v, %v", due, err)
	}
}

func TestDailyWindowOvernightAndStopExclusive(t *testing.T) {
	w := DailyWindow{ID: "overnight", Timezone: "UTC", Start: "23:00", Stop: "01:00", Misfire: MisfireCatchUp}
	inside := mustTime(t, "2026-08-26T00:30:00Z")
	occ, ok, err := w.OccurrenceAt(inside)
	if err != nil || !ok || !occ.StartsAt.Equal(mustTime(t, "2026-08-25T23:00:00Z")) {
		t.Fatalf("overnight = %#v, %v, %v", occ, ok, err)
	}
	if _, ok, err := w.OccurrenceAt(mustTime(t, "2026-08-26T01:00:00Z")); err != nil || ok {
		t.Fatalf("stop boundary ok=%v err=%v", ok, err)
	}
}

func TestDailyWindowDueDoesNotDuplicateAppliedStart(t *testing.T) {
	w := DailyWindow{ID: "work", Timezone: "UTC", Start: "09:30", Stop: "10:00", Misfire: MisfireCatchUp}
	start := mustTime(t, "2026-08-25T09:30:00Z")
	due, err := w.Due(start, mustTime(t, "2026-08-25T09:45:00Z"))
	if err != nil || len(due) != 0 {
		t.Fatalf("duplicate due = %#v, %v", due, err)
	}
}
