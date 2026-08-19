package sessionjournal

import (
	"testing"
	"time"
)

func TestEventsAsOfInclusiveAndRejectsInvalidTime(t *testing.T) {
	events := []Event{{ID: "before", TS: "2026-08-18T00:00:00Z"}, {ID: "at", TS: "2026-08-18T01:00:00Z"}, {ID: "after", TS: "2026-08-18T01:00:01Z"}, {ID: "invalid", TS: "bad"}}
	got := EventsAsOf(events, time.Date(2026, 8, 18, 1, 0, 0, 0, time.UTC))
	if len(got) != 2 || got[0].ID != "before" || got[1].ID != "at" {
		t.Fatalf("got=%+v", got)
	}
}
