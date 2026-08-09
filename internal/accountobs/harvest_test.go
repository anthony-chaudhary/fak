package accountobs

import (
	"net/http"
	"testing"
	"time"
)

func TestHarvesterPersistsAcrossProcessInstances(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	first := NewHarvester(store, "seat-a")
	first.Now = func() time.Time { return time.Unix(10, 0) }
	if err := first.Observe(200, http.Header{
		"X-Ratelimit-Tokens-Limit":     {"1000"},
		"X-Ratelimit-Tokens-Remaining": {"900"},
		"X-Ratelimit-Tokens-Reset":     {"60"},
	}); err != nil {
		t.Fatal(err)
	}
	families := first.Snapshot().Families()
	if len(families) != 1 || families[0].Name != "tokens" {
		t.Fatalf("families = %#v", families)
	}
	if got := families[0].Remaining; got != 900 {
		t.Fatalf("process tracker remaining = %v, want 900", got)
	}

	// A new Harvester represents a subsequent guard process. Its status-only
	// response must not erase the quota window persisted by the first process.
	second := NewHarvester(store, "seat-a")
	second.Now = func() time.Time { return time.Unix(20, 0) }
	if err := second.Observe(429, http.Header{"Retry-After": {"5"}}); err != nil {
		t.Fatal(err)
	}
	record, err := store.Load("seat-a")
	if err != nil {
		t.Fatal(err)
	}
	if record.Headers["x-ratelimit-tokens-remaining"] != "900" || record.Headers["x-ratelimit-tokens-reset"] != "60" {
		t.Fatalf("second process erased live quota window: %#v", record.Headers)
	}
	if record.LastStatus != 429 {
		t.Fatalf("last status = %d, want 429", record.LastStatus)
	}
}

func TestHarvesterRejectsOlderCrossProcessObservation(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	newer := NewHarvester(store, "seat-a")
	newer.Now = func() time.Time { return time.Unix(20, 0) }
	if err := newer.Observe(200, http.Header{"X-Ratelimit-Tokens-Remaining": {"900"}}); err != nil {
		t.Fatal(err)
	}
	older := NewHarvester(store, "seat-a")
	older.Now = func() time.Time { return time.Unix(10, 0) }
	if err := older.Observe(200, http.Header{"X-Ratelimit-Tokens-Remaining": {"1"}}); err != nil {
		t.Fatal(err)
	}
	record, err := store.Load("seat-a")
	if err != nil {
		t.Fatal(err)
	}
	if record.Headers["x-ratelimit-tokens-remaining"] != "900" {
		t.Fatalf("older process clobbered newer quota: %#v", record.Headers)
	}
}
