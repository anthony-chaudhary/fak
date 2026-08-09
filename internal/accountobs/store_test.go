package accountobs

import (
	"net/http"
	"testing"
	"time"
)

func TestStorePersistsAndCoalescesPartialObservations(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	first := http.Header{"X-Ratelimit-Remaining-Tokens": {"900"}, "X-Ratelimit-Reset-Tokens": {"60"}}
	if err := store.Observe("seat-a", 200, first, time.Unix(10, 0)); err != nil {
		t.Fatal(err)
	}
	if err := store.Observe("seat-a", 429, http.Header{"Retry-After": {"5"}}, time.Unix(20, 0)); err != nil {
		t.Fatal(err)
	}
	got, err := store.Load("seat-a")
	if err != nil {
		t.Fatal(err)
	}
	if got.Headers["x-ratelimit-remaining-tokens"] != "900" || got.Headers["x-ratelimit-reset-tokens"] != "60" {
		t.Fatalf("coalesced headers = %#v", got.Headers)
	}
	if got.LastStatus != 429 {
		t.Fatalf("last status = %d", got.LastStatus)
	}
	if err := store.Observe("seat-a", 200, http.Header{"X-Ratelimit-Remaining-Tokens": {"1"}}, time.Unix(15, 0)); err != nil {
		t.Fatal(err)
	}
	got, _ = store.Load("seat-a")
	if got.Headers["x-ratelimit-remaining-tokens"] != "900" {
		t.Fatalf("older write clobbered record: %#v", got.Headers)
	}
}
