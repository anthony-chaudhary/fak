package goalpark

import (
	"errors"
	"net/http"
	"reflect"
	"testing"
	"time"
)

func TestLongRetryAfterSurvivesRestartAndResumesExactlyOnce(t *testing.T) {
	dir := t.TempDir()
	start := time.Unix(1_800_000_000, 0)
	store := Store{Dir: dir}
	original := Record{Goal: "issue-4805", Lane: "guard", Account: "seat-redacted", Pool: "claude", Lease: "lease-77", Witness: "dos verify commit", Command: []string{"fak", "guard", "--", "claude", "-p", "goal"}}
	h := http.Header{"Retry-After": []string{"4444"}}
	parked, err := store.RecordLongRetry(429, h, start, original)
	if err != nil || !parked {
		t.Fatalf("parked=%v err=%v", parked, err)
	}
	// A fresh Store is the process-restart witness; no in-memory state survives.
	restarted := Store{Dir: dir}
	listed, err := restarted.List()
	if err != nil || len(listed) != 1 || listed[0].Goal != original.Goal {
		t.Fatalf("status list=%+v err=%v", listed, err)
	}
	got, err := restarted.Load(original.Goal)
	if err != nil {
		t.Fatal(err)
	}
	if got.ParkedUntil != start.Unix()+4444 || got.Account != original.Account || got.Pool != original.Pool || got.Lease != original.Lease || got.Witness != original.Witness || !reflect.DeepEqual(got.Command, original.Command) {
		t.Fatalf("identity lost: %+v", got)
	}
	if _, err = restarted.ClaimDue(original.Goal, "supervisor-a", start.Add(4443*time.Second)); !errors.Is(err, ErrNotDue) {
		t.Fatalf("early claim err=%v", err)
	}
	claimed, err := restarted.ClaimDue(original.Goal, "supervisor-a", start.Add(4444*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(claimed.Command, original.Command) || claimed.ClaimedAt == 0 {
		t.Fatalf("bad claim: %+v", claimed)
	}
	if _, err = restarted.ClaimDue(original.Goal, "supervisor-b", start.Add(4445*time.Second)); !errors.Is(err, ErrClaimed) {
		t.Fatalf("second claim err=%v", err)
	}
}

func TestOrdinaryRetryClassesDoNotEnterLongPark(t *testing.T) {
	s := Store{Dir: t.TempDir()}
	now := time.Unix(10, 0)
	r := Record{Goal: "g", Command: []string{"worker"}}
	for _, tc := range []struct {
		status int
		value  string
	}{{500, "4444"}, {429, "30"}, {429, ""}} {
		h := http.Header{}
		if tc.value != "" {
			h.Set("Retry-After", tc.value)
		}
		parked, err := s.RecordLongRetry(tc.status, h, now, r)
		if err != nil || parked {
			t.Fatalf("status=%d retry=%q parked=%v err=%v", tc.status, tc.value, parked, err)
		}
	}
}
