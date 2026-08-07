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

// A park is one ACCOUNT's wall on a goal, not the goal's. Before this, the check
// was account-blind, so a single account's 1h Retry-After stopped every account on
// the lane for as long as the park lasted.
func TestParkWallsOnlyTheAccountThatWasRateLimited(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	live := Record{Schema: Schema, Goal: "quality", Account: "seat-a", ParkedUntil: now.Unix() + 3600}
	for _, tc := range []struct {
		name    string
		rec     Record
		account string
		want    bool
	}{
		{"the walled account is stopped", live, "seat-a", true},
		{"a sibling account is not", live, "seat-b", false},
		{"an unnamed caller is not", live, "", false},
		{"whitespace still matches the same seat", live, "  seat-a ", true},
		{"an unattributed record stops nobody", Record{Schema: Schema, Goal: "quality", ParkedUntil: now.Unix() + 3600}, "seat-a", false},
		{"two unattributed sides are not the same account", Record{Schema: Schema, Goal: "quality", ParkedUntil: now.Unix() + 3600}, "", false},
		{"an elapsed wait stops nobody", Record{Schema: Schema, Goal: "quality", Account: "seat-a", ParkedUntil: now.Unix() - 1}, "seat-a", false},
		{"a claimed resume stops nobody", Record{Schema: Schema, Goal: "quality", Account: "seat-a", ParkedUntil: now.Unix() + 3600, ClaimedAt: now.Unix()}, "seat-a", false},
		{"a foreign schema stops nobody", Record{Goal: "quality", Account: "seat-a", ParkedUntil: now.Unix() + 3600}, "seat-a", false},
	} {
		if got := tc.rec.Blocks(tc.account, now); got != tc.want {
			t.Errorf("%s: Blocks(%q)=%v want %v", tc.name, tc.account, got, tc.want)
		}
	}
}

// Resolve is the supervisor seam: it must wall only the walled account, and it
// must RETIRE a due park by claiming it. Nothing ever called ClaimDue in the
// product before, so claimed_at stayed 0 forever and a park never resumed.
func TestResolveScopesByAccountAndRetiresADuePark(t *testing.T) {
	dir := t.TempDir()
	store := Store{Dir: dir}
	start := time.Unix(1_800_000_000, 0)
	rec := Record{Goal: "quality", Lane: "quality", Account: "seat-a", Command: []string{"claude", "-p"}}
	if parked, err := store.RecordLongRetry(429, http.Header{"Retry-After": []string{"3600"}}, start, rec); err != nil || !parked {
		t.Fatalf("parked=%v err=%v", parked, err)
	}

	// The walled account waits; every sibling account walks straight through.
	if _, blocked := store.Resolve("quality", "seat-a", "sup", start.Add(time.Minute)); !blocked {
		t.Fatal("the rate-limited account was not walled by its own park")
	}
	if _, blocked := store.Resolve("quality", "seat-b", "sup", start.Add(time.Minute)); blocked {
		t.Fatal("a sibling account was walled by another account's park")
	}
	if got, err := store.Load("quality"); err != nil || got.ClaimedAt != 0 {
		t.Fatalf("a sibling's pass-through must not claim the live park: %+v err=%v", got, err)
	}

	// Past parked_until the park retires itself instead of lingering unclaimed.
	after := start.Add(3600 * time.Second)
	resumed, blocked := store.Resolve("quality", "seat-a", "sup-a", after)
	if blocked {
		t.Fatal("a park whose wait elapsed still walled its account")
	}
	if resumed.ClaimedAt != after.Unix() || resumed.ClaimedBy != "sup-a" {
		t.Fatalf("a due park was not claimed/retired: %+v", resumed)
	}
	// Exactly once: a second supervisor cannot re-claim the same resume.
	again, blocked := store.Resolve("quality", "seat-a", "sup-b", after.Add(time.Second))
	if blocked || again.ClaimedBy != "sup-a" {
		t.Fatalf("resume was not exactly-once: blocked=%v rec=%+v", blocked, again)
	}
	if _, err := store.ClaimDue("quality", "sup-b", after.Add(time.Second)); !errors.Is(err, ErrClaimed) {
		t.Fatalf("claim ledger did not hold: %v", err)
	}
}

// A missing/unreadable park must fail OPEN: over-parking is the failure this seam exists to prevent.
func TestResolveFailsOpenWithoutAReadableRecord(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	if _, blocked := store.Resolve("never-parked", "seat-a", "sup", time.Unix(1_800_000_000, 0)); blocked {
		t.Fatal("a goal with no park record walled its account")
	}
}

// An oversized/mis-scaled Retry-After must not become a multi-day wall.
func TestLongRetryAfterIsCappedAtMaxWait(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	now := time.Unix(1_800_000_000, 0)
	rec := Record{Goal: "g", Account: "seat-a", Command: []string{"worker"}}
	if parked, err := store.RecordLongRetry(429, http.Header{"Retry-After": []string{"99999999"}}, now, rec); err != nil || !parked {
		t.Fatalf("parked=%v err=%v", parked, err)
	}
	got, err := store.Load("g")
	if err != nil {
		t.Fatal(err)
	}
	if want := now.Add(MaxWait).Unix(); got.ParkedUntil != want {
		t.Fatalf("parked_until=%d want %d (capped at MaxWait)", got.ParkedUntil, want)
	}
	// A legitimate long wait under the cap is untouched.
	if parked, err := store.RecordLongRetry(429, http.Header{"Retry-After": []string{"7200"}}, now, rec); err != nil || !parked {
		t.Fatalf("parked=%v err=%v", parked, err)
	}
	if got, err = store.Load("g"); err != nil || got.ParkedUntil != now.Unix()+7200 {
		t.Fatalf("an under-cap wait was clipped: %+v err=%v", got, err)
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
