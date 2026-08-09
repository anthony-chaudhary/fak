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

// #5870: the account scoping only does anything if the PRODUCER of the fleet's
// worker units names a seat. Every one of the 27 on-disk park records carried
// account="" because DISPATCH_ACCOUNT was stamped on the Go dispatch path only,
// while the live producer of the `resolve-*.log` units is
// tools/issue_resolve_dispatch.py -- so the park landed inert.
//
// This walks the whole seam the way guard does (RecordLongRetry -> Load ->
// Blocks/Resolve) over the two identity shapes that producer emits, pinned by
// tools/issue_resolve_dispatch_test.py's DispatchAccountStampTest: the switcher
// tag, and the config dir's base name for a seat carrying no tag. SameAccount is
// a plain string compare, so a producer that stamped a DIFFERENT format would
// fail exactly here -- silently, with Blocks always false, which is the bug this
// pins shut.
func TestParkFromADispatchedWorkerNamesItsSeatAndWallsNoOther(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	h := http.Header{}
	h.Set("Retry-After", "5400") // 1h30m: a real long wall, above LongWaitFloor
	for _, account := range []string{"aug5-netra", ".claude-july16-netra"} {
		store := Store{Dir: t.TempDir()}
		parked, err := store.RecordLongRetry(http.StatusTooManyRequests, h, now, Record{
			Goal: "gateway", Lane: "gateway", Account: account,
			Command: []string{"claude", "-p"},
		})
		if err != nil || !parked {
			t.Fatalf("%s: RecordLongRetry parked=%v err=%v", account, parked, err)
		}
		rec, err := store.Load("gateway")
		if err != nil {
			t.Fatalf("%s: reload: %v", account, err)
		}
		if rec.Account == "" {
			t.Fatalf("%s: persisted account is empty — this park would wall NOBODY", account)
		}
		if !rec.Blocks(account, now) {
			t.Errorf("%s: Blocks(own seat)=false, want true", account)
		}
		if rec.Blocks("some-other-seat", now) {
			t.Errorf("%s: Blocks(sibling seat)=true, want false", account)
		}
		if rec.Blocks("", now) {
			t.Errorf("%s: Blocks(unnamed caller)=true, want false", account)
		}
		// Resolve is the seam every supervisor actually consults.
		if _, blocked := store.Resolve("gateway", account, "test", now); !blocked {
			t.Errorf("%s: Resolve(own seat) did not block", account)
		}
		if _, blocked := store.Resolve("gateway", "some-other-seat", "test", now); blocked {
			t.Errorf("%s: Resolve(sibling seat) blocked", account)
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

// A park must not seal itself shut for its whole announced window.
//
// This is the ONE test here written against the pre-existing API only (Resolve,
// RecordLongRetry), so its fail-before is a genuine wrong-behaviour assertion
// rather than a missing symbol: before the probe slots existed, Resolve returned
// blocked=true for every call between ParkedAt and ParkedUntil, so the run that
// would have shown the wall was gone never happened. Measured over
// 2026-08-04..08-07, 25 of the 41 units torn down on this branch had another pool
// account with logged successful turns within ±45 minutes.
func TestParkAdmitsAProbeInsteadOfSealingItsWholeWindow(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	start := time.Unix(1_800_000_000, 0)
	rec := Record{Goal: "gateway", Lane: "gateway", Account: "seat-a", Command: []string{"claude", "-p"}}
	// A 3h announced wait: 3h/ProbeBudget is 45m, under LongWaitFloor, so the
	// floor governs and slots open at +1h and +2h.
	if parked, err := store.RecordLongRetry(429, http.Header{"Retry-After": []string{"10800"}}, start, rec); err != nil || !parked {
		t.Fatalf("parked=%v err=%v", parked, err)
	}
	// Inside the first interval the wall holds: no probe may be admitted early,
	// because a probe that walks straight back into the same 429 costs a worker
	// unit and learns nothing.
	for _, early := range []time.Duration{time.Minute, 30 * time.Minute, 59 * time.Minute} {
		if _, blocked := store.Resolve("gateway", "seat-a", "sup", start.Add(early)); !blocked {
			t.Fatalf("a probe was admitted %s into the wall, under the %s floor", early, LongWaitFloor)
		}
	}
	// The slot opens on the floor and exactly one run goes through.
	if _, blocked := store.Resolve("gateway", "seat-a", "sup", start.Add(time.Hour)); blocked {
		t.Fatal("the park sealed its whole window: no probe was ever admitted")
	}
	// ...and the wall immediately closes again behind it.
	if _, blocked := store.Resolve("gateway", "seat-a", "sup", start.Add(time.Hour+time.Minute)); !blocked {
		t.Fatal("the park stayed open after its probe: a probe is a one-shot pass, not a clear")
	}
	if _, blocked := store.Resolve("gateway", "seat-a", "sup", start.Add(2*time.Hour)); blocked {
		t.Fatal("the second probe slot never opened")
	}
}

// THE FALSE-CLEAR GUARD. A probe is evidence-gathering, not a verdict: a park
// whose condition still holds must survive its own probe completely intact —
// still walling its account, still unclaimed, still ending at the announced
// parked_until — or the probe becomes a way to launder a live wall into a clear.
func TestParkProbeDoesNotClearAParkWhoseConditionStillHolds(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	start := time.Unix(1_800_000_000, 0)
	rec := Record{Goal: "quality", Lane: "quality", Account: "seat-a", Command: []string{"claude", "-p"}}
	if parked, err := store.RecordLongRetry(429, http.Header{"Retry-After": []string{"14400"}}, start, rec); err != nil || !parked {
		t.Fatalf("parked=%v err=%v", parked, err)
	}
	before, err := store.Load("quality")
	if err != nil {
		t.Fatal(err)
	}
	probed, ok := store.AdmitProbe("quality", "sup-a", start.Add(time.Hour))
	if !ok {
		t.Fatal("no probe slot opened at the floor")
	}
	if probed.Probes != 1 {
		t.Fatalf("probes=%d want 1", probed.Probes)
	}
	after, err := store.Load("quality")
	if err != nil {
		t.Fatal(err)
	}
	if after.ClaimedAt != 0 || after.ClaimedBy != "" {
		t.Fatalf("a probe claimed/retired a live park: %+v", after)
	}
	if after.ParkedUntil != before.ParkedUntil || after.ParkedAt != before.ParkedAt {
		t.Fatalf("a probe moved the announced window: %d..%d want %d..%d",
			after.ParkedAt, after.ParkedUntil, before.ParkedAt, before.ParkedUntil)
	}
	if after.Reason != before.Reason || !reflect.DeepEqual(after.Command, before.Command) {
		t.Fatalf("a probe rewrote the park's identity: %+v", after)
	}
	// The wall is still up for its account for the rest of the window, and still
	// down for every sibling account.
	if !after.Blocks("seat-a", start.Add(2*time.Hour)) {
		t.Fatal("a probed park stopped walling the account it was recorded for")
	}
	if after.Blocks("seat-b", start.Add(2*time.Hour)) {
		t.Fatal("a probed park started walling a sibling account")
	}
	if _, blocked := store.Resolve("quality", "seat-a", "sup-b", start.Add(time.Hour+time.Second)); !blocked {
		t.Fatal("the wall did not close behind the probe")
	}
}

// One probe slot is one worker unit, so concurrent supervisors must not both
// spend it. Same O_EXCL discipline as ClaimDue, and every error path must refuse
// rather than manufacture a probe.
func TestParkProbeSlotIsAdmittedExactlyOnce(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	start := time.Unix(1_800_000_000, 0)
	rec := Record{Goal: "g", Account: "seat-a", Command: []string{"worker"}}
	if parked, err := store.RecordLongRetry(429, http.Header{"Retry-After": []string{"14400"}}, start, rec); err != nil || !parked {
		t.Fatalf("parked=%v err=%v", parked, err)
	}
	at := start.Add(time.Hour)
	if _, ok := store.AdmitProbe("g", "sup-a", at); !ok {
		t.Fatal("first probe was refused")
	}
	// A second supervisor at the same instant, and a fresh Store standing in for
	// another process, both find the slot spent.
	if _, ok := store.AdmitProbe("g", "sup-b", at); ok {
		t.Fatal("two supervisors both spent one probe slot")
	}
	if _, ok := (Store{Dir: store.Dir}).AdmitProbe("g", "sup-c", at.Add(time.Minute)); ok {
		t.Fatal("a second process re-spent an already-taken probe slot")
	}
	if got, err := store.Load("g"); err != nil || got.Probes != 1 {
		t.Fatalf("probes=%+v err=%v want exactly 1", got, err)
	}
	// A goal with no park record can never yield a probe.
	if _, ok := store.AdmitProbe("never-parked", "sup", at); ok {
		t.Fatal("an absent park record manufactured a probe")
	}
}

// The bound: spacing comes from the provider's announced wait, the budget caps
// total cost however long the wall is, and neither a degenerate window nor an
// exhausted budget may open a slot.
func TestParkProbeSpacingHonorsTheAnnouncedWaitAndStaysBounded(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	mk := func(wait time.Duration, probes int, last time.Time) Record {
		r := Record{Schema: Schema, Goal: "g", Account: "seat-a", ParkedAt: now.Unix(), Probes: probes}
		r.ParkedUntil = now.Add(wait).Unix()
		if !last.IsZero() {
			r.LastProbeAt = last.Unix()
		}
		return r
	}
	for _, tc := range []struct {
		name string
		wait time.Duration
		want time.Duration
	}{
		{"a short wall is governed by the floor, not by wait/budget", 3 * time.Hour, LongWaitFloor},
		{"the announced wait governs once it clears the floor", 24 * time.Hour, 6 * time.Hour},
		{"the clamped weekly reset probes four times a day", MaxWait, MaxWait / ProbeBudget},
		{"a degenerate window falls back to the floor", 0, LongWaitFloor},
	} {
		if got := mk(tc.wait, 0, time.Time{}).ProbeInterval(); got != tc.want {
			t.Errorf("%s: ProbeInterval(wait=%s)=%s want %s", tc.name, tc.wait, got, tc.want)
		}
	}
	long := 24 * time.Hour
	for _, tc := range []struct {
		name   string
		probes int
		last   time.Time
		at     time.Duration
		want   bool
	}{
		{"not due before one interval has passed", 0, time.Time{}, 5 * time.Hour, false},
		{"due on the interval", 0, time.Time{}, 6 * time.Hour, true},
		{"spacing measures from the last probe, not from ParkedAt", 1, now.Add(6 * time.Hour), 11 * time.Hour, false},
		{"the next slot opens one interval after the last probe", 1, now.Add(6 * time.Hour), 12 * time.Hour, true},
		{"an exhausted budget never opens another slot", ProbeBudget, now, 100 * time.Hour, false},
	} {
		if got := mk(long, tc.probes, tc.last).ProbeDue(now.Add(tc.at)); got != tc.want {
			t.Errorf("%s: ProbeDue=%v want %v", tc.name, got, tc.want)
		}
	}
}

// A re-park is a NEW wall under a freshly announced wait, so it must start with a
// full probe budget — and its slot sidecars must not collide with the spent slots
// of the generation it replaced, or the re-armed park would be unprobeable and
// would seal exactly the window this seam exists to keep open.
func TestReparkReArmsTheProbeBudget(t *testing.T) {
	store := Store{Dir: t.TempDir()}
	start := time.Unix(1_800_000_000, 0)
	rec := Record{Goal: "g", Account: "seat-a", Command: []string{"worker"}}
	if _, err := store.RecordLongRetry(429, http.Header{"Retry-After": []string{"14400"}}, start, rec); err != nil {
		t.Fatal(err)
	}
	if _, ok := store.AdmitProbe("g", "sup", start.Add(time.Hour)); !ok {
		t.Fatal("first-generation probe refused")
	}
	// The probe walked back into the same wall and the provider announced a new
	// wait: this is the re-arm path.
	reparked := start.Add(time.Hour)
	if parked, err := store.RecordLongRetry(429, http.Header{"Retry-After": []string{"7200"}}, reparked, rec); err != nil || !parked {
		t.Fatalf("re-park parked=%v err=%v", parked, err)
	}
	got, err := store.Load("g")
	if err != nil {
		t.Fatal(err)
	}
	if got.Probes != 0 || got.LastProbeAt != 0 {
		t.Fatalf("a re-park inherited the previous generation's spent budget: %+v", got)
	}
	if got.ParkedUntil != reparked.Unix()+7200 {
		t.Fatalf("re-park did not adopt the newly announced wait: %+v", got)
	}
	// Slot 0 of the NEW generation is available even though slot 0 of the old one
	// was spent.
	if _, ok := store.AdmitProbe("g", "sup", reparked.Add(time.Hour)); !ok {
		t.Fatal("the re-armed park could not be probed: slot sidecars collided across generations")
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
