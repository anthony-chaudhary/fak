package resumebackoff

import (
	"testing"
	"time"
)

func TestSameSignatureBackoffEscalates(t *testing.T) {
	now := time.Unix(1000, 0)
	h := []Event{{"s", "x", now.Add(-time.Minute)}, {"s", "x", now.Add(-30 * time.Second)}}
	d := Decide(Input{Session: "s", Signature: "x", Now: now, History: h, Base: time.Minute, Ceiling: time.Hour})
	if d.Eligible || d.Reason != ReasonBackoff || d.Delay != 2*time.Minute {
		t.Fatalf("d=%+v", d)
	}
}
func TestSharedSignatureParksOnce(t *testing.T) {
	now := time.Unix(1000, 0)
	h := []Event{{"a", "x", now}, {"b", "x", now}}
	d := Decide(Input{Session: "c", Signature: "x", Now: now, History: h, ParkThreshold: 3})
	if d.Eligible || !d.Parked || d.Reason != ReasonSignatureParked || len(d.Sessions) != 3 {
		t.Fatalf("d=%+v", d)
	}
}
func TestDifferentSignatureResetsBackoff(t *testing.T) {
	now := time.Unix(1000, 0)
	h := []Event{{"s", "old", now.Add(-time.Second)}}
	d := Decide(Input{Session: "s", Signature: "new", Now: now, History: h})
	if !d.Eligible || d.Repeat != 0 {
		t.Fatalf("d=%+v", d)
	}
}

// Runtime estimate is unmeasured; six fixed-clock cases use only the pure fold.
// fak-test:runtime fast est=10ms lane=default
func TestChangedSignatureStartsNewConsecutiveRun(t *testing.T) {
	now := time.Unix(10_000, 0)
	event := func(signature string, secondsAgo int) Event {
		return Event{Session: "s", Signature: signature, At: now.Add(-time.Duration(secondsAgo) * time.Second)}
	}
	for _, tc := range []struct {
		name        string
		history     []Event
		repeat      int
		delay       time.Duration
		quarantined bool
	}{
		{
			name:    "first repeat after reset",
			history: []Event{event("x", 180), event("x", 120), event("y", 60), event("x", 30)},
			repeat:  1,
			delay:   time.Minute,
		},
		{
			name:    "unordered history",
			history: []Event{event("x", 120), event("x", 30), event("x", 180), event("y", 60)},
			repeat:  1,
			delay:   time.Minute,
		},
		{
			name:    "second repeat after reset",
			history: []Event{event("x", 240), event("x", 180), event("y", 120), event("x", 90), event("x", 30)},
			repeat:  2,
			delay:   2 * time.Minute,
		},
		{
			name:        "new consecutive run still quarantines at budget",
			history:     []Event{event("x", 300), event("y", 240), event("x", 180), event("x", 90), event("x", 30)},
			repeat:      3,
			quarantined: true,
		},
		{
			name:    "equal timestamps keep first event tie choice",
			history: []Event{event("x", 30), event("y", 30), event("x", 90)},
			repeat:  1,
			delay:   time.Minute,
		},
		{
			name:    "other session does not reset this run",
			history: []Event{event("x", 90), {Session: "other", Signature: "y", At: now.Add(-time.Minute)}, event("x", 30)},
			repeat:  2,
			delay:   2 * time.Minute,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := append([]Event(nil), tc.history...)
			d := Decide(Input{Session: "s", Signature: "x", Now: now, History: tc.history, CrashLoopBudget: 3})
			if d.Eligible || d.Repeat != tc.repeat || d.Quarantined != tc.quarantined {
				t.Fatalf("decision = %+v, want repeat=%d quarantined=%v", d, tc.repeat, tc.quarantined)
			}
			if tc.quarantined {
				if !d.Parked || d.Reason != ReasonCrashLoopQuarantined {
					t.Fatalf("consecutive crash run escaped quarantine: %+v", d)
				}
			} else if d.Parked || d.Reason != ReasonBackoff || d.Delay != tc.delay || !d.NextEligible.Equal(now.Add(-30*time.Second).Add(tc.delay)) {
				t.Fatalf("reset backoff = %+v, want delay=%v from latest event", d, tc.delay)
			}
			for i, want := range before {
				if tc.history[i] != want {
					t.Fatalf("history[%d] mutated: got %+v, want %+v", i, tc.history[i], want)
				}
			}
		})
	}
}

func TestCrashLoopQuarantinedWhenBudgetExceeded(t *testing.T) {
	now := time.Unix(1000, 0)
	h := []Event{
		{"s", "crash-sig", now.Add(-10 * time.Minute)},
		{"s", "crash-sig", now.Add(-5 * time.Minute)},
		{"s", "crash-sig", now.Add(-time.Minute)},
	}
	// With budget 3 and 3 previous launches for the same signature, must quarantine.
	d := Decide(Input{Session: "s", Signature: "crash-sig", Now: now, History: h, CrashLoopBudget: 3})
	if d.Eligible || !d.Parked || !d.Quarantined || d.Reason != ReasonCrashLoopQuarantined || d.Repeat != 3 {
		t.Fatalf("expected crash loop quarantined: d=%+v", d)
	}

	// Witness: changing the witnessed signature resets consecutive repeats and permits one bounded attempt.
	dNew := Decide(Input{Session: "s", Signature: "repaired-sig", Now: now, History: h, CrashLoopBudget: 3})
	if !dNew.Eligible || dNew.Parked || dNew.Quarantined || dNew.Reason != "" {
		t.Fatalf("expected eligible on changed signature: d=%+v", dNew)
	}
}
