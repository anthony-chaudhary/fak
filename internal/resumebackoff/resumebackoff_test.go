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

