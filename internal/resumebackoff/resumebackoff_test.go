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
