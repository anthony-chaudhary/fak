package fleetbus

import (
	"testing"
	"time"
)

func TestDrainRetainedDirectiveDoesNotReachPostPublishInstance(t *testing.T) {
	bus, err := OpenDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 6, 12, 0, 0, 0, time.UTC)
	old, refusal := NewInstance("old", "host", "serve", 1, "", []Op{"pause"}, now)
	if refusal != nil {
		t.Fatal(refusal)
	}
	late, refusal := NewInstance("late", "host", "serve", 2, "", []Op{"pause"}, now.Add(time.Second))
	if refusal != nil {
		t.Fatal(refusal)
	}
	d, refusal := NewDirective("issuer", "pause", "", Selector{All: true}, time.Minute, "test", now)
	if refusal != nil {
		t.Fatal(refusal)
	}
	d = d.WithTargets([]Instance{old})
	if err := bus.Publish(d); err != nil {
		t.Fatal(err)
	}
	calls := 0
	rep, err := Drain(bus, late, ApplierFunc(func(Directive) Outcome { calls++; return OutcomeApplied("bad", 1) }), now.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Matched != 0 || calls != 0 || len(rep.Acks) != 0 {
		t.Fatalf("late drain=%+v calls=%d", rep, calls)
	}
	rep, err = Drain(bus, old, ApplierFunc(func(Directive) Outcome { calls++; return OutcomeApplied("old", 1) }), now.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Applied != 1 || calls != 1 {
		t.Fatalf("old drain=%+v calls=%d", rep, calls)
	}
}
