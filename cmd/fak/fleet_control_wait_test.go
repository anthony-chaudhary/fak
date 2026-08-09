package main

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
)

func TestAwaitFleetControlAdvancesThroughInjectedSleep(t *testing.T) {
	bus, err := fleetbus.OpenDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	base := time.Date(2026, 8, 6, 12, 0, 0, 0, time.UTC)
	now := base
	oldNow, oldSleep := fleetControlNow, fleetControlSleep
	fleetControlNow = func() time.Time { return now }
	var sleeps []time.Duration
	fleetControlSleep = func(d time.Duration) { sleeps = append(sleeps, d); now = now.Add(d) }
	t.Cleanup(func() { fleetControlNow, fleetControlSleep = oldNow, oldSleep })
	inst, refusal := fleetbus.NewInstance("serve-1", "host", "serve", 1, "", []fleetbus.Op{"pause"}, base)
	if refusal != nil {
		t.Fatal(refusal)
	}
	if err := bus.Announce(inst); err != nil {
		t.Fatal(err)
	}
	d, refusal := fleetbus.NewDirective("issuer", "pause", "", fleetbus.Selector{All: true}, time.Minute, "test", base)
	if refusal != nil {
		t.Fatal(refusal)
	}
	d = d.WithTargets([]fleetbus.Instance{inst})
	if err := bus.Publish(d); err != nil {
		t.Fatal(err)
	}
	report, waited, err := awaitFleetControl(bus, d, 600*time.Millisecond, fleetbus.DefaultInstanceTTL)
	if err != nil {
		t.Fatal(err)
	}
	if report.Complete || report.Outstanding != 1 || waited != 600*time.Millisecond {
		t.Fatalf("report=%+v waited=%s", report, waited)
	}
	want := []time.Duration{250 * time.Millisecond, 250 * time.Millisecond, 100 * time.Millisecond}
	if len(sleeps) != len(want) {
		t.Fatalf("sleeps=%v want=%v", sleeps, want)
	}
	for i := range want {
		if sleeps[i] != want[i] {
			t.Fatalf("sleeps=%v want=%v", sleeps, want)
		}
	}
}
