package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
)

func TestFleetControlSendRosterTTLChangesTargeting(t *testing.T) {
	dir := t.TempDir()
	bus, err := fleetbus.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 6, 12, 0, 0, 0, time.UTC)
	old := fleetControlNow
	fleetControlNow = func() time.Time { return now }
	t.Cleanup(func() { fleetControlNow = old })
	inst, refusal := fleetbus.NewInstance("serve-1", "host", "serve", 1, "", []fleetbus.Op{"pause"}, now.Add(-2*time.Minute))
	if refusal != nil {
		t.Fatal(refusal)
	}
	if err := bus.Announce(inst); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runFleetControlSend(&out, &errOut, []string{"--op", "pause", "--all", "--bus", dir, "--roster-ttl", "1m"}); code != 2 || !strings.Contains(errOut.String(), "NO_TARGET") {
		t.Fatalf("short ttl code=%d stderr=%q", code, errOut.String())
	}
	out.Reset()
	errOut.Reset()
	if code := runFleetControlSend(&out, &errOut, []string{"--op", "pause", "--all", "--bus", dir, "--roster-ttl", "3m", "--wait", "0"}); code != 1 || !strings.Contains(out.String(), "targeted=1") {
		t.Fatalf("long ttl code=%d stdout=%q stderr=%q", code, out.String(), errOut.String())
	}
}
