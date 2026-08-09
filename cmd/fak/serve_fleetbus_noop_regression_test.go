package main

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// TestFleetBusLifecycleAffectedCountsStateChanges pins #5822 at the operator
// boundary: Affected counts sessions whose run state moved, not every
// non-terminal session that accepted a write. An all-no-op fan remains applied
// with a zero count; a mixed fan reports only the one session that changed.
func TestFleetBusLifecycleAffectedCountsStateChanges(t *testing.T) {
	tbl := &session.Table{}
	for _, trace := range []string{"noop-1", "noop-2", "noop-3"} {
		seedFleetBusSession(t, tbl, trace, sessionctl.BroadcastMeta{})
	}
	ap := &fleetBusApplier{tbl: tbl, ctx: context.Background()}

	out := ap.Apply(fleetBusDirective(string(sessionctl.OpResume), fleetbus.Selector{All: true}))
	if out.Status != fleetbus.AckApplied || out.Affected != 0 {
		t.Fatalf("all-no-op resume = %#v, want applied with affected=0", out)
	}
	if strings.Contains(out.Witness+" "+out.Detail, "took it") {
		t.Fatalf("all-no-op resume claims a state change: %#v", out)
	}

	if _, ok := tbl.Transition("noop-2", session.Paused, "test"); !ok {
		t.Fatal("could not seed the mixed-state arm")
	}
	out = ap.Apply(fleetBusDirective(string(sessionctl.OpResume), fleetbus.Selector{All: true}))
	if out.Status != fleetbus.AckApplied || out.Affected != 1 {
		t.Fatalf("mixed resume = %#v, want applied with affected=1", out)
	}
	if !strings.Contains(out.Witness, "2 were already running") {
		t.Fatalf("mixed resume hides the no-op tail: %#v", out)
	}
}
