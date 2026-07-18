// Tests for rung H6 (issue #1899): the RecontinueHook binding of the driver's
// Recontinue seam to a live session.Table — leg == Generation++, ParentTrace link,
// closing leg Stopped with its relay reason.
package relay

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// recontinueTestBudget is the fresh budget every rotation in these tests re-arms with.
func recontinueTestBudget() session.Budget {
	return session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 100}
}

func TestRecontinueWiringIncrementsGenerationAndLinksParentStopped(t *testing.T) {
	tbl := session.NewTable()
	parent := "leg-0"
	hook := RecontinueHook(tbl, recontinueTestBudget())

	b := Baton{
		Schema:      Schema,
		RelayID:     "RLY-1899",
		Leg:         0,
		ParentTrace: parent,
		Tombstone:   Tombstone{Reason: "RELAY_ROTATED", AtSHA: "abc123"},
	}
	successor, err := hook(b)
	if err != nil {
		t.Fatalf("hook(baton) error = %v, want nil", err)
	}
	if successor == "" {
		t.Fatal("hook(baton) returned an empty successor trace")
	}
	if successor == parent {
		t.Fatalf("hook(baton) returned the parent trace %q as the successor", successor)
	}

	childSt := tbl.Get(successor)
	parentSt := tbl.Get(parent)
	if childSt.Generation != parentSt.Generation+1 {
		t.Fatalf("child Generation = %d, want parent Generation+1 = %d", childSt.Generation, parentSt.Generation+1)
	}
	if childSt.ParentTrace != parent {
		t.Fatalf("child ParentTrace = %q, want %q", childSt.ParentTrace, parent)
	}
	if parentSt.Run != session.Stopped {
		t.Fatalf("parent Run = %v, want session.Stopped", parentSt.Run)
	}
	if childSt.TraceID != successor {
		t.Fatalf("child TraceID = %q, want the returned successor %q", childSt.TraceID, successor)
	}
}

func TestRecontinueWiringChainsGenerationsAcrossLegs(t *testing.T) {
	tbl := session.NewTable()
	hook := RecontinueHook(tbl, recontinueTestBudget())

	prior := "leg-0"
	traces := []string{prior}
	for leg := 0; leg < 2; leg++ {
		b := Baton{
			Schema:      Schema,
			RelayID:     "RLY-1899",
			Leg:         leg,
			ParentTrace: prior,
			Tombstone:   Tombstone{Reason: "RELAY_ROTATED", AtSHA: "abc123"},
		}
		successor, err := hook(b)
		if err != nil {
			t.Fatalf("leg %d: hook(baton) error = %v, want nil", leg, err)
		}
		traces = append(traces, successor)
		prior = successor
	}

	for i := 1; i < len(traces); i++ {
		childSt := tbl.Get(traces[i])
		priorSt := tbl.Get(traces[i-1])
		if childSt.Generation != i {
			t.Fatalf("leg %d: Generation = %d, want %d", i, childSt.Generation, i)
		}
		if childSt.ParentTrace != traces[i-1] {
			t.Fatalf("leg %d: ParentTrace = %q, want prior leg %q", i, childSt.ParentTrace, traces[i-1])
		}
		if priorSt.Run != session.Stopped {
			t.Fatalf("leg %d: prior leg %q Run = %v, want session.Stopped", i, traces[i-1], priorSt.Run)
		}
	}
}

func TestRecontinueWiringRejectsBatonWithoutParentTrace(t *testing.T) {
	tbl := session.NewTable()
	hook := RecontinueHook(tbl, recontinueTestBudget())

	successor, err := hook(Baton{Schema: Schema, RelayID: "RLY-1899"})
	if err == nil {
		t.Fatal("hook(baton without ParentTrace) error = nil, want non-nil")
	}
	if successor != "" {
		t.Fatalf("hook(baton without ParentTrace) successor = %q, want empty", successor)
	}
	if !strings.Contains(err.Error(), "ParentTrace") {
		t.Fatalf("error %q does not mention ParentTrace", err)
	}
}
