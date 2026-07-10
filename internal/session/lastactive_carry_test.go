package session

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
)

// lastactive_carry_test.go witnesses #4141 (dormancy-clock wiring, epic #1178): the
// durable LastActiveAt stamp is written by the Recontinue drive-write, carried forward
// across the in-process reset monotonically, and survives the descriptor round-trip a
// process restart re-attaches through. The stamp is advisory (no behavior gates on it),
// so these tests assert only that the VALUE is preserved — the property a later
// rehydration rung (#1181-#1186) will read via dormancy.Stamp.HorizonAt.

// TestLastActiveCarry_RecontinueStampsAndCarries witnesses the write + lineage-carry
// half: a Recontinue (a) stamps a non-zero LastActive even from an unstamped parent and
// (d) carries the parent lineage's stamp forward, advancing monotonically to now and
// never regressing on a backwards wall-clock.
func TestLastActiveCarry_RecontinueStampsAndCarries(t *testing.T) {
	base := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	unbounded := Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded}

	// (a) An unstamped parent: the Recontinue write itself marks LastActive at now, so a
	// child is never born "never-active" (which would bucket to Ancient).
	t.Run("first mark from unstamped parent", func(t *testing.T) {
		tbl := NewTable()
		tbl.Restore("parent", State{Run: Running, Budget: unbounded}) // LastActive zero
		child := tbl.RecontinueAt("parent", "child", unbounded, base)
		if child.LastActive.IsZero() {
			t.Fatalf("Recontinue must stamp a non-zero LastActive; got the zero stamp")
		}
		if got := child.LastActive.Time(); !got.Equal(base) {
			t.Fatalf("first mark should be now=%v; got %v", base, got)
		}
	})

	// (d) A parent stamped at T, reset at the SAME instant: the child carries the parent
	// lineage's exact stamp (monotonic Refresh keeps it — the reset is not later activity).
	t.Run("carries parent stamp when reset at same instant", func(t *testing.T) {
		tbl := NewTable()
		tbl.Restore("parent", State{Run: Running, Budget: unbounded, LastActive: dormancy.At(base)})
		child := tbl.RecontinueAt("parent", "child", unbounded, base)
		if child.LastActive != dormancy.At(base) {
			t.Fatalf("child must carry parent lineage stamp %v; got %v", dormancy.At(base), child.LastActive)
		}
	})

	// (d) A reset at a LATER instant advances the clock — the reset IS activity.
	t.Run("advances monotonically on later reset", func(t *testing.T) {
		tbl := NewTable()
		tbl.Restore("parent", State{Run: Running, Budget: unbounded, LastActive: dormancy.At(base)})
		later := base.Add(2 * time.Hour)
		child := tbl.RecontinueAt("parent", "child", unbounded, later)
		if got := child.LastActive.Time(); !got.Equal(later) {
			t.Fatalf("later reset should advance LastActive to %v; got %v", later, got)
		}
	})

	// A reset at an EARLIER instant (NTP slew / restore onto a skewed host) must NOT move
	// the carried stamp backwards, or an elapsed gap would be spuriously inflated.
	t.Run("does not regress on backwards clock", func(t *testing.T) {
		tbl := NewTable()
		tbl.Restore("parent", State{Run: Running, Budget: unbounded, LastActive: dormancy.At(base)})
		earlier := base.Add(-time.Hour)
		child := tbl.RecontinueAt("parent", "child", unbounded, earlier)
		if child.LastActive != dormancy.At(base) {
			t.Fatalf("backwards-clock reset must keep parent stamp %v; got %v", dormancy.At(base), child.LastActive)
		}
	})

	// The nil-Table degenerate path (no live parent) still stamps its first mark at now.
	t.Run("nil table stamps first mark", func(t *testing.T) {
		var tbl *Table
		child := tbl.RecontinueAt("parent", "child", unbounded, base)
		if child.LastActive != dormancy.At(base) {
			t.Fatalf("nil-table Recontinue should first-mark at now=%v; got %v", dormancy.At(base), child.LastActive)
		}
	})
}

// TestLastActiveCarry_DescriptorRoundTrip witnesses the persistence half: the stamp
// survives Snapshot -> descriptorFromState -> RestoredState -> Restore unchanged, so a
// process restart re-attaches a session at its real dormancy stamp rather than a zero
// (which HorizonAt would read as Ancient, forcing needless full revalidation).
func TestLastActiveCarry_DescriptorRoundTrip(t *testing.T) {
	base := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	st := State{
		TraceID:    "trace-x",
		Run:        Running,
		Budget:     Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded},
		LastActive: dormancy.At(base),
	}

	// (b) descriptorFromState mirrors the stamp verbatim.
	d := descriptorFromState(st)
	if d.LastActive != st.LastActive {
		t.Fatalf("descriptorFromState must mirror LastActive %v; got %v", st.LastActive, d.LastActive)
	}

	// (c) RestoredState is the inverse — the stamp round-trips.
	got := d.RestoredState()
	if got.LastActive != st.LastActive {
		t.Fatalf("RestoredState must preserve LastActive %v; got %v", st.LastActive, got.LastActive)
	}

	// The full loop through a live Table keeps the stamp queryable after re-attach.
	tbl := NewTable()
	tbl.Restore(got.TraceID, got)
	if reloaded := tbl.Get(got.TraceID); reloaded.LastActive != st.LastActive {
		t.Fatalf("table round-trip must preserve LastActive %v; got %v", st.LastActive, reloaded.LastActive)
	}

	// The point of preserving it: a live stamp yields a derivable dormancy band. A
	// 10-minute gap buckets Cool; a dropped (zero) stamp would falsely read Ancient.
	if band := got.LastActive.HorizonAt(base.Add(10 * time.Minute)); band != dormancy.Cool {
		t.Fatalf("10-min gap should bucket Cool; got %v", band)
	}
	if (dormancy.Stamp{}).HorizonAt(base) != dormancy.Ancient {
		t.Fatalf("a dropped stamp must read Ancient (the failure this carry prevents)")
	}
}
