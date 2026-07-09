package slackoutbox

import (
	"testing"
	"time"
)

func TestLimitsReportsWindowsAndOccupancy(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := noSleepOutbox(t, base)

	// One fresh posted row (terminal, within window) and one stale superseded row (terminal,
	// already droppable). The clock is 2h past base: past the 1h settled window only.
	nPost, _ := o.Enqueue(Row{Channel: "C1", Text: "posted"})
	setState(t, o, nPost, statePosted, "1.1", base)
	nSup, _ := o.Enqueue(Row{Channel: "C1", Text: "superseded"})
	setState(t, o, nSup, stateSuperseded, "", base)
	o.now = func() time.Time { return base.Add(2 * time.Hour) }

	lim, err := o.Limits()
	if err != nil {
		t.Fatal(err)
	}

	// Windows echo the package defaults, in seconds.
	if lim.RetainSettledS != int64(DefaultRetainSettled/time.Second) ||
		lim.RetainPostedS != int64(DefaultRetainPosted/time.Second) ||
		lim.RetainDeadS != int64(DefaultRetainDead/time.Second) ||
		lim.RetainCardsS != int64(DefaultRetainCards/time.Second) {
		t.Fatalf("retention windows wrong: %+v", lim)
	}
	if lim.CompactRowThreshold != CompactRowThreshold ||
		lim.CompactMinIntervalS != int64(CompactMinInterval/time.Second) {
		t.Fatalf("compaction thresholds wrong: %+v", lim)
	}

	// Occupancy: both rows are terminal; only the superseded one is past its window.
	if lim.TerminalRows != 2 {
		t.Fatalf("terminal rows = %d, want 2", lim.TerminalRows)
	}
	if lim.DroppableRows != 1 {
		t.Fatalf("droppable rows = %d, want 1 (the past-window superseded row)", lim.DroppableRows)
	}
	if !lim.CompactionDue {
		t.Fatal("a pass should be due with a droppable row and no recent compaction")
	}
}

func TestLimitsDeadRowNotDroppableUntilLongWindow(t *testing.T) {
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o := noSleepOutbox(t, base)

	nDead, _ := o.Enqueue(Row{Channel: "C1", Text: "dead"})
	setState(t, o, nDead, stateDead, "", base)

	// 13 days on: terminal, but still inside the 14d dead window — not droppable, not due.
	o.now = func() time.Time { return base.Add(13 * 24 * time.Hour) }
	lim, err := o.Limits()
	if err != nil {
		t.Fatal(err)
	}
	if lim.TerminalRows != 1 || lim.DroppableRows != 0 || lim.CompactionDue {
		t.Fatalf("dead row inside window should not be droppable/due: %+v", lim)
	}

	// 15 days on: past the window — now droppable and a pass is due.
	o.now = func() time.Time { return base.Add(15 * 24 * time.Hour) }
	lim, err = o.Limits()
	if err != nil {
		t.Fatal(err)
	}
	if lim.DroppableRows != 1 || !lim.CompactionDue {
		t.Fatalf("dead row past window should be droppable/due: %+v", lim)
	}
}
