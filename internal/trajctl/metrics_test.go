package trajctl

import (
	"path/filepath"
	"testing"
)

func TestMetricsBoundsLabelsAndFoldsHealth(t *testing.T) {
	root := Objective{ID: "root", Status: StatusActive}
	child := Objective{ID: "child", ParentID: "root", Status: StatusPaused}
	scorer := Objective{ID: "scorer", Status: StatusMet, Meta: &MetaTarget{Method: "judge"}}
	s := Fold([]Row{ObjectiveRecord(root), ObjectiveRecord(child), ObjectiveRecord(scorer), ScoreRecord(ScoreRow{ObjectiveID: "root", Method: CommitScorerMethod, Value: .8, Witness: W3}), ScoreRecord(ScoreRow{ObjectiveID: "child", Method: CommitScorerMethod, Value: .2, Witness: W3}), SteerRecord(SteerDecision{ObjectiveID: "child", Action: ActionNudge, Signal: SignalStall, Reason: "stall", Packet: "p", Delivered: true})})
	m := s.Metrics()
	if m.Objectives[StatusActive] != 1 || m.Objectives[StatusPaused] != 1 || m.Objectives[StatusMet] != 1 {
		t.Fatalf("objectives=%v", m.Objectives)
	}
	if m.Scores["root"] != .8 || m.Scores["child"] != .2 {
		t.Fatalf("scores=%v", m.Scores)
	}
	if m.Nudges["delivered"] != 1 {
		t.Fatalf("nudges=%v", m.Nudges)
	}
	if len(m.Objectives) != 4 || len(m.Scores) != 3 || len(m.Signals) != 3 || len(m.Nudges) != 2 {
		t.Fatalf("unbounded labels: %+v", m)
	}
}

func TestMetricsFileRefreshesAfterLedgerChange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trajctl.jsonl")
	f := NewMetricsFile(path)
	if got := f.Snapshot().Objectives[StatusActive]; got != 0 {
		t.Fatal(got)
	}
	if err := Append(path, ObjectiveRecord(Objective{ID: "o", Statement: "ship", Status: StatusActive})); err != nil {
		t.Fatal(err)
	}
	if got := f.Snapshot().Objectives[StatusActive]; got != 1 {
		t.Fatalf("active=%d", got)
	}
	if got := f.Snapshot().Objectives[StatusActive]; got != 1 {
		t.Fatalf("cached active=%d", got)
	}
}
