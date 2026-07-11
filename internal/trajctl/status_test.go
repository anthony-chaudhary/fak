package trajctl

import "testing"

func TestStatusWorstFirstAndWitnessComposition(t *testing.T) {
	st := Fold([]Row{
		ObjectiveRecord(Objective{ID: "healthy", Statement: "h", Status: StatusActive}),
		ObjectiveRecord(Objective{ID: "stall", Statement: "s", Status: StatusActive}),
		ScoreRecord(ScoreRow{ObjectiveID: "stall", Method: "commit-progress", Version: "1", Witness: W3, Value: 0, UnixMillis: 1}),
		ScoreRecord(ScoreRow{ObjectiveID: "stall", Method: ActivityDivergenceScorerMethod, Version: "1", Witness: W2, Value: 1, UnixMillis: 1}),
		ScoreRecord(ScoreRow{ObjectiveID: "healthy", Method: "self", Version: "1", Witness: W0, Value: 1, UnixMillis: 1}),
	})
	got := st.Status()
	if len(got.Open) != 2 || got.Open[0].ObjectiveID != "stall" || got.Open[0].Signal != SignalStall {
		t.Fatalf("open=%+v", got.Open)
	}
	if len(got.Witnesses) != 3 || got.Witnesses[0].WitnessRung != W3 || got.Witnesses[2].WitnessRung != W0 {
		t.Fatalf("witnesses=%+v", got.Witnesses)
	}
}
func TestEmptyStatusHasNoLines(t *testing.T) {
	got := Fold(nil).Status()
	if !got.Empty() || len(got.Lines()) != 0 {
		t.Fatalf("status=%+v lines=%v", got, got.Lines())
	}
}
