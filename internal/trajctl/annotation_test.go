package trajctl

import "testing"

func TestSignalAnnotationSurvivesLedgerFoldAndCurve(t *testing.T) {
	rows := []Row{
		ObjectiveRecord(Objective{ID: "ship", Statement: "ship it", Status: StatusActive}),
		ScoreRecord(ScoreRow{ObjectiveID: "ship", Method: "commit-progress", Version: "1", Witness: W3, Value: 0.4, UnixMillis: 1}),
		ScoreRecord(ScoreRow{ObjectiveID: "ship", Method: "commit-progress", Version: "1", Witness: W3, Value: 0.4, UnixMillis: 2}),
		ScoreRecord(ScoreRow{ObjectiveID: "ship", Method: "activity-progress-divergence", Version: "1", Witness: W2, Value: 1, UnixMillis: 2}),
	}
	st := Fold(rows)
	annotations := st.SignalAnnotations(3)
	if len(annotations) != 1 || annotations[0].Annotation.Signal != SignalStall {
		t.Fatalf("annotations = %+v", annotations)
	}
	st = Fold(append(rows, annotations...))
	curve, ok := st.CurveFor("ship")
	if !ok || len(curve.Annotations) != 1 {
		t.Fatalf("curve = %+v ok=%v", curve, ok)
	}
	if curve.Annotations[0].Signal != SignalStall || curve.Annotations[0].Detail == "" {
		t.Fatalf("annotation = %+v", curve.Annotations[0])
	}
}

func TestSignalAnnotationsIgnoreHealthyCurve(t *testing.T) {
	st := Fold([]Row{ObjectiveRecord(Objective{ID: "ok", Statement: "healthy", Status: StatusActive})})
	if got := st.SignalAnnotations(1); len(got) != 0 {
		t.Fatalf("healthy annotations = %+v", got)
	}
}
