package overtonscore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestSpine drives the generated leaf's real surface end to end.
func TestSpine(t *testing.T) {
	if !Ready() {
		t.Fatal("generated leaf spine did not reach Ready")
	}
}

func TestEvaluateMetric(t *testing.T) {
	band := WindowBand{
		OrthodoxMin:   10.0,
		OrthodoxMax:   20.0,
		AcceptableMin: 5.0,
		AcceptableMax: 25.0,
		ElevatedMin:   0.0,
		ElevatedMax:   30.0,
	}

	tests := []struct {
		name            string
		observed        float64
		wantDisposition Disposition
		wantPoints      int
		checkDeficit    func(t *testing.T, def float64)
		checkSurplus    func(t *testing.T, sur float64)
	}{
		{
			name:            "orthodox midpoint",
			observed:        15.0,
			wantDisposition: DispositionOrthodoxClean,
			wantPoints:      0,
			checkDeficit: func(t *testing.T, def float64) {
				if def != 0.0 {
					t.Errorf("expected deficit 0, got %f", def)
				}
			},
			checkSurplus: func(t *testing.T, sur float64) {
				if sur != 5.0 {
					t.Errorf("expected surplus 5.0, got %f", sur)
				}
			},
		},
		{
			name:            "orthodox lower bound",
			observed:        10.0,
			wantDisposition: DispositionOrthodoxClean,
			wantPoints:      0,
			checkDeficit: func(t *testing.T, def float64) {
				if def != 0.0 {
					t.Errorf("expected deficit 0, got %f", def)
				}
			},
			checkSurplus: func(t *testing.T, sur float64) {
				if sur != 0.0 {
					t.Errorf("expected surplus 0 at bound, got %f", sur)
				}
			},
		},
		{
			name:            "orthodox upper bound",
			observed:        20.0,
			wantDisposition: DispositionOrthodoxClean,
			wantPoints:      0,
			checkDeficit: func(t *testing.T, def float64) {
				if def != 0.0 {
					t.Errorf("expected deficit 0, got %f", def)
				}
			},
			checkSurplus: func(t *testing.T, sur float64) {
				if sur != 0.0 {
					t.Errorf("expected surplus 0 at bound, got %f", sur)
				}
			},
		},
		{
			name:            "acceptable upper",
			observed:        23.0,
			wantDisposition: DispositionOrthodoxClean,
			wantPoints:      0,
			checkDeficit: func(t *testing.T, def float64) {
				if def != 0.0 {
					t.Errorf("expected deficit 0, got %f", def)
				}
			},
			checkSurplus: func(t *testing.T, sur float64) {
				if sur != 0.0 {
					t.Errorf("expected surplus 0 outside orthodox, got %f", sur)
				}
			},
		},
		{
			name:            "acceptable lower",
			observed:        7.0,
			wantDisposition: DispositionOrthodoxClean,
			wantPoints:      0,
			checkDeficit: func(t *testing.T, def float64) {
				if def != 0.0 {
					t.Errorf("expected deficit 0, got %f", def)
				}
			},
			checkSurplus: func(t *testing.T, sur float64) {
				if sur != 0.0 {
					t.Errorf("expected surplus 0 outside orthodox, got %f", sur)
				}
			},
		},
		{
			name:            "elevated upper",
			observed:        28.0,
			wantDisposition: DispositionAcceptedTemporary,
			wantPoints:      1,
			checkDeficit: func(t *testing.T, def float64) {
				if def != 3.0 {
					t.Errorf("expected deficit 3.0, got %f", def)
				}
			},
			checkSurplus: func(t *testing.T, sur float64) {
				if sur != 0.0 {
					t.Errorf("expected surplus 0 in elevated, got %f", sur)
				}
			},
		},
		{
			name:            "elevated lower",
			observed:        2.0,
			wantDisposition: DispositionAcceptedTemporary,
			wantPoints:      1,
			checkDeficit: func(t *testing.T, def float64) {
				if def != 3.0 {
					t.Errorf("expected deficit 3.0, got %f", def)
				}
			},
			checkSurplus: func(t *testing.T, sur float64) {
				if sur != 0.0 {
					t.Errorf("expected surplus 0 in elevated, got %f", sur)
				}
			},
		},
		{
			name:            "unacceptable above elevated",
			observed:        35.0,
			wantDisposition: DispositionAccidentalUnaccepted,
			wantPoints:      2,
			checkDeficit: func(t *testing.T, def float64) {
				if def != 15.0 {
					t.Errorf("expected deficit 15.0, got %f", def)
				}
			},
			checkSurplus: func(t *testing.T, sur float64) {
				if sur != 0.0 {
					t.Errorf("expected surplus 0 in unaccepted, got %f", sur)
				}
			},
		},
		{
			name:            "unacceptable below elevated",
			observed:        -5.0,
			wantDisposition: DispositionAccidentalUnaccepted,
			wantPoints:      2,
			checkDeficit: func(t *testing.T, def float64) {
				if def != 15.0 {
					t.Errorf("expected deficit 15.0, got %f", def)
				}
			},
			checkSurplus: func(t *testing.T, sur float64) {
				if sur != 0.0 {
					t.Errorf("expected surplus 0 in unaccepted, got %f", sur)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ev := EvaluateMetric("compute", "latency", tc.observed, band)
			if ev.Disposition != tc.wantDisposition {
				t.Fatalf("disposition got %q, want %q", ev.Disposition, tc.wantDisposition)
			}
			if ev.Points != tc.wantPoints {
				t.Fatalf("points got %d, want %d", ev.Points, tc.wantPoints)
			}
			if tc.checkDeficit != nil {
				tc.checkDeficit(t, ev.Deficit)
			}
			if tc.checkSurplus != nil {
				tc.checkSurplus(t, ev.Surplus)
			}
		})
	}
}

func TestBuildReport(t *testing.T) {
	rep := Build("")
	if rep.Schema != Schema {
		t.Fatalf("schema got %q, want %q", rep.Schema, Schema)
	}
	if len(rep.Evaluations) == 0 {
		t.Fatal("expected evaluations to be non-empty")
	}
	subsystems := make(map[string]bool)
	for _, ev := range rep.Evaluations {
		subsystems[ev.Subsystem] = true
	}
	for _, wantSub := range []string{"compute", "vcache", "transport", "rules", "traces"} {
		if !subsystems[wantSub] {
			t.Errorf("expected evaluation for standard subsystem %q", wantSub)
		}
	}
	if rep.Grade == "" {
		t.Error("expected non-empty grade")
	}
	if rep.Dispositions == nil {
		t.Error("expected dispositions map to be populated")
	}

	// Verify JSON marshaling conformant to Schema
	raw, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("failed to marshal report: %v", err)
	}
	var roundTrip Report
	if err := json.Unmarshal(raw, &roundTrip); err != nil {
		t.Fatalf("failed to unmarshal report: %v", err)
	}
	if roundTrip.Schema != Schema {
		t.Errorf("roundtrip schema got %q, want %q", roundTrip.Schema, Schema)
	}
}

func TestBuildReportWithCustomConfig(t *testing.T) {
	tmp := t.TempDir()
	customSpecs := []MetricSpec{
		{
			Subsystem: "compute",
			Metric:    "custom_metric",
			Observed:  12.0,
			Band: WindowBand{
				OrthodoxMin:   10.0,
				OrthodoxMax:   20.0,
				AcceptableMin: 5.0,
				AcceptableMax: 25.0,
				ElevatedMin:   0.0,
				ElevatedMax:   30.0,
			},
		},
	}
	data, err := json.Marshal(customSpecs)
	if err != nil {
		t.Fatalf("marshal custom specs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "overton.json"), data, 0o644); err != nil {
		t.Fatalf("write custom config: %v", err)
	}

	rep := Build(tmp)
	if len(rep.Evaluations) != 1 {
		t.Fatalf("evaluations count got %d, want 1", len(rep.Evaluations))
	}
	if rep.Evaluations[0].Metric != "custom_metric" {
		t.Errorf("metric got %q, want %q", rep.Evaluations[0].Metric, "custom_metric")
	}
}

func TestPressureAndSlackCalculation(t *testing.T) {
	band := WindowBand{
		OrthodoxMin:   10.0,
		OrthodoxMax:   20.0,
		AcceptableMin: 5.0,
		AcceptableMax: 25.0,
		ElevatedMin:   0.0,
		ElevatedMax:   30.0,
	}

	// 1 orthodox (observed 15 -> surplus 5, deficit 0, debt 0)
	// 1 elevated (observed 27 -> deficit 2, surplus 0, debt 1)
	// 1 unaccepted (observed 35 -> deficit 15, surplus 0, debt 2)
	ev1 := EvaluateMetric("compute", "m1", 15.0, band)
	ev2 := EvaluateMetric("vcache", "m2", 27.0, band)
	ev3 := EvaluateMetric("transport", "m3", 35.0, band)

	rep := FoldReport([]MetricEvaluation{ev1, ev2, ev3})

	expectedSlack := 5.0
	expectedPressure := 2.0 + 15.0 // 17.0
	if rep.Slack != expectedSlack {
		t.Errorf("slack got %f, want %f", rep.Slack, expectedSlack)
	}
	if rep.Pressure != expectedPressure {
		t.Errorf("pressure got %f, want %f", rep.Pressure, expectedPressure)
	}
	if rep.OvertonPoints != 3 {
		t.Errorf("overton points got %d, want 3", rep.OvertonPoints)
	}
	if rep.Dispositions[string(DispositionOrthodoxClean)] != 1 {
		t.Errorf("orthodox_clean count got %d, want 1", rep.Dispositions[string(DispositionOrthodoxClean)])
	}
	if rep.Dispositions[string(DispositionAcceptedTemporary)] != 1 {
		t.Errorf("accepted_temporary count got %d, want 1", rep.Dispositions[string(DispositionAcceptedTemporary)])
	}
	if rep.Dispositions[string(DispositionAccidentalUnaccepted)] != 1 {
		t.Errorf("accidental_unaccepted count got %d, want 1", rep.Dispositions[string(DispositionAccidentalUnaccepted)])
	}
}
