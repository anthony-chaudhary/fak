package modelperfobs

import (
	"context"
	"testing"
	"time"
)

func TestRooflineBenchmarkBounds(t *testing.T) {
	o := RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: MinRooflineTrials, TargetDuration: 10 * time.Millisecond, Threads: 1}
	if err := ValidateRooflineBenchmarkOptions(o); err != nil {
		t.Fatal(err)
	}
	o.WorkingSetBytes = MinRooflineWorkingSet - 1
	if err := ValidateRooflineBenchmarkOptions(o); err == nil {
		t.Fatal("expected working-set bound")
	}
}

func TestMeasureHostMemoryRooflineAccounting(t *testing.T) {
	o := RooflineBenchmarkOptions{WorkingSetBytes: MinRooflineWorkingSet, Trials: 3, TargetDuration: 10 * time.Millisecond, Threads: 1}
	got, err := MeasureHostMemoryRoofline(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != RooflineMeasurementSchema || got.Scope != "host-memory" || got.MeasuredSustainableGBS <= 0 || got.DRAMIsolation != "not-proven" || len(got.Caveats) != 3 || len(got.Trials) != 3 {
		t.Fatalf("%+v", got)
	}
	for _, trial := range got.Trials {
		want := o.WorkingSetBytes * 2 * trial.Iterations
		if trial.TrafficBytes != want || trial.GBS <= 0 || trial.DurationMS <= 0 {
			t.Fatalf("trial=%+v want traffic=%d", trial, want)
		}
	}
}
