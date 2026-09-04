package scorecardpane

import (
	"math"
	"strings"
	"testing"
)

func TestAffordanceRatio(t *testing.T) {
	t.Run("empty records defaults to 100 percent pass", func(t *testing.T) {
		kpi := ComputeAffordanceRatio(nil)
		if kpi.TotalInterventions != 0 {
			t.Fatalf("expected 0 total interventions, got %d", kpi.TotalInterventions)
		}
		if kpi.InterventionsWithNextAction != 0 {
			t.Fatalf("expected 0 actionable interventions, got %d", kpi.InterventionsWithNextAction)
		}
		if kpi.BareDenialCount != 0 {
			t.Fatalf("expected 0 bare denials, got %d", kpi.BareDenialCount)
		}
		if kpi.AffordanceRatio != 1.0 {
			t.Fatalf("expected affordance ratio 1.0, got %f", kpi.AffordanceRatio)
		}
		if kpi.TargetRatio != 1.0 {
			t.Fatalf("expected target ratio 1.0, got %f", kpi.TargetRatio)
		}
		if !kpi.Pass {
			t.Fatalf("expected pass to be true for empty interventions")
		}
		if kpi.Deficit() != 0.0 {
			t.Fatalf("expected deficit 0.0, got %f", kpi.Deficit())
		}
	})

	t.Run("all interventions provide actionable next actions", func(t *testing.T) {
		records := []InterventionRecord{
			{Tool: "read", Reason: "restricted", NextAction: "use fak_read with authorized path"},
			{Tool: "shell", Reason: "blocked", ActionableNextStep: "run fak recover LANE_DRAINED"},
			{Tool: "write", Reason: "out_of_tree", HasNextAction: true},
			{Tool: "edit", Reason: "concurrency", Actionable: true},
		}

		kpi := ComputeAffordanceRatio(records)
		if kpi.TotalInterventions != 4 {
			t.Fatalf("expected 4 total interventions, got %d", kpi.TotalInterventions)
		}
		if kpi.InterventionsWithNextAction != 4 {
			t.Fatalf("expected 4 actionable interventions, got %d", kpi.InterventionsWithNextAction)
		}
		if kpi.BareDenialCount != 0 {
			t.Fatalf("expected 0 bare denials, got %d", kpi.BareDenialCount)
		}
		if kpi.AffordanceRatio != 1.0 {
			t.Fatalf("expected affordance ratio 1.0, got %f", kpi.AffordanceRatio)
		}
		if !kpi.Pass {
			t.Fatalf("expected pass to be true")
		}
		if kpi.Deficit() != 0.0 {
			t.Fatalf("expected deficit 0.0, got %f", kpi.Deficit())
		}
	})

	t.Run("all interventions are bare denials", func(t *testing.T) {
		records := []InterventionRecord{
			{Tool: "shell", Reason: "POLICY_BLOCK"},
			{Tool: "eval", Reason: "UNCLASSIFIED"},
		}

		kpi := ComputeAffordanceRatio(records)
		if kpi.TotalInterventions != 2 {
			t.Fatalf("expected 2 total interventions, got %d", kpi.TotalInterventions)
		}
		if kpi.InterventionsWithNextAction != 0 {
			t.Fatalf("expected 0 actionable interventions, got %d", kpi.InterventionsWithNextAction)
		}
		if kpi.BareDenialCount != 2 {
			t.Fatalf("expected 2 bare denials, got %d", kpi.BareDenialCount)
		}
		if kpi.AffordanceRatio != 0.0 {
			t.Fatalf("expected affordance ratio 0.0, got %f", kpi.AffordanceRatio)
		}
		if kpi.Pass {
			t.Fatalf("expected pass to be false")
		}
		if math.Abs(kpi.Deficit()-1.0) > 1e-9 {
			t.Fatalf("expected deficit 1.0, got %f", kpi.Deficit())
		}
	})

	t.Run("mixed interventions evaluate ratio and compliance", func(t *testing.T) {
		records := []InterventionRecord{
			{Tool: "shell", NextAction: "run fak recover"},
			{Tool: "shell", Reason: "POLICY_BLOCK"},
			{Tool: "read", ActionableNextStep: "inspect permitted roots"},
			{Tool: "write", HasNextAction: true},
		}

		kpi := ComputeAffordanceRatio(records)
		if kpi.TotalInterventions != 4 {
			t.Fatalf("expected 4 total interventions, got %d", kpi.TotalInterventions)
		}
		if kpi.InterventionsWithNextAction != 3 {
			t.Fatalf("expected 3 actionable interventions, got %d", kpi.InterventionsWithNextAction)
		}
		if kpi.BareDenialCount != 1 {
			t.Fatalf("expected 1 bare denial, got %d", kpi.BareDenialCount)
		}
		if math.Abs(kpi.AffordanceRatio-0.75) > 1e-9 {
			t.Fatalf("expected affordance ratio 0.75, got %f", kpi.AffordanceRatio)
		}
		if kpi.Pass {
			t.Fatalf("expected pass to be false when target is 1.0")
		}
		if math.Abs(kpi.Deficit()-0.25) > 1e-9 {
			t.Fatalf("expected deficit 0.25, got %f", kpi.Deficit())
		}

		withCustomTarget := ComputeAffordanceRatioWithTarget(records, 0.75)
		if !withCustomTarget.Pass {
			t.Fatalf("expected pass to be true when target is 0.75")
		}
	})

	t.Run("summary formatting contains key metrics", func(t *testing.T) {
		kpi := ComputeAffordanceRatio([]InterventionRecord{
			{NextAction: "fak recover"},
			{Reason: "bare block"},
		})
		s := kpi.Summary()
		for _, substr := range []string{"affordance-ratio=50.0%", "1/2 interventions", "1 bare denials", "pass=false"} {
			if !strings.Contains(s, substr) {
				t.Errorf("summary missing expected substring %q: %s", substr, s)
			}
		}
	})
}

func TestAffordanceRatioEmpty(t *testing.T) {
	kpi := NewAffordanceRatioKPI()
	if kpi.AffordanceRatio != 1.0 || !kpi.Pass {
		t.Fatalf("expected initial ratio 1.0 and pass=true, got ratio=%f pass=%t", kpi.AffordanceRatio, kpi.Pass)
	}
	kpi.Calculate()
	if kpi.AffordanceRatio != 1.0 || !kpi.Pass {
		t.Fatalf("expected calculated empty ratio 1.0 and pass=true, got ratio=%f pass=%t", kpi.AffordanceRatio, kpi.Pass)
	}
}

func TestAffordanceRatioAllWithAction(t *testing.T) {
	var kpi AffordanceRatioKPI
	kpi.RecordNextAction()
	kpi.RecordNextAction()
	if kpi.TotalInterventions != 2 || kpi.InterventionsWithNextAction != 2 || kpi.BareDenialCount != 0 {
		t.Fatalf("unexpected counts: %+v", kpi)
	}
	if kpi.AffordanceRatio != 1.0 || !kpi.Pass {
		t.Fatalf("expected 1.0 and pass=true, got ratio=%f pass=%t", kpi.AffordanceRatio, kpi.Pass)
	}
}

func TestAffordanceRatioMixedCompliance(t *testing.T) {
	kpi := NewAffordanceRatioKPI()
	kpi.RecordIntervention(true)
	kpi.RecordIntervention(false)
	if kpi.TotalInterventions != 2 {
		t.Fatalf("expected total 2, got %d", kpi.TotalInterventions)
	}
	if kpi.InterventionsWithNextAction != 1 {
		t.Fatalf("expected 1 with next action, got %d", kpi.InterventionsWithNextAction)
	}
	if kpi.BareDenialCount != 1 {
		t.Fatalf("expected 1 bare denial, got %d", kpi.BareDenialCount)
	}
	if math.Abs(kpi.AffordanceRatio-0.5) > 1e-9 {
		t.Fatalf("expected ratio 0.5, got %f", kpi.AffordanceRatio)
	}
	if kpi.Pass {
		t.Fatalf("expected pass=false against target 1.0")
	}

	// Change target to 0.5 and re-evaluate
	kpi.TargetRatio = 0.5
	if !kpi.EvaluateCompliance() {
		t.Fatalf("expected pass=true against target 0.5")
	}
}

func TestAffordanceRatioRecordingMethods(t *testing.T) {
	kpi := NewAffordanceRatioKPI()
	kpi.Record(InterventionRecord{Tool: "read", NextAction: "use permitted path"})
	kpi.RecordBareDenial()
	kpi.RecordIntervention(true)

	if kpi.TotalInterventions != 3 {
		t.Fatalf("expected total 3, got %d", kpi.TotalInterventions)
	}
	if kpi.InterventionsWithNextAction != 2 {
		t.Fatalf("expected 2 with next action, got %d", kpi.InterventionsWithNextAction)
	}
	if kpi.BareDenialCount != 1 {
		t.Fatalf("expected 1 bare denial, got %d", kpi.BareDenialCount)
	}

	expectedRatio := 2.0 / 3.0
	if math.Abs(kpi.AffordanceRatio-expectedRatio) > 1e-9 {
		t.Fatalf("expected ratio %f, got %f", expectedRatio, kpi.AffordanceRatio)
	}
	if kpi.Pass {
		t.Fatalf("expected pass=false")
	}
}
