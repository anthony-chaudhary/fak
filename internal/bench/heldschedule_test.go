package bench

import "testing"

func heldFixture() []HeldScheduleJob {
	return []HeldScheduleJob{
		{ID: "cache-heavy", CachedInputTokens: 1000, DecodeTokens: 1, ServiceMS: 2},
		{ID: "decode-heavy", UncachedInputTokens: 5, DecodeTokens: 100, ServiceMS: 100},
		{ID: "prefill", UncachedInputTokens: 200, DecodeTokens: 1, ServiceMS: 20},
	}
}

func TestEvaluateHeldScheduleUsesMeasuredTruthOnlyForCompletion(t *testing.T) {
	jobs := heldFixture()
	cal := HeldSchedulePolicy{Name: "calibrated", PrefillRate: .1, CacheReadRate: .001, DecodeRate: 1, DecisionMS: .003}
	scalar := HeldSchedulePolicy{Name: "scalar", PrefillRate: .05, CacheReadRate: .05, DecodeRate: .05, DecisionMS: .001}
	got, err := EvaluateHeldSchedule(jobs, cal, scalar)
	if err != nil {
		t.Fatal(err)
	}
	if got.Calibrated.Order[0] != "cache-heavy" || got.ScalarTotal.Order[0] != "decode-heavy" {
		t.Fatalf("orders calibrated=%v scalar=%v", got.Calibrated.Order, got.ScalarTotal.Order)
	}
	if !got.CalibratedBeatsScalarTotal || got.NetMeanCompletionValueMS <= 0 {
		t.Fatalf("expected net held-queue value: %+v", got)
	}
	if got.Calibrated.MakespanMS != got.ScalarTotal.MakespanMS {
		t.Fatal("ordering must not change total measured service")
	}
}

func TestEvaluateHeldScheduleRejectsUnwitnessedInputs(t *testing.T) {
	_, err := EvaluateHeldSchedule([]HeldScheduleJob{{ID: "one", ServiceMS: 1}}, HeldSchedulePolicy{Name: "a"}, HeldSchedulePolicy{Name: "b"})
	if err == nil {
		t.Fatal("expected too-small held set refusal")
	}
	jobs := heldFixture()
	jobs[1].ServiceMS = 0
	_, err = EvaluateHeldSchedule(jobs, HeldSchedulePolicy{Name: "a"}, HeldSchedulePolicy{Name: "b"})
	if err == nil {
		t.Fatal("expected missing measured service refusal")
	}
}

func TestMeasureHeldDecisionOverhead(t *testing.T) {
	cal, scalar, err := MeasureHeldDecisionOverhead(heldFixture(), HeldSchedulePolicy{Name: "c", DecodeRate: 1}, HeldSchedulePolicy{Name: "s", PrefillRate: 1, CacheReadRate: 1, DecodeRate: 1}, 100)
	if err != nil {
		t.Fatal(err)
	}
	if cal.DecisionMS < 0 || scalar.DecisionMS < 0 || cal.DecisionSamples != 300 || scalar.DecisionSamples != 300 {
		t.Fatalf("bad overhead evidence cal=%+v scalar=%+v", cal, scalar)
	}
}
