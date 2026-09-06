package dojo

import (
	"testing"
)

func TestMicrocontextEpisodesHonestUnmeasured(t *testing.T) {
	// 1. Zero raw tokens -> unmeasured
	inputs := MicrocontextEpisodes(MicrocontextLedger{
		RawToolOutputTokens:    0,
		ElidedToolOutputTokens: 0,
		ElisionRecorded:        true,
	})
	if len(inputs) != 2 {
		t.Fatalf("expected 2 episodes, got %d", len(inputs))
	}
	if inputs[0].Outcome.Measured {
		t.Errorf("expected elision ratio unmeasured for 0 tokens")
	}

	// 2. Unrecorded elision -> unmeasured
	inputs = MicrocontextEpisodes(MicrocontextLedger{
		RawToolOutputTokens:    5000,
		ElidedToolOutputTokens: 2500,
		ElisionRecorded:        false,
	})
	if inputs[0].Outcome.Measured {
		t.Errorf("expected elision ratio unmeasured when not recorded")
	}
}

func TestMicrocontextEpisodesMeasuredAndFloorBreach(t *testing.T) {
	// Healthy elision, zero loops
	inputs := MicrocontextEpisodes(MicrocontextLedger{
		RawToolOutputTokens:    10000,
		ElidedToolOutputTokens: 5000,
		ElisionRecorded:        true,
		TotalSubturns:          20,
		ResubmissionLoops:      0,
		LoopRecorded:           true,
	})
	if len(inputs) != 2 {
		t.Fatalf("expected 2 inputs, got %d", len(inputs))
	}

	// Elision ratio: 0.50 claimed, 0.50 realized -> Calibrated
	elisionEp := Score("test-scenario", inputs[0].Prediction, inputs[0].Outcome, DefaultCalibBand())
	if elisionEp.Verdict != VerdictCalibrated {
		t.Errorf("expected elision ratio Calibrated, got %s", elisionEp.Verdict)
	}

	// Loop rate: 0.0 floor, 0.0 realized -> holds
	loopEp := Score("test-scenario", inputs[1].Prediction, inputs[1].Outcome, DefaultCalibBand())
	if FloorRespectErr(loopEp) != 0.0 {
		t.Errorf("expected 0 floor breach for 0 loops, got %f", FloorRespectErr(loopEp))
	}

	// Toxic elision: induced 2 client loops
	toxicInputs := MicrocontextEpisodes(MicrocontextLedger{
		RawToolOutputTokens:    10000,
		ElidedToolOutputTokens: 8000,
		ElisionRecorded:        true,
		TotalSubturns:          10,
		ResubmissionLoops:      2,
		LoopRecorded:           true,
	})
	toxicLoopEp := Score("test-scenario", toxicInputs[1].Prediction, toxicInputs[1].Outcome, DefaultCalibBand())
	if FloorRespectErr(toxicLoopEp) <= 0.0 {
		t.Errorf("expected positive floor breach for toxic loops, got %f", FloorRespectErr(toxicLoopEp))
	}

	// Fold should auto-flag client_resubmission_loop_detected
	report := Fold([]Episode{elisionEp, toxicLoopEp}, FoldOpts{})
	if report.Finding != "client_resubmission_loop_detected" {
		t.Errorf("expected finding client_resubmission_loop_detected, got %s", report.Finding)
	}
}
