package dojo

import (
	"testing"
)

func TestSubturnYieldEpisodesHonestUnmeasured(t *testing.T) {
	// Zero yield events -> unmeasured
	inputs := SubturnYieldEpisodes(SubturnYieldLedger{
		YieldEvents:       0,
		TokensBeforeYield: 0,
		TokensAfterYield:  0,
		YieldRecorded:     true,
	})
	if len(inputs) != 2 {
		t.Fatalf("expected 2 inputs, got %d", len(inputs))
	}
	if inputs[0].Outcome.Measured {
		t.Errorf("expected yield efficacy unmeasured for 0 events")
	}
}

func TestSubturnYieldEpisodesMeasuredAndBreach(t *testing.T) {
	// 40% compaction efficacy, 0 uncompacted resubmissions
	inputs := SubturnYieldEpisodes(SubturnYieldLedger{
		YieldEvents:          10,
		TokensBeforeYield:    100000,
		TokensAfterYield:     60000,
		YieldRecorded:        true,
		UncompactedResubmits: 0,
	})

	efficacyEp := Score("test-scenario", inputs[0].Prediction, inputs[0].Outcome, DefaultCalibBand())
	if efficacyEp.Verdict != VerdictCalibrated {
		t.Errorf("expected efficacy Calibrated (40%% shed), got %s", efficacyEp.Verdict)
	}

	loopEp := Score("test-scenario", inputs[1].Prediction, inputs[1].Outcome, DefaultCalibBand())
	if FloorRespectErr(loopEp) != 0.0 {
		t.Errorf("expected 0 floor breach for 0 uncompacted resubmits, got %f", FloorRespectErr(loopEp))
	}

	// Broken yield: client echoed prompt uncompacted (TokensAfterYield >= TokensBeforeYield)
	toxicInputs := SubturnYieldEpisodes(SubturnYieldLedger{
		YieldEvents:          10,
		TokensBeforeYield:    100000,
		TokensAfterYield:     100000,
		YieldRecorded:        true,
		UncompactedResubmits: 4,
	})

	toxicLoopEp := Score("test-scenario", toxicInputs[1].Prediction, toxicInputs[1].Outcome, DefaultCalibBand())
	if FloorRespectErr(toxicLoopEp) <= 0.0 {
		t.Errorf("expected positive floor breach for uncompacted resubmits, got %f", FloorRespectErr(toxicLoopEp))
	}

	report := Fold([]Episode{efficacyEp, toxicLoopEp}, FoldOpts{})
	if report.Finding != "client_resubmission_loop_detected" {
		t.Errorf("expected finding client_resubmission_loop_detected, got %s", report.Finding)
	}
}

func TestDetectResubmissionLoops(t *testing.T) {
	samples := []ResubmissionLoopSample{
		{Turn: 1, ToolName: "read_file", ArgumentsHash: "hash_a", WasElided: true},
		{Turn: 2, ToolName: "read_file", ArgumentsHash: "hash_a", WasElided: false},
		{Turn: 3, ToolName: "read_file", ArgumentsHash: "hash_a", WasElided: false}, // Streak of 3 identical -> loop!
		{Turn: 4, ToolName: "other_tool", ArgumentsHash: "hash_b", WasElided: false},
		{Turn: 5, YieldIssued: true, PromptTokens: 1000},
		{Turn: 6, PromptTokens: 1200}, // Continuation has >= 1000 tokens -> uncompacted yield loop!
	}

	toolLoops, yieldLoops := DetectResubmissionLoops(samples)
	if toolLoops != 1 {
		t.Errorf("toolLoops = %d, want 1", toolLoops)
	}
	if yieldLoops != 1 {
		t.Errorf("yieldLoops = %d, want 1", yieldLoops)
	}
}
