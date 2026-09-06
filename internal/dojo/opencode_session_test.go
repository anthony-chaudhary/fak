package dojo

import (
	"math"
	"testing"
)

// TestOpencodeSessionLeverRecognized asserts that the opencode-session lever
// is registered in the dojo lever registry and its claims resolve via the claim registry.
func TestOpencodeSessionLeverRecognized(t *testing.T) {
	// 1. Lever registry lookup
	lv, ok := LookupLever(OpencodeSessionLeverName)
	if !ok {
		t.Fatalf("LookupLever(%q) = false; opencode-session lever not recognized", OpencodeSessionLeverName)
	}
	if lv.Name() != OpencodeSessionLeverName {
		t.Fatalf("lever.Name() = %q, want %q", lv.Name(), OpencodeSessionLeverName)
	}

	// 2. Lever list contains opencode-session
	levers := RegisteredLevers()
	found := false
	for _, l := range levers {
		if l.Name() == OpencodeSessionLeverName {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("RegisteredLevers() did not contain %q: %+v", OpencodeSessionLeverName, levers)
	}

	// 3. Claims registry lookup for the 3 metrics
	expectedClaims := []struct {
		metric  string
		claimed float64
		unit    string
	}{
		{"cache_read_share", 0.80, "fraction"},
		{"turns_per_task", 16.0, "turns"},
		{"compaction_shed_ratio", 0.40, "fraction"},
	}

	for _, tc := range expectedClaims {
		c, ok := Registry.Lookup(OpencodeSessionLeverName, tc.metric)
		if !ok {
			t.Fatalf("Registry.Lookup(%q, %q) failed", OpencodeSessionLeverName, tc.metric)
		}
		if c.Claimed != tc.claimed {
			t.Errorf("%s/%s Claimed = %v, want %v", OpencodeSessionLeverName, tc.metric, c.Claimed, tc.claimed)
		}
		if c.IntentionalFloor {
			t.Errorf("%s/%s is an estimate, must not be marked IntentionalFloor", OpencodeSessionLeverName, tc.metric)
		}
		if c.LowerIsBetter {
			t.Errorf("%s/%s must not be marked LowerIsBetter", OpencodeSessionLeverName, tc.metric)
		}
		if c.Basis == "" {
			t.Errorf("%s/%s missing prose basis", OpencodeSessionLeverName, tc.metric)
		}

		pred, ok := Registry.Predict(OpencodeSessionLeverName, tc.metric, tc.unit)
		if !ok || pred.Claimed != tc.claimed || pred.Unit != tc.unit {
			t.Errorf("Registry.Predict(%q, %q, %q) = %+v, ok=%v", OpencodeSessionLeverName, tc.metric, tc.unit, pred, ok)
		}

		// Verify additive seam: must NOT be in central Registry literal
		if _, central := Registry[claimKey{OpencodeSessionLeverName, tc.metric}]; central {
			t.Errorf("%s/%s landed in central Registry literal; must be additive", OpencodeSessionLeverName, tc.metric)
		}
	}
}

// TestOpencodeSessionEpisodesUnmeasured verifies that unrecorded or empty telemetry
// yields honest UNMEASURED episodes without inventing numbers.
func TestOpencodeSessionEpisodesUnmeasured(t *testing.T) {
	inputs := OpencodeSessionEpisodes(OpencodeSessionLedger{})
	if len(inputs) != 3 {
		t.Fatalf("expected 3 inputs, got %d", len(inputs))
	}
	for i, in := range inputs {
		if in.Outcome.Measured {
			t.Errorf("input %d (%s) must be unmeasured", i, in.Prediction.Metric)
		}
		ep := Score("test-unmeasured", in.Prediction, in.Outcome, DefaultCalibBand())
		if ep.Verdict != VerdictUnmeasured {
			t.Errorf("input %d verdict = %s, want UNMEASURED", i, ep.Verdict)
		}
		if ep.Grade != gradeNA {
			t.Errorf("input %d grade = %s, want n/a", i, ep.Grade)
		}
		if ep.CalibErr != 0.0 {
			t.Errorf("input %d calib_err = %f, want 0.0", i, ep.CalibErr)
		}
	}

	report := OpencodeSessionReport("test-scenario", OpencodeSessionLedger{}, FoldOpts{Date: "2026-09-06"}, nil)
	if report.OK {
		t.Errorf("unmeasured report OK = %v, want false", report.OK)
	}
	if report.Finding != "dojo_unmeasured" {
		t.Errorf("unmeasured report Finding = %q, want dojo_unmeasured", report.Finding)
	}
}

// TestOpencodeSessionCalibrationErrorsAgainstEpisodes verifies exact calibration error
// computation for perfect and off-target episodes.
func TestOpencodeSessionCalibrationErrorsAgainstEpisodes(t *testing.T) {
	band := DefaultCalibBand()

	// 1. Perfectly calibrated episodes: realized == claimed -> calibErr == 0.0, VerdictCalibrated (A)
	calibratedLedger := OpencodeSessionLedger{
		SessionID:              "ses-calibrated",
		InputTokens:            2000,
		CacheReadTokens:        8000,
		CacheCreationTokens:    0,
		CacheRecorded:          true,
		TotalTurns:             32,
		CompletedTasks:         2,
		TurnsRecorded:          true,
		TokensBeforeCompaction: 100000,
		TokensAfterCompaction:  60000,
		CompactionEvents:       1,
		CompactionRecorded:     true,
	}

	inputs := OpencodeSessionEpisodes(calibratedLedger)
	if len(inputs) != 3 {
		t.Fatalf("expected 3 inputs, got %d", len(inputs))
	}

	for _, in := range inputs {
		ep := Score("test-scenario", in.Prediction, in.Outcome, band)
		if ep.Verdict != VerdictCalibrated {
			t.Errorf("%s: verdict = %s, want %s", in.Prediction.Metric, ep.Verdict, VerdictCalibrated)
		}
		if ep.Grade != "A" {
			t.Errorf("%s: grade = %s, want A", in.Prediction.Metric, ep.Grade)
		}
		if math.Abs(ep.CalibErr) > 1e-9 {
			t.Errorf("%s: calib_err = %f, want 0.0", in.Prediction.Metric, ep.CalibErr)
		}
	}

	// 2. Off-target episodes (25% relative error for all 3):
	// - cache_read_share: 6000 / 10000 = 0.60 vs claim 0.80 -> |0.60-0.80|/0.80 = 0.25 (OVER_CLAIM, Grade C)
	// - turns_per_task: 20 / 1 = 20.0 vs claim 16.0 -> |20.0-16.0|/16.0 = 0.25 (OVER_CLAIM, Grade C)
	// - compaction_shed_ratio: (100000-70000)/100000 = 0.30 vs claim 0.40 -> |0.30-0.40|/0.40 = 0.25 (OVER_CLAIM, Grade C)
	offTargetLedger := OpencodeSessionLedger{
		SessionID:              "ses-offtarget",
		InputTokens:            4000,
		CacheReadTokens:        6000,
		CacheCreationTokens:    0,
		CacheRecorded:          true,
		TotalTurns:             20,
		CompletedTasks:         1,
		TurnsRecorded:          true,
		TokensBeforeCompaction: 100000,
		TokensAfterCompaction:  70000,
		CompactionEvents:       1,
		CompactionRecorded:     true,
	}

	offInputs := OpencodeSessionEpisodes(offTargetLedger)
	for _, in := range offInputs {
		ep := Score("test-scenario", in.Prediction, in.Outcome, band)
		if math.Abs(ep.CalibErr-0.25) > 1e-9 {
			t.Errorf("%s: calib_err = %f, want 0.25", in.Prediction.Metric, ep.CalibErr)
		}
		if ep.Grade != "C" {
			t.Errorf("%s: grade = %s, want C", in.Prediction.Metric, ep.Grade)
		}
		var wantVerdict string
		switch in.Prediction.Metric {
		case "turns_per_task":
			wantVerdict = VerdictUnderClaim
		default:
			wantVerdict = VerdictOverClaim
		}
		if ep.Verdict != wantVerdict {
			t.Errorf("%s: verdict = %s, want %s", in.Prediction.Metric, ep.Verdict, wantVerdict)
		}
	}

	// 3. Multi-session aggregation
	multiInputs := MultiOpencodeSessionEpisodes([]OpencodeSessionLedger{
		{
			InputTokens:            1000,
			CacheReadTokens:        4000,
			CacheRecorded:          true,
			TotalTurns:             16,
			CompletedTasks:         1,
			TurnsRecorded:          true,
			TokensBeforeCompaction: 50000,
			TokensAfterCompaction:  30000,
			CompactionEvents:       1,
			CompactionRecorded:     true,
		},
		{
			InputTokens:            1000,
			CacheReadTokens:        4000,
			CacheRecorded:          true,
			TotalTurns:             16,
			CompletedTasks:         1,
			TurnsRecorded:          true,
			TokensBeforeCompaction: 50000,
			TokensAfterCompaction:  30000,
			CompactionEvents:       1,
			CompactionRecorded:     true,
		},
	})
	for _, in := range multiInputs {
		ep := Score("test-multi", in.Prediction, in.Outcome, band)
		if ep.Verdict != VerdictCalibrated {
			t.Errorf("multi %s: verdict = %s, want %s", in.Prediction.Metric, ep.Verdict, VerdictCalibrated)
		}
	}
}

// TestOpencodeSessionReportAndTrending asserts that the opencode-session lever emits
// valid dojo claim reports with calibration error trending across ticks.
func TestOpencodeSessionReportAndTrending(t *testing.T) {
	// Tick 1: Off-target baseline (mean calib err ~0.25)
	led1 := OpencodeSessionLedger{
		SessionID:              "ses-1",
		InputTokens:            4000,
		CacheReadTokens:        6000,
		CacheRecorded:          true,
		TotalTurns:             20,
		CompletedTasks:         1,
		TurnsRecorded:          true,
		TokensBeforeCompaction: 100000,
		TokensAfterCompaction:  70000,
		CompactionEvents:       1,
		CompactionRecorded:     true,
	}

	rep1 := OpencodeSessionReport("test-scenario", led1, FoldOpts{
		Workspace:   "/work/fak",
		Commit:      "commit-001",
		GeneratedAt: "2026-09-06T10:00:00Z",
		Date:        "2026-09-06",
	}, nil)

	if rep1.Schema != Schema {
		t.Fatalf("rep1 schema = %q, want %q", rep1.Schema, Schema)
	}
	if !rep1.OK || rep1.Verdict != "OK" || rep1.Finding != "dojo_recorded" {
		t.Fatalf("rep1 status invalid: ok=%v verdict=%s finding=%s", rep1.OK, rep1.Verdict, rep1.Finding)
	}
	if rep1.EpisodeCount != 3 || rep1.Measured != 3 {
		t.Fatalf("rep1 counts: total=%d measured=%d", rep1.EpisodeCount, rep1.Measured)
	}
	if math.Abs(rep1.MeanCalibErr-0.25) > 1e-9 {
		t.Fatalf("rep1 mean_calib_err = %f, want 0.25", rep1.MeanCalibErr)
	}

	row1 := RowFromReport(rep1)
	trend1 := TrendVsLast(row1, nil)
	rep1 = rep1.WithTrend(trend1)
	if rep1.Trend.Direction != "new" {
		t.Errorf("rep1 trend direction = %s, want 'new'", rep1.Trend.Direction)
	}

	// Tick 2: Improvement (calibrated, mean calib err 0.0)
	led2 := OpencodeSessionLedger{
		SessionID:              "ses-2",
		InputTokens:            2000,
		CacheReadTokens:        8000,
		CacheRecorded:          true,
		TotalTurns:             32,
		CompletedTasks:         2,
		TurnsRecorded:          true,
		TokensBeforeCompaction: 100000,
		TokensAfterCompaction:  60000,
		CompactionEvents:       1,
		CompactionRecorded:     true,
	}

	rep2 := OpencodeSessionReport("test-scenario", led2, FoldOpts{
		Workspace:   "/work/fak",
		Commit:      "commit-002",
		GeneratedAt: "2026-09-06T11:00:00Z",
		Date:        "2026-09-06",
	}, []LedgerRow{row1})

	if rep2.Trend == nil {
		t.Fatal("rep2 trend is nil")
	}
	if rep2.Trend.Direction != "improved" {
		t.Errorf("rep2 trend direction = %s, want 'improved'", rep2.Trend.Direction)
	}
	if rep2.Trend.CalibErrDelta >= 0 {
		t.Errorf("rep2 trend delta = %f, want negative (improved)", rep2.Trend.CalibErrDelta)
	}
	if rep2.Calibrated != 3 {
		t.Errorf("rep2 calibrated = %d, want 3", rep2.Calibrated)
	}

	// Tick 3: Regression (poor calibration, mean calib err 0.50)
	led3 := OpencodeSessionLedger{
		SessionID:              "ses-3",
		InputTokens:            6000,
		CacheReadTokens:        4000,
		CacheRecorded:          true, // share = 0.40 vs 0.80 -> calib_err = 0.50
		TotalTurns:             24,
		CompletedTasks:         1,
		TurnsRecorded:          true, // tpt = 24.0 vs 16.0 -> calib_err = 0.50
		TokensBeforeCompaction: 100000,
		TokensAfterCompaction:  80000,
		CompactionEvents:       1,
		CompactionRecorded:     true, // shed = 0.20 vs 0.40 -> calib_err = 0.50
	}

	row2 := RowFromReport(rep2)
	rep3 := OpencodeSessionReport("test-scenario", led3, FoldOpts{
		Workspace:   "/work/fak",
		Commit:      "commit-003",
		GeneratedAt: "2026-09-06T12:00:00Z",
		Date:        "2026-09-06",
	}, []LedgerRow{row1, row2})

	if rep3.Trend == nil {
		t.Fatal("rep3 trend is nil")
	}
	if rep3.Trend.Direction != "regressed" {
		t.Errorf("rep3 trend direction = %s, want 'regressed'", rep3.Trend.Direction)
	}
	if rep3.Trend.CalibErrDelta <= 0 {
		t.Errorf("rep3 trend delta = %f, want positive (regressed)", rep3.Trend.CalibErrDelta)
	}
}

// TestOpencodeSessionLeverRunIntegration tests running the lever through the gym's Run engine.
func TestOpencodeSessionLeverRunIntegration(t *testing.T) {
	led := OpencodeSessionLedger{
		SessionID:              "ses-run",
		InputTokens:            2000,
		CacheReadTokens:        8000,
		CacheRecorded:          true,
		TotalTurns:             16,
		CompletedTasks:         1,
		TurnsRecorded:          true,
		TokensBeforeCompaction: 100000,
		TokensAfterCompaction:  60000,
		CompactionEvents:       1,
		CompactionRecorded:     true,
	}

	lever := NewOpencodeSessionLever(led)
	scenarios := []Scenario{
		{Name: "opencode-agent-corpus", Mode: "offline", Note: "evaluation on opencode session runs"},
	}

	episodes, errs := Run(scenarios, []Lever{lever}, DefaultCalibBand())
	if len(errs) != 0 {
		t.Fatalf("Run produced unexpected errors: %+v", errs)
	}
	if len(episodes) != 3 {
		t.Fatalf("Run produced %d episodes, want 3", len(episodes))
	}

	for _, ep := range episodes {
		if ep.Scenario != "opencode-agent-corpus" {
			t.Errorf("episode scenario = %q, want 'opencode-agent-corpus'", ep.Scenario)
		}
		if ep.Lever != OpencodeSessionLeverName {
			t.Errorf("episode lever = %q, want %q", ep.Lever, OpencodeSessionLeverName)
		}
		if ep.Verdict != VerdictCalibrated {
			t.Errorf("episode %s verdict = %s, want %s", ep.Metric, ep.Verdict, VerdictCalibrated)
		}
	}
}
