package harnesscreationstudy

import (
	"strings"
	"testing"
)

func frozenStudy() Study {
	return Study{
		Schema: Schema,
		ID:     "study-1",
		Protocol: Protocol{Frozen: true, TenMinuteLimitSeconds: 600,
			AssistancePolicy: "task-card-and-help-only", FailuresInDenominator: true},
		Baseline: Baseline{ID: "tuned-alt", Runnable: true, Tuned: true, Frozen: true, Evidence: "receipts/baseline.json"},
	}
}

func TestEvaluateKeepsFailuresAndExcludesCalibration(t *testing.T) {
	s := frozenStudy()
	s.Runs = []Run{
		{ID: "maintainer", ParticipantID: "maintainer-1", Track: "ten-minute", ParticipantClass: "maintainer-calibration", Outcome: "success", ElapsedSeconds: 10, Receipt: "receipts/m.json"},
		{ID: "builder-a", ParticipantID: "builder-a", Track: "ten-minute", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 540, Receipt: "receipts/a.json"},
		{ID: "builder-b", ParticipantID: "builder-b", Track: "ten-minute", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "timeout", ElapsedSeconds: 600, Receipt: "receipts/b.json"},
	}
	r := Evaluate(s)
	if r.Calibration != 1 || r.TenMinute.EligibleRuns != 2 || r.TenMinute.Successes != 1 || r.TenMinute.Failures != 1 || r.TenMinute.PassRate != .5 {
		t.Fatalf("unexpected fold: %+v", r)
	}
	if r.TenMinute.ClaimStatus != "not_yet" {
		t.Fatalf("one success must not unlock claim: %+v", r.TenMinute)
	}
}

func TestEvaluateSupportsOnlyCompleteIndependentEnvelopes(t *testing.T) {
	s := frozenStudy()
	s.Runs = []Run{
		{ID: "builder-a", ParticipantID: "builder-a", Track: "ten-minute", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 500, Receipt: "receipts/a.json"},
		{ID: "builder-b", ParticipantID: "builder-b", Track: "ten-minute", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 580, Receipt: "receipts/b.json"},
		{ID: "builder-c", ParticipantID: "builder-c", Track: "weekend", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 7200, IndependentlyAuthored: true, ConformancePassed: true, Receipt: "receipts/c.json"},
	}
	r := Evaluate(s)
	if r.TenMinute.ClaimStatus != "supported" || r.Weekend.ClaimStatus != "supported" {
		t.Fatalf("complete envelopes rejected: %+v", r)
	}
	if r.TenMinute.MedianSuccessSeconds == nil || *r.TenMinute.MedianSuccessSeconds != 540 {
		t.Fatalf("median=%v", r.TenMinute.MedianSuccessSeconds)
	}
}

func TestParseFailsClosedOnPIIShapedIDsAndMutableProtocol(t *testing.T) {
	raw := `{"schema":"fak.harness-creation-study/v1alpha1","id":"study","protocol":{"frozen":false,"ten_minute_limit_seconds":600,"assistance_policy":"task-card-and-help-only","failures_in_denominator":true},"baseline":{"id":"alt","runnable":true,"tuned":true,"frozen":true,"evidence":"x"},"runs":[]}`
	if _, err := Parse([]byte(raw)); err == nil || !strings.Contains(err.Error(), "protocol") {
		t.Fatalf("mutable protocol accepted: %v", err)
	}
	raw = strings.Replace(raw, `"frozen":false`, `"frozen":true`, 1)
	raw = strings.Replace(raw, `"runs":[]`, `"runs":[{"id":"r","participant_id":"person@example.com","track":"ten-minute","participant_class":"unfamiliar-builder","independent":true,"outcome":"failure","elapsed_seconds":600,"receipt":"x"}]`, 1)
	if _, err := Parse([]byte(raw)); err == nil || !strings.Contains(err.Error(), "privacy-safe") {
		t.Fatalf("PII-shaped id accepted: %v", err)
	}
}
