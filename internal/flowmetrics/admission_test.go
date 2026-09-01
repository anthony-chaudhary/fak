package flowmetrics

import (
	"testing"
	"time"
)

func TestAdmitWIPUsesArrivalServiceWindow(t *testing.T) {
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	since := now.Add(-10 * 24 * time.Hour)
	closed := now.Add(-24 * time.Hour)
	spans := []Span{
		{OpenedAt: now.Add(-5 * 24 * time.Hour), ClosedAt: &closed},
		{OpenedAt: now.Add(-4 * 24 * time.Hour)},
		{OpenedAt: now.Add(-3 * 24 * time.Hour)},
	}
	got := AdmitWIP(IntentFresh, MeasureArrivalService(spans, since, now), ArrivalServiceRatioCeiling)
	if got.Verdict != "REFUSE" || got.ReasonCode != OverloadReasonCode {
		t.Fatalf("verdict=%s reason_code=%s, want overload refusal", got.Verdict, got.ReasonCode)
	}
	if got.Observed.Opened != 3 || got.Observed.Closed != 1 || got.Observed.WindowDays != 10 {
		t.Fatalf("observed=%+v", got.Observed)
	}
	if got.Observed.Ratio == nil || *got.Observed.Ratio != 3 {
		t.Fatalf("ratio=%v, want 3", got.Observed.Ratio)
	}
}

func TestAdmitWIPAllowsEqualOrImprovingWindows(t *testing.T) {
	for _, tc := range []struct {
		name   string
		opened int
		closed int
	}{
		{name: "equal", opened: 4, closed: 4},
		{name: "improving", opened: 3, closed: 4},
		{name: "threshold boundary", opened: 11, closed: 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ratio := float64(tc.opened) / float64(tc.closed)
			got := AdmitWIP(IntentFresh, ArrivalServiceWindow{Opened: tc.opened, Closed: tc.closed, Ratio: &ratio}, ArrivalServiceRatioCeiling)
			if got.Verdict != "ADMIT" {
				t.Fatalf("verdict=%s reason=%s", got.Verdict, got.Reason)
			}
		})
	}
}

func TestAdmitWIPAllowsRecoveryLandingSafetyAndContinuationDuringOverload(t *testing.T) {
	ratio := 2.0
	observed := ArrivalServiceWindow{Opened: 4, Closed: 2, Ratio: &ratio}
	for _, intent := range []AdmissionIntent{IntentRecovery, IntentLanding, IntentSafety, IntentContinuation} {
		t.Run(string(intent), func(t *testing.T) {
			got := AdmitWIP(intent, observed, ArrivalServiceRatioCeiling)
			if got.Verdict != "ADMIT" || got.ReasonCode != "" {
				t.Fatalf("receipt=%+v", got)
			}
		})
	}
}

func TestAdmitWIPRefusesWriteOnlyFreshIntake(t *testing.T) {
	got := AdmitWIP(IntentFresh, ArrivalServiceWindow{Opened: 2}, ArrivalServiceRatioCeiling)
	if got.Verdict != "REFUSE" || got.ReasonCode != OverloadReasonCode {
		t.Fatalf("receipt=%+v", got)
	}
}

func TestParseAdmissionIntentRejectsUnknownIntent(t *testing.T) {
	if _, err := ParseAdmissionIntent("maintenance"); err == nil {
		t.Fatal("expected unknown intent error")
	}
}
