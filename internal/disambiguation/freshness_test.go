package disambiguation

import "testing"

func TestEvaluateFreshnessFourStateTable(t *testing.T) {
	t.Parallel()
	base := FreshnessProbe{Probe: "public-fixture/1", CheckedAt: "2026-08-11T00:00:00Z"}
	tests := []struct {
		name        string
		observation FreshnessProbe
		verdict     FreshnessVerdict
		reason      string
	}{
		{name: "fresh", observation: FreshnessProbe{Probe: base.Probe, CheckedAt: base.CheckedAt, Available: true, EvidenceValid: true, Current: true}, verdict: FreshnessFresh, reason: FreshnessReasonSourceCurrent},
		{name: "stale", observation: FreshnessProbe{Probe: base.Probe, CheckedAt: base.CheckedAt, Available: true, EvidenceValid: true}, verdict: FreshnessStale, reason: FreshnessReasonSourceOutdated},
		{name: "unknown", observation: FreshnessProbe{Probe: base.Probe, CheckedAt: base.CheckedAt, Available: false, Current: true}, verdict: FreshnessUnknown, reason: FreshnessReasonProbeUnavailable},
		{name: "invalid", observation: FreshnessProbe{Probe: base.Probe, CheckedAt: base.CheckedAt, Available: true, EvidenceValid: false}, verdict: FreshnessInvalid, reason: FreshnessReasonEvidenceMalformed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EvaluateFreshness(tt.observation)
			if got.Verdict != tt.verdict || got.ReasonCode != tt.reason {
				t.Fatalf("EvaluateFreshness() = %s/%s, want %s/%s", got.Verdict, got.ReasonCode, tt.verdict, tt.reason)
			}
			if err := validateFreshness(got); err != nil {
				t.Fatalf("public result did not validate: %v", err)
			}
		})
	}
}

func TestFreshnessSelfCheckCoversAllStates(t *testing.T) {
	t.Parallel()
	report := FreshnessSelfCheck()
	if !report.Passed || len(report.Cases) != 4 {
		t.Fatalf("FreshnessSelfCheck() = %+v", report)
	}
	for i, want := range []FreshnessVerdict{FreshnessFresh, FreshnessStale, FreshnessUnknown, FreshnessInvalid} {
		if report.Cases[i].Verdict != want || !report.Cases[i].Passed {
			t.Fatalf("case %d = %+v, want passed %s", i, report.Cases[i], want)
		}
	}
}
