package recall

import "testing"

func TestStaleRecallRefreshPlanTriggersRefreshBeforeEffect(t *testing.T) {
	body := []byte("durable recalled fact whose source was refuted")
	page := cleanPage(body)
	syn := pageSyndromeWith(page, body, fakeOracle{revoked: map[string]bool{page.Witness: true}})

	report := PlanStaleRecallRefresh([]PageSyndrome{syn})
	if report.EffectSafe {
		t.Fatalf("stale recalled context must not be effect-safe: %+v", report)
	}
	if len(report.Decisions) != 1 {
		t.Fatalf("decisions = %+v, want one refresh decision", report.Decisions)
	}
	decision := report.Decisions[0]
	if decision.Action != StaleRecallRefreshSource || decision.Axis != EvidenceTrustEpoch.String() {
		t.Fatalf("decision = %+v, want trust-epoch refresh", decision)
	}
	if decision.SourceRef == "" || decision.Reason == "" {
		t.Fatalf("decision must carry source ref and reason: %+v", decision)
	}
}

func TestStaleRecallRefreshPlanIgnoresReusablePages(t *testing.T) {
	body := []byte("durable live recalled fact")
	syn := pageSyndromeWith(cleanPage(body), body, noRevocations)

	report := PlanStaleRecallRefresh([]PageSyndrome{syn})
	if !report.EffectSafe || len(report.Decisions) != 0 {
		t.Fatalf("clean recall should remain effect-safe with no refresh decisions: %+v", report)
	}
}
