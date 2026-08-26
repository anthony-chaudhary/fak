package trajectoryassurance

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestReceiptAdaptersMatchGoldenAcrossHarnessesAndDelegation(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	point := now.Add(-time.Minute).UnixMilli()
	traj, err := DecodeTrajctlCurve(strings.NewReader(`{"schema":"fak-trajctl-curve/1","objectives":[{"objective_id":"issue-8828","signal":"HEALTHY","latest":0.8,"delta":0.3,"detail":"witnessed","methods":[{"points":[{"unix_millis":`+jsonInt(point)+`,"session_id":"claude-session","run_id":"run-1"}]}]}]}`), now, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	audit, err := DecodeTrajectoryAudit(strings.NewReader("{\"schema\":\"fak-trajectory-audit/1\",\"kind\":\"session\",\"source\":\"claude\",\"session_id\":\"run-1\",\"usage_records\":2}\n{\"schema\":\"fak-trajectory-audit/1\",\"kind\":\"session\",\"source\":\"codex\",\"session_id\":\"codex-session\",\"usage_records\":1}\n"), "run-1")
	if err != nil {
		t.Fatal(err)
	}
	dojo, err := DecodeDojoIteration(strings.NewReader(`{"schema":"fak-dojo-rsi/1","kept":true,"reason":"strict gain","witness":{"ok":true,"outcome":true,"constraints_satisfied":true,"parent_units":100,"child_units":[30,20],"accounting_complete":true}}`))
	if err != nil {
		t.Fatal(err)
	}
	effects := `{"schema":"fak.orchestration_effect_receipt.v1","run_id":"run-1","child_id":"a","state":"VERIFIED","reconciliation":"RECONCILED","observed_at":"2026-08-25T11:59:00Z","witness":{"authority_id":"observer-a","author_child_id":"b"}}` + "\n" + `{"schema":"fak.orchestration_effect_receipt.v1","run_id":"run-1","child_id":"b","state":"VERIFIED","reconciliation":"RECONCILED","observed_at":"2026-08-25T11:59:00Z","witness":{"authority_id":"observer-b","author_child_id":"a"}}`
	delegation, err := DecodeEffectReceipts(strings.NewReader(effects), now, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	var in Input
	for _, part := range []Input{traj, audit, dojo, delegation} {
		if err := MergeInput(&in, part); err != nil {
			t.Fatal(err)
		}
	}
	passed := true
	in.SemanticReview = Observation{State: Pass, Evidence: testAdapterEvidence("semantic")}
	in.DeterministicFloor = append(in.DeterministicFloor, DeterministicCheck{Name: "tests", Passed: &passed, Evidence: testAdapterEvidence("tests")})
	got := Assess(in)
	if got.State != Pass || got.Recommendation != RecommendationContinue || got.ObjectiveID != "issue-8828" || got.TrajectoryID != "run-1" {
		t.Fatalf("receipt=%+v", got)
	}
	if got.Layers[2].Usage == nil || got.Layers[2].Usage.TotalUnits != 150 {
		t.Fatalf("efficiency=%+v", got.Layers[2])
	}
}

func TestAdaptersUnknownStaleSchemaAccountingAndPrivacy(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	unavailable := Assess(UnavailableInput("trajctl", "not found"))
	if unavailable.Layers[1].State != Unknown || unavailable.Layers[1].ReasonToken != ReasonSourceUnavailable {
		t.Fatalf("unavailable=%+v", unavailable.Layers[1])
	}
	cases := []struct {
		name  string
		input Input
		token string
	}{
		{"stale", mustTraj(t, `{"schema":"fak-trajctl-curve/1","objectives":[{"objective_id":"o","signal":"HEALTHY","methods":[{"points":[{"unix_millis":1,"run_id":"r"}]}]}]}`, now), ReasonSourceStale},
		{"schema", mustTraj(t, `{"schema":"future","objectives":[]}`, now), ReasonSchemaUnsupported},
		{"dojo accounting", mustDojo(t, `{"schema":"fak-dojo-rsi/1","kept":true,"reason":"x","witness":{"ok":true}}`), ReasonAccountingIncomplete},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := Assess(tc.input)
			b, _ := json.Marshal(r)
			if !strings.Contains(string(b), tc.token) {
				t.Fatalf("receipt=%s", b)
			}
		})
	}
	canary := "PROMPT_CANARY_TOOL_PAYLOAD_8828"
	payload := `{"schema":"fak-trajctl-curve/1","objectives":[{"objective_id":"o","signal":"HEALTHY","detail":"` + canary + `","methods":[{"points":[{"unix_millis":` + jsonInt(now.UnixMilli()) + `,"run_id":"r"}]}]}]}`
	in := mustTraj(t, payload, now)
	b, _ := json.Marshal(Assess(in))
	if strings.Contains(string(b), canary) {
		t.Fatalf("private payload leaked: %s", b)
	}
}

func TestEffectFailureHoldsForOperatorAndIdentityMismatchRefusesMerge(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	in, err := DecodeEffectReceipts(strings.NewReader(`{"schema":"fak.orchestration_effect_receipt.v1","run_id":"r","child_id":"a","state":"FAILED","reconciliation":"FAILED","observed_at":"2026-08-25T11:59:00Z","witness":{"authority_id":"observer","author_child_id":"b"}}`), now, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	receipt := Assess(in)
	if receipt.Layers[3].State != Fail || receipt.Layers[3].ReasonToken != ReasonEffectReadbackFailed || receipt.Recommendation != RecommendationOperatorReview {
		t.Fatalf("receipt=%+v", receipt)
	}
	dst := Input{TrajectoryID: "claude"}
	if err := MergeInput(&dst, Input{TrajectoryID: "codex"}); err == nil {
		t.Fatal("identity mismatch accepted")
	}
}

func mustTraj(t *testing.T, p string, now time.Time) Input {
	t.Helper()
	v, e := DecodeTrajctlCurve(strings.NewReader(p), now, time.Hour)
	if e != nil {
		t.Fatal(e)
	}
	return v
}
func mustDojo(t *testing.T, p string) Input {
	t.Helper()
	v, e := DecodeDojoIteration(strings.NewReader(p))
	if e != nil {
		t.Fatal(e)
	}
	return v
}
func testAdapterEvidence(reason string) Evidence {
	return Evidence{Source: "test", Provenance: "fixture", Authority: "test", Freshness: "current", Reason: reason}
}
func jsonInt(v int64) string { b, _ := json.Marshal(v); return string(b) }
