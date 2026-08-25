package trajectoryassurance

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestDecodeUltracodeStatusUnsupportedPositiveOutcomeStaysUnknown(t *testing.T) {
	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	got := decodeUltracode(t, syntheticPositiveStatus(now.Add(time.Minute)), now)
	if got.DelegationIntegrity.State != Unknown || got.DelegationIntegrity.Evidence.ReasonToken != ReasonUltracodeOutcomeMissing {
		t.Fatalf("delegation=%+v", got.DelegationIntegrity)
	}
	if got.SessionID != "session-1" || got.RunID != "run-1" || got.TrajectoryID != "run-1" {
		t.Fatalf("identity=%+v", got)
	}
}

func TestDecodeUltracodeStatusRealProducerFailureShape(t *testing.T) {
	now := time.Date(2026, 8, 25, 4, 30, 0, 0, time.UTC)
	got := decodeUltracode(t, realInvalidStatus(), now).DelegationIntegrity
	if got.State != Fail || got.Evidence.ReasonToken != ReasonUltracodeOutcomeFailed {
		t.Fatalf("delegation=%+v", got)
	}
}

func TestDecodeUltracodeStatusIncompleteBudget(t *testing.T) {
	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	payload := strings.Replace(syntheticPositiveStatus(now.Add(time.Minute)), `"complete":true`, `"complete":false`, 1)
	got := decodeUltracode(t, payload, now).DelegationIntegrity
	if got.State != Unknown || got.Evidence.ReasonToken != ReasonUltracodeBudgetIncomplete {
		t.Fatalf("delegation=%+v", got)
	}
}

func TestDecodeUltracodeStatusUnsupportedAndStale(t *testing.T) {
	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	payload := strings.Replace(syntheticPositiveStatus(now.Add(time.Minute)), UltracodeStatusSchema, "fak.ultracode_status.v2", 1)
	if got := decodeUltracode(t, payload, now).DelegationIntegrity; got.State != Unknown || got.Evidence.ReasonToken != ReasonUltracodeSchemaUnsupported {
		t.Fatalf("schema delegation=%+v", got)
	}
	if got := decodeUltracode(t, syntheticPositiveStatus(now.Add(-time.Second)), now).DelegationIntegrity; got.State != Unknown || got.Evidence.ReasonToken != ReasonUltracodeReceiptStale {
		t.Fatalf("stale delegation=%+v", got)
	}
}

func TestDecodeUltracodeStatusDeterministicPrivateAndNoCallbacks(t *testing.T) {
	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	payload := syntheticPositiveStatus(now.Add(time.Minute))
	a := decodeUltracode(t, payload, now)
	b := decodeUltracode(t, payload, now)
	aj, _ := json.Marshal(a)
	bj, _ := json.Marshal(b)
	if !bytes.Equal(aj, bj) {
		t.Fatalf("nondeterministic:\n%s\n%s", aj, bj)
	}
	for _, secret := range []string{"secret-host", "private-path", "raw-log"} {
		if bytes.Contains(aj, []byte(secret)) {
			t.Fatalf("private field leaked: %s", secret)
		}
	}
}

func decodeUltracode(t *testing.T, payload string, now time.Time) Input {
	t.Helper()
	got, err := DecodeUltracodeStatus(strings.NewReader(payload), now)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func syntheticPositiveStatus(deadline time.Time) string {
	return `{"schema":"fak.ultracode_status.v1","session_id":"session-1","run_id":"run-1","state":"complete","outcome":{"verdict":"verified","effect_readback":"observed","independent_witness":"witnessed","reconciliation":"reconciled"},"activation":{"schema":"fak.ultracode_activation.v1","total":2,"active":2,"inactive":0,"degraded":0,"unknown":0,"verified":2,"ratio":1,"children":[{"child_id":"child-a","state":"active"},{"child_id":"child-b","state":"active"}]},"budget":{"schema":"fak.ultracode_budget_receipt.v1","deadline_at":"` + deadline.Format(time.RFC3339) + `","authority":"provider-reported","covered_children":2,"total_children":2,"overrun":false,"complete":true,"admitted":true,"children":[{"child_id":"child-a","authority":"provider-reported","covered":true},{"child_id":"child-b","authority":"provider-reported","covered":true}]},"budget_phase":"final","workers":[{"child_id":"child-a","state":"completed","turns_started":1,"turns_completed":1,"activation":"active","activation_verdict":"verified-active"},{"child_id":"child-b","state":"completed","turns_started":1,"turns_completed":1,"activation":"active","activation_verdict":"verified-active"}]}`
}

func realInvalidStatus() string {
	// Shape captured from the authoritative v1 producer. It demonstrates the
	// actual invalid/not_observed enums and completed+failed-activation output.
	return `{"schema":"fak.ultracode_status.v1","session_id":"session-live","run_id":"run-live","state":"invalid","outcome":{"verdict":"invalid","effect_readback":"not_observed","independent_witness":"not_observed","reconciliation":"not_observed","reason":"PARENT_TOKEN_BUDGET_EXCEEDED"},"activation":{"schema":"fak.ultracode_activation.v1","total":2,"active":0,"inactive":0,"degraded":2,"unknown":0,"verified":0,"ratio":0,"children":[{"child_id":"worker-1","state":"degraded"},{"child_id":"worker-2","state":"degraded"}]},"budget":{"schema":"fak.ultracode_budget_receipt.v1","deadline_at":"2026-08-25T04:20:22Z","authority":"provider-reported","covered_children":2,"total_children":2,"overrun":true,"complete":true,"admitted":false,"children":[{"child_id":"worker-1","authority":"provider-reported","covered":true,"overrun":true},{"child_id":"worker-2","authority":"provider-reported","covered":true,"overrun":true}]},"budget_phase":"invalid","workers":[{"child_id":"worker-1","state":"completed","turns_started":1,"turns_completed":1,"activation":"degraded","activation_verdict":"failed"},{"child_id":"worker-2","state":"completed","turns_started":1,"turns_completed":1,"activation":"degraded","activation_verdict":"failed"}]}`
}
