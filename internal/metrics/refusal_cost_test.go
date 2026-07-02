package metrics

import (
	"encoding/json"
	"testing"
)

func TestFoldRefusalCostBuildsAgentEnvelope(t *testing.T) {
	envelope := FoldRefusalCost("off-trunk", "commit directly to main", []RefusalClearanceObservation{
		{Reason: "OFF_TRUNK", Cleared: true, TurnsToClear: 2},
		{Reason: "off trunk", Cleared: true, TurnsToClear: 6},
		{Reason: "OFF-TRUNK", Cleared: false},
		{Reason: "OUT_OF_TREE_WRITE", Cleared: true, TurnsToClear: 1},
	})

	if envelope.Reason != "OFF_TRUNK" {
		t.Fatalf("Reason = %q, want OFF_TRUNK", envelope.Reason)
	}
	if envelope.Fix != "commit directly to main" {
		t.Fatalf("Fix = %q, want trimmed fix", envelope.Fix)
	}
	if envelope.Samples != 3 || envelope.Recovered != 2 {
		t.Fatalf("samples/recovered = %d/%d, want 3/2", envelope.Samples, envelope.Recovered)
	}
	if envelope.MedianTurnsToClear != 4 {
		t.Fatalf("MedianTurnsToClear = %v, want 4", envelope.MedianTurnsToClear)
	}
	if got, want := envelope.RecoveryRate, 2.0/3.0; got != want {
		t.Fatalf("RecoveryRate = %v, want %v", got, want)
	}

	raw, err := json.Marshal(envelope)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var shaped map[string]any
	if err := json.Unmarshal(raw, &shaped); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	for _, key := range []string{"reason", "fix", "median_turns_to_clear", "recovery_rate"} {
		if _, ok := shaped[key]; !ok {
			t.Fatalf("refusal envelope JSON missing %q: %s", key, raw)
		}
	}
}

func TestRefusalCostRecoveryRecommendation(t *testing.T) {
	quick := FoldRefusalCost("STALE_RECALL", "refresh from source witness", []RefusalClearanceObservation{
		{Reason: "STALE_RECALL", Cleared: true, TurnsToClear: 1},
		{Reason: "STALE_RECALL", Cleared: true, TurnsToClear: 3},
	})
	if !quick.RecommendRecovery(3, 0.75) {
		t.Fatalf("quick envelope should recommend local recovery: %+v", quick)
	}

	sink := FoldRefusalCost("MESSAGE_RACE", "inspect intact commit", []RefusalClearanceObservation{
		{Reason: "MESSAGE_RACE", Cleared: false},
		{Reason: "MESSAGE_RACE", Cleared: true, TurnsToClear: 9},
	})
	if sink.RecommendRecovery(3, 0.75) {
		t.Fatalf("costly envelope should not recommend local recovery: %+v", sink)
	}

	unknown := FoldRefusalCost("INDETERMINATE", "consult a stronger rung", nil)
	if unknown.RecommendRecovery(10, 0.1) {
		t.Fatalf("unknown envelope should not recommend local recovery: %+v", unknown)
	}
}
