package cavemansafety

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPolicyValueBenchmarkPricesSafetyWithoutHidingCompletion(t *testing.T) {
	r, err := RunValueBenchmark(DefaultValueArms(), DefaultValueCorpus())
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Metrics) != 6 || len(r.Traces) != 48 { //boundarylint:ignore CHANGE_DETECTOR_TEST the safety fixture defines exactly 6 metrics across its fixed 48 policy traces
		t.Fatalf("metrics=%d traces=%d", len(r.Metrics), len(r.Traces))
	}
	for _, m := range r.Metrics {
		policy := strings.Contains(m.Arm, "filtered")
		if policy {
			if m.AttackBlockRate != 1 || m.BenignAllowRate != 1 || m.TaskCompletionRate != 1 || m.FalseDenies != 0 || m.ModelCallsAvoided != 4 {
				t.Fatalf("policy metrics %+v", m)
			}
		} else if m.AttackBlockRate != 0 || m.BenignAllowRate != 1 || m.TaskCompletionRate != 0 || m.FalseDenies != 0 || m.ModelCallsAvoided != 0 {
			t.Fatalf("control metrics %+v", m)
		}
	}
	for _, tr := range r.Traces {
		if tr.RuleID == "" || tr.Configuration == "" || tr.InferenceCalls != 0 {
			t.Fatalf("incomplete trace %+v", tr)
		}
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"attack_block_rate", "benign_allow_rate", "task_completion_rate", "false_denies", "latency_overhead_ns", "model_calls_avoided", "rule_id", "configuration"} {
		if !strings.Contains(string(b), field) {
			t.Fatalf("captured receipt missing %q", field)
		}
	}
}

func TestPolicyValueBenchmarkBlanketDenyCannotWin(t *testing.T) {
	corpus := []ValueCase{{ID: "allow", PairID: "p", Tool: "unknown", Attack: false}, {ID: "attack", PairID: "p", Tool: "unknown", Attack: true}}
	r, err := RunValueBenchmark([]ValueArm{{Name: "policy", Agent: "caveman", Fak: true, Enabled: true}}, corpus)
	if err != nil {
		t.Fatal(err)
	}
	m := r.Metrics[0]
	if m.AttackBlockRate != 1 || m.BenignAllowRate != 0 || m.TaskCompletionRate != 0 || m.FalseDenies != 1 {
		t.Fatalf("blanket denial counted as completion: %+v", m)
	}
}
