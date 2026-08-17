package valuechain

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadAgenticPacketPreservesUnknownCost(t *testing.T) {
	p := filepath.Join(t.TempDir(), "packet.json")
	body := `{"schema":"fak.agentic-benchmark-result-packet.v1","status":"PASS_RESULT","result_claim_allowed":true,"benchmark_native":true,"same_task_ids":true,"same_model":true,"same_budget":true,"official_grader":{"available":true},"arms":[{"role":"raw"},{"role":"fak"}],"metric_categories":{"task_success":true,"safe_success":true,"cost_or_token_budget":true,"latency":true,"policy_events":true,"evidence_completeness":true},"artifacts":["raw.json","fak.json"],"value_chain":[{"role":"raw","trace_id":"r","pair_id":"p","turns":4,"outcomes":{"passed":1},"provenance":"official-grader"},{"role":"fak","trace_id":"f","pair_id":"p","turns":3,"cost_usd":0.3,"outcomes":{"passed":1},"provenance":"official-grader+bill"}]}`
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	in, err := ReadAgenticPacket(p, "benchmark")
	if err != nil {
		t.Fatal(err)
	}
	if len(in.Observations) != 2 || in.Observations[0].CostUSD != nil || in.Observations[1].CostUSD == nil {
		t.Fatalf("observations=%#v", in.Observations)
	}
}
func TestReadAgenticPacketRefusesPendingPacket(t *testing.T) {
	p := filepath.Join(t.TempDir(), "packet.json")
	if err := os.WriteFile(p, []byte(`{"schema":"fak.agentic-benchmark-result-packet.v1","status":"PENDING","result_claim_allowed":false}`), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadAgenticPacket(p, "benchmark"); err == nil {
		t.Fatal("wanted graduation refusal")
	}
}

func TestReadAgenticPacketRefusesClaimBitWithoutGraduationEvidence(t *testing.T) {
	p := filepath.Join(t.TempDir(), "packet.json")
	body := `{"schema":"fak.agentic-benchmark-result-packet.v1","status":"PASS_RESULT","result_claim_allowed":true,"value_chain":[{"role":"raw","trace_id":"r","turns":1,"outcomes":{"passed":1},"provenance":"self-report"}]}`
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadAgenticPacket(p, "benchmark"); err == nil {
		t.Fatal("claim bit without benchmark-native graduation evidence was accepted")
	}
}
