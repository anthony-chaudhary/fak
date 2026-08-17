package valuechain

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadAgenticPacketPreservesUnknownCost(t *testing.T) {
	p := filepath.Join(t.TempDir(), "packet.json")
	body := `{"schema":"fak.agentic-benchmark-result-packet.v1","status":"PASS_RESULT","result_claim_allowed":true,"value_chain":[{"role":"raw","trace_id":"r","pair_id":"p","turns":4,"outcomes":{"passed":1},"provenance":"official-grader"},{"role":"fak","trace_id":"f","pair_id":"p","turns":3,"cost_usd":0.3,"outcomes":{"passed":1},"provenance":"official-grader+bill"}]}`
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
