package compute

import (
	"os"
	"strings"
	"testing"
)

func TestKVReplayArtifactWitnessesCostAwarePinAndRestore(t *testing.T) {
	data, err := os.ReadFile("testdata/kvbm_agent_replay_issue2666.json")
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := ParseKVReplayArtifact(data)
	if err != nil {
		t.Fatal(err)
	}
	report, err := ReplayKVArtifact(artifact)
	if err != nil {
		t.Fatal(err)
	}

	if !report.CostAwareAtLeastLRU() {
		t.Fatalf("cost-aware replay regressed below LRU: LRU=%d cost-aware=%d accessed=%d",
			report.LRU.HitTokens, report.CostAware.HitTokens, report.CostAware.AccessTokens)
	}
	if !report.Oracle.Exact || report.Oracle.HitTokens != report.CostAware.HitTokens {
		t.Fatalf("oracle = %+v, want exact upper bound matching cost-aware hits %d", report.Oracle, report.CostAware.HitTokens)
	}
	if lead := report.CostAware.HitTokens - report.LRU.HitTokens; lead < 50 {
		t.Fatalf("cost-aware lead = %d tokens, want at least the restored hot-prefix span", lead)
	}
	if report.LRU.PinnedSkips == 0 || report.CostAware.PinnedSkips == 0 {
		t.Fatalf("artifact did not exercise pinned pressure: LRU=%+v cost-aware=%+v", report.LRU, report.CostAware)
	}
	if report.PinViolations() != 0 {
		t.Fatalf("pin safety violated: LRU=%+v cost-aware=%+v", report.LRU, report.CostAware)
	}
	if report.LRU.Restores == 0 {
		t.Fatalf("artifact did not exercise restore-on-access after eviction: LRU=%+v", report.LRU)
	}
	if report.BitDriftMismatches() != 0 {
		t.Fatalf("restore/evict bit drift detected: LRU=%+v cost-aware=%+v", report.LRU, report.CostAware)
	}
}

func TestParseKVReplayArtifactRejectsInvalidRows(t *testing.T) {
	_, err := ParseKVReplayArtifact([]byte(`{
		"schema":"fak.kvbm.replay/v1",
		"name":"bad",
		"budget_bytes":100,
		"events":[{"span_id":"x","tokens":0,"payload":"x"}]
	}`))
	if err == nil || !strings.Contains(err.Error(), "non-positive tokens") {
		t.Fatalf("ParseKVReplayArtifact invalid row error = %v, want non-positive token refusal", err)
	}
}

func TestKVReplayArtifactDetectsRestoreBitDrift(t *testing.T) {
	artifact := KVReplayArtifact{
		Schema:      KVReplayArtifactSchema,
		Name:        "drift",
		BudgetBytes: 1,
		Events: []KVReplayArtifactEvent{
			{SpanID: "hot", Tokens: 1, Bytes: 1, Payload: "original"},
			{SpanID: "cold", Tokens: 1, Bytes: 1, Payload: "cold"},
			{SpanID: "hot", Tokens: 1, Bytes: 1, Payload: "changed"},
		},
	}
	report, err := ReplayKVArtifact(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if report.LRU.Restores == 0 {
		t.Fatalf("drift fixture did not exercise restore path: %+v", report.LRU)
	}
	if report.LRU.BitDriftMismatches == 0 {
		t.Fatalf("restore drift was not detected: %+v", report.LRU)
	}
}
