package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachewitness"
)

func TestRunSwebenchCacheWitnessSubtractsBaseline(t *testing.T) {
	dir := t.TempDir()
	before := filepath.Join(dir, "before.prom")
	after := filepath.Join(dir, "after.prom")
	const beforeBody = `fak_gateway_kv_prefix_turns_total 10
fak_gateway_kv_prefix_prompt_tokens_total 10000
fak_gateway_kv_prefix_reused_tokens_total 4000
fak_gateway_kv_prefix_turns_by_regime_total{regime="frozen"} 2
fak_gateway_kv_prefix_turns_by_regime_total{regime="partial"} 3
fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} 5
fak_gateway_inference_cached_prompt_tokens_total 100
`
	const afterBody = `fak_gateway_kv_prefix_turns_total 13
fak_gateway_kv_prefix_prompt_tokens_total 19000
fak_gateway_kv_prefix_reused_tokens_total 8500
fak_gateway_kv_prefix_turns_by_regime_total{regime="frozen"} 3
fak_gateway_kv_prefix_turns_by_regime_total{regime="partial"} 5
fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} 5
fak_gateway_inference_cached_prompt_tokens_total 150
`
	if err := os.WriteFile(before, []byte(beforeBody), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(after, []byte(afterBody), 0o600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	rc := runSwebenchCacheWitness(&stdout, &stderr, []string{"--metrics-file", after, "--baseline", before})
	if rc != 0 {
		t.Fatalf("runSwebenchCacheWitness rc=%d stderr=%s stdout=%s", rc, stderr.String(), stdout.String())
	}
	var rec cachewitness.Record
	if err := json.Unmarshal(stdout.Bytes(), &rec); err != nil {
		t.Fatalf("stdout JSON: %v\n%s", err, stdout.String())
	}
	if rec.KVPrefix.Turns != 3 || rec.KVPrefix.PromptTokens != 9000 || rec.KVPrefix.ReusedTokens != 4500 {
		t.Fatalf("record kv delta = %+v, want 3 turns / 9000 prompt / 4500 reused", rec.KVPrefix)
	}
	if rec.ProviderCacheReadTokens != 50 || rec.GatewayUptimeTurns != 13 {
		t.Fatalf("provider/uptime = %d/%d, want 50/13", rec.ProviderCacheReadTokens, rec.GatewayUptimeTurns)
	}
	if rec.WitnessWindow == nil || rec.WitnessWindow.StartScrape != "file://"+before || rec.WitnessWindow.EndScrape != "file://"+after {
		t.Fatalf("witness window = %+v, want before->after file labels", rec.WitnessWindow)
	}
	if !strings.Contains(stderr.String(), "witness window:") || !strings.Contains(stderr.String(), "gateway uptime turns 13") {
		t.Fatalf("stderr missing witness-window summary:\n%s", stderr.String())
	}
}
