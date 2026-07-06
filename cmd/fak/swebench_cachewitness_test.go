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
	if rec.CacheBitScope != cachewitness.CacheBitScopeAggregateRun {
		t.Fatalf("cache bit scope = %q, want %q", rec.CacheBitScope, cachewitness.CacheBitScopeAggregateRun)
	}
	if rec.WitnessWindow == nil || rec.WitnessWindow.StartScrape != "file://"+before || rec.WitnessWindow.EndScrape != "file://"+after {
		t.Fatalf("witness window = %+v, want before->after file labels", rec.WitnessWindow)
	}
	if !strings.Contains(stderr.String(), "witness window:") || !strings.Contains(stderr.String(), "gateway uptime turns 13") {
		t.Fatalf("stderr missing witness-window summary:\n%s", stderr.String())
	}
	if !strings.Contains(stderr.String(), "WITNESSED, aggregate-run-kv-prefix-reuse") {
		t.Fatalf("stderr missing aggregate cache-bit scope:\n%s", stderr.String())
	}
}

// #3053: --warmup-tax-seconds emits the first-request warmup cost as a SEPARATE
// OBSERVED signal in the record and annotates it in the human summary as warmup-
// dominated (not cache-accelerated), de-conflated from the aggregate cache_bit.
func TestRunSwebenchCacheWitnessEmitsWarmupTaxSeparately(t *testing.T) {
	dir := t.TempDir()
	body := filepath.Join(dir, "metrics.prom")
	const metricsBody = `fak_gateway_kv_prefix_turns_total 7
fak_gateway_kv_prefix_prompt_tokens_total 16384
fak_gateway_kv_prefix_reused_tokens_total 15000
fak_gateway_kv_prefix_turns_by_regime_total{regime="frozen"} 5
fak_gateway_kv_prefix_turns_by_regime_total{regime="partial"} 1
fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} 1
fak_gateway_inference_cached_prompt_tokens_total 2048
`
	if err := os.WriteFile(body, []byte(metricsBody), 0o600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	rc := runSwebenchCacheWitness(&stdout, &stderr, []string{"--metrics-file", body, "--warmup-tax-seconds", "511.3"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var rec cachewitness.Record
	if err := json.Unmarshal(stdout.Bytes(), &rec); err != nil {
		t.Fatalf("stdout JSON: %v\n%s", err, stdout.String())
	}
	if rec.WarmupTax == nil {
		t.Fatalf("record missing warmup tax:\n%s", stdout.String())
	}
	if rec.WarmupTax.TimeToFirstReadySeconds != 511.3 || rec.WarmupTax.Provenance != cachewitness.Observed {
		t.Fatalf("warmup tax = %+v, want 511.3s OBSERVED", rec.WarmupTax)
	}
	if rec.WarmupTax.Scope != cachewitness.WarmupTaxScope {
		t.Fatalf("warmup scope = %q, want %q", rec.WarmupTax.Scope, cachewitness.WarmupTaxScope)
	}
	// The WITNESSED reuse is untouched and distinct from the OBSERVED warmup latency.
	if rec.KVPrefix.ReusedTokens != 15000 {
		t.Fatalf("warmup tax mutated reused_tokens: %d, want 15000", rec.KVPrefix.ReusedTokens)
	}
	if !strings.Contains(stderr.String(), "first-request warmup tax (OBSERVED") ||
		!strings.Contains(stderr.String(), "NOT cache-accelerated") {
		t.Fatalf("stderr missing de-conflated warmup-tax annotation:\n%s", stderr.String())
	}
}
