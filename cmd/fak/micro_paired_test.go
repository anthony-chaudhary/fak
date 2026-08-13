package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFoldPairedPassesExecutionButRefusesUnsupportedCostClaim(t *testing.T) {
	cost := 0.02
	r := foldPaired(pairedReport{
		Schema: "fak-micro-paired/1",
		Micro:  pairedArm{Correct: true, InputTokens: 7, OutputTokens: 1, CostStatus: "provider-unsupported"},
		CLI:    pairedArm{Correct: true, InputTokens: 10, OutputTokens: 1, CostUSD: &cost, CostStatus: "provider-reported"},
	})
	if r.ExecutionVerdict != "PASS" || r.ValueVerdict != "NOT_YET" {
		t.Fatalf("receipt=%+v", r)
	}
	if r.Micro.CostUSD != nil {
		t.Fatalf("unsupported micro cost became a numeric claim: %+v", r.Micro)
	}
}

func TestClaudeResultDecodesProviderCostAndUsage(t *testing.T) {
	var got claudeResult
	if err := json.Unmarshal([]byte(`{"result":"READY","total_cost_usd":0.02,"usage":{"input_tokens":12,"output_tokens":1,"cache_read_input_tokens":3}}`), &got); err != nil {
		t.Fatal(err)
	}
	if got.Result != "READY" || got.TotalCostUSD == nil || *got.TotalCostUSD != 0.02 || got.Usage.InputTokens != 12 || got.Usage.CacheReadInputTokens != 3 {
		t.Fatalf("decoded=%+v", got)
	}
}

func TestClaudeResultMissingCostStaysNull(t *testing.T) {
	var got claudeResult
	if err := json.Unmarshal([]byte(`{"result":"READY","usage":{"input_tokens":12,"output_tokens":1}}`), &got); err != nil {
		t.Fatal(err)
	}
	if got.TotalCostUSD != nil {
		t.Fatalf("missing provider cost became numeric: %v", *got.TotalCostUSD)
	}
}

func TestFilteredPairedEnvDropsOnlyStaleRegistrationContext(t *testing.T) {
	got := filteredPairedEnv([]string{
		"FAK_REGISTRATION_ID=stale",
		"FAK_ROOT_REGISTRATION_ID=root",
		"FAK_PARENT_REGISTRATION_ID=parent",
		"FAK_SPAWN_GRANT_ID=grant",
		"ANTHROPIC_API_KEY=keep",
	})
	joined := strings.Join(got, "\n")
	if strings.Contains(joined, "REGISTRATION_ID=") || strings.Contains(joined, "SPAWN_GRANT_ID=") {
		t.Fatalf("stale registration context survived: %q", joined)
	}
	if !strings.Contains(joined, "ANTHROPIC_API_KEY=keep") || !strings.Contains(joined, "FLEET_CLAUDE_GUARD_CRASH_RESTART_LIMIT=0") {
		t.Fatalf("required environment missing: %q", joined)
	}
}
