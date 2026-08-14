package main

import (
	"encoding/json"
	"os"
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
	if !strings.Contains(joined, "ANTHROPIC_API_KEY=keep") || !strings.Contains(joined, "FLEET_CLAUDE_GUARD_CRASH_RESTART_LIMIT=0") || !strings.Contains(joined, "FAK_GUARD_AFFORDANCE_MODE=off") {
		t.Fatalf("required benchmark environment missing: %q", joined)
	}
}

func TestFoldPairedUsesOnlyProviderReportedCostForValueVerdict(t *testing.T) {
	microCost, baselineCost := 0.01, 0.02
	r := foldPaired(pairedReport{
		Schema: "fak-micro-paired/1",
		Micro:  pairedArm{Correct: true, InputTokens: 7, OutputTokens: 1, CostUSD: &microCost, CostStatus: "provider-reported"},
		CLI:    pairedArm{Correct: true, InputTokens: 10, OutputTokens: 1, CostUSD: &baselineCost, CostStatus: "provider-reported"},
	})
	if r.ExecutionVerdict != "PASS" || r.ValueVerdict != "MICRO_WINS" {
		t.Fatalf("receipt=%+v", r)
	}
}

func TestRunPairedBaselineAddsBoundedGuardEnvelope(t *testing.T) {
	if pairedBaselineTimeout <= 0 {
		t.Fatalf("baseline timeout must be positive: %s", pairedBaselineTimeout)
	}
	// The parent CommandContext and the managed guard receive the same wall-clock
	// envelope: the guard can emit a typed TIME_BUDGET_EXHAUSTED result, while the
	// parent deadline remains the final process-tree backstop.
	source, err := os.ReadFile("micro_paired.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	for _, want := range []string{"context.WithTimeout(ctx, pairedBaselineTimeout)", `"--max-duration", pairedBaselineTimeout.String()`} {
		if !strings.Contains(text, want) {
			t.Fatalf("baseline command lacks %q", want)
		}
	}
}
