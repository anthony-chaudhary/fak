package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
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

func TestClaudeErrorResultDoesNotReportZeroCost(t *testing.T) {
	var got claudeResult
	if err := json.Unmarshal([]byte(`{"is_error":true,"api_error_status":429,"result":"usage limit; retry in 1h","total_cost_usd":0}`), &got); err != nil {
		t.Fatal(err)
	}
	if !got.IsError || got.APIErrorStatus != 429 || got.TotalCostUSD == nil {
		t.Fatalf("decoded=%+v", got)
	}
	arm := pairedArm{CostStatus: "provider-unreported"}
	if !got.IsError && got.TotalCostUSD != nil {
		arm.CostUSD = got.TotalCostUSD
	}
	if arm.CostUSD != nil || arm.CostStatus != "provider-unreported" {
		t.Fatalf("pre-inference error became a cost claim: %+v", arm)
	}
}

func TestFilteredPairedEnvDropsOnlyStaleRegistrationContext(t *testing.T) {
	got := filteredPairedEnv([]string{
		"FAK_REGISTRATION_ID=stale",
		"FAK_ROOT_REGISTRATION_ID=root",
		"FAK_PARENT_REGISTRATION_ID=parent",
		"FAK_SPAWN_GRANT_ID=grant",
		"FAK_GUARD_CAP_PARK=1",
		"ANTHROPIC_API_KEY=keep",
	})
	joined := strings.Join(got, "\n")
	if strings.Contains(joined, "REGISTRATION_ID=") || strings.Contains(joined, "SPAWN_GRANT_ID=") {
		t.Fatalf("stale registration context survived: %q", joined)
	}
	if strings.Contains(joined, "FAK_GUARD_CAP_PARK=1") {
		t.Fatalf("inherited cap recovery survived: %q", joined)
	}
	if !strings.Contains(joined, "ANTHROPIC_API_KEY=keep") || !strings.Contains(joined, "FLEET_CLAUDE_GUARD_CRASH_RESTART_LIMIT=0") || !strings.Contains(joined, "FAK_GUARD_CAP_PARK=off") || !strings.Contains(joined, "FAK_GUARD_AFFORDANCE_MODE=off") {
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
	if pairedBaselineGuardTimeout <= 0 || pairedBaselineParentTimeout <= pairedBaselineGuardTimeout {
		t.Fatalf("baseline envelopes invalid: guard=%s parent=%s", pairedBaselineGuardTimeout, pairedBaselineParentTimeout)
	}
	// The parent CommandContext and the managed guard receive the same wall-clock
	// envelope: the guard can emit a typed TIME_BUDGET_EXHAUSTED result, while the
	// parent deadline remains the final process-tree backstop.
	source, err := os.ReadFile("micro_paired.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	for _, want := range []string{"context.WithTimeout(ctx, pairedBaselineParentTimeout)", `"--rotate", "off"`, `"--max-duration", pairedBaselineGuardTimeout.String()`} {
		if !strings.Contains(text, want) {
			t.Fatalf("baseline command lacks %q", want)
		}
	}
}

func TestPairedEnvLookupPreservesExplicitSeat(t *testing.T) {
	env := []string{"Path=x", "CLAUDE_CONFIG_DIR=C:\\seat"}
	if !pairedEnvHas(env, "claude_config_dir") || pairedEnvValue(env, "CLAUDE_CONFIG_DIR") != `C:\seat` {
		t.Fatalf("seat lookup failed: %v", env)
	}
}

func TestPairedReadyClaudeConfigDirUsesServableAccount(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture is POSIX-only; Windows path is live-witnessed")
	}
	dir := t.TempDir()
	exe := filepath.Join(dir, "fak")
	if err := os.WriteFile(exe, []byte("#!/bin/sh\nprintf '%s' '{\"dir\":\"/seat/ready\",\"can_serve\":true}'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := pairedReadyClaudeConfigDir(context.Background(), exe); got != "/seat/ready" {
		t.Fatalf("config dir=%q", got)
	}
}

func TestReconcilePairedBaselineCooldownPersistsExactSeat(t *testing.T) {
	t.Setenv("FLEET_STATE_DIR", t.TempDir())
	now := time.Date(2026, 8, 14, 11, 0, 0, 0, time.UTC)
	root := t.TempDir()
	home := mkFailoverHome(t, root, ".claude-paired", "paired@example.test", "paired-account", "token", now.Add(time.Hour).UnixMilli())
	got := claudeResult{IsError: true, APIErrorStatus: 429, Result: "API Error 429: usage limit reached; try again in 1h6m"}
	if !reconcilePairedBaselineCooldown([]string{"CLAUDE_CONFIG_DIR=" + home.Dir}, got, now) {
		t.Fatal("provider cap was not persisted")
	}
	store, err := accounts.LoadCooldownStore(defaultCooldownStorePath())
	if err != nil {
		t.Fatal(err)
	}
	entry, ok := store.CooledDown(home.Identity.AccountKey(), now.Add(time.Minute))
	if !ok || entry.ResetAt.Before(now.Add(time.Hour)) {
		t.Fatalf("cooldown=%+v ok=%v", entry, ok)
	}
}

func TestReconcilePairedBaselineCooldownIgnoresSuccessfulZeroCost(t *testing.T) {
	t.Setenv("FLEET_STATE_DIR", t.TempDir())
	zero := 0.0
	got := claudeResult{TotalCostUSD: &zero, Result: "READY"}
	if reconcilePairedBaselineCooldown([]string{"CLAUDE_CONFIG_DIR=C:\\seat"}, got, time.Now()) {
		t.Fatal("successful provider-reported zero became a cooldown")
	}
}
