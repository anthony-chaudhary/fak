package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func TestOrchestrationLaunchConservesOneParentEnvelope(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	task, err := os.ReadFile(filepath.Join("..", "..", "docs", "_witnesses", "issue-8168-ultracode-live", "task.txt"))
	if err != nil {
		t.Fatal(err)
	}
	started := time.Date(2026, 8, 22, 4, 27, 54, 0, time.UTC)
	t.Setenv("CODEX_THREAD_ID", "session-budget-envelope")
	oldNow := orchestrationLaunchNow
	orchestrationLaunchNow = func() time.Time {
		return started
	}
	t.Cleanup(func() { orchestrationLaunchNow = oldNow })
	oldLauncher := orchestrationWorkerLauncher
	var requests []orchestrationWorkerLaunchRequest
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		requests = append(requests, req)
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 100 + len(requests), Status: "started"}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = oldLauncher })

	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "ultracode", "--task-text", string(task),
		"--max-tokens", "65536", "--max-wall", "3m", "--codex-home", home, "--launch", "--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if len(requests) != 3 {
		t.Fatalf("launched %d children, want 3", len(requests))
	}
	var reserved int64
	for _, req := range requests {
		reserved += req.TokenBudget
		wantRemaining := 3 * time.Minute
		if req.DeadlineAt != started.Add(3*time.Minute) || req.RemainingWall != wantRemaining {
			t.Fatalf("child %s wall envelope = deadline %s remaining %s", req.Role.ID, req.DeadlineAt, req.RemainingWall)
		}
		args := strings.Join(orchestrationWorkerArgs(req, "audit.jsonl"), " ")
		for _, want := range []string{
			"--context-budget-tokens " + strconv.FormatInt(req.TokenBudget, 10),
			"--max-duration " + wantRemaining.String(),
			"--session-id " + req.RunID + "-" + req.Role.ID,
		} {
			if !strings.Contains(args, want) {
				t.Fatalf("child %s args missing %q: %s", req.Role.ID, want, args)
			}
		}
	}
	if reserved != 65_536 {
		t.Fatalf("child token budgets sum to %d, want one 65536-token parent envelope", reserved)
	}
	receipt, ok := readCodexOrchestrationLaunchReceipt(home, "session-budget-envelope")
	if !ok || receipt.Budget.DeclaredTokens != 65_536 || receipt.Budget.TotalChildren != 3 || receipt.Budget.DeadlineAt != started.Add(3*time.Minute) {
		t.Fatalf("persisted launch budget = %+v ok=%v", receipt.Budget, ok)
	}
	t.Logf("redacted #8168 rerun: children=%d declared_tokens=%d shared_deadline=%s", receipt.Budget.TotalChildren, receipt.Budget.DeclaredTokens, receipt.Budget.DeadlineAt.Format(time.RFC3339))
}

func TestOrchestrationLaunchDeclinesAfterSharedDeadlineExpires(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	started := time.Date(2026, 8, 22, 4, 27, 54, 0, time.UTC)
	t.Setenv("CODEX_THREAD_ID", "session-budget-deadline")
	oldNow := orchestrationLaunchNow
	first := true
	orchestrationLaunchNow = func() time.Time {
		if first {
			first = false
			return started
		}
		return started.Add(time.Millisecond)
	}
	t.Cleanup(func() { orchestrationLaunchNow = oldNow })
	oldLauncher := orchestrationWorkerLauncher
	launched := false
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		launched = true
		return codexOrchestrationWorkerLaunch{}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = oldLauncher })

	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "ultracode", "--task-text", "review three independent invariants",
		"--max-tokens", "65536", "--max-wall", "1ms", "--codex-home", home, "--launch", "--json",
	})
	if code != 1 || launched || !strings.Contains(stderr.String(), orchestration.UltracodeBudgetReasonWallOverrun) {
		t.Fatalf("code=%d launched=%t stderr=%s", code, launched, stderr.String())
	}
	receipt, ok := readCodexOrchestrationLaunchReceipt(home, "session-budget-deadline")
	if !ok || receipt.Status != "invalid" || receipt.DeclineReason != orchestration.UltracodeBudgetReasonWallOverrun {
		t.Fatalf("persisted deadline refusal = %+v ok=%v", receipt, ok)
	}
}

func TestUltracodeStatusPersistsProviderAggregateOverrun(t *testing.T) {
	home := t.TempDir()
	runID := "orch-budget-overrun"
	runDir := filepath.Join(home, "fak-orchestration-runs", runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	started := time.Date(2026, 8, 22, 4, 27, 54, 0, time.UTC)
	budget, err := orchestration.NewUltracodeEnvelopeReceipt(65_536, 3*time.Minute, started, []string{"worker-1", "worker-2", "worker-3"})
	if err != nil {
		t.Fatal(err)
	}
	usage := [][3]int64{{59_253, 38_400, 377}, {59_179, 38_400, 300}, {117_954, 96_256, 661}}
	workers := make([]codexOrchestrationWorkerLaunch, 0, len(usage))
	for i, tokens := range usage {
		roleID := "worker-" + strconv.Itoa(i+1)
		logRel := filepath.Join("fak-orchestration-runs", runID, roleID+".jsonl")
		line := `{"type":"turn.started"}` + "\n" +
			`{"type":"turn.completed","usage":{"input_tokens":` + strconv.FormatInt(tokens[0], 10) +
			`,"cached_input_tokens":` + strconv.FormatInt(tokens[1], 10) +
			`,"output_tokens":` + strconv.FormatInt(tokens[2], 10) + `}}` + "\n"
		if err := os.WriteFile(filepath.Join(home, logRel), []byte(line), 0o600); err != nil {
			t.Fatal(err)
		}
		workers = append(workers, codexOrchestrationWorkerLaunch{RoleID: roleID, PID: 99_999_999, Status: "started", LogPath: logRel})
	}
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: "session-budget-status", RunID: runID,
		LaunchedAt: started, RequestedProfile: "ultracode", ResolvedProfile: "ultracode",
		Status: "launched", Workers: workers, Budget: budget,
	}
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		t.Fatal(err)
	}
	oldNow := ultracodeStatusNow
	ultracodeStatusNow = func() time.Time { return started.Add(34 * time.Second) }
	t.Cleanup(func() { ultracodeStatusNow = oldNow })

	var stdout, stderr bytes.Buffer
	if code := runUltracodeStatus(&stdout, &stderr, []string{"--home", home, "--session", "session-budget-status", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got ultracodeStatus
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.State != "invalid" || got.Budget.ConsumedTokens != 237_724 || got.Budget.CoveredChildren != 3 || !got.Budget.Overrun || got.Budget.Admitted || got.Budget.Reason != orchestration.UltracodeBudgetReasonTokenOverrun {
		t.Fatalf("status admitted #8168 overrun: %+v", got)
	}
	if strings.Contains(stdout.String(), "log_path") {
		t.Fatalf("redacted status leaked a worker path:\n%s", stdout.String())
	}
	persisted, ok := readCodexOrchestrationLaunchReceipt(home, "session-budget-status")
	if !ok || persisted.Budget.ConsumedTokens != 237_724 || !persisted.Budget.Overrun || persisted.Budget.Authority != orchestration.UltracodeBudgetAuthorityProvider {
		t.Fatalf("persisted budget receipt = %+v ok=%v", persisted.Budget, ok)
	}
}

func TestUltracodeBenchAbstainsOnOverrunOrIncompleteFleetReceipt(t *testing.T) {
	pair := ultracodeBenchSelfcheckPair()
	pair.Single.Identity.TokenBudget, pair.Fleet.Identity.TokenBudget = 65_536, 65_536
	pair.Single.Identity.WallBudgetMS, pair.Fleet.Identity.WallBudgetMS = 180_000, 180_000
	baseReport, err := ultracodebench.Evaluate(pair)
	if err != nil {
		t.Fatal(err)
	}
	if baseReport.Verdict != "GAIN" {
		t.Fatalf("fixture must begin with a value verdict, got %s: %v", baseReport.Verdict, baseReport.Reasons)
	}
	started := time.Date(2026, 8, 22, 4, 27, 54, 0, time.UTC)
	receipt, err := orchestration.NewUltracodeEnvelopeReceipt(65_536, 3*time.Minute, started, []string{"worker-1", "worker-2", "worker-3"})
	if err != nil {
		t.Fatal(err)
	}
	receipt, err = orchestration.FoldUltracodeEnvelopeReceipt(receipt, []orchestration.UltracodeChildUsage{
		{ChildID: "worker-1", ProviderTokens: 59_630, Authority: orchestration.UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-2", ProviderTokens: 59_479, Authority: orchestration.UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-3", ProviderTokens: 118_615, Authority: orchestration.UltracodeBudgetAuthorityProvider},
	}, started.Add(34*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	overrun := applyUltracodeBenchBudgetReceipt(baseReport, pair, ultracodeBenchBudgetReceipt{Budget: receipt})
	if overrun.Verdict != "ABSTAIN" || len(overrun.Reasons) != 1 || overrun.Reasons[0] != orchestration.UltracodeBudgetReasonTokenOverrun {
		t.Fatalf("bench admitted #8168 overrun: verdict=%s reasons=%v", overrun.Verdict, overrun.Reasons)
	}
	pairJSON, err := json.Marshal(pair)
	if err != nil {
		t.Fatal(err)
	}
	var pairDocument map[string]json.RawMessage
	if err := json.Unmarshal(pairJSON, &pairDocument); err != nil {
		t.Fatal(err)
	}
	pairDocument["budget"], err = json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	pairJSON, err = json.Marshal(pairDocument)
	if err != nil {
		t.Fatal(err)
	}
	pairPath := filepath.Join(t.TempDir(), "issue-8168-overrun-pair.json")
	if err := os.WriteFile(pairPath, pairJSON, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runUltracodeBench(&stdout, &stderr, []string{"--pair", pairPath, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var cliReport ultracodebench.Report
	if err := json.Unmarshal(stdout.Bytes(), &cliReport); err != nil {
		t.Fatal(err)
	}
	if cliReport.Verdict != "ABSTAIN" || len(cliReport.Reasons) != 1 || cliReport.Reasons[0] != orchestration.UltracodeBudgetReasonTokenOverrun {
		t.Fatalf("CLI admitted #8168 overrun: verdict=%s reasons=%v", cliReport.Verdict, cliReport.Reasons)
	}

	incomplete := applyUltracodeBenchBudgetReceipt(baseReport, pair, ultracodeBenchBudgetReceipt{})
	if incomplete.Verdict != "ABSTAIN" || len(incomplete.Reasons) != 1 || incomplete.Reasons[0] != orchestration.UltracodeBudgetReasonIncomplete {
		t.Fatalf("bench admitted missing receipt: verdict=%s reasons=%v", incomplete.Verdict, incomplete.Reasons)
	}
}
