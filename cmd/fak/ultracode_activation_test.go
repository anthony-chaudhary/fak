package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func TestOrchestrationPersistsRunningProvisionalReceiptBeforeWorkerWait(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	t.Setenv("CODEX_THREAD_ID", "session-provisional")
	launchedAt := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	oldLaunchNow := orchestrationLaunchNow
	orchestrationLaunchNow = func() time.Time { return launchedAt }
	t.Cleanup(func() { orchestrationLaunchNow = oldLaunchNow })
	oldStatusNow := ultracodeStatusNow
	ultracodeStatusNow = func() time.Time { return launchedAt.Add(2 * time.Second) }
	t.Cleanup(func() { ultracodeStatusNow = oldStatusNow })
	oldLauncher := orchestrationWorkerLauncher
	started := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseWorker := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseWorker)
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		beforeStart, ok := readCodexOrchestrationLaunchReceipt(home, "session-provisional")
		if !ok {
			return codexOrchestrationWorkerLaunch{}, fmt.Errorf("missing pre-spawn receipt")
		}
		beforeStartStatus, err := projectUltracodeStatus(home, beforeStart)
		if err != nil {
			return codexOrchestrationWorkerLaunch{}, err
		}
		if len(beforeStart.Workers) != 1 || beforeStart.Workers[0].Status != "starting" || beforeStart.Workers[0].PID != 0 || beforeStartStatus.State != "launching" || beforeStartStatus.Workers[0].ActivationVerdict != ultracodeActivationVerdictPending {
			return codexOrchestrationWorkerLaunch{}, fmt.Errorf("untruthful pre-spawn receipt/status: receipt=%+v status=%+v", beforeStart, beforeStartStatus)
		}
		provisional := codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: os.Getpid(), Status: "started"}
		if err := req.RecordStarted(provisional); err != nil {
			return provisional, err
		}
		close(started)
		<-release // deterministic stand-in for a live launcher's cmd.Wait
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 99_999_999, Status: "joined"}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = oldLauncher })

	type launchResult struct {
		code   int
		stderr string
	}
	done := make(chan launchResult, 1)
	go func() {
		var stdout, stderr bytes.Buffer
		code := runOrchestration(&stdout, &stderr, []string{
			"plan", "--profile", "ultracode", "--max-workers", "2",
			"--task-text", "implement and independently verify the activation receipt",
			"--codex-home", home, "--launch", "--json",
		})
		done <- launchResult{code: code, stderr: stderr.String()}
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not reach the simulated wait")
	}

	// This is the state after cmd.Start and before the launcher's simulated
	// cmd.Wait. It must survive a controller crash at this exact point.
	got, ok := readCodexOrchestrationLaunchReceipt(home, "session-provisional")
	if !ok {
		t.Fatal("missing provisional receipt")
	}
	status, err := projectUltracodeStatus(home, got)
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != "launching" || len(got.Workers) != 1 || got.Workers[0].PID != os.Getpid() || status.State != "running" {
		t.Fatalf("provisional receipt/status = workers=%+v state=%s, want one durable running child", got.Workers, status.State)
	}
	if status.BudgetPhase != ultracodeBudgetPhaseProvisional || len(status.Workers) != 1 || status.Workers[0].Activation != ultracodebench.ActivationUnknown || status.Workers[0].ActivationVerdict != ultracodeActivationVerdictPending || status.Workers[0].ActivationReason != ultracodeActivationReasonPending || status.Workers[0].ActivationAgeMS != 2000 || status.Workers[0].ActivationDeadline != got.Budget.DeadlineAt {
		t.Fatalf("provisional status = %+v", status)
	}
	if len(got.Activations) != 1 || got.Activations[0].ChildID != got.Workers[0].RoleID || !got.Activations[0].Injected {
		t.Fatalf("provisional activation join = %+v", got.Activations)
	}
	if got.Budget.Schema != orchestration.UltracodeEnvelopeReceiptSchema || got.Budget.TotalChildren != 1 || got.Budget.CoveredChildren != 0 || got.Budget.Complete || got.Budget.Authority != orchestration.UltracodeBudgetAuthorityIncomplete || got.Budget.Reason != orchestration.UltracodeBudgetReasonIncomplete || len(got.Budget.Children) != 1 || got.Budget.Children[0].ReservedTokens != got.Budget.DeclaredTokens {
		t.Fatalf("provisional envelope = %+v", got.Budget)
	}
	ultracodeStatusNow = func() time.Time { return got.Budget.DeadlineAt }
	timedOut, err := projectUltracodeStatus(home, got)
	if err != nil {
		t.Fatal(err)
	}
	if timedOut.Workers[0].ActivationVerdict != ultracodeActivationVerdictFailed || timedOut.Workers[0].ActivationReason != ultracodeActivationReasonDeadlineExceeded || timedOut.Workers[0].Activation != ultracodebench.ActivationDegraded || timedOut.Activation.Unknown != 0 || timedOut.Activation.Degraded != 1 {
		t.Fatalf("deadline did not bound pending activation: %+v", timedOut)
	}
	ultracodeStatusNow = func() time.Time { return launchedAt.Add(2 * time.Second) }

	releaseWorker()
	result := <-done
	if result.code != 0 {
		t.Fatalf("launch code=%d stderr=%s", result.code, result.stderr)
	}
	final, ok := readCodexOrchestrationLaunchReceipt(home, "session-provisional")
	if !ok || final.Status != "launched" || len(final.Workers) != 1 || final.Workers[0].PID != 99_999_999 || final.Workers[0].Status != "joined" {
		t.Fatalf("final joined receipt = %+v ok=%v", final, ok)
	}
	if len(final.Activations) != 1 || final.Budget.DeclaredTokens != got.Budget.DeclaredTokens || len(final.Budget.Children) != 1 {
		t.Fatalf("final receipt lost launch facts: activation=%+v budget=%+v", final.Activations, final.Budget)
	}
	finalStatus, err := projectUltracodeStatus(home, final)
	if err != nil {
		t.Fatal(err)
	}
	if finalStatus.Workers[0].ActivationVerdict != ultracodeActivationVerdictFailed || finalStatus.Workers[0].ActivationReason != ultracodeActivationReasonChildExited || finalStatus.Activation.Unknown != 0 || finalStatus.Activation.Degraded != 1 {
		t.Fatalf("joined child without activation evidence was treated as healthy: %+v", finalStatus)
	}
}

func TestUltracodeStatusProjectsVerifiedFinalReceipts(t *testing.T) {
	home := t.TempDir()
	runID := "orch-final"
	runDir := filepath.Join(home, "fak-orchestration-runs", runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	logRel := filepath.Join("fak-orchestration-runs", runID, "worker-1.jsonl")
	log := "{\"type\":\"turn.started\"}\n{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":100,\"output_tokens\":20}}\n"
	if err := os.WriteFile(filepath.Join(home, logRel), []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	started := time.Date(2026, 8, 24, 13, 0, 0, 0, time.UTC)
	budget, err := orchestration.NewUltracodeEnvelopeReceipt(4096, time.Minute, started, []string{"worker-1"})
	if err != nil {
		t.Fatal(err)
	}
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: "session-final", RunID: runID,
		LaunchedAt: started, RequestedProfile: "ultracode", ResolvedProfile: "ultracode", Status: "launched",
		Workers: []codexOrchestrationWorkerLaunch{{RoleID: "worker-1", PID: 99_999_999, Status: "joined", LogPath: logRel, DeadlineAt: budget.DeadlineAt}},
		Budget:  budget,
	}
	activation, err := codexOrchestrationActivation(receipt, "worker-1")
	if err != nil {
		t.Fatal(err)
	}
	activation, err = ultracodebench.Acknowledge(activation, ultracodebench.ObservableActive, ultracodebench.SourceExplicitAcknowledgement)
	if err != nil {
		t.Fatal(err)
	}
	receipt.Activations = []ultracodebench.ActivationReceipt{activation}
	oldNow := ultracodeStatusNow
	ultracodeStatusNow = func() time.Time { return started.Add(10 * time.Second) }
	t.Cleanup(func() { ultracodeStatusNow = oldNow })

	status, err := projectUltracodeStatus(home, receipt)
	if err != nil {
		t.Fatal(err)
	}
	if status.State != "complete" || status.BudgetPhase != ultracodeBudgetPhaseFinal || !status.Budget.Complete || !status.Budget.Admitted || status.Budget.CoveredChildren != 1 {
		t.Fatalf("final envelope status = %+v", status)
	}
	if len(status.Workers) != 1 || status.Workers[0].Activation != ultracodebench.ActivationActive || status.Workers[0].ActivationVerdict != ultracodeActivationVerdictVerifiedActive || status.Workers[0].ActivationReason != "" || status.Activation.Active != 1 || status.Activation.Unknown != 0 {
		t.Fatalf("final activation status = %+v", status)
	}
}

func TestOrchestrationPersistsActivationBeforeSpawn(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	t.Setenv("CODEX_THREAD_ID", "session-activation")
	old := orchestrationWorkerLauncher
	launches := 0
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		launches++
		persisted, ok := readCodexOrchestrationLaunchReceipt(home, "session-activation")
		if !ok || len(persisted.Activations) != launches {
			t.Fatalf("pre-spawn receipt missing at launch %d: ok=%v receipt=%+v", launches, ok, persisted)
		}
		if len(persisted.Workers) != launches || persisted.Workers[launches-1].RoleID != req.Role.ID || persisted.Workers[launches-1].Status != "starting" || persisted.Workers[launches-1].PID != 0 {
			t.Fatalf("truthful pre-spawn worker missing at launch %d: %+v", launches, persisted.Workers)
		}
		activation := persisted.Activations[launches-1]
		if activation.ChildID != req.Role.ID || activation.State() != ultracodebench.ActivationUnknown || !activation.Injected {
			t.Fatalf("activation=%+v request=%+v", activation, req)
		}
		raw, _ := json.Marshal(activation)
		for _, forbidden := range []string{"path", "prompt", "account", "host", "settings", "argv"} {
			if strings.Contains(string(raw), forbidden) {
				t.Fatalf("activation retained %q: %s", forbidden, raw)
			}
		}
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 100 + launches, Status: "started"}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = old })

	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "ultracode", "--task-text", "implement and verify the activation receipt", "--codex-home", home, "--launch", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if launches < 2 {
		t.Fatalf("launches=%d", launches)
	}
}

func TestUltracodeStatusReportsActivationCoverageWithoutPrivatePaths(t *testing.T) {
	home := t.TempDir()
	active, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "orch-status", ChildID: "active", Harness: "codex", Requested: ultracodebench.SettingOn, Resolved: ultracodebench.SettingOn, Injected: true})
	if err != nil {
		t.Fatal(err)
	}
	active, err = ultracodebench.Acknowledge(active, ultracodebench.ObservableActive, ultracodebench.SourceExplicitAcknowledgement)
	if err != nil {
		t.Fatal(err)
	}
	degraded, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "orch-status", ChildID: "degraded", Harness: "codex", Requested: ultracodebench.SettingOn, Resolved: ultracodebench.SettingOn, Degradations: []string{"harness_cannot_inject"}})
	if err != nil {
		t.Fatal(err)
	}
	unknown, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "orch-status", ChildID: "unknown", Harness: "codex", Requested: ultracodebench.SettingOn, Resolved: ultracodebench.SettingOn, Injected: true})
	if err != nil {
		t.Fatal(err)
	}
	inactive, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "orch-status", ChildID: "inactive", Harness: "codex", Requested: ultracodebench.SettingOff, Resolved: ultracodebench.SettingOff})
	if err != nil {
		t.Fatal(err)
	}
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: "activation-status", RunID: "orch-status",
		RequestedProfile: "auto", ResolvedProfile: "ultracode", Status: "launched",
		Workers: []codexOrchestrationWorkerLaunch{
			{RoleID: "active", Status: "started", LogPath: `C:\private\active.jsonl`},
			{RoleID: "degraded", Status: "started", LogPath: `C:\private\degraded.jsonl`},
			{RoleID: "unknown", Status: "started", LogPath: `C:\private\unknown.jsonl`},
			{RoleID: "inactive", Status: "started", LogPath: `C:\private\inactive.jsonl`},
		},
		Activations: []ultracodebench.ActivationReceipt{active, degraded, unknown, inactive},
	}
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if code := runUltracodeStatus(&out, &stderr, []string{"--home", home, "--session", "activation-status", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got ultracodeStatus
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != ultracodeStatusSchema || got.Activation.Active != 1 || got.Activation.Degraded != 2 || got.Activation.Unknown != 0 || got.Activation.Inactive != 1 {
		t.Fatalf("status=%+v", got)
	}
	if got.Workers[2].ActivationVerdict != ultracodeActivationVerdictFailed || got.Workers[2].ActivationReason != ultracodeActivationReasonChildExited {
		t.Fatalf("exited child retained unknown activation: %+v", got.Workers[2])
	}
	for _, forbidden := range []string{"log_path", "C:\\\\private", "prompt", "account", "host", "raw_settings", "argv"} {
		if strings.Contains(out.String(), forbidden) {
			t.Fatalf("status retained %q:\n%s", forbidden, out.String())
		}
	}
}

func TestUltracodeStatusMarksLegacyChildrenUnknown(t *testing.T) {
	home := t.TempDir()
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: "legacy", RunID: "orch-legacy",
		Workers: []codexOrchestrationWorkerLaunch{{RoleID: "worker-1", Status: "started"}},
	}
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		t.Fatal(err)
	}
	got, err := projectUltracodeStatus(home, receipt)
	if err != nil {
		t.Fatal(err)
	}
	if got.Activation.Total != 1 || got.Activation.Unknown != 1 || got.Workers[0].Activation != ultracodebench.ActivationUnknown {
		t.Fatalf("legacy status=%+v", got)
	}
}

func TestUltracodeStatusAccountsNestedCodexRootGoalExactlyOnce(t *testing.T) {
	home := t.TempDir()
	runID := "orch-nested-usage"
	runDir := filepath.Join(home, "fak-orchestration-runs", runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	logRel := filepath.Join("fak-orchestration-runs", runID, "worker-1.jsonl")
	started := time.Date(2026, 8, 31, 16, 0, 0, 0, time.UTC)
	log := strings.Join([]string{
		`{"type":"turn.started"}`,
		`{"type":"event_msg","payload":{"type":"thread_goal_updated","goal":{"threadId":"root-resumed","createdAt":` + fmt.Sprint(started.Add(-time.Hour).Unix()) + `,"tokensUsed":100}}}`,
		`{"type":"event_msg","payload":{"type":"thread_goal_updated","goal":{"threadId":"root-resumed","createdAt":` + fmt.Sprint(started.Add(-time.Hour).Unix()) + `,"tokensUsed":160}}}`,
		`{"type":"event_msg","payload":{"type":"thread_goal_updated","goal":{"threadId":"root-resumed","createdAt":` + fmt.Sprint(started.Add(-time.Hour).Unix()) + `,"tokensUsed":190}}}`,
		`{"type":"turn.completed","usage":{"input_tokens":40,"output_tokens":10}}`,
		`{"type":"event_msg","payload":{"type":"thread_goal_updated","goal":{"threadId":"root-replacement","createdAt":` + fmt.Sprint(started.Add(time.Second).Unix()) + `,"tokensUsed":20}}}`,
		`{"type":"event_msg","payload":{"type":"thread_goal_updated","goal":{"threadId":"root-replacement","createdAt":` + fmt.Sprint(started.Add(time.Second).Unix()) + `,"tokensUsed":35}}}`,
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(home, logRel), []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	budget, err := orchestration.NewUltracodeEnvelopeReceipt(4096, time.Minute, started, []string{"worker-1"})
	if err != nil {
		t.Fatal(err)
	}
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: "session-nested", RunID: runID,
		LaunchedAt: started, RequestedProfile: "ultracode", ResolvedProfile: "ultracode", Status: "launched",
		Workers: []codexOrchestrationWorkerLaunch{{RoleID: "worker-1", PID: 99_999_999, Status: "joined", LogPath: logRel, DeadlineAt: budget.DeadlineAt}},
		Budget:  budget,
	}
	oldNow := ultracodeStatusNow
	ultracodeStatusNow = func() time.Time { return started.Add(10 * time.Second) }
	t.Cleanup(func() { ultracodeStatusNow = oldNow })

	status, err := projectUltracodeStatus(home, receipt)
	if err != nil {
		t.Fatal(err)
	}
	// The resumed goal contributes 190-100=90 and the replacement goal
	// contributes its cumulative 35 once. Direct turn usage (50) is retained
	// for attribution/lower-bound checking but is not added to the root total.
	if got, want := status.Budget.ConsumedTokens, int64(125); got != want {
		t.Fatalf("nested root-goal usage=%d, want %d exactly once: %+v", got, want, status.Budget)
	}
	if !status.Budget.Complete || status.Budget.Authority != orchestration.UltracodeBudgetAuthorityProvider || status.Budget.CoveredChildren != 1 {
		t.Fatalf("nested root-goal authority=%+v", status.Budget)
	}
}

func TestInspectCodexRootGoalUsagePreservesDirectSessionAccounting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "worker.jsonl")
	started := time.Date(2026, 8, 31, 16, 0, 0, 0, time.UTC)
	log := `{"type":"event_msg","payload":{"type":"thread_goal_updated","goal":{"threadId":"root","createdAt":` + fmt.Sprint(started.Unix()) + `,"tokensUsed":30}}}` + "\n"
	if err := os.WriteFile(path, []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	got, covered, err := inspectCodexRootGoalUsage(path, started, 50)
	if err != nil {
		t.Fatal(err)
	}
	if !covered || got != 50 {
		t.Fatalf("root usage=(%d,%v), want direct-session lower bound (50,true)", got, covered)
	}
}
