package goalrunner

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestFormatGoalPrompt(t *testing.T) {
	// 1. Under 4000: prepends /goal if missing
	t.Run("prepend missing goal prefix", func(t *testing.T) {
		res, err := FormatGoalPrompt("solve issue 42")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res != "/goal solve issue 42" {
			t.Fatalf("expected '/goal solve issue 42', got %q", res)
		}
	})

	// 2. Under 4000: preserves existing /goal prefix without duplicating
	t.Run("preserve existing goal prefix", func(t *testing.T) {
		res, err := FormatGoalPrompt("/goal solve issue 42")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res != "/goal solve issue 42" {
			t.Fatalf("expected '/goal solve issue 42', got %q", res)
		}
	})

	// 3. Exactly 4000 chars should pass
	t.Run("boundary 4000 characters passes", func(t *testing.T) {
		prefix := "/goal "
		body := strings.Repeat("a", 4000-len(prefix))
		res, err := FormatGoalPrompt(body)
		if err != nil {
			t.Fatalf("expected 4000 chars to succeed, got error: %v", err)
		}
		if len(res) != 4000 {
			t.Fatalf("expected length 4000, got %d", len(res))
		}
	})

	// 4. Over 4000 chars must be rejected
	t.Run("boundary 4001 characters fails", func(t *testing.T) {
		prefix := "/goal "
		body := strings.Repeat("a", 4001-len(prefix))
		_, err := FormatGoalPrompt(body)
		if err == nil {
			t.Fatalf("expected error for 4001 chars, got nil")
		}
		if !strings.Contains(err.Error(), ">4000 cap") {
			t.Fatalf("expected error to mention '>4000 cap', got %q", err.Error())
		}
	})

	t.Run("large body 5000 characters fails", func(t *testing.T) {
		body := strings.Repeat("x", 5000)
		_, err := FormatGoalPrompt(body)
		if err == nil {
			t.Fatalf("expected error for 5000 chars, got nil")
		}
	})
}

func TestScrubParentAuthEnv(t *testing.T) {
	inputEnv := []string{
		"ANTHROPIC_API_KEY=sk-ant-test-key-12345",
		"ANTHROPIC_BASE_URL=http://localhost:8080",
		"anthropic_custom_header=token",
		"CLAUDE_CODE_SESSION_ID=sess_98765",
		"CLAUDE_CODE_CHILD_SESSION=child_54321",
		"CLAUDE_CONFIG_DIR=/path/to/claude-config",
		"CLAUDE_CODE_OAUTH_TOKEN=oauth_valid_token",
		"PATH=/usr/bin:/bin",
		"USER=testuser",
	}

	scrubbed := ScrubParentAuthEnv(inputEnv)

	// Verify ANTHROPIC_* removed
	for _, entry := range scrubbed {
		upper := strings.ToUpper(entry)
		if strings.HasPrefix(upper, "ANTHROPIC_") {
			t.Errorf("expected ANTHROPIC_* to be stripped, found: %s", entry)
		}
		if strings.HasPrefix(upper, "CLAUDE_CODE_SESSION_ID=") {
			t.Errorf("expected CLAUDE_CODE_SESSION_ID to be stripped, found: %s", entry)
		}
		if strings.HasPrefix(upper, "CLAUDE_CODE_CHILD_SESSION=") {
			t.Errorf("expected CLAUDE_CODE_CHILD_SESSION to be stripped, found: %s", entry)
		}
	}

	// Verify CLAUDE_CONFIG_DIR and other safe env vars preserved
	configDirFound := false
	oauthTokenFound := false
	pathFound := false
	userFound := false

	for _, entry := range scrubbed {
		if entry == "CLAUDE_CONFIG_DIR=/path/to/claude-config" {
			configDirFound = true
		}
		if entry == "CLAUDE_CODE_OAUTH_TOKEN=oauth_valid_token" {
			oauthTokenFound = true
		}
		if entry == "PATH=/usr/bin:/bin" {
			pathFound = true
		}
		if entry == "USER=testuser" {
			userFound = true
		}
	}

	if !configDirFound {
		t.Errorf("expected CLAUDE_CONFIG_DIR to be preserved")
	}
	if !oauthTokenFound {
		t.Errorf("expected CLAUDE_CODE_OAUTH_TOKEN to be preserved")
	}
	if !pathFound {
		t.Errorf("expected PATH to be preserved")
	}
	if !userFound {
		t.Errorf("expected USER to be preserved")
	}
}

func TestResolveFakExe(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Explicit path not found
	_, err := ResolveFakExe(tempDir, filepath.Join(tempDir, "nonexistent-fak.exe"))
	if err == nil {
		t.Fatalf("expected error for nonexistent explicit fak binary, got nil")
	}

	// 2. Explicit path exists
	explicitFile := filepath.Join(tempDir, "my-fak.exe")
	if err := os.WriteFile(explicitFile, []byte("binary"), 0755); err != nil {
		t.Fatal(err)
	}
	resolved, err := ResolveFakExe(tempDir, explicitFile)
	if err != nil {
		t.Fatalf("unexpected error resolving explicit fak: %v", err)
	}
	expectedAbs, _ := filepath.Abs(explicitFile)
	if resolved != expectedAbs {
		t.Fatalf("expected %q, got %q", expectedAbs, resolved)
	}

	// 3. Probing candidates: tools/.bin/fak.exe
	toolsBin := filepath.Join(tempDir, "tools", ".bin")
	if err := os.MkdirAll(toolsBin, 0755); err != nil {
		t.Fatal(err)
	}
	candidateFile := filepath.Join(toolsBin, "fak.exe")
	if err := os.WriteFile(candidateFile, []byte("bin"), 0755); err != nil {
		t.Fatal(err)
	}
	probed, err := ResolveFakExe(tempDir, "")
	if err != nil {
		t.Fatalf("unexpected error probing candidate fak: %v", err)
	}
	expectedCandidateAbs, _ := filepath.Abs(candidateFile)
	if probed != expectedCandidateAbs {
		t.Fatalf("expected %q, got %q", expectedCandidateAbs, probed)
	}
}

func TestIsProcessLive(t *testing.T) {
	// Current process must be live
	currPID := os.Getpid()
	if !IsProcessLive(currPID) {
		t.Fatalf("expected current pid %d to be live", currPID)
	}

	// Non-positive PIDs must return false
	if IsProcessLive(0) {
		t.Fatalf("expected pid 0 to be dead")
	}
	if IsProcessLive(-1) {
		t.Fatalf("expected pid -1 to be dead")
	}

	// Unlikely high PID should return false
	if IsProcessLive(99999999) {
		t.Fatalf("expected pid 99999999 to be dead")
	}
}

func TestPidBreadcrumbsSweep(t *testing.T) {
	tempDir := t.TempDir()
	logDir := filepath.Join(tempDir, ".goal-runs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		t.Fatal(err)
	}

	livePID := os.Getpid()
	deadPID := 99999999

	liveFile := filepath.Join(logDir, "live-worker.pid")
	deadFile := filepath.Join(logDir, "dead-worker.pid")
	corruptFile := filepath.Join(logDir, "corrupt-worker.pid")
	emptyFile := filepath.Join(logDir, "empty-worker.pid")

	if err := os.WriteFile(liveFile, []byte(fmt.Sprintf("%d\n", livePID)), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(deadFile, []byte(fmt.Sprintf("%d\n", deadPID)), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(corruptFile, []byte("not_a_valid_pid"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(emptyFile, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	swept, err := SweepDeadPidBreadcrumbs(tempDir)
	if err != nil {
		t.Fatalf("unexpected error during sweep: %v", err)
	}

	// Live file should be preserved
	if _, err := os.Stat(liveFile); os.IsNotExist(err) {
		t.Errorf("expected live pid file %s to be preserved, but it was removed", liveFile)
	}

	// Dead file should be swept
	if _, err := os.Stat(deadFile); !os.IsNotExist(err) {
		t.Errorf("expected dead pid file %s to be removed, but it exists", deadFile)
	}

	// Corrupt file should be swept
	if _, err := os.Stat(corruptFile); !os.IsNotExist(err) {
		t.Errorf("expected corrupt pid file %s to be removed, but it exists", corruptFile)
	}

	// Empty file should be swept
	if _, err := os.Stat(emptyFile); !os.IsNotExist(err) {
		t.Errorf("expected empty pid file %s to be removed, but it exists", emptyFile)
	}

	// Swept slice should contain deadPID
	foundDead := false
	for _, p := range swept {
		if p == deadPID {
			foundDead = true
		}
	}
	if !foundDead {
		t.Errorf("expected swept pids %v to contain %d", swept, deadPID)
	}
}

func TestFleetPlanOrdering(t *testing.T) {
	contracts := []GoalContract{
		{N: 9, Lane: "compute", Host: "blocked: CUDA node"},
		{N: 5, Lane: "model", Host: "partial: perf->GPU node"},
		{N: 10, Lane: "compute", Host: "blocked: AMD Vulkan node"},
		{N: 21, Lane: "gateway", Host: "full"},
		{N: 12, Lane: "ci", Host: "partial: -race->cgo node"},
		{N: 11, Lane: "bench", Host: "partial: N=1000->bench node"},
		{N: 7, Lane: "model", Host: "blocked: 2x GPU node"},
	}

	ordered := OrderContracts(contracts)

	if len(ordered) != len(contracts) {
		t.Fatalf("expected %d contracts, got %d", len(contracts), len(ordered))
	}

	// Host tractability order: full first, partial second, blocked last
	if ordered[0].N != 21 || ordered[0].Host != "full" {
		t.Errorf("expected contract #21 (full) at index 0, got #%d (%s)", ordered[0].N, ordered[0].Host)
	}

	// Indexes 1..3 should all be partial
	for i := 1; i <= 3; i++ {
		if !strings.HasPrefix(ordered[i].Host, "partial") {
			t.Errorf("expected partial contract at index %d, got #%d (%s)", i, ordered[i].N, ordered[i].Host)
		}
	}

	// Indexes 4..6 should all be blocked
	for i := 4; i <= 6; i++ {
		if !strings.HasPrefix(ordered[i].Host, "blocked") {
			t.Errorf("expected blocked contract at index %d, got #%d (%s)", i, ordered[i].N, ordered[i].Host)
		}
	}
}

func TestLaunchDetachedWorkerPlanOnly(t *testing.T) {
	tempDir := t.TempDir()

	pointerFile := filepath.Join(tempDir, "solve-issue.md")
	if err := os.WriteFile(pointerFile, []byte("Fix the buffer overflow in gateway"), 0644); err != nil {
		t.Fatal(err)
	}

	opt := LaunchOptions{
		Workspace:   tempDir,
		PointerFile: pointerFile,
		PlanOnly:    true,
	}

	res, err := LaunchDetachedWorker(opt)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !res.PlanOnly {
		t.Errorf("expected PlanOnly to be true")
	}
	if res.PID != 0 {
		t.Errorf("expected PID to be 0 for PlanOnly, got %d", res.PID)
	}
	if !strings.HasPrefix(res.LaunchWitness, "PLAN_ONLY") {
		t.Errorf("expected witness starting with PLAN_ONLY, got %q", res.LaunchWitness)
	}

	// Verify input prompt file was written with /goal prefix
	inData, err := os.ReadFile(res.PromptFile)
	if err != nil {
		t.Fatalf("failed to read prompt file: %v", err)
	}
	if !strings.HasPrefix(string(inData), "/goal Fix the buffer overflow") {
		t.Fatalf("expected prompt file to start with '/goal ', got %q", string(inData))
	}
	if res.Receipt == nil {
		t.Fatalf("expected non-nil Receipt in LaunchResult")
	}
	if res.Receipt.Outcome != "plan_only" {
		t.Errorf("expected Receipt Outcome 'plan_only', got %q", res.Receipt.Outcome)
	}
}

func TestLaunchDetachedWorkerReceipt(t *testing.T) {
	tempDir := t.TempDir()

	pointerFile := filepath.Join(tempDir, "solve-issue.md")
	content := "Fix the memory leak in engine"
	if err := os.WriteFile(pointerFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	opt := LaunchOptions{
		Workspace:           tempDir,
		PointerFile:         pointerFile,
		Product:             "codex",
		WorkKind:            "gardening",
		Tier:                "t2",
		Account:             "acct-qa",
		AccountTag:          "qa-team",
		Guarded:             true,
		ContextBudgetTokens: 1500000,
		MaxDuration:         "30m",
		PreflightMaxWorkers: 8,
		PlanOnly:            true,
	}

	res, err := LaunchDetachedWorker(opt)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res.Receipt == nil {
		t.Fatalf("expected non-nil Receipt in LaunchResult")
	}

	rec := res.Receipt
	if rec.Schema != GoalLaunchReceiptSchema {
		t.Errorf("schema = %q, want %q", rec.Schema, GoalLaunchReceiptSchema)
	}
	if rec.Outcome != "plan_only" {
		t.Errorf("outcome = %q, want plan_only", rec.Outcome)
	}
	if rec.Product != "codex" {
		t.Errorf("product = %q, want codex", rec.Product)
	}
	if rec.WorkKind != "gardening" {
		t.Errorf("work_kind = %q, want gardening", rec.WorkKind)
	}
	if rec.Tier != "t2" {
		t.Errorf("tier = %q, want t2", rec.Tier)
	}
	if rec.Account != "acct-qa" {
		t.Errorf("account = %q, want acct-qa", rec.Account)
	}
	if rec.AccountTag != "qa-team" {
		t.Errorf("account_tag = %q, want qa-team", rec.AccountTag)
	}
	if !rec.Guarded {
		t.Errorf("guarded = false, want true")
	}
	if rec.BudgetTokens != 1500000 {
		t.Errorf("budget_tokens = %d, want 1500000", rec.BudgetTokens)
	}
	if rec.MaxDuration != "30m" {
		t.Errorf("max_duration = %q, want 30m", rec.MaxDuration)
	}
	expectedPrompt := "/goal " + content
	if rec.PromptChars != len(expectedPrompt) {
		t.Errorf("prompt_chars = %d, want %d", rec.PromptChars, len(expectedPrompt))
	}
	if rec.ShiftLeftPreflight.Verdict != "SPAWN_OK" {
		t.Errorf("shift_left_preflight verdict = %q, want SPAWN_OK", rec.ShiftLeftPreflight.Verdict)
	}
	if rec.ShiftLeftPreflight.LiveCount != 0 {
		t.Errorf("shift_left_preflight live_count = %d, want 0", rec.ShiftLeftPreflight.LiveCount)
	}
	if rec.ShiftLeftPreflight.HostCap != 8 {
		t.Errorf("shift_left_preflight host_cap = %d, want 8", rec.ShiftLeftPreflight.HostCap)
	}
	if rec.ReceiptPath == "" {
		t.Fatalf("receipt_path is empty")
	}

	// Verify disk receipt at ReceiptPath
	receiptBytes, err := os.ReadFile(rec.ReceiptPath)
	if err != nil {
		t.Fatalf("failed to read receipt file at %s: %v", rec.ReceiptPath, err)
	}
	var diskRec GoalLaunchReceipt
	if err := json.Unmarshal(receiptBytes, &diskRec); err != nil {
		t.Fatalf("failed to parse disk receipt: %v", err)
	}
	if diskRec.Schema != GoalLaunchReceiptSchema || diskRec.Outcome != "plan_only" {
		t.Errorf("parsed disk receipt mismatch: %+v", diskRec)
	}

	// Verify .fak/goal-launch-receipt.json update
	fakReceiptPath := filepath.Join(tempDir, ".fak", "goal-launch-receipt.json")
	fakBytes, err := os.ReadFile(fakReceiptPath)
	if err != nil {
		t.Fatalf("failed to read .fak receipt file: %v", err)
	}
	var fakRec GoalLaunchReceipt
	if err := json.Unmarshal(fakBytes, &fakRec); err != nil {
		t.Fatalf("failed to parse .fak receipt: %v", err)
	}
	if fakRec.RunID != rec.RunID || fakRec.Outcome != "plan_only" {
		t.Errorf("parsed .fak receipt mismatch: %+v", fakRec)
	}
}

func TestLaunchDetachedWorkerCustomReceiptPath(t *testing.T) {
	tempDir := t.TempDir()
	customPath := filepath.Join(tempDir, "custom", "receipts", "custom-run.receipt.json")

	opt := LaunchOptions{
		Workspace:      tempDir,
		PointerContent: "inline task condition",
		ReceiptPath:    customPath,
		PlanOnly:       true,
	}

	res, err := LaunchDetachedWorker(opt)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Receipt == nil {
		t.Fatalf("expected non-nil Receipt")
	}
	if res.Receipt.ReceiptPath != customPath {
		t.Errorf("receipt path = %q, want %q", res.Receipt.ReceiptPath, customPath)
	}

	if _, err := os.Stat(customPath); os.IsNotExist(err) {
		t.Fatalf("custom receipt file does not exist at %s", customPath)
	}
}

func TestLaunchDetachedWorkerShiftLeftPreflightFailures(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Workspace does not exist
	t.Run("nonexistent workspace fails preflight", func(t *testing.T) {
		_, err := LaunchDetachedWorker(LaunchOptions{
			Workspace:      filepath.Join(tempDir, "nonexistent-dir"),
			PointerContent: "task",
			PlanOnly:       true,
		})
		if err == nil {
			t.Fatalf("expected error for nonexistent workspace, got nil")
		}
		if !strings.Contains(err.Error(), "workspace") {
			t.Errorf("expected error to mention workspace, got: %v", err)
		}
	})

	// 2. Goal prompt length > 4000 fails preflight
	t.Run("prompt length exceeding 4000 fails preflight", func(t *testing.T) {
		_, err := LaunchDetachedWorker(LaunchOptions{
			Workspace:      tempDir,
			PointerContent: strings.Repeat("a", 4001),
			PlanOnly:       true,
		})
		if err == nil {
			t.Fatalf("expected error for oversized prompt, got nil")
		}
		if !strings.Contains(err.Error(), ">4000 cap") {
			t.Errorf("expected error to mention >4000 cap, got: %v", err)
		}
	})

	// 3. Preflight cap reached
	t.Run("preflight cap exceeded refuses spawn and writes failed receipt", func(t *testing.T) {
		ws := t.TempDir()
		logDir := filepath.Join(ws, ".goal-runs")
		if err := os.MkdirAll(logDir, 0755); err != nil {
			t.Fatal(err)
		}
		// Write a live PID file using current process ID
		livePID := os.Getpid()
		if err := os.WriteFile(filepath.Join(logDir, "live-worker.pid"), []byte(fmt.Sprintf("%d\n", livePID)), 0644); err != nil {
			t.Fatal(err)
		}

		_, err := LaunchDetachedWorker(LaunchOptions{
			Workspace:           ws,
			PointerContent:      "work task",
			PreflightMaxWorkers: 1, // live = 1 >= cap 1 => refusal
			PlanOnly:            true,
		})
		if err == nil {
			t.Fatalf("expected error when live workers >= cap, got nil")
		}
		if !strings.Contains(err.Error(), "REFUSE_AT_CAP") {
			t.Fatalf("expected error to mention REFUSE_AT_CAP, got: %v", err)
		}

		// Verify that a failed receipt was written to .fak/goal-launch-receipt.json
		fakReceipt := filepath.Join(ws, ".fak", "goal-launch-receipt.json")
		b, err := os.ReadFile(fakReceipt)
		if err != nil {
			t.Fatalf("expected .fak receipt to be written on failure: %v", err)
		}
		var rec GoalLaunchReceipt
		if err := json.Unmarshal(b, &rec); err != nil {
			t.Fatalf("failed to decode receipt: %v", err)
		}
		if rec.Outcome != "failed" {
			t.Errorf("outcome = %q, want failed", rec.Outcome)
		}
		if rec.ShiftLeftPreflight.Verdict != "REFUSE_AT_CAP" {
			t.Errorf("preflight verdict = %q, want REFUSE_AT_CAP", rec.ShiftLeftPreflight.Verdict)
		}
		if rec.ShiftLeftPreflight.LiveCount != 1 {
			t.Errorf("preflight live_count = %d, want 1", rec.ShiftLeftPreflight.LiveCount)
		}
		if rec.Error == "" {
			t.Errorf("expected receipt Error to be populated")
		}
	})

	// 4. SkipPreflight bypasses cap refusal
	t.Run("skip preflight bypasses cap refusal", func(t *testing.T) {
		ws := t.TempDir()
		logDir := filepath.Join(ws, ".goal-runs")
		if err := os.MkdirAll(logDir, 0755); err != nil {
			t.Fatal(err)
		}
		livePID := os.Getpid()
		if err := os.WriteFile(filepath.Join(logDir, "live-worker.pid"), []byte(fmt.Sprintf("%d\n", livePID)), 0644); err != nil {
			t.Fatal(err)
		}

		res, err := LaunchDetachedWorker(LaunchOptions{
			Workspace:           ws,
			PointerContent:      "work task",
			PreflightMaxWorkers: 1,
			SkipPreflight:       true,
			PlanOnly:            true,
		})
		if err != nil {
			t.Fatalf("unexpected error with SkipPreflight: %v", err)
		}
		if res.Receipt == nil {
			t.Fatalf("expected non-nil Receipt")
		}
		if res.Receipt.ShiftLeftPreflight.Verdict != "SKIPPED" {
			t.Errorf("preflight verdict = %q, want SKIPPED", res.Receipt.ShiftLeftPreflight.Verdict)
		}
	})
}

func TestSerializePlanAndRollup(t *testing.T) {
	plan := &FleetPlan{
		Name:      "test-fleet",
		Workspace: "/test/workspace",
		RunRoot:   "/test/workspace/.goal-runs/test-root",
		Contracts: []GoalContract{
			{N: 21, Lane: "gateway", Host: "full"},
			{N: 5, Lane: "model", Host: "partial"},
		},
	}

	jsonStr, err := SerializePlanJSON(plan)
	if err != nil {
		t.Fatalf("failed to serialize plan: %v", err)
	}

	var parsed FleetPlan
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		t.Fatalf("failed to parse serialized plan: %v", err)
	}
	if parsed.Name != plan.Name || len(parsed.Contracts) != 2 {
		t.Fatalf("parsed plan mismatch: %+v", parsed)
	}

	witnessResults := []*WitnessResult{
		{
			Issue:          21,
			Outcome:        "met (witnessed ship)",
			ShippedSHA:     "abc1234",
			ShippedVerdict: "OK",
			ShippedWitness: "diff-witnessed",
			IssueState:     "closed",
		},
		{
			Issue:   5,
			Outcome: "host-precluded-block (host=partial)",
		},
	}

	rollup := RenderRollup(plan, witnessResults)
	if !strings.Contains(rollup, "# P0 goal-fleet serial run") {
		t.Errorf("expected rollup header, got:\n%s", rollup)
	}
	if !strings.Contains(rollup, "## #21") || !strings.Contains(rollup, "met (witnessed ship)") {
		t.Errorf("expected issue 21 details in rollup, got:\n%s", rollup)
	}
	if !strings.Contains(rollup, "## #5") || !strings.Contains(rollup, "host-precluded-block") {
		t.Errorf("expected issue 5 details in rollup, got:\n%s", rollup)
	}
}

func TestWitnessIssue(t *testing.T) {
	// 1. Shipped without sweep
	w1, err := WitnessIssue("", 100, "", "", "full", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if w1.Outcome != "no-witnessed-effect" {
		t.Errorf("expected no-witnessed-effect, got %s", w1.Outcome)
	}

	// 2. Blocked host
	w2, err := WitnessIssue("", 101, "", "", "blocked: CUDA node", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(w2.Outcome, "host-precluded-block") {
		t.Errorf("expected host-precluded-block, got %s", w2.Outcome)
	}

	// 3. Partial host
	w3, err := WitnessIssue("", 102, "", "", "partial: 2x GPU", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(w3.Outcome, "host-precluded-block") {
		t.Errorf("expected host-precluded-block, got %s", w3.Outcome)
	}
}

func TestRunFleetPlanDryRun(t *testing.T) {
	tempDir := t.TempDir()

	plan := &FleetPlan{
		Name:      "dry-run-fleet",
		Workspace: tempDir,
		DryRun:    true,
		Contracts: []GoalContract{
			{N: 1, Host: "full"},
			{N: 2, Host: "blocked"},
		},
	}

	results, err := RunFleetPlan(context.Background(), plan, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for _, r := range results {
		if r.Outcome != "dry-run (skipped launch)" {
			t.Errorf("expected dry-run outcome, got %s", r.Outcome)
		}
	}

	// Verify rollup file was created
	if _, err := os.Stat(plan.RollupPath); os.IsNotExist(err) {
		t.Errorf("expected rollup file at %s, but not found", plan.RollupPath)
	}
}

func TestMonitorProcessTimeout(t *testing.T) {
	// A non-existent PID should immediately report exited
	exited, err := MonitorProcess(context.Background(), 99999999, 100*time.Millisecond, 10*time.Millisecond)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !exited {
		t.Errorf("expected dead PID to exit immediately")
	}

	// A running dummy child process should timeout and be killed
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("powershell", "-NoProfile", "-Command", "Start-Sleep -Seconds 30")
	} else {
		cmd = exec.Command("sleep", "30")
	}
	if err := cmd.Start(); err != nil {
		t.Skipf("skipping child process spawn: %v", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}()

	childPID := cmd.Process.Pid
	exitedChild, err := MonitorProcess(context.Background(), childPID, 50*time.Millisecond, 10*time.Millisecond)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if exitedChild {
		t.Errorf("expected running child process not to report exited before timeout")
	}

	// Verify the child process was killed upon timeout
	_ = cmd.Wait()
	if IsProcessLive(childPID) {
		t.Errorf("expected child process to be terminated after timeout")
	}
}
