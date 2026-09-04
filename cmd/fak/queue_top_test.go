package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func TestQueueTop_JSONSchemaConformance(t *testing.T) {
	dir := setupTestGitRepo(t)

	var stdout, stderr bytes.Buffer
	code := runQueueTop(&stdout, &stderr, []string{"top", "--dir", dir, "--json"})
	if code != 0 {
		t.Fatalf("runQueueTop --json exited %d, stderr: %s", code, stderr.String())
	}

	var raw map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &raw); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v\nOutput:\n%s", err, stdout.String())
	}

	if raw["schema"] != QueueTopSchema {
		t.Errorf("raw schema = %v, want %q", raw["schema"], QueueTopSchema)
	}

	var report QueueTopReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to unmarshal QueueTopReport: %v", err)
	}

	if report.Schema != QueueTopSchema {
		t.Errorf("report.Schema = %q, want %q", report.Schema, QueueTopSchema)
	}
	if report.GeneratedAt == "" {
		t.Errorf("report.GeneratedAt is empty")
	}
	if report.ActiveContracts == nil {
		t.Errorf("report.ActiveContracts is nil, want non-nil slice")
	}
	if report.Slots.TotalSlots != defaultQueueTopSlots {
		t.Errorf("report.Slots.TotalSlots = %d, want %d", report.Slots.TotalSlots, defaultQueueTopSlots)
	}
	if report.Slots.AvailableSlots != defaultQueueTopSlots {
		t.Errorf("report.Slots.AvailableSlots = %d, want %d", report.Slots.AvailableSlots, defaultQueueTopSlots)
	}
	if report.Slots.UsedSlots != 0 {
		t.Errorf("report.Slots.UsedSlots = %d, want 0", report.Slots.UsedSlots)
	}
	if report.Backlog.TotalLive != 0 || report.Backlog.Active != 0 {
		t.Errorf("backlog non-zero: %+v", report.Backlog)
	}
	if report.Pacing.TokensHeadroom != 0 || report.Pacing.HeadroomPct != 100.0 {
		t.Errorf("pacing unexpected: %+v", report.Pacing)
	}
}

func TestQueueTop_EmptyQueue(t *testing.T) {
	dir := setupTestGitRepo(t)

	var stdout, stderr bytes.Buffer
	code := runQueueTop(&stdout, &stderr, []string{"top", "--dir", dir, "--snapshot"})
	if code != 0 {
		t.Fatalf("runQueueTop --snapshot exited %d, stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	for _, want := range []string{
		"fak queue top — contract states, pacing headroom & slots",
		"QUEUE BACKLOG SUMMARY:",
		"Total Live: 0 | Active: 0",
		"Slots: 0/16 used (16 available)",
		"ACTIVE CONTRACTS:",
		"(no active contracts)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\nFull output:\n%s", want, out)
		}
	}

	// Test bare command defaults to snapshot when stdout is not a TTY.
	stdout.Reset()
	stderr.Reset()
	code = runQueueTop(&stdout, &stderr, []string{"--dir", dir})
	if code != 0 {
		t.Fatalf("runQueueTop exited %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "(no active contracts)") {
		t.Errorf("output missing '(no active contracts)'\nFull output:\n%s", stdout.String())
	}
}

func TestQueueTop_SimulatedContracts(t *testing.T) {
	dir := setupTestGitRepo(t)
	ctx := context.Background()
	store := leaseref.NewInDir(dir)
	now := time.Now()

	// 1. Contract 1: EXECUTING, frontier tier
	_, err := store.AcquireContract(ctx, leaseref.ContractRecord{
		TicketID:    "issue-1101",
		Holder:      "agent-frontier",
		SessionID:   "sess-1",
		State:       leaseref.ContractStateExecuting,
		PaceTier:    leaseref.PaceTierFrontier,
		TokenBudget: 50000,
		TokensUsed:  20000,
		VerifyCmd:   "go test ./cmd/fak",
		WorktreeDir: "/tmp/wt-1101",
		TTLSeconds:  3600,
	}, now)
	if err != nil {
		t.Fatalf("acquire contract 1: %v", err)
	}

	// 2. Contract 2: YIELDED_IO, commodity tier
	_, err = store.AcquireContract(ctx, leaseref.ContractRecord{
		TicketID:    "issue-1102",
		Holder:      "agent-io",
		SessionID:   "sess-2",
		State:       leaseref.ContractStateYieldedIO,
		PaceTier:    leaseref.PaceTierCommodity,
		TokenBudget: 30000,
		TokensUsed:  10000,
		VerifyCmd:   "go test ./internal/...",
		TTLSeconds:  3600,
	}, now)
	if err != nil {
		t.Fatalf("acquire contract 2: %v", err)
	}

	// 3. Contract 3: VERIFYING, commodity tier
	_, err = store.AcquireContract(ctx, leaseref.ContractRecord{
		TicketID:    "issue-1103",
		Holder:      "agent-verify",
		SessionID:   "sess-3",
		State:       leaseref.ContractStateVerifying,
		PaceTier:    leaseref.PaceTierCommodity,
		TokenBudget: 20000,
		TokensUsed:  15000,
		VerifyCmd:   "make ci",
		TTLSeconds:  3600,
	}, now)
	if err != nil {
		t.Fatalf("acquire contract 3: %v", err)
	}

	// 4. Contract 4: Closed / SUCCEEDED (terminal)
	_, err = store.AcquireContract(ctx, leaseref.ContractRecord{
		TicketID:    "issue-1104",
		Holder:      "agent-done",
		State:       leaseref.ContractStateSucceeded,
		TokenBudget: 10000,
		TokensUsed:  8000,
		TTLSeconds:  3600,
	}, now)
	if err != nil {
		t.Fatalf("acquire contract 4: %v", err)
	}

	// 5. Contract 5: PENDING
	_, err = store.AcquireContract(ctx, leaseref.ContractRecord{
		TicketID:    "issue-1105",
		Holder:      "agent-queued",
		State:       leaseref.ContractStatePending,
		TokenBudget: 25000,
		TokensUsed:  0,
		TTLSeconds:  3600,
	}, now)
	if err != nil {
		t.Fatalf("acquire contract 5: %v", err)
	}

	// Test Text / Snapshot rendering
	var stdout, stderr bytes.Buffer
	code := runQueueTop(&stdout, &stderr, []string{"top", "--dir", dir, "--snapshot", "--slots", "10"})
	if code != 0 {
		t.Fatalf("runQueueTop --snapshot exited %d, stderr: %s", code, stderr.String())
	}

	textOut := stdout.String()
	for _, want := range []string{
		"issue-1101",
		"issue-1102",
		"issue-1103",
		"EXECUTING",
		"YIELDED_IO",
		"VERIFYING",
		"agent-frontier",
		"agent-io",
		"agent-verify",
		"20000/50000",
		"10000/30000",
		"15000/20000",
		"frontier",
		"commodity",
		"Active: 3 (Executing: 1, Yielded IO: 1, Verifying: 1)",
		"Pending: 1",
		"Slots: 3/10 used (7 available)",
		"3 active contract(s)",
	} {
		if !strings.Contains(textOut, want) {
			t.Errorf("text output missing %q\nFull output:\n%s", want, textOut)
		}
	}

	// Succeeded contract should not appear in ACTIVE CONTRACTS table
	if strings.Contains(textOut, "issue-1104") {
		t.Errorf("text output should not contain terminal contract issue-1104")
	}

	// Test JSON rendering and schema verification
	stdout.Reset()
	stderr.Reset()
	code = runQueueTop(&stdout, &stderr, []string{"top", "--dir", dir, "--json", "--slots", "10"})
	if code != 0 {
		t.Fatalf("runQueueTop --json exited %d, stderr: %s", code, stderr.String())
	}

	var rep QueueTopReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if rep.Schema != QueueTopSchema {
		t.Errorf("rep.Schema = %q, want %q", rep.Schema, QueueTopSchema)
	}
	if rep.Backlog.TotalLive != 5 {
		t.Errorf("rep.Backlog.TotalLive = %d, want 5", rep.Backlog.TotalLive)
	}
	if rep.Backlog.Active != 3 {
		t.Errorf("rep.Backlog.Active = %d, want 3", rep.Backlog.Active)
	}
	if rep.Backlog.Executing != 1 {
		t.Errorf("rep.Backlog.Executing = %d, want 1", rep.Backlog.Executing)
	}
	if rep.Backlog.YieldedIO != 1 {
		t.Errorf("rep.Backlog.YieldedIO = %d, want 1", rep.Backlog.YieldedIO)
	}
	if rep.Backlog.Verifying != 1 {
		t.Errorf("rep.Backlog.Verifying = %d, want 1", rep.Backlog.Verifying)
	}
	if rep.Backlog.Pending != 1 {
		t.Errorf("rep.Backlog.Pending = %d, want 1", rep.Backlog.Pending)
	}
	if rep.Backlog.Succeeded != 1 {
		t.Errorf("rep.Backlog.Succeeded = %d, want 1", rep.Backlog.Succeeded)
	}
	if rep.Slots.TotalSlots != 10 {
		t.Errorf("rep.Slots.TotalSlots = %d, want 10", rep.Slots.TotalSlots)
	}
	if rep.Slots.UsedSlots != 3 {
		t.Errorf("rep.Slots.UsedSlots = %d, want 3", rep.Slots.UsedSlots)
	}
	if rep.Slots.AvailableSlots != 7 {
		t.Errorf("rep.Slots.AvailableSlots = %d, want 7", rep.Slots.AvailableSlots)
	}
	if rep.Pacing.FrontierCount != 1 {
		t.Errorf("rep.Pacing.FrontierCount = %d, want 1", rep.Pacing.FrontierCount)
	}
	if rep.Pacing.CommodityCount != 2 {
		t.Errorf("rep.Pacing.CommodityCount = %d, want 2", rep.Pacing.CommodityCount)
	}
	if rep.Pacing.EvalOnlyCount != 0 {
		t.Errorf("rep.Pacing.EvalOnlyCount = %d, want 0", rep.Pacing.EvalOnlyCount)
	}

	// 50000 + 30000 + 20000 = 100000
	if rep.Pacing.TotalTokenBudget != 100000 {
		t.Errorf("rep.Pacing.TotalTokenBudget = %d, want 100000", rep.Pacing.TotalTokenBudget)
	}
	// 20000 + 10000 + 15000 = 45000
	if rep.Pacing.TotalTokensUsed != 45000 {
		t.Errorf("rep.Pacing.TotalTokensUsed = %d, want 45000", rep.Pacing.TotalTokensUsed)
	}
	// 100000 - 45000 = 55000
	if rep.Pacing.TokensHeadroom != 55000 {
		t.Errorf("rep.Pacing.TokensHeadroom = %d, want 55000", rep.Pacing.TokensHeadroom)
	}
	if rep.Pacing.HeadroomPct < 54.99 || rep.Pacing.HeadroomPct > 55.01 {
		t.Errorf("rep.Pacing.HeadroomPct = %f, want ~55.0", rep.Pacing.HeadroomPct)
	}

	if len(rep.ActiveContracts) != 3 {
		t.Fatalf("len(rep.ActiveContracts) = %d, want 3", len(rep.ActiveContracts))
	}
	if rep.ActiveContracts[0].TicketID != "issue-1101" || rep.ActiveContracts[0].State != "EXECUTING" {
		t.Errorf("contract 0 = %+v", rep.ActiveContracts[0])
	}
	if rep.ActiveContracts[1].TicketID != "issue-1102" || rep.ActiveContracts[1].State != "YIELDED_IO" {
		t.Errorf("contract 1 = %+v", rep.ActiveContracts[1])
	}
	if rep.ActiveContracts[2].TicketID != "issue-1103" || rep.ActiveContracts[2].State != "VERIFYING" {
		t.Errorf("contract 2 = %+v", rep.ActiveContracts[2])
	}
}

func TestQueueTop_WatchModeFrames(t *testing.T) {
	dir := setupTestGitRepo(t)

	var stdout, stderr bytes.Buffer
	code := runQueueTop(&stdout, &stderr, []string{"top", "--dir", dir, "--watch", "--frames", "2", "--interval", "10ms"})
	if code != 0 {
		t.Fatalf("runQueueTop watch exited %d, stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, queueTopClearScreen) {
		t.Errorf("watch mode should emit clear screen escape sequence")
	}
	if !strings.Contains(out, "fak queue top") {
		t.Errorf("watch mode output missing title")
	}
}

func TestQueueTop_HelpAndValidation(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Help flag
	code := runQueueTop(&stdout, &stderr, []string{"top", "--help"})
	if code != 0 {
		t.Errorf("runQueueTop --help code = %d, want 0", code)
	}
	if !strings.Contains(stdout.String(), "fak queue top") {
		t.Errorf("--help output missing usage")
	}

	// Help word
	stdout.Reset()
	stderr.Reset()
	code = runQueueTop(&stdout, &stderr, []string{"top", "help"})
	if code != 0 {
		t.Errorf("runQueueTop help code = %d, want 0", code)
	}

	// Unexpected argument
	stdout.Reset()
	stderr.Reset()
	code = runQueueTop(&stdout, &stderr, []string{"top", "stray-arg"})
	if code != 2 {
		t.Errorf("runQueueTop stray-arg code = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "unexpected argument") {
		t.Errorf("stderr missing unexpected argument message: %s", stderr.String())
	}
}
