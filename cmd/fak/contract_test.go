package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func setupTestGitRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "test@example.com"},
		{"config", "user.name", "test"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}

func TestContractUsageAndValidation(t *testing.T) {
	cases := []struct {
		name     string
		argv     []string
		wantCode int
	}{
		{"no args", nil, 2},
		{"unknown sub", []string{"bogus"}, 2},
		{"help flag", []string{"--help"}, 0},
		{"help word", []string{"help"}, 0},
		{"acquire no args", []string{"acquire"}, 2},
		{"acquire missing flags", []string{"acquire", "issue-1"}, 2},
		{"acquire zero budget", []string{"acquire", "issue-1", "--budget-tokens", "0", "--verify-cmd", "go test"}, 2},
		{"acquire negative budget", []string{"acquire", "issue-1", "--budget-tokens", "-10", "--verify-cmd", "go test"}, 2},
		{"acquire empty verify-cmd", []string{"acquire", "issue-1", "--budget-tokens", "1000", "--verify-cmd", ""}, 2},
		{"yield no ticket", []string{"yield"}, 2},
		{"resume no ticket", []string{"resume"}, 2},
		{"verify no ticket", []string{"verify"}, 2},
		{"close no ticket", []string{"close"}, 2},
		{"close missing status", []string{"close", "issue-1"}, 2},
		{"close invalid status", []string{"close", "issue-1", "--status", "INVALID"}, 2},
		{"close negative tokens", []string{"close", "issue-1", "--status", "SUCCEEDED", "--tokens-used", "-1"}, 2},
		{"list extra arg", []string{"list", "extra"}, 2},
		{"reap extra arg", []string{"reap", "extra"}, 2},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runContract(&stdout, &stderr, tc.argv)
			if code != tc.wantCode {
				t.Fatalf("runContract(%v) = %d, want %d (stderr=%q, stdout=%q)",
					tc.argv, code, tc.wantCode, stderr.String(), stdout.String())
			}
		})
	}
}

func TestContractLifecycle(t *testing.T) {
	dir := setupTestGitRepo(t)
	ctx := context.Background()
	store := leaseref.NewInDir(dir)

	// 1. Acquire contract
	var stdout, stderr bytes.Buffer
	code := runContract(&stdout, &stderr, []string{
		"acquire", "issue-11167",
		"--budget-tokens", "50000",
		"--verify-cmd", "go test -v ./cmd/fak -run TestContract",
		"--tier", "commodity",
		"--holder", "worker-main",
		"--dir", dir,
	})
	if code != 0 {
		t.Fatalf("acquire failed: exit=%d stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "acquired contract issue-11167") {
		t.Fatalf("acquire output missing confirmation: %q", stdout.String())
	}

	rec, ok, err := store.GetContract(ctx, "issue-11167")
	if err != nil || !ok {
		t.Fatalf("GetContract ok=%v err=%v", ok, err)
	}
	if rec.State != leaseref.ContractStateExecuting {
		t.Fatalf("expected state EXECUTING, got %q", rec.State)
	}
	if rec.TokenBudget != 50000 {
		t.Fatalf("expected TokenBudget=50000, got %d", rec.TokenBudget)
	}
	if rec.VerifyCmd != "go test -v ./cmd/fak -run TestContract" {
		t.Fatalf("unexpected VerifyCmd: %q", rec.VerifyCmd)
	}
	if rec.Generation != 1 {
		t.Fatalf("expected Generation=1, got %d", rec.Generation)
	}

	// 2. Yield contract (transitions to YIELDED_IO)
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{"yield", "issue-11167", "--dir", dir})
	if code != 0 {
		t.Fatalf("yield failed: exit=%d stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	rec, _, err = store.GetContract(ctx, "issue-11167")
	if err != nil {
		t.Fatalf("GetContract: %v", err)
	}
	if rec.State != leaseref.ContractStateYieldedIO {
		t.Fatalf("expected state YIELDED_IO, got %q", rec.State)
	}

	// 3. Resume contract (transitions back to EXECUTING)
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{"resume", "issue-11167", "--dir", dir})
	if code != 0 {
		t.Fatalf("resume failed: exit=%d stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	rec, _, err = store.GetContract(ctx, "issue-11167")
	if err != nil {
		t.Fatalf("GetContract: %v", err)
	}
	if rec.State != leaseref.ContractStateExecuting {
		t.Fatalf("expected state EXECUTING, got %q", rec.State)
	}

	// 4. Verify contract (transitions to VERIFYING)
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{"verify", "issue-11167", "--dir", dir})
	if code != 0 {
		t.Fatalf("verify failed: exit=%d stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	rec, _, err = store.GetContract(ctx, "issue-11167")
	if err != nil {
		t.Fatalf("GetContract: %v", err)
	}
	if rec.State != leaseref.ContractStateVerifying {
		t.Fatalf("expected state VERIFYING, got %q", rec.State)
	}

	// 5. Close contract with SUCCEEDED status and tokens-used
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{
		"close", "issue-11167",
		"--status", "SUCCEEDED",
		"--tokens-used", "12500",
		"--dir", dir,
	})
	if code != 0 {
		t.Fatalf("close failed: exit=%d stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	rec, _, err = store.GetContract(ctx, "issue-11167")
	if err != nil {
		t.Fatalf("GetContract: %v", err)
	}
	if rec.State != leaseref.ContractStateSucceeded {
		t.Fatalf("expected state SUCCEEDED, got %q", rec.State)
	}
	if rec.TokensUsed != 12500 {
		t.Fatalf("expected TokensUsed=12500, got %d", rec.TokensUsed)
	}

	// 6. Close another contract with FAILED status
	stdout.Reset()
	stderr.Reset()
	if code := runContract(&stdout, &stderr, []string{
		"acquire", "issue-fail",
		"--budget-tokens", "10000",
		"--verify-cmd", "false",
		"--holder", "worker-fail",
		"--dir", dir,
	}); code != 0 {
		t.Fatalf("acquire fail contract: %d stderr=%s", code, stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{
		"close", "issue-fail",
		"--status", "FAILED",
		"--tokens-used", "3500",
		"--dir", dir,
	})
	if code != 0 {
		t.Fatalf("close fail contract: exit=%d stderr=%q", code, stderr.String())
	}
	recFail, _, _ := store.GetContract(ctx, "issue-fail")
	if recFail.State != leaseref.ContractStateFailed {
		t.Fatalf("expected state FAILED, got %q", recFail.State)
	}
	if recFail.TokensUsed != 3500 {
		t.Fatalf("expected TokensUsed=3500, got %d", recFail.TokensUsed)
	}
}

func TestContractListJSONAndSchema(t *testing.T) {
	dir := setupTestGitRepo(t)
	ctx := context.Background()
	store := leaseref.NewInDir(dir)

	// In an empty repo, contract list --json should return []
	var stdout, stderr bytes.Buffer
	code := runContract(&stdout, &stderr, []string{"list", "--json", "--dir", dir})
	if code != 0 {
		t.Fatalf("empty list --json exit=%d stderr=%q", code, stderr.String())
	}
	var emptyList []leaseref.ContractRecord
	if err := json.Unmarshal(stdout.Bytes(), &emptyList); err != nil {
		t.Fatalf("unmarshal empty list: %v\nout=%s", err, stdout.String())
	}
	if len(emptyList) != 0 {
		t.Fatalf("expected 0 contracts, got %d", len(emptyList))
	}

	// Empty non-json list
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{"list", "--dir", dir})
	if code != 0 {
		t.Fatalf("empty list exit=%d stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "no contracts") {
		t.Fatalf("expected 'no contracts' message, got: %q", stdout.String())
	}

	// Acquire a live contract
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{
		"acquire", "issue-live-1",
		"--budget-tokens", "40000",
		"--verify-cmd", "go test ./...",
		"--tier", "frontier",
		"--holder", "worker-alpha",
		"--dir", dir,
	})
	if code != 0 {
		t.Fatalf("acquire live: %d stderr=%q", code, stderr.String())
	}

	// Insert an expired contract into the store directly
	pastTime := time.Now().Add(-2 * time.Hour)
	_, err := store.AcquireContract(ctx, leaseref.ContractRecord{
		TicketID:    "issue-expired-1",
		Holder:      "worker-old",
		State:       leaseref.ContractStateExecuting,
		TokenBudget: 20000,
		VerifyCmd:   "make check",
		TTLSeconds:  60,
	}, pastTime)
	if err != nil {
		t.Fatalf("AcquireContract expired: %v", err)
	}

	// Test list --json: should contain both records and match schema
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{"list", "--json", "--dir", dir})
	if code != 0 {
		t.Fatalf("list --json exit=%d stderr=%q", code, stderr.String())
	}

	var all []leaseref.ContractRecord
	if err := json.Unmarshal(stdout.Bytes(), &all); err != nil {
		t.Fatalf("unmarshal list JSON: %v\nout=%s", err, stdout.String())
	}
	if len(all) != 2 {
		t.Fatalf("expected 2 contracts in list, got %d", len(all))
	}

	// Schema validation: check unmarshaled fields
	var liveRec, expRec *leaseref.ContractRecord
	for i := range all {
		if all[i].TicketID == "issue-live-1" {
			liveRec = &all[i]
		} else if all[i].TicketID == "issue-expired-1" {
			expRec = &all[i]
		}
	}
	if liveRec == nil || expRec == nil {
		t.Fatalf("missing expected records: live=%v exp=%v", liveRec, expRec)
	}
	if liveRec.Holder != "worker-alpha" || liveRec.TokenBudget != 40000 || liveRec.PaceTier != "frontier" {
		t.Fatalf("live record schema fields unexpected: %+v", liveRec)
	}
	if liveRec.Generation != 1 || liveRec.AcquiredAt <= 0 {
		t.Fatalf("live record generation/acquired timestamp missing: %+v", liveRec)
	}

	// Test list --live --json: should only contain issue-live-1
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{"list", "--live", "--json", "--dir", dir})
	if code != 0 {
		t.Fatalf("list --live --json exit=%d stderr=%q", code, stderr.String())
	}
	var liveOnly []leaseref.ContractRecord
	if err := json.Unmarshal(stdout.Bytes(), &liveOnly); err != nil {
		t.Fatalf("unmarshal liveOnly JSON: %v\nout=%s", err, stdout.String())
	}
	if len(liveOnly) != 1 || liveOnly[0].TicketID != "issue-live-1" {
		t.Fatalf("expected only issue-live-1, got: %+v", liveOnly)
	}

	// Test list text format: shows LIVE and EXPIRED
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{"list", "--dir", dir})
	if code != 0 {
		t.Fatalf("list text exit=%d stderr=%q", code, stderr.String())
	}
	outText := stdout.String()
	if !strings.Contains(outText, "issue-live-1") || !strings.Contains(outText, "LIVE") {
		t.Fatalf("list text missing live contract: %s", outText)
	}
	if !strings.Contains(outText, "issue-expired-1") || !strings.Contains(outText, "EXPIRED") {
		t.Fatalf("list text missing expired contract: %s", outText)
	}
}

func TestContractReap(t *testing.T) {
	dir := setupTestGitRepo(t)
	ctx := context.Background()
	store := leaseref.NewInDir(dir)

	// Add an expired contract
	pastTime := time.Now().Add(-10 * time.Minute)
	_, err := store.AcquireContract(ctx, leaseref.ContractRecord{
		TicketID:    "issue-to-reap",
		Holder:      "worker-stale",
		State:       leaseref.ContractStateExecuting,
		TokenBudget: 5000,
		VerifyCmd:   "test",
		TTLSeconds:  30,
	}, pastTime)
	if err != nil {
		t.Fatalf("AcquireContract: %v", err)
	}

	// Add a live contract
	_, err = store.AcquireContract(ctx, leaseref.ContractRecord{
		TicketID:    "issue-stay-alive",
		Holder:      "worker-live",
		State:       leaseref.ContractStateExecuting,
		TokenBudget: 5000,
		VerifyCmd:   "test",
		TTLSeconds:  3600,
	}, time.Now())
	if err != nil {
		t.Fatalf("AcquireContract: %v", err)
	}

	// Run contract reap
	var stdout, stderr bytes.Buffer
	code := runContract(&stdout, &stderr, []string{"reap", "--dir", dir})
	if code != 0 {
		t.Fatalf("reap exit=%d stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "reaped 1 expired contract(s)") {
		t.Fatalf("reap output unexpected: %s", stdout.String())
	}

	// Verify issue-to-reap is gone
	_, ok, err := store.GetContract(ctx, "issue-to-reap")
	if err != nil || ok {
		t.Fatalf("issue-to-reap should be gone, ok=%v err=%v", ok, err)
	}

	// Verify issue-stay-alive is still present
	_, ok, err = store.GetContract(ctx, "issue-stay-alive")
	if err != nil || !ok {
		t.Fatalf("issue-stay-alive should be present, ok=%v err=%v", ok, err)
	}

	// Idempotent reap again
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{"reap", "--json", "--dir", dir})
	if code != 0 {
		t.Fatalf("reap --json exit=%d stderr=%q", code, stderr.String())
	}
	var reapResp map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &reapResp); err != nil {
		t.Fatalf("unmarshal reap JSON: %v", err)
	}
	if count, _ := reapResp["count"].(float64); count != 0 {
		t.Fatalf("expected count 0 on second reap, got %v", count)
	}
}

func TestContractCollisionAndRefusal(t *testing.T) {
	dir := setupTestGitRepo(t)

	// Worker A acquires contract
	var stdout, stderr bytes.Buffer
	code := runContract(&stdout, &stderr, []string{
		"acquire", "issue-collision",
		"--budget-tokens", "10000",
		"--verify-cmd", "go test",
		"--holder", "worker-A",
		"--dir", dir,
	})
	if code != 0 {
		t.Fatalf("worker A acquire: exit=%d stderr=%q", code, stderr.String())
	}

	// Worker B attempts to acquire same contract -> exit code 3 (fence/collision refusal)
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{
		"acquire", "issue-collision",
		"--budget-tokens", "10000",
		"--verify-cmd", "go test",
		"--holder", "worker-B",
		"--dir", dir,
	})
	if code != 3 {
		t.Fatalf("worker B acquire expected exit=3 (refusal), got %d (stderr=%q, stdout=%q)",
			code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stderr.String(), "held by") {
		t.Fatalf("stderr should mention held by: %s", stderr.String())
	}

	// Worker A re-acquiring succeeds (heartbeat / renewal)
	stdout.Reset()
	stderr.Reset()
	code = runContract(&stdout, &stderr, []string{
		"acquire", "issue-collision",
		"--budget-tokens", "20000",
		"--verify-cmd", "go test",
		"--holder", "worker-A",
		"--dir", dir,
	})
	if code != 0 {
		t.Fatalf("worker A re-acquire expected exit=0, got %d (stderr=%q)", code, stderr.String())
	}
}

func TestContractNotFoundErrors(t *testing.T) {
	dir := setupTestGitRepo(t)

	verbs := [][]string{
		{"yield", "non-existent-ticket", "--dir", dir},
		{"resume", "non-existent-ticket", "--dir", dir},
		{"verify", "non-existent-ticket", "--dir", dir},
		{"close", "non-existent-ticket", "--status", "SUCCEEDED", "--dir", dir},
	}

	for _, v := range verbs {
		var stdout, stderr bytes.Buffer
		code := runContract(&stdout, &stderr, v)
		if code != 1 {
			t.Fatalf("runContract(%v) exit=%d, want 1 for not found error (stderr=%q)",
				v, code, stderr.String())
		}
		if !strings.Contains(stderr.String(), "not found") {
			t.Fatalf("runContract(%v) expected 'not found' in stderr, got: %q",
				v, stderr.String())
		}
	}
}
