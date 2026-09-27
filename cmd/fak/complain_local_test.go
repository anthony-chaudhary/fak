package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardcomplaint"
)

func TestComplainLiveLocalFallbackAndReplayCleanup(t *testing.T) {
	workspace := t.TempDir()
	summary := "legitimate complaint details remain recoverable offline"
	benignRationale := "the documented recovery was unavailable"
	secret := "sk-ant-api03-abcdefghijklmnop"
	args := []string{
		"--summary", summary,
		"--rationale", benignRationale + " credential " + secret,
		"--reason", "FILE_ADMISSION",
		"--tool", "Bash",
		"--workspace", workspace,
		"--live",
		"--json",
	}

	originalFetch := complainFetchExisting
	originalSync := complainSync
	t.Cleanup(func() {
		complainFetchExisting = originalFetch
		complainSync = originalSync
	})
	complainFetchExisting = func(string, int) ([]dogfoodissues.Issue, error) {
		return nil, errors.New("offline for test")
	}

	code, out, _ := runComplainCapture(args)
	if code == 0 {
		t.Fatalf("offline live complaint exit = 0, want nonzero")
	}
	var fallback struct {
		Mode        string `json:"mode"`
		PendingPath string `json:"pending_path"`
		Synced      []any  `json:"synced"`
	}
	if err := json.Unmarshal([]byte(out), &fallback); err != nil {
		t.Fatalf("offline result is not JSON: %v\n%s", err, out)
	}
	if fallback.Mode != "pending-local" || fallback.PendingPath == "" {
		t.Fatalf("offline mode/path = %q/%q, want pending-local and a path", fallback.Mode, fallback.PendingPath)
	}
	if len(fallback.Synced) != 0 {
		t.Fatalf("offline result claimed a remote sync: %+v", fallback.Synced)
	}
	pendingDir := filepath.Join(workspace, ".fak", "complaints", "pending")
	rel, err := filepath.Rel(pendingDir, fallback.PendingPath)
	if err != nil || rel == "." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		t.Fatalf("pending_path %q is outside %q", fallback.PendingPath, pendingDir)
	}
	receipt, err := os.ReadFile(fallback.PendingPath)
	if err != nil {
		t.Fatalf("read pending receipt: %v", err)
	}
	receiptText := string(receipt)
	if !strings.Contains(receiptText, summary) || !strings.Contains(receiptText, benignRationale) {
		t.Fatalf("pending receipt lost recoverable complaint text: %s", receipt)
	}
	if strings.Contains(receiptText, secret) || !strings.Contains(receiptText, "[REDACTED_API_KEY]") {
		t.Fatalf("pending receipt did not redact secret-like material: %s", receipt)
	}

	code, listed, errs := runComplainCapture([]string{"--pending", "--workspace", workspace, "--json"})
	if code != 0 {
		t.Fatalf("pending list exit = %d, want 0 (stderr=%q)", code, errs)
	}
	if !strings.Contains(listed, filepath.Base(fallback.PendingPath)) || !strings.Contains(listed, summary) || !strings.Contains(listed, benignRationale) {
		t.Fatalf("pending list must expose recoverable receipt details: %s", listed)
	}
	if strings.Contains(listed, secret) {
		t.Fatalf("pending list leaked secret-like material: %s", listed)
	}

	complainFetchExisting = func(string, int) ([]dogfoodissues.Issue, error) { return nil, nil }
	complainSync = func(row guardcomplaint.PlanRow, _ string, _ []string, _ dogfoodissues.Runner) dogfoodissues.SyncRow {
		return dogfoodissues.SyncRow{Key: row.Key, Action: row.Action, OK: true, Verified: true}
	}
	code, _, errs = runComplainCapture(args)
	if code != 0 {
		t.Fatalf("successful replay exit = %d, want 0 (stderr=%q)", code, errs)
	}
	if _, err := os.Stat(fallback.PendingPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("successful replay left pending receipt: %v", err)
	}
}

func TestComplainDryRunLocalCreatesNothing(t *testing.T) {
	t.Setenv("FAK_COMPLAIN_LIVE", "")
	workspace := t.TempDir()
	code, _, errs := runComplainCapture([]string{
		"--summary", "dry run remains side effect free",
		"--reason", "FILE_ADMISSION",
		"--workspace", workspace,
		"--json",
	})
	if code != 0 {
		t.Fatalf("dry-run exit = %d, want 0 (stderr=%q)", code, errs)
	}
	pendingDir := filepath.Join(workspace, ".fak", "complaints", "pending")
	if _, err := os.Stat(pendingDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("dry-run created pending storage: %v", err)
	}
}

func TestComplainUnverifiedSyncRetainsLocalReceipt(t *testing.T) {
	workspace := t.TempDir()
	originalFetch := complainFetchExisting
	originalSync := complainSync
	t.Cleanup(func() {
		complainFetchExisting = originalFetch
		complainSync = originalSync
	})
	complainFetchExisting = func(string, int) ([]dogfoodissues.Issue, error) { return nil, nil }
	complainSync = func(row guardcomplaint.PlanRow, _ string, _ []string, _ dogfoodissues.Runner) dogfoodissues.SyncRow {
		return dogfoodissues.SyncRow{Key: row.Key, Action: row.Action, OK: true, Verified: false}
	}

	code, out, _ := runComplainCapture([]string{
		"--summary", "remote sync could not be verified",
		"--reason", "FILE_ADMISSION",
		"--workspace", workspace,
		"--live",
		"--json",
	})
	if code == 0 {
		t.Fatalf("unverified sync exit = 0, want nonzero")
	}
	var result struct {
		Mode        string `json:"mode"`
		PendingPath string `json:"pending_path"`
		Synced      []any  `json:"synced"`
	}
	if err := json.Unmarshal([]byte(out), &result); err != nil {
		t.Fatalf("unverified sync must emit one JSON result: %v\n%s", err, out)
	}
	if result.Mode != "pending-local" || result.PendingPath == "" {
		t.Fatalf("unverified sync mode/path = %q/%q, want pending-local and a path", result.Mode, result.PendingPath)
	}
	if len(result.Synced) != 0 {
		t.Fatalf("unverified sync result claimed remote success: %+v", result.Synced)
	}
	if _, err := os.Stat(result.PendingPath); err != nil {
		t.Fatalf("unverified sync did not retain pending receipt: %v", err)
	}
}
