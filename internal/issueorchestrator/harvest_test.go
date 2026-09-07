package issueorchestrator

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarvest_FourStateTaxonomy(t *testing.T) {
	leaves := []LeafReceipt{
		{IssueNumber: 101, Title: "cleared feature", Lane: "gateway"},
		{IssueNumber: 102, Title: "residual feature", Lane: "policy"},
		{IssueNumber: 103, Title: "quiet incomplete feature", Lane: "model"},
		{IssueNumber: 104, Title: "stalled feature", Lane: "compute"},
	}

	opts := HarvestOptions{
		Leaves: leaves,
		WitnessChecker: func(leaf LeafReceipt) (bool, string, error) {
			switch leaf.IssueNumber {
			case 101:
				return true, "c0ffee1", nil
			case 102:
				return false, "c0ffee2", ErrResidualReview
			case 103:
				return false, "", ErrQuietIncomplete
			case 104:
				return false, "", ErrSpinningStalled
			default:
				return false, "", nil
			}
		},
	}

	receipt, err := ReconcileWave(opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if receipt.TotalLeaves != 4 {
		t.Fatalf("expected 4 total leaves, got %d", receipt.TotalLeaves)
	}
	if receipt.ClearedCount != 1 {
		t.Fatalf("expected 1 cleared count, got %d", receipt.ClearedCount)
	}
	if receipt.ResidualCount != 1 {
		t.Fatalf("expected 1 residual count, got %d", receipt.ResidualCount)
	}
	if receipt.QuietCount != 1 {
		t.Fatalf("expected 1 quiet count, got %d", receipt.QuietCount)
	}
	if receipt.StalledCount != 1 {
		t.Fatalf("expected 1 stalled count, got %d", receipt.StalledCount)
	}
	if receipt.ClearRate != 0.25 {
		t.Fatalf("expected clear rate 0.25, got %f", receipt.ClearRate)
	}

	if receipt.Leaves[0].State != StateVerifiedCleared {
		t.Errorf("leaf 101 expected StateVerifiedCleared, got %s", receipt.Leaves[0].State)
	}
	if receipt.Leaves[1].State != StateResidualReview {
		t.Errorf("leaf 102 expected StateResidualReview, got %s", receipt.Leaves[1].State)
	}
	if receipt.Leaves[2].State != StateQuietIncomplete {
		t.Errorf("leaf 103 expected StateQuietIncomplete, got %s", receipt.Leaves[2].State)
	}
	if receipt.Leaves[3].State != StateSpinningStalled {
		t.Errorf("leaf 104 expected StateSpinningStalled, got %s", receipt.Leaves[3].State)
	}
}

func TestHarvest_ProgressiveReconciliation(t *testing.T) {
	leaves := []LeafReceipt{
		{IssueNumber: 201, Title: "cleared 1", Lane: "gateway"},
		{IssueNumber: 202, Title: "cleared 2", Lane: "model"},
		{IssueNumber: 203, Title: "cleared 3", Lane: "engine"},
		{IssueNumber: 204, Title: "residual 1", Lane: "policy"},
	}

	var landed []string
	opts := HarvestOptions{
		Leaves:   leaves,
		AutoLand: true,
		WitnessChecker: func(leaf LeafReceipt) (bool, string, error) {
			if leaf.IssueNumber == 204 {
				return false, "sha-residual-204", ErrResidualReview
			}
			return true, fmt.Sprintf("sha-cleared-%d", leaf.IssueNumber), nil
		},
		GitLander: func(sha string) error {
			landed = append(landed, sha)
			return nil
		},
	}

	receipt, err := ReconcileWave(opts)
	if err != nil {
		t.Fatalf("progressive reconciliation should not fail or halt: %v", err)
	}

	if receipt.ClearedCount != 3 {
		t.Errorf("expected 3 cleared, got %d", receipt.ClearedCount)
	}
	if receipt.ResidualCount != 1 {
		t.Errorf("expected 1 residual, got %d", receipt.ResidualCount)
	}
	if receipt.ClearRate != 0.75 {
		t.Errorf("expected clear rate 0.75, got %f", receipt.ClearRate)
	}

	// Verify the 3 cleared leaves landed immediately
	if len(landed) != 3 {
		t.Fatalf("expected 3 landed SHAs, got %d", len(landed))
	}
	if len(receipt.LandedSHAs) != 3 {
		t.Fatalf("expected receipt.LandedSHAs to have 3 SHAs, got %d", len(receipt.LandedSHAs))
	}

	// Verify the 1 residual is in review queue
	if len(receipt.ReviewQueue) != 1 {
		t.Fatalf("expected 1 leaf in review queue, got %d", len(receipt.ReviewQueue))
	}
	if receipt.ReviewQueue[0].IssueNumber != 204 {
		t.Errorf("expected issue 204 in review queue, got #%d", receipt.ReviewQueue[0].IssueNumber)
	}

	// Verify advisory log
	expectedAdvisory := "[issueorchestrator:harvest] ADVISORY: leaf #204 in residual review; continuing reconciliation"
	foundAdvisory := false
	for _, log := range receipt.AdvisoryLogs {
		if strings.Contains(log, expectedAdvisory) {
			foundAdvisory = true
			break
		}
	}
	if !foundAdvisory {
		t.Errorf("expected advisory log %q, got: %v", expectedAdvisory, receipt.AdvisoryLogs)
	}
}

func TestHarvest_AutoLand(t *testing.T) {
	leaves := []LeafReceipt{
		{IssueNumber: 301, Title: "cleared A", Lane: "gateway"},
		{IssueNumber: 302, Title: "cleared B", Lane: "model"},
	}

	var landed []string
	opts := HarvestOptions{
		Leaves:   leaves,
		AutoLand: true,
		WitnessChecker: func(leaf LeafReceipt) (bool, string, error) {
			return true, fmt.Sprintf("sha-%d", leaf.IssueNumber), nil
		},
		GitLander: func(sha string) error {
			landed = append(landed, sha)
			return nil
		},
	}

	receipt, err := ReconcileWave(opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(receipt.LandedSHAs) != 2 {
		t.Fatalf("expected 2 LandedSHAs, got %d", len(receipt.LandedSHAs))
	}
	if len(landed) != 2 || landed[0] != "sha-301" || landed[1] != "sha-302" {
		t.Fatalf("expected landed SHAs [sha-301, sha-302], got %v", landed)
	}

	// Now test AutoLand: false
	landed = nil
	opts.AutoLand = false
	receiptNoLand, err := ReconcileWave(opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(receiptNoLand.LandedSHAs) != 0 {
		t.Errorf("expected 0 LandedSHAs when AutoLand is false, got %d", len(receiptNoLand.LandedSHAs))
	}
	if len(landed) != 0 {
		t.Errorf("expected GitLander not called when AutoLand is false, got %v", landed)
	}
}

func TestHarvest_GracefulSpinner(t *testing.T) {
	leaves := []LeafReceipt{
		{IssueNumber: 401, Title: "healthy leaf", Lane: "gateway"},
		{IssueNumber: 402, Title: "runaway process", Lane: "compute"},
	}

	opts := HarvestOptions{
		Leaves: leaves,
		WitnessChecker: func(leaf LeafReceipt) (bool, string, error) {
			if leaf.IssueNumber == 402 {
				return false, "", ErrSpinningStalled
			}
			return true, "sha-healthy-401", nil
		},
	}

	receipt, err := ReconcileWave(opts)
	if err != nil {
		t.Fatalf("spinning stalled worker should not cause error: %v", err)
	}

	if receipt.StalledCount != 1 {
		t.Errorf("expected 1 stalled leaf, got %d", receipt.StalledCount)
	}
	if receipt.ClearedCount != 1 {
		t.Errorf("expected 1 cleared leaf, got %d", receipt.ClearedCount)
	}
	if receipt.Leaves[1].State != StateSpinningStalled {
		t.Errorf("expected leaf 402 state to be StateSpinningStalled, got %s", receipt.Leaves[1].State)
	}

	foundAdvisory := false
	for _, log := range receipt.AdvisoryLogs {
		if strings.Contains(log, "402") && strings.Contains(log, "spinning stalled") {
			foundAdvisory = true
			break
		}
	}
	if !foundAdvisory {
		t.Errorf("expected advisory for spinning stalled leaf #402, got %v", receipt.AdvisoryLogs)
	}
}

func TestHarvest_LoadFromReceiptPath(t *testing.T) {
	tmpDir := t.TempDir()
	receiptPath := filepath.Join(tmpDir, "wave_receipt.json")

	content := `{
		"wave_id": "wave-file-test",
		"leaves": [
			{"issue_number": 501, "title": "file leaf 1", "lane": "gateway"},
			{"issue_number": 502, "title": "file leaf 2", "lane": "model"}
		]
	}`
	if err := os.WriteFile(receiptPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write receipt file: %v", err)
	}

	opts := HarvestOptions{
		WaveReceiptPath: receiptPath,
		WitnessChecker: func(leaf LeafReceipt) (bool, string, error) {
			return true, fmt.Sprintf("sha-%d", leaf.IssueNumber), nil
		},
	}

	receipt, err := ReconcileWave(opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if receipt.WaveID != "wave-file-test" {
		t.Errorf("expected wave_id %q, got %q", "wave-file-test", receipt.WaveID)
	}
	if receipt.TotalLeaves != 2 {
		t.Errorf("expected 2 total leaves, got %d", receipt.TotalLeaves)
	}
	if receipt.ClearedCount != 2 {
		t.Errorf("expected 2 cleared leaves, got %d", receipt.ClearedCount)
	}
}

func TestHarvestTaxonomy(t *testing.T) {
	leaves := []LeafHarvest{
		{
			IssueNumber:    101,
			Lane:           "gateway",
			State:          LeafStateVerifiedCleared,
			Cleared:        true,
			WitnessCommand: "go test -v ./internal/gateway",
			WitnessOutput:  "PASS",
		},
		{
			IssueNumber: 102,
			Lane:        "policy",
			State:       LeafStateResidualReview,
			Cleared:     false,
			Reason:      "subject-only claim without diff witness",
		},
		{
			IssueNumber: 103,
			Lane:        "model",
			State:       LeafStateQuietIncomplete,
			Cleared:     false,
			Reason:      "claimed done but no commit found",
		},
		{
			IssueNumber: 104,
			Lane:        "compute",
			State:       LeafStateSpinningStalled,
			Cleared:     false,
			Reason:      "worker spinning without net progress",
		},
	}

	plan := ComputeHarvestPlan("wave-tax-1", leaves)
	if plan.WaveID != "wave-tax-1" {
		t.Fatalf("expected wave_id %q, got %q", "wave-tax-1", plan.WaveID)
	}
	if plan.TotalLeaves != 4 {
		t.Fatalf("expected 4 total leaves, got %d", plan.TotalLeaves)
	}
	if plan.ClearedCount != 1 {
		t.Fatalf("expected 1 cleared, got %d", plan.ClearedCount)
	}
	if plan.ResidualCount != 1 {
		t.Fatalf("expected 1 residual, got %d", plan.ResidualCount)
	}
	if plan.QuietCount != 1 {
		t.Fatalf("expected 1 quiet, got %d", plan.QuietCount)
	}
	if plan.StalledCount != 1 {
		t.Fatalf("expected 1 stalled, got %d", plan.StalledCount)
	}
	if plan.ClearRate != 0.25 {
		t.Fatalf("expected clear_rate 0.25, got %f", plan.ClearRate)
	}

	result := ReconcileHarvest(plan, 0.25)
	if result.Schema != HarvestReceiptSchema {
		t.Fatalf("expected schema %q, got %q", HarvestReceiptSchema, result.Schema)
	}
	if result.WaveID != "wave-tax-1" {
		t.Fatalf("expected wave_id %q, got %q", "wave-tax-1", result.WaveID)
	}
	if result.OverallStatus != "PARTIAL" {
		t.Fatalf("expected overall_status PARTIAL, got %q", result.OverallStatus)
	}
	if len(result.AdmittedToTrunk) != 1 || result.AdmittedToTrunk[0].IssueNumber != 101 {
		t.Fatalf("expected leaf 101 admitted to trunk, got %+v", result.AdmittedToTrunk)
	}
	if len(result.QueuedForReview) != 1 || result.QueuedForReview[0].IssueNumber != 102 {
		t.Fatalf("expected leaf 102 queued for review, got %+v", result.QueuedForReview)
	}
	if len(result.RequeuedToBacklog) != 1 || result.RequeuedToBacklog[0].IssueNumber != 103 {
		t.Fatalf("expected leaf 103 requeued to backlog, got %+v", result.RequeuedToBacklog)
	}
	if len(result.StalledReaped) != 1 || result.StalledReaped[0].IssueNumber != 104 {
		t.Fatalf("expected leaf 104 stalled reaped, got %+v", result.StalledReaped)
	}

	// Verify JSON marshaling roundtrip
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("failed to marshal ReconcileResult: %v", err)
	}
	var unmarshaled ReconcileResult
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("failed to unmarshal ReconcileResult: %v", err)
	}
	if unmarshaled.OverallStatus != "PARTIAL" {
		t.Fatalf("expected unmarshaled status PARTIAL, got %q", unmarshaled.OverallStatus)
	}
}

func TestHarvestMinClearRatePartialAcceptance(t *testing.T) {
	leaves := []LeafHarvest{
		{IssueNumber: 201, Lane: "gateway", State: LeafStateVerifiedCleared, Cleared: true},
		{IssueNumber: 202, Lane: "model", State: LeafStateVerifiedCleared, Cleared: true},
		{IssueNumber: 203, Lane: "policy", State: LeafStateResidualReview, Cleared: false},
		{IssueNumber: 204, Lane: "compute", State: LeafStateQuietIncomplete, Cleared: false},
	}

	plan := ComputeHarvestPlan("wave-accept-1", leaves)
	if plan.TotalLeaves != 4 {
		t.Fatalf("expected 4 total leaves, got %d", plan.TotalLeaves)
	}
	if plan.ClearedCount != 2 {
		t.Fatalf("expected 2 cleared leaves, got %d", plan.ClearedCount)
	}
	if plan.ClearRate != 0.50 {
		t.Fatalf("expected clear_rate 0.50, got %f", plan.ClearRate)
	}

	result := ReconcileHarvest(plan, 0.50)
	if result.OverallStatus != "PARTIAL" {
		t.Fatalf("expected overall_status PARTIAL, got %q", result.OverallStatus)
	}
	if len(result.AdmittedToTrunk) != 2 {
		t.Fatalf("expected 2 admitted to trunk, got %d", len(result.AdmittedToTrunk))
	}
	if result.AdmittedToTrunk[0].IssueNumber != 201 || result.AdmittedToTrunk[1].IssueNumber != 202 {
		t.Fatalf("unexpected admitted leaves: %+v", result.AdmittedToTrunk)
	}
	if result.ActualClearRate != 0.50 {
		t.Fatalf("expected actual_clear_rate 0.50, got %f", result.ActualClearRate)
	}
}

func TestHarvestFailsClosedWhenBelowMinClearRate(t *testing.T) {
	leaves := []LeafHarvest{
		{IssueNumber: 301, Lane: "gateway", State: LeafStateVerifiedCleared, Cleared: true},
		{IssueNumber: 302, Lane: "policy", State: LeafStateResidualReview, Cleared: false},
		{IssueNumber: 303, Lane: "model", State: LeafStateQuietIncomplete, Cleared: false},
		{IssueNumber: 304, Lane: "compute", State: LeafStateSpinningStalled, Cleared: false},
	}

	plan := ComputeHarvestPlan("wave-fail-1", leaves)
	if plan.ClearedCount != 1 {
		t.Fatalf("expected 1 cleared, got %d", plan.ClearedCount)
	}
	if plan.ClearRate != 0.25 {
		t.Fatalf("expected clear_rate 0.25, got %f", plan.ClearRate)
	}

	result := ReconcileHarvest(plan, 0.50)
	if result.OverallStatus != "FAILED" {
		t.Fatalf("expected overall_status FAILED, got %q", result.OverallStatus)
	}
	if len(result.AdmittedToTrunk) != 0 {
		t.Fatalf("expected 0 admitted to trunk, got %d", len(result.AdmittedToTrunk))
	}
}

func TestHarvestAllClearedPasses(t *testing.T) {
	leaves := []LeafHarvest{
		{IssueNumber: 401, Lane: "gateway", State: LeafStateVerifiedCleared, Cleared: true},
		{IssueNumber: 402, Lane: "model", State: LeafStateVerifiedCleared, Cleared: true},
		{IssueNumber: 403, Lane: "policy", State: LeafStateVerifiedCleared, Cleared: true},
		{IssueNumber: 404, Lane: "compute", State: LeafStateVerifiedCleared, Cleared: true},
	}

	plan := ComputeHarvestPlan("wave-pass-1", leaves)
	if plan.TotalLeaves != 4 {
		t.Fatalf("expected 4 total leaves, got %d", plan.TotalLeaves)
	}
	if plan.ClearedCount != 4 {
		t.Fatalf("expected 4 cleared, got %d", plan.ClearedCount)
	}
	if plan.ClearRate != 1.0 {
		t.Fatalf("expected clear_rate 1.0, got %f", plan.ClearRate)
	}

	result := ReconcileHarvest(plan, 0.80)
	if result.OverallStatus != "PASSED" {
		t.Fatalf("expected overall_status PASSED, got %q", result.OverallStatus)
	}
	if len(result.AdmittedToTrunk) != 4 {
		t.Fatalf("expected 4 admitted to trunk, got %d", len(result.AdmittedToTrunk))
	}
}
