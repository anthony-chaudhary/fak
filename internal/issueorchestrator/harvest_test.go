package issueorchestrator

import (
	"encoding/json"
	"testing"
)

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
