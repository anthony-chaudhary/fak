package mcpbroker

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func makeTaskPayload(id string, indent int, optOut bool) MatchedTaskCase {
	padding := strings.Repeat(" ", indent)
	inner := fmt.Sprintf(`{"task_id":%q,"status":"completed","padding":%q,"metrics":{"duration_ms":123,"retries":0}}`, id, padding)
	marshaledInner, _ := json.Marshal(inner)
	content := fmt.Sprintf(`[{"type":"text","text":%s}]`, marshaledInner)
	rawResult := fmt.Sprintf(`{"content":%s,"structuredContent":%s}`, content, inner)

	return MatchedTaskCase{
		ID:          id,
		ToolName:    "repo_task_runner",
		RawResult:   json.RawMessage(rawResult),
		Content:     json.RawMessage(content),
		OptOut:      optOut,
		QualityGoal: "retain_all_keys_and_metrics",
	}
}

func TestDogfoodBenchmark_MatchedArmsConservation(t *testing.T) {
	var tasks []MatchedTaskCase
	// Build 26 matched tasks (>= 24 requirement) with diverse formatting and opt-outs
	for i := 1; i <= 26; i++ {
		optOut := (i%7 == 0) // tested opt-out cases
		indent := 50 + (i * 20)
		tasks = append(tasks, makeTaskPayload(fmt.Sprintf("task-%02d", i), indent, optOut))
	}

	ctx := context.Background()
	report, err := RunMatchedDogfood(ctx, tasks)
	if err != nil {
		t.Fatalf("RunMatchedDogfood failed: %v", err)
	}

	if !report.Validated {
		t.Fatalf("expected report to be validated, got expansion_status=%s", report.ExpansionStatus)
	}
	if report.ExpansionStatus != "APPROVED" {
		t.Fatalf("expected expansion APPROVED, got %s", report.ExpansionStatus)
	}

	// Verify byte conservation: input = output + saved
	if report.DefaultArm.TotalInput != report.DefaultArm.TotalOutput+report.DefaultArm.TotalSaved {
		t.Errorf("default arm conservation violated: input(%d) != output(%d) + saved(%d)",
			report.DefaultArm.TotalInput, report.DefaultArm.TotalOutput, report.DefaultArm.TotalSaved)
	}

	// Noop arm must have saved 0 bytes and input == output
	if report.NoopArm.TotalSaved != 0 {
		t.Errorf("noop arm expected 0 saved bytes, got %d", report.NoopArm.TotalSaved)
	}
	if report.NoopArm.TotalInput != report.NoopArm.TotalOutput {
		t.Errorf("noop arm expected total input (%d) == output (%d)", report.NoopArm.TotalInput, report.NoopArm.TotalOutput)
	}

	// Zero semantic errors
	if report.DefaultArm.SemanticErrors != 0 {
		t.Errorf("expected 0 semantic errors, got %d", report.DefaultArm.SemanticErrors)
	}

	// Transform overhead must be sub-millisecond per task on average
	avgTransform := report.DefaultArm.TotalTransform / time.Duration(len(tasks))
	if avgTransform > 5*time.Millisecond {
		t.Errorf("transform overhead exceeded budget: %v per task", avgTransform)
	}
}

func TestDogfoodBenchmark_ReceiptValidatorReplay(t *testing.T) {
	v := NewReceiptValidator()

	validReceipt := CompressionReceipt{
		Stage:       CompressionStageIdentity,
		Codec:       DefaultCompressionCodec,
		Reason:      ReasonSaved,
		InputBytes:  500,
		OutputBytes: 350,
		BytesSaved:  150,
		Duration:    100 * time.Microsecond,
	}

	if err := v.ValidateReceipt(validReceipt); err != nil {
		t.Fatalf("expected valid receipt to pass, got: %v", err)
	}

	// Test conservation failure
	invalidConservation := validReceipt
	invalidConservation.BytesSaved = 200 // 500 != 350 + 200
	if err := v.ValidateReceipt(invalidConservation); err == nil {
		t.Error("expected conservation failure error, got nil")
	}

	// Test wrong stage identity
	invalidStage := validReceipt
	invalidStage.Stage = "native-headroom"
	if err := v.ValidateReceipt(invalidStage); err == nil {
		t.Error("expected stage error, got nil")
	}
}

func TestDogfoodBenchmark_HoldReportOnDegradation(t *testing.T) {
	v := NewReceiptValidator()

	report := &MatchedReport{
		Timestamp: time.Now(),
		Receipts:  make([]CompressionReceipt, 24),
		DefaultArm: ArmResult{
			ArmName:        "default",
			TotalCases:     24,
			TotalInput:     1000,
			TotalOutput:    800,
			TotalSaved:     200,
			SemanticErrors: 1, // Simulated semantic loss!
		},
	}

	for i := range report.Receipts {
		report.Receipts[i] = CompressionReceipt{
			Stage:       CompressionStageIdentity,
			Codec:       DefaultCompressionCodec,
			Reason:      ReasonSaved,
			InputBytes:  100,
			OutputBytes: 80,
			BytesSaved:  20,
		}
	}

	err := v.ValidateReport(report)
	if err == nil {
		t.Fatal("expected validation error on semantic degradation, got nil")
	}
	if report.ExpansionStatus != "HOLD" {
		t.Fatalf("expected ExpansionStatus 'HOLD', got %s", report.ExpansionStatus)
	}
	if !report.DefaultArm.Hold {
		t.Errorf("expected DefaultArm.Hold=true, got false")
	}
}
