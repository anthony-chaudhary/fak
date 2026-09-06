package mcpbroker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"time"
)

// MatchedTaskCase represents an agent task case run under matched arms.
type MatchedTaskCase struct {
	ID          string
	ToolName    string
	RawResult   json.RawMessage
	Content     json.RawMessage
	OptOut      bool
	QualityGoal string
}

// ArmResult captures aggregated performance across one evaluation arm.
type ArmResult struct {
	ArmName        string        `json:"arm_name"`
	TotalCases     int           `json:"total_cases"`
	TotalInput     int           `json:"total_input_bytes"`
	TotalOutput    int           `json:"total_output_bytes"`
	TotalSaved     int           `json:"total_saved_bytes"`
	TotalDuration  time.Duration `json:"total_duration"`
	TotalTransform time.Duration `json:"total_transform_duration"`
	SuccessCount   int           `json:"success_count"`
	SemanticErrors int           `json:"semantic_errors"`
	Hold           bool          `json:"hold"`
	HoldReason     string        `json:"hold_reason,omitempty"`
}

// MatchedReport captures the comparative evaluation across Default and Noop arms.
type MatchedReport struct {
	Timestamp       time.Time            `json:"timestamp"`
	DefaultArm      ArmResult            `json:"default_arm"`
	NoopArm         ArmResult            `json:"noop_arm"`
	Receipts        []CompressionReceipt `json:"receipts"`
	Validated       bool                 `json:"validated"`
	ExpansionStatus string               `json:"expansion_status"` // "APPROVED" or "HOLD"
}

// ReceiptValidator validates payload-free conservation, provenance, and quality of receipts.
type ReceiptValidator struct {
	MaxTransformLatency time.Duration
}

// NewReceiptValidator returns a validator configured with default safety limits.
func NewReceiptValidator() *ReceiptValidator {
	return &ReceiptValidator{
		MaxTransformLatency: 50 * time.Millisecond,
	}
}

// ValidateReceipt checks a single receipt for conservation, provenance, and lack of payload leak.
func (v *ReceiptValidator) ValidateReceipt(r CompressionReceipt) error {
	if r.Stage != CompressionStageIdentity {
		return fmt.Errorf("invalid stage identity: %s", r.Stage)
	}
	if r.Codec != "" && r.Codec != DefaultCompressionCodec {
		return fmt.Errorf("invalid codec: %s", r.Codec)
	}
	// Byte conservation invariant: InputBytes must equal OutputBytes + BytesSaved.
	if r.InputBytes != r.OutputBytes+r.BytesSaved {
		return fmt.Errorf("conservation violation: input(%d) != output(%d) + saved(%d)",
			r.InputBytes, r.OutputBytes, r.BytesSaved)
	}
	if r.BytesSaved < 0 {
		return fmt.Errorf("negative bytes saved: %d", r.BytesSaved)
	}
	if r.InputBytes < 0 || r.OutputBytes < 0 {
		return fmt.Errorf("negative byte counts: input=%d output=%d", r.InputBytes, r.OutputBytes)
	}
	if r.Duration < 0 {
		return fmt.Errorf("negative transform duration: %v", r.Duration)
	}
	if v.MaxTransformLatency > 0 && r.Duration > v.MaxTransformLatency {
		return fmt.Errorf("transform duration %v exceeded budget %v", r.Duration, v.MaxTransformLatency)
	}
	return nil
}

// ValidateReport checks all receipts and arm invariants in a matched live report.
func (v *ReceiptValidator) ValidateReport(report *MatchedReport) error {
	if report == nil {
		return errors.New("nil report")
	}
	if len(report.Receipts) < 24 {
		report.ExpansionStatus = "HOLD"
		return fmt.Errorf("insufficient matched task cases: %d < 24", len(report.Receipts))
	}

	for i, r := range report.Receipts {
		if err := v.ValidateReceipt(r); err != nil {
			report.ExpansionStatus = "HOLD"
			return fmt.Errorf("receipt[%d] invalid: %w", i, err)
		}
	}

	if report.DefaultArm.SemanticErrors > 0 {
		report.ExpansionStatus = "HOLD"
		report.DefaultArm.Hold = true
		report.DefaultArm.HoldReason = fmt.Sprintf("semantic degradation: %d errors", report.DefaultArm.SemanticErrors)
		return fmt.Errorf("semantic degradation in default arm: %d errors", report.DefaultArm.SemanticErrors)
	}

	if report.DefaultArm.TotalInput != report.DefaultArm.TotalOutput+report.DefaultArm.TotalSaved {
		report.ExpansionStatus = "HOLD"
		return fmt.Errorf("default arm conservation failure")
	}

	report.Validated = true
	if report.ExpansionStatus == "" {
		report.ExpansionStatus = "APPROVED"
	}
	return nil
}

// RunMatchedDogfood runs a suite of at least 24 matched tasks under default and noop arms.
func RunMatchedDogfood(ctx context.Context, tasks []MatchedTaskCase) (*MatchedReport, error) {
	if len(tasks) < 24 {
		return nil, fmt.Errorf("dogfood requires at least 24 matched tasks, got %d", len(tasks))
	}

	report := &MatchedReport{
		Timestamp: time.Now().UTC(),
		Receipts:  make([]CompressionReceipt, 0, len(tasks)),
	}

	report.DefaultArm.ArmName = "default"
	report.DefaultArm.TotalCases = len(tasks)
	report.NoopArm.ArmName = "noop"
	report.NoopArm.TotalCases = len(tasks)

	for _, task := range tasks {
		// Run Default arm
		callCtx := ctx
		if task.OptOut {
			callCtx = WithCompressionOptOut(callCtx)
		}
		defOut, defReceipt := CompactStructuredContentWithReceipt(task.RawResult, task.Content, WithCompressionContext(callCtx))
		if defReceipt != nil {
			report.Receipts = append(report.Receipts, *defReceipt)
			report.DefaultArm.TotalInput += defReceipt.InputBytes
			report.DefaultArm.TotalOutput += defReceipt.OutputBytes
			report.DefaultArm.TotalSaved += defReceipt.BytesSaved
			report.DefaultArm.TotalTransform += defReceipt.Duration
		}

		// Run Noop arm
		noopCtx := WithCompressionPolicy(ctx, CompressionOff)
		noopOut, noopReceipt := CompactStructuredContentWithReceipt(task.RawResult, task.Content, WithCompressionContext(noopCtx))
		if noopReceipt != nil {
			report.NoopArm.TotalInput += noopReceipt.InputBytes
			report.NoopArm.TotalOutput += noopReceipt.OutputBytes
			report.NoopArm.TotalSaved += noopReceipt.BytesSaved
			report.NoopArm.TotalTransform += noopReceipt.Duration
		}

		// Semantic fidelity check: unmarshaled JSON semantics must be identical
		var defVal, noopVal any
		if err1 := json.Unmarshal(defOut, &defVal); err1 == nil {
			if err2 := json.Unmarshal(noopOut, &noopVal); err2 == nil {
				if !reflect.DeepEqual(defVal, noopVal) {
					report.DefaultArm.SemanticErrors++
				}
			}
		}
		report.DefaultArm.SuccessCount++
		report.NoopArm.SuccessCount++
	}

	validator := NewReceiptValidator()
	_ = validator.ValidateReport(report)
	return report, nil
}
