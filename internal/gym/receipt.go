package gym

import (
	"fmt"
	"time"
)

// GymReceiptSchema defines the canonical schema identifier for gym simulation receipts.
const GymReceiptSchema = "fak.gym.receipt.v1"

const (
	OutcomePass = "PASS"
	OutcomeFail = "FAIL"
)

// GymReceipt records the execution summary, telemetry, and verification status of a gym simulation scenario.
type GymReceipt struct {
	Schema              string    `json:"schema"`
	ScenarioID          string    `json:"scenario_id"`
	Timestamp           time.Time `json:"timestamp"`
	TurnsExecuted       int       `json:"turns_executed"`
	TotalToolCalls      int       `json:"total_tool_calls"`
	ElisionsObserved    int       `json:"elisions_observed"`
	RestoresObserved    int       `json:"restores_observed"`
	YieldsObserved      int       `json:"yields_observed"`
	LivelockDetected    bool      `json:"livelock_detected"`
	ZeroProgressTripped bool      `json:"zero_progress_tripped"`
	PeakPromptTokens    int       `json:"peak_prompt_tokens"`
	NetTokenSavings     int       `json:"net_token_savings"`
	MultiTurnPass       bool      `json:"multi_turn_pass"`
	Outcome             string    `json:"outcome"` // "PASS", "FAIL"
	FailureReason       string    `json:"failure_reason,omitempty"`
	TranscriptDigest    string    `json:"transcript_digest"`
}

// VerifyReceipt asserts that the receipt conforms to the valid schema, observed non-zero turns,
// suffered no livelocks, matches the expected scenario (if provided), and achieved a PASS outcome.
func (r *GymReceipt) VerifyReceipt(expectedScenario string) (bool, string) {
	if r == nil {
		return false, "receipt is nil"
	}
	if r.Schema != GymReceiptSchema {
		return false, fmt.Sprintf("invalid schema: expected %q, got %q", GymReceiptSchema, r.Schema)
	}
	if expectedScenario != "" && r.ScenarioID != expectedScenario {
		return false, fmt.Sprintf("scenario mismatch: expected %q, got %q", expectedScenario, r.ScenarioID)
	}
	if r.TurnsExecuted <= 0 {
		return false, fmt.Sprintf("zero turns executed (%d)", r.TurnsExecuted)
	}
	if r.LivelockDetected {
		return false, "livelock detected"
	}
	if r.Outcome != "PASS" {
		reason := r.FailureReason
		if reason == "" {
			reason = "unspecified failure"
		}
		return false, fmt.Sprintf("outcome not PASS: %s (%s)", r.Outcome, reason)
	}
	return true, ""
}

// VerifyReceipt is the package-level helper that validates a GymReceipt.
func VerifyReceipt(r GymReceipt, expectedScenario string) (bool, string) {
	return r.VerifyReceipt(expectedScenario)
}
