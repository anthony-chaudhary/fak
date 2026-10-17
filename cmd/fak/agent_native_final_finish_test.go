package main

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=1s
func TestHeadlessReceiptFinalFinish(t *testing.T) {
	for _, tc := range []struct {
		name       string
		reason     string
		truncated  bool
		circuit    bool
		turnCap    bool
		runErr     error
		wantStatus string
	}{
		{name: "length", reason: "length", truncated: true, wantStatus: "final_answer_truncated"},
		{name: "stop", reason: "stop", wantStatus: "completed"},
		{name: "omitted reason", wantStatus: "completed"},
		{name: "run error precedence", reason: "length", truncated: true, circuit: true, turnCap: true, runErr: errors.New("fixture failure"), wantStatus: "failed"},
		{name: "circuit precedence", reason: "length", truncated: true, circuit: true, turnCap: true, wantStatus: "stopped_circuit_breaker"},
		{name: "synthesis turn cap precedence", reason: "length", truncated: true, turnCap: true, wantStatus: "turn_cap_exceeded"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			metrics := agent.ArmMetrics{
				Arm: "fak", FinalAnswer: "Partial result remains visible",
				FinalFinishReason: tc.reason, FinalAnswerTruncated: tc.truncated,
				PromptTokens: 11, CompletionTokens: 7,
				CircuitBreakerTripped: tc.circuit, HitTurnCap: tc.turnCap,
				// Tool/effect completion must not override truncated prose.
				TaskCompleted: true,
			}
			body, err := json.Marshal(newHeadlessAgentReceipt("task", "fixture", metrics, nil, "", tc.runErr))
			if err != nil {
				t.Fatal(err)
			}
			var got nativeAgentReceipt
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatal(err)
			}
			if got.Status != tc.wantStatus {
				t.Fatalf("serialized status = %q, want %q", got.Status, tc.wantStatus)
			}
			if got.Schema != nativeAgentReceiptSchema || got.FinalAnswer != metrics.FinalAnswer || got.Metrics.FinalAnswer != metrics.FinalAnswer {
				t.Fatalf("receipt identity or partial text lost: %s", body)
			}
			if got.Metrics.PromptTokens != 11 || got.Metrics.CompletionTokens != 7 || got.Metrics.FinalFinishReason != tc.reason || got.Metrics.FinalAnswerTruncated != tc.truncated || !got.Metrics.TaskCompleted {
				t.Fatalf("usage or final classification lost: %s", body)
			}
		})
	}
}
