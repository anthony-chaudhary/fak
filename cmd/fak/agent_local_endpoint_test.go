package main

import (
	"bytes"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestEnsureAgentLocalEndpointDefaults(t *testing.T) {
	const planner, stall = "FAK_PLANNER_TIMEOUT_S", "FAK_STREAM_STALL_TIMEOUT_S"

	t.Run("loopback sets both", func(t *testing.T) {
		t.Setenv(planner, "")
		t.Setenv(stall, "")
		var stderr bytes.Buffer
		ensureAgentLocalEndpointDefaults("http://127.0.0.1:18560/v1", &stderr)
		if got := os.Getenv(planner); got != "1800" {
			t.Fatalf("%s = %q, want 1800", planner, got)
		}
		if got := os.Getenv(stall); got != "600" {
			t.Fatalf("%s = %q, want 600", stall, got)
		}
		line := stderr.String()
		if strings.Count(line, "\n") != 1 || !strings.Contains(line, planner) || !strings.Contains(line, stall) {
			t.Fatalf("want one stderr line naming both knobs, got %q", line)
		}
	})

	t.Run("preset env is untouched", func(t *testing.T) {
		t.Setenv(planner, "90")
		t.Setenv(stall, "0")
		var stderr bytes.Buffer
		ensureAgentLocalEndpointDefaults("http://localhost:18560/v1", &stderr)
		if got := os.Getenv(planner); got != "90" {
			t.Fatalf("%s = %q, want 90", planner, got)
		}
		if got := os.Getenv(stall); got != "0" {
			t.Fatalf("%s = %q, want 0", stall, got)
		}
		if stderr.Len() != 0 {
			t.Fatalf("want no stderr when nothing defaulted, got %q", stderr.String())
		}
	})

	t.Run("hosted sets nothing", func(t *testing.T) {
		t.Setenv(planner, "")
		t.Setenv(stall, "")
		var stderr bytes.Buffer
		ensureAgentLocalEndpointDefaults("https://api.openai.com/v1", &stderr)
		if os.Getenv(planner) != "" || os.Getenv(stall) != "" {
			t.Fatalf("hosted endpoint changed env: %s=%q %s=%q", planner, os.Getenv(planner), stall, os.Getenv(stall))
		}
		if stderr.Len() != 0 {
			t.Fatalf("want no stderr for hosted, got %q", stderr.String())
		}
	})
}

func TestNativeAgentReceiptStatus(t *testing.T) {
	runErr := errors.New("stopped")
	cases := []struct {
		name     string
		metrics  agent.ArmMetrics
		err      error
		native   string
		headless string
	}{
		{"clean", agent.ArmMetrics{}, nil, "", "completed"},
		{"untyped error", agent.ArmMetrics{}, runErr, "failed", "failed"},
		{"identical retry", agent.ArmMetrics{EndedBy: agent.EndedByIdenticalRetry}, runErr, "identical_retry", "identical_retry"},
		{"context overflow", agent.ArmMetrics{EndedBy: agent.EndedByContextOverflow}, runErr, "context_overflow", "context_overflow"},
		{"tool call truncated", agent.ArmMetrics{EndedBy: agent.EndedByToolCallTruncated}, runErr, "tool_call_truncated", "tool_call_truncated"},
		{"turn cap", agent.ArmMetrics{HitTurnCap: true}, nil, "", "turn_cap_exceeded"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := newNativeAgentReceiptWithError("task", "model", tc.metrics, tc.err).Status; got != tc.native {
				t.Errorf("native status = %q, want %q", got, tc.native)
			}
			if got := newHeadlessAgentReceipt("task", "model", tc.metrics, nil, "", tc.err).Status; got != tc.headless {
				t.Errorf("headless status = %q, want %q", got, tc.headless)
			}
		})
	}
}
