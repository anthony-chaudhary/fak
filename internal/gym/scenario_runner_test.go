package gym

import (
	"context"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestEphemeralGateway verifies initialization, temporary directory provisioning,
// loopback endpoint serving, and graceful cleanup of an in-process ephemeral gateway.
func TestEphemeralGateway(t *testing.T) {
	ctx := context.Background()

	opts := EphemeralGatewayOptions{
		DeferColdTools:      true,
		MaxSubturnToolCalls: 5,
		MaxSubturnTokens:    2000,
	}

	eg, err := NewEphemeralGateway(opts)
	if err != nil {
		t.Fatalf("NewEphemeralGateway failed: %v", err)
	}

	url := eg.URL()
	if !strings.HasPrefix(url, "http://") {
		t.Fatalf("expected HTTP URL, got %q", url)
	}

	if eg.Server() == nil {
		t.Fatal("expected non-nil gateway server")
	}

	casDir := eg.CASDir()
	if casDir == "" {
		t.Fatal("expected non-empty CAS directory")
	}

	if info, err := os.Stat(casDir); err != nil || !info.IsDir() {
		t.Fatalf("CAS directory %q does not exist: %v", casDir, err)
	}

	// Verify health check endpoint responds
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url+"/healthz", nil)
	if err != nil {
		t.Fatalf("failed to construct /healthz request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("health check GET /healthz failed: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK from /healthz, got %d", resp.StatusCode)
	}

	// Verify close reaps temporary directory and is idempotent
	if err := eg.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	if _, err := os.Stat(casDir); !os.IsNotExist(err) {
		t.Errorf("expected temporary CAS directory %q to be removed, stat err: %v", casDir, err)
	}

	if err := eg.Close(); err != nil {
		t.Fatalf("idempotent second Close failed: %v", err)
	}
}

// TestScenarioRunner executes comprehensive closed-loop multi-turn harness simulations
// covering all 4 stress profiles:
// 1. Deep Subturn Burst
// 2. Tool Elision & Context Restore (>1KiB payloads, CAS stashing, in-band restore)
// 3. Context Threshold Yield Valve Recovery (resumption after subturn yield)
// 4. Livelock Circuit Breaker (runaway repetition loop detection)
func TestScenarioRunner(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	runner := NewScenarioRunner()

	// -----------------------------------------------------------------------
	// Stress Profile 1: Deep Subturn Burst
	// -----------------------------------------------------------------------
	t.Run("DeepSubturnBurst", func(t *testing.T) {
		scenario := Scenario{
			ID:      "scenario-deep-burst",
			Name:    "Deep Subturn Burst Simulation",
			Dialect: DialectCodexResponses,
			Profile: StressProfile{
				MaxTurns:        10,
				TargetToolCalls: 6,
			},
		}

		receipt, err := runner.Run(ctx, scenario)
		if err != nil {
			t.Fatalf("Run failed: %v", err)
		}

		if ok, reason := receipt.VerifyReceipt(scenario.ID); !ok {
			t.Fatalf("receipt verification failed: %s (receipt: %+v)", reason, receipt)
		}

		if receipt.TotalToolCalls < 6 {
			t.Errorf("expected >= 6 tool calls, observed %d", receipt.TotalToolCalls)
		}
		if receipt.TurnsExecuted < 6 {
			t.Errorf("expected >= 6 turns executed, observed %d", receipt.TurnsExecuted)
		}
		if !receipt.MultiTurnPass {
			t.Errorf("expected MultiTurnPass == true")
		}
		if receipt.LivelockDetected {
			t.Errorf("expected LivelockDetected == false")
		}
		if receipt.TranscriptDigest == "" {
			t.Errorf("expected non-empty TranscriptDigest")
		}
	})

	// -----------------------------------------------------------------------
	// Stress Profile 2: Tool Elision & Context Restore
	// -----------------------------------------------------------------------
	t.Run("ToolElisionAndContextRestore", func(t *testing.T) {
		scenario := Scenario{
			ID:      "scenario-elide-restore",
			Name:    "Tool Elision and In-Band Context Restore Simulation",
			Dialect: DialectCodexResponses,
			Profile: StressProfile{
				MaxTurns:         12,
				TargetToolCalls:  6,
				PayloadSizeBytes: 35000, // >32KiB to trigger both tool result elision (>1024B) and durable CAS persistence (>=32KiB)
				ExpectRestore:    true,
			},
		}

		receipt, err := runner.Run(ctx, scenario)
		if err != nil {
			t.Fatalf("Run failed: %v", err)
		}

		if ok, reason := receipt.VerifyReceipt(scenario.ID); !ok {
			t.Fatalf("receipt verification failed: %s (receipt: %+v)", reason, receipt)
		}

		if receipt.ElisionsObserved == 0 {
			t.Errorf("expected ElisionsObserved > 0, observed %d", receipt.ElisionsObserved)
		}
		if receipt.RestoresObserved == 0 {
			t.Errorf("expected RestoresObserved > 0, observed %d", receipt.RestoresObserved)
		}
		if receipt.NetTokenSavings <= 0 {
			t.Errorf("expected NetTokenSavings > 0, observed %d", receipt.NetTokenSavings)
		}
		if !receipt.MultiTurnPass {
			t.Errorf("expected MultiTurnPass == true")
		}
	})

	// -----------------------------------------------------------------------
	// Stress Profile 3: Context Threshold Yield Valve Recovery
	// -----------------------------------------------------------------------
	t.Run("ContextThresholdYieldValveRecovery", func(t *testing.T) {
		scenario := Scenario{
			ID:      "scenario-yield-recovery",
			Name:    "Context Threshold Yield Valve Recovery Simulation",
			Dialect: DialectCodexResponses,
			Profile: StressProfile{
				MaxTurns:        15,
				TargetToolCalls: 6,
				InduceYield:     true,
			},
		}

		receipt, err := runner.Run(ctx, scenario)
		if err != nil {
			t.Fatalf("Run failed: %v", err)
		}

		if ok, reason := receipt.VerifyReceipt(scenario.ID); !ok {
			t.Fatalf("receipt verification failed: %s (receipt: %+v)", reason, receipt)
		}

		if receipt.YieldsObserved < 1 {
			t.Errorf("expected YieldsObserved >= 1, observed %d", receipt.YieldsObserved)
		}
		if !receipt.MultiTurnPass {
			t.Errorf("expected MultiTurnPass == true")
		}
		if receipt.LivelockDetected {
			t.Errorf("expected LivelockDetected == false")
		}
	})

	// -----------------------------------------------------------------------
	// Stress Profile 4: Livelock Circuit Breaker
	// -----------------------------------------------------------------------
	t.Run("LivelockCircuitBreaker", func(t *testing.T) {
		scenario := Scenario{
			ID:      "scenario-livelock",
			Name:    "Livelock Circuit Breaker Simulation",
			Dialect: DialectCodexResponses,
			Profile: StressProfile{
				MaxTurns:        10,
				SimulateRunaway: true,
			},
		}

		receipt, err := runner.Run(ctx, scenario)
		if err != nil {
			t.Fatalf("Run unexpected error: %v", err)
		}

		if !receipt.LivelockDetected {
			t.Errorf("expected LivelockDetected == true")
		}
		if !receipt.ZeroProgressTripped {
			t.Errorf("expected ZeroProgressTripped == true")
		}
		if receipt.Outcome != "FAIL" {
			t.Errorf("expected Outcome == FAIL, got %s", receipt.Outcome)
		}

		ok, reason := receipt.VerifyReceipt(scenario.ID)
		if ok {
			t.Errorf("expected VerifyReceipt to fail on livelock scenario, but passed")
		}
		if !strings.Contains(reason, "livelock") {
			t.Errorf("expected failure reason to mention livelock, got %q", reason)
		}
	})
}

// TestScenarioRunnerDialects verifies closed-loop simulation across all supported client dialects.
func TestScenarioRunnerDialects(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	runner := NewScenarioRunner()

	dialects := []string{
		DialectClaudeMessages,
		DialectOpenCodeChat,
	}

	for _, dialect := range dialects {
		t.Run(dialect, func(t *testing.T) {
			scenario := Scenario{
				ID:      "scenario-" + dialect,
				Name:    "Dialect Test " + dialect,
				Dialect: dialect,
				Profile: StressProfile{
					MaxTurns:        6,
					TargetToolCalls: 2,
				},
				ClientTools: []agent.ToolDef{
					{
						Type: "function",
						Function: agent.ToolDefFunction{
							Name:        "read_echo",
							Description: "Echo test tool",
						},
					},
				},
			}

			receipt, err := runner.Run(ctx, scenario)
			if err != nil {
				t.Fatalf("Run failed for dialect %s: %v", dialect, err)
			}

			if ok, reason := receipt.VerifyReceipt(scenario.ID); !ok {
				t.Fatalf("receipt verification failed for dialect %s: %s (receipt: %+v)", dialect, reason, receipt)
			}

			if receipt.TotalToolCalls < 2 {
				t.Errorf("expected >= 2 tool calls, got %d", receipt.TotalToolCalls)
			}
			if !receipt.MultiTurnPass {
				t.Errorf("expected MultiTurnPass == true")
			}
		})
	}
}

// TestReceiptVerificationEdgeCases ensures VerifyReceipt correctly validates receipts.
func TestReceiptVerificationEdgeCases(t *testing.T) {
	r := &GymReceipt{
		Schema:        GymReceiptSchema,
		ScenarioID:    "valid-id",
		TurnsExecuted: 3,
		Outcome:       "PASS",
	}

	if ok, reason := r.VerifyReceipt("valid-id"); !ok {
		t.Fatalf("expected valid receipt to pass, got: %s", reason)
	}

	// Mismatched scenario ID
	if ok, reason := r.VerifyReceipt("other-id"); ok || !strings.Contains(reason, "mismatch") {
		t.Errorf("expected mismatch error, got ok=%v, reason=%s", ok, reason)
	}

	// Invalid schema
	invalidSchema := *r
	invalidSchema.Schema = "bad.schema"
	if ok, reason := invalidSchema.VerifyReceipt("valid-id"); ok || !strings.Contains(reason, "schema") {
		t.Errorf("expected schema error, got ok=%v, reason=%s", ok, reason)
	}

	// Zero turns
	zeroTurns := *r
	zeroTurns.TurnsExecuted = 0
	if ok, reason := zeroTurns.VerifyReceipt("valid-id"); ok || !strings.Contains(reason, "zero turns") {
		t.Errorf("expected zero turns error, got ok=%v, reason=%s", ok, reason)
	}

	// Livelock detected
	livelock := *r
	livelock.LivelockDetected = true
	if ok, reason := livelock.VerifyReceipt("valid-id"); ok || !strings.Contains(reason, "livelock") {
		t.Errorf("expected livelock error, got ok=%v, reason=%s", ok, reason)
	}

	// Outcome FAIL
	failOutcome := *r
	failOutcome.Outcome = "FAIL"
	failOutcome.FailureReason = "some error"
	if ok, reason := failOutcome.VerifyReceipt("valid-id"); ok || !strings.Contains(reason, "some error") {
		t.Errorf("expected fail outcome error, got ok=%v, reason=%s", ok, reason)
	}
}
