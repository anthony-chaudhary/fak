package model

import (
	"sync"
	"testing"
)

func TestQwen38MTP_KillSwitch_LiveRollbackDrill(t *testing.T) {
	ks := NewQwen38MTPKillSwitch()

	cfg := Qwen38MTPDrillConfig{
		DrillID:            "test-drill-001",
		InitialTokens:      []int{151644, 872, 198},
		DraftTokens:        []int{201, 202, 203},
		TargetContinuation: []int{301, 302, 303, 304},
		HiddenSize:         32,
		MaxDraftDepth:      3,
	}

	receipt, err := ks.ExecuteRollbackDrill(cfg)
	if err != nil {
		t.Fatalf("ExecuteRollbackDrill failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt.Validate() failed: %v", err)
	}

	if !receipt.RollbackSuccessful {
		t.Error("expected RollbackSuccessful = true")
	}
	if !receipt.CleanContinuation {
		t.Error("expected CleanContinuation = true")
	}
	if !receipt.Reenabled {
		t.Error("expected Reenabled = true")
	}
	if receipt.PreKillEngine != Qwen38EngineMTP {
		t.Errorf("PreKillEngine = %q, want %q", receipt.PreKillEngine, Qwen38EngineMTP)
	}
	if receipt.PostKillEngine != Qwen38EngineTargetDecode {
		t.Errorf("PostKillEngine = %q, want %q", receipt.PostKillEngine, Qwen38EngineTargetDecode)
	}
	if receipt.DowngradeReason != Qwen38MTPDisabledByPolicy {
		t.Errorf("DowngradeReason = %q, want %q", receipt.DowngradeReason, Qwen38MTPDisabledByPolicy)
	}
}

func TestQwen38MTP_KillSwitch_StateTransitionsAndReenablePolicy(t *testing.T) {
	ks := NewQwen38MTPKillSwitch()

	// Initial status
	if ks.IsEngaged() {
		t.Fatal("kill switch should initialize disengaged")
	}
	eligible, reason := ks.CheckEligible()
	if !eligible || reason != Qwen38MTPEligible {
		t.Fatalf("CheckEligible expected true, got %v, %v", eligible, reason)
	}

	// Engage
	if err := ks.Engage("test operator intervention"); err != nil {
		t.Fatalf("Engage failed: %v", err)
	}
	if !ks.IsEngaged() {
		t.Fatal("kill switch should be engaged")
	}

	eligible, reason = ks.CheckEligible()
	if eligible || reason != Qwen38MTPDisabledByPolicy {
		t.Fatalf("CheckEligible while engaged expected false and disabled_by_policy, got %v, %v", eligible, reason)
	}

	// Admit check
	in := &Qwen38MTPEligibilityInput{OperatorEnabled: true}
	admitted, admReason := ks.Admit(in)
	if admitted || admReason != Qwen38MTPDisabledByPolicy || in.OperatorEnabled {
		t.Fatalf("Admit failed to enforce kill switch: admitted=%v, reason=%v, in.OperatorEnabled=%v", admitted, admReason, in.OperatorEnabled)
	}

	// Re-enabling without policy permission fails closed
	if err := ks.Disengage(false); err == nil {
		t.Fatal("expected Disengage(false) to fail closed")
	}
	if !ks.IsEngaged() {
		t.Fatal("kill switch should remain engaged after failed disengage attempt")
	}

	// Re-enabling with policy permission succeeds
	if err := ks.Disengage(true); err != nil {
		t.Fatalf("Disengage(true) failed: %v", err)
	}
	if ks.IsEngaged() {
		t.Fatal("kill switch should be disengaged after policy permitted restore")
	}
}

func TestQwen38MTP_KillSwitch_InFlightRegistrationRollback(t *testing.T) {
	ks := NewQwen38MTPKillSwitch()

	hiddenSize := 16
	initialKV := [][]float32{make([]float32, hiddenSize), make([]float32, hiddenSize)}
	initialRec := [][]float32{make([]float32, hiddenSize)}
	initialConv := [][]float32{make([]float32, hiddenSize)}
	baselineState := NewMTPState(initialKV, initialRec, initialConv)
	baselineState.Position = 2

	txCfg := MTPTransactionConfig{
		HiddenSize:    hiddenSize,
		MaxDraftDepth: 2,
		Backend:       Qwen38MTPBackendMetal,
	}
	tx := NewMTPTransactionWithState(baselineState, txCfg)
	_, unreg := ks.RegisterTransaction(tx)
	defer unreg()

	// Begin speculative round and mutate state
	_, err := tx.BeginRound()
	if err != nil {
		t.Fatalf("BeginRound failed: %v", err)
	}
	stepKV := [][]float32{make([]float32, hiddenSize)}
	stepRec := [][]float32{make([]float32, hiddenSize)}
	stepConv := [][]float32{make([]float32, hiddenSize)}
	if err := tx.AppendStep(stepKV, stepRec, stepConv, nil); err != nil {
		t.Fatalf("AppendStep failed: %v", err)
	}

	// Engage kill switch: should trigger immediate rollback and downgrade
	if err := ks.Engage("operator cutoff"); err != nil {
		t.Fatalf("Engage failed: %v", err)
	}

	if !tx.IsDowngraded() {
		t.Fatal("in-flight transaction was not downgraded upon kill switch engagement")
	}
	if tx.Engine() != Qwen38EngineTargetDecode {
		t.Fatalf("tx.Engine() = %q, want %q", tx.Engine(), Qwen38EngineTargetDecode)
	}

	rolledBackState := tx.State()
	if rolledBackState.Position != baselineState.Position {
		t.Fatalf("rolledBackState.Position = %d, want %d", rolledBackState.Position, baselineState.Position)
	}
	if !rolledBackState.Equal(baselineState) {
		t.Fatal("rolledBackState is not bit-exact equal to baseline state")
	}

	// Registering a new transaction while kill switch is engaged immediately aborts it
	tx2 := NewMTPTransactionWithState(baselineState, txCfg)
	_, unreg2 := ks.RegisterTransaction(tx2)
	defer unreg2()

	if !tx2.IsDowngraded() {
		t.Fatal("transaction registered while engaged should be immediately downgraded")
	}
}

func TestQwen38MTP_KillSwitch_ConcurrentSafety(t *testing.T) {
	ks := NewQwen38MTPKillSwitch()
	var wg sync.WaitGroup
	workers := 16
	iterations := 100

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				if (workerID+j)%3 == 0 {
					_ = ks.Engage("concurrent test cutoff")
				} else if (workerID+j)%3 == 1 {
					_ = ks.Disengage(true)
				} else {
					_ = ks.IsEngaged()
					_, _ = ks.CheckEligible()
					in := &Qwen38MTPEligibilityInput{OperatorEnabled: true}
					_, _ = ks.Admit(in)
				}
			}
		}(i)
	}

	wg.Wait()
}
