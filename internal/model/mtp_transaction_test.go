package model

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestMTPTransaction_CommitAccepted(t *testing.T) {
	initState := &MTPState{
		Position:  10,
		KV:        [][]float32{{1.0, 2.0, 3.0, 4.0}},
		Recurrent: [][]float32{{0.1, 0.2}, {0.3, 0.4}},
		Conv:      [][]float32{{0.01, 0.02}},
		Hidden:    [][]float32{{0.5, 0.6, 0.7, 0.8}},
	}

	cfg := MTPTransactionConfig{
		HiddenSize:    4,
		VocabSize:     100,
		MaxDraftDepth: 4,
		Backend:       Qwen38MTPBackendMetal,
	}

	tx := NewMTPTransactionWithState(initState, cfg)
	defer tx.Close()

	// Round 1: Full acceptance of 3 draft tokens.
	cp, err := tx.BeginRound()
	if err != nil {
		t.Fatalf("BeginRound error = %v", err)
	}
	if cp == nil {
		t.Fatal("expected non-nil checkpoint")
	}

	draft := []int{10, 20, 30}
	hidden := [][]float32{{1, 2, 3, 4}}
	if err := tx.Propose(draft, hidden); err != nil {
		t.Fatalf("Propose error = %v", err)
	}

	for i := 0; i < 3; i++ {
		stepKV := [][]float32{{float32(i + 1), float32(i + 2)}}
		stepRec := [][]float32{{float32(i+1) * 0.1, float32(i+2) * 0.1}}
		stepConv := [][]float32{{float32(i+1) * 0.01}}
		stepHidden := []float32{float32(i), float32(i), float32(i), float32(i)}
		if err := tx.AppendStep(stepKV, stepRec, stepConv, stepHidden); err != nil {
			t.Fatalf("AppendStep(%d) error = %v", i, err)
		}
	}

	targetTokens := []int{10, 20, 30}
	accepted, err := tx.VerifyTokens(targetTokens)
	if err != nil {
		t.Fatalf("VerifyTokens error = %v", err)
	}
	if accepted != 3 {
		t.Fatalf("accepted = %d, want 3", accepted)
	}

	if err := tx.Commit(accepted); err != nil {
		t.Fatalf("Commit error = %v", err)
	}

	state := tx.State()
	if state.Position != 13 {
		t.Errorf("state.Position = %d, want 13", state.Position)
	}
	if len(state.KV) != 4 { // 1 initial + 3 appended
		t.Errorf("len(state.KV) = %d, want 4", len(state.KV))
	}
	if len(state.Hidden) != 4 {
		t.Errorf("len(state.Hidden) = %d, want 4", len(state.Hidden))
	}

	acc := tx.Accounting()
	if acc.AcceptedCount != 3 || acc.RejectedCount != 0 {
		t.Errorf("accounting counts mismatch: accepted=%d rejected=%d", acc.AcceptedCount, acc.RejectedCount)
	}

	// Round 2: Partial acceptance (2 out of 3 accepted).
	if _, err := tx.BeginRound(); err != nil {
		t.Fatalf("BeginRound 2 error = %v", err)
	}

	draft2 := []int{40, 50, 60}
	if err := tx.Propose(draft2, hidden); err != nil {
		t.Fatalf("Propose 2 error = %v", err)
	}

	for i := 0; i < 3; i++ {
		stepKV := [][]float32{{float32(i + 10)}}
		stepRec := [][]float32{{float32(i+10) * 0.1}}
		stepConv := [][]float32{{float32(i+10) * 0.01}}
		stepHidden := []float32{float32(i + 10), 0, 0, 0}
		if err := tx.AppendStep(stepKV, stepRec, stepConv, stepHidden); err != nil {
			t.Fatalf("AppendStep 2 (%d) error = %v", i, err)
		}
	}

	// Target evaluation has token 40, 50, but then 99 (divergence at 3rd token).
	targetTokens2 := []int{40, 50, 99}
	accepted2, err := tx.VerifyTokens(targetTokens2)
	if err != nil {
		t.Fatalf("VerifyTokens 2 error = %v", err)
	}
	if accepted2 != 2 {
		t.Fatalf("accepted2 = %d, want 2", accepted2)
	}

	if err := tx.Commit(accepted2); err != nil {
		t.Fatalf("Commit 2 error = %v", err)
	}

	state2 := tx.State()
	if state2.Position != 15 {
		t.Errorf("state2.Position = %d, want 15 (13 + 2)", state2.Position)
	}
	if len(state2.KV) != 6 { // 4 previous + 2 accepted
		t.Errorf("len(state2.KV) = %d, want 6", len(state2.KV))
	}

	acc2 := tx.Accounting()
	if acc2.AcceptedCount != 5 || acc2.RejectedCount != 1 {
		t.Errorf("accounting counts mismatch after round 2: accepted=%d rejected=%d", acc2.AcceptedCount, acc2.RejectedCount)
	}
}

func TestMTPTransaction_ExactRollbackOnRejection(t *testing.T) {
	initState := &MTPState{
		Position:  5,
		KV:        [][]float32{{1.1, 1.2}, {2.1, 2.2}, {3.1, 3.2}},
		Recurrent: [][]float32{{10.1, 10.2}, {20.1, 20.2}},
		Conv:      [][]float32{{0.5, 0.6}},
		Hidden:    [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}},
	}

	cfg := MTPTransactionConfig{
		HiddenSize:    4,
		VocabSize:     100,
		MaxDraftDepth: 4,
		Backend:       Qwen38MTPBackendMetal,
	}

	tx := NewMTPTransactionWithState(initState, cfg)
	defer tx.Close()

	expectedBaseline := initState.Clone()
	initialBytes := initState.ByteSize()

	// Scenario 1: Rejection via Commit(0).
	if _, err := tx.BeginRound(); err != nil {
		t.Fatalf("BeginRound error = %v", err)
	}

	draft := []int{11, 22, 33}
	hidden := [][]float32{{1, 2, 3, 4}}
	if err := tx.Propose(draft, hidden); err != nil {
		t.Fatalf("Propose error = %v", err)
	}

	// Mutate speculative state with intermediate steps.
	for i := 0; i < 3; i++ {
		stepKV := [][]float32{{99.9, 99.9}}
		stepRec := [][]float32{{88.8, 88.8}}
		stepConv := [][]float32{{77.7, 77.7}}
		stepHidden := []float32{66.6, 66.6, 66.6, 66.6}
		if err := tx.AppendStep(stepKV, stepRec, stepConv, stepHidden); err != nil {
			t.Fatalf("AppendStep error = %v", err)
		}
	}

	// Active round has grown in memory.
	midAccounting := tx.Accounting()
	if midAccounting.CurrentMemoryBytes <= initialBytes {
		t.Errorf("CurrentMemoryBytes during round = %d, want > %d", midAccounting.CurrentMemoryBytes, initialBytes)
	}

	// Target rejects all draft tokens.
	targetTokens := []int{99, 98, 97}
	accepted, err := tx.VerifyTokens(targetTokens)
	if err != nil {
		t.Fatalf("VerifyTokens error = %v", err)
	}
	if accepted != 0 {
		t.Fatalf("accepted = %d, want 0", accepted)
	}

	if err := tx.Commit(accepted); err != nil {
		t.Fatalf("Commit(0) error = %v", err)
	}

	// Verify exact bit-for-bit parity with expected baseline.
	postState := tx.State()
	if !postState.Equal(expectedBaseline) {
		t.Fatalf("post-rollback state does not equal expected baseline:\ngot  %+v\nwant %+v", postState, expectedBaseline)
	}

	// Verify zero memory leaks: current memory equals initial baseline.
	postAccounting := tx.Accounting()
	if postAccounting.CurrentMemoryBytes != initialBytes {
		t.Errorf("CurrentMemoryBytes after rollback = %d, want %d (zero leak)", postAccounting.CurrentMemoryBytes, initialBytes)
	}
	if postAccounting.RejectedCount != 3 {
		t.Errorf("RejectedCount = %d, want 3", postAccounting.RejectedCount)
	}
	if postAccounting.RollbackCount != 1 {
		t.Errorf("RollbackCount = %d, want 1", postAccounting.RollbackCount)
	}

	// Scenario 2: Direct Rollback() call during active round.
	if _, err := tx.BeginRound(); err != nil {
		t.Fatalf("BeginRound 2 error = %v", err)
	}
	if err := tx.Propose([]int{55}, hidden); err != nil {
		t.Fatalf("Propose 2 error = %v", err)
	}
	if err := tx.AppendStep([][]float32{{100}}, [][]float32{{200}}, [][]float32{{300}}, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("AppendStep 2 error = %v", err)
	}

	if err := tx.Rollback(); err != nil {
		t.Fatalf("Rollback() error = %v", err)
	}

	postState2 := tx.State()
	if !postState2.Equal(expectedBaseline) {
		t.Fatalf("state after direct Rollback does not match baseline:\ngot  %+v\nwant %+v", postState2, expectedBaseline)
	}
	if tx.Accounting().CurrentMemoryBytes != initialBytes {
		t.Errorf("CurrentMemoryBytes = %d, want %d", tx.Accounting().CurrentMemoryBytes, initialBytes)
	}
}

func TestMTPTransaction_FailClosedDowngrade(t *testing.T) {
	cfg := MTPTransactionConfig{
		HiddenSize:    4,
		VocabSize:     50,
		MaxDraftDepth: 3,
		Backend:       Qwen38MTPBackendMetal,
	}

	t.Run("unsupported_dimensions_hidden", func(t *testing.T) {
		tx := NewMTPTransaction(cfg)
		defer tx.Close()

		_, _ = tx.BeginRound()
		// Wrong hidden size: length 3 instead of 4.
		err := tx.Propose([]int{1, 2}, [][]float32{{1, 2, 3}})
		if err == nil {
			t.Fatal("expected error on hidden dimension mismatch")
		}
		if !tx.IsDowngraded() {
			t.Fatal("expected tx to be downgraded")
		}
		if tx.DowngradeReason() != MTPDowngradeDimensionMismatch {
			t.Errorf("DowngradeReason = %q, want %q", tx.DowngradeReason(), MTPDowngradeDimensionMismatch)
		}
		if tx.Engine() != Qwen38EngineTargetDecode {
			t.Errorf("Engine = %q, want %q", tx.Engine(), Qwen38EngineTargetDecode)
		}
		assertNoForeignFallback(t, tx)
	})

	t.Run("unsupported_dimensions_vocab", func(t *testing.T) {
		tx := NewMTPTransaction(cfg)
		defer tx.Close()

		_, _ = tx.BeginRound()
		// Token 60 is outside vocab [0, 50).
		err := tx.Propose([]int{1, 60}, [][]float32{{1, 2, 3, 4}})
		if err == nil {
			t.Fatal("expected error on vocab overflow")
		}
		if tx.DowngradeReason() != MTPDowngradeDimensionMismatch {
			t.Errorf("DowngradeReason = %q, want %q", tx.DowngradeReason(), MTPDowngradeDimensionMismatch)
		}
		assertNoForeignFallback(t, tx)
	})

	t.Run("unsupported_dimensions_depth", func(t *testing.T) {
		tx := NewMTPTransaction(cfg)
		defer tx.Close()

		_, _ = tx.BeginRound()
		// 4 tokens exceeds MaxDraftDepth 3.
		err := tx.Propose([]int{1, 2, 3, 4}, [][]float32{{1, 2, 3, 4}})
		if err == nil {
			t.Fatal("expected error on depth overflow")
		}
		if tx.DowngradeReason() != MTPDowngradeDimensionMismatch {
			t.Errorf("DowngradeReason = %q, want %q", tx.DowngradeReason(), MTPDowngradeDimensionMismatch)
		}
		assertNoForeignFallback(t, tx)
	})

	t.Run("numerical_nan_in_hidden", func(t *testing.T) {
		tx := NewMTPTransaction(cfg)
		defer tx.Close()

		_, _ = tx.BeginRound()
		nanHidden := [][]float32{{1.0, float32(math.NaN()), 3.0, 4.0}}
		err := tx.Propose([]int{1, 2}, nanHidden)
		if err == nil {
			t.Fatal("expected error on NaN hidden state")
		}
		if tx.DowngradeReason() != MTPDowngradeNumericalNaN {
			t.Errorf("DowngradeReason = %q, want %q", tx.DowngradeReason(), MTPDowngradeNumericalNaN)
		}
		assertNoForeignFallback(t, tx)
	})

	t.Run("numerical_nan_in_step", func(t *testing.T) {
		tx := NewMTPTransaction(cfg)
		defer tx.Close()

		_, _ = tx.BeginRound()
		_ = tx.Propose([]int{1, 2}, [][]float32{{1, 2, 3, 4}})
		err := tx.AppendStep([][]float32{{float32(math.NaN())}}, nil, nil, []float32{1, 2, 3, 4})
		if err == nil {
			t.Fatal("expected error on NaN step")
		}
		if tx.DowngradeReason() != MTPDowngradeNumericalNaN {
			t.Errorf("DowngradeReason = %q, want %q", tx.DowngradeReason(), MTPDowngradeNumericalNaN)
		}
		assertNoForeignFallback(t, tx)
	})

	t.Run("numerical_nan_in_verify", func(t *testing.T) {
		tx := NewMTPTransaction(cfg)
		defer tx.Close()

		_, _ = tx.BeginRound()
		_ = tx.Propose([]int{1, 2}, [][]float32{{1, 2, 3, 4}})
		nanLogits := make([][]float32, 2)
		nanLogits[0] = make([]float32, 50)
		nanLogits[1] = make([]float32, 50)
		nanLogits[0][5] = float32(math.NaN())

		_, err := tx.Verify(nanLogits)
		if err == nil {
			t.Fatal("expected error on NaN in verify logits")
		}
		if tx.DowngradeReason() != MTPDowngradeNumericalNaN {
			t.Errorf("DowngradeReason = %q, want %q", tx.DowngradeReason(), MTPDowngradeNumericalNaN)
		}
		assertNoForeignFallback(t, tx)
	})

	t.Run("memory_pressure_critical", func(t *testing.T) {
		critCfg := cfg
		critCfg.MemoryPressure = Qwen38MTPPressureCritical
		tx := NewMTPTransaction(critCfg)
		defer tx.Close()

		_, err := tx.BeginRound()
		if err == nil {
			t.Fatal("expected error on critical memory pressure")
		}
		if tx.DowngradeReason() != MTPDowngradeMemoryPressure {
			t.Errorf("DowngradeReason = %q, want %q", tx.DowngradeReason(), MTPDowngradeMemoryPressure)
		}
		assertNoForeignFallback(t, tx)
	})

	t.Run("metal_step_panic_recovery", func(t *testing.T) {
		tx := NewMTPTransaction(cfg)
		defer tx.Close()

		_, _ = tx.BeginRound()
		// Panic inside metal step closure should not crash the process.
		_, err := tx.ExecuteDraftStep(func(pos int, priorHidden []float32) ([][]float32, [][]float32, [][]float32, []float32, []float32, int, error) {
			panic("metal device memory fault simulator")
		})
		if err == nil {
			t.Fatal("expected error on panic in step")
		}
		if !tx.IsDowngraded() {
			t.Fatal("expected transaction to downgrade cleanly after panic")
		}
		if tx.DowngradeReason() != MTPDowngradeExecutionFailed {
			t.Errorf("DowngradeReason = %q, want %q", tx.DowngradeReason(), MTPDowngradeExecutionFailed)
		}
		assertNoForeignFallback(t, tx)
	})
}

func TestMTPTransaction_BoundedMemory(t *testing.T) {
	cfg := MTPTransactionConfig{
		HiddenSize:    8,
		VocabSize:     100,
		MaxDraftDepth: 4,
		Backend:       Qwen38MTPBackendMetal,
	}

	initState := &MTPState{
		Position:  0,
		KV:        [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}},
		Recurrent: [][]float32{{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}},
		Conv:      [][]float32{{0.01, 0.02}},
		Hidden:    [][]float32{{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}},
	}

	tx := NewMTPTransactionWithState(initState, cfg)
	defer tx.Close()

	hidden := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}

	// Execute 100 speculative rounds with alternating outcomes:
	// - Round % 3 == 0: full accept
	// - Round % 3 == 1: partial accept
	// - Round % 3 == 2: total rejection (rollback)
	for round := 0; round < 100; round++ {
		_, err := tx.BeginRound()
		if err != nil {
			t.Fatalf("round %d BeginRound error = %v", round, err)
		}

		draft := []int{10, 20, 30}
		if err := tx.Propose(draft, hidden); err != nil {
			t.Fatalf("round %d Propose error = %v", round, err)
		}

		for i := 0; i < 3; i++ {
			stepKV := [][]float32{{1, 2, 3, 4}}
			stepRec := [][]float32{{0.1, 0.2}}
			stepConv := [][]float32{{0.01}}
			stepHidden := []float32{1, 2, 3, 4, 5, 6, 7, 8}
			if err := tx.AppendStep(stepKV, stepRec, stepConv, stepHidden); err != nil {
				t.Fatalf("round %d AppendStep error = %v", round, err)
			}
		}

		switch round % 3 {
		case 0:
			// Full accept
			if err := tx.Commit(3); err != nil {
				t.Fatalf("round %d Commit(3) error = %v", round, err)
			}
		case 1:
			// Partial accept
			if err := tx.Commit(1); err != nil {
				t.Fatalf("round %d Commit(1) error = %v", round, err)
			}
		case 2:
			// Total rejection
			if err := tx.Commit(0); err != nil {
				t.Fatalf("round %d Commit(0) error = %v", round, err)
			}
		}

		// Ensure step states and checkpoint are freed after every round.
		acc := tx.Accounting()
		if tx.checkpoint != nil {
			t.Fatalf("round %d: checkpoint leaked after round completion", round)
		}
		if len(tx.stepStates) != 0 {
			t.Fatalf("round %d: stepStates leaked after round completion", round)
		}
		// Memory growth must stay proportional only to accepted tokens, never accumulating rejected suffix leaks.
		expectedStateBytes := tx.State().ByteSize()
		if acc.CurrentMemoryBytes != expectedStateBytes {
			t.Fatalf("round %d: CurrentMemoryBytes (%d) != expectedStateBytes (%d)", round, acc.CurrentMemoryBytes, expectedStateBytes)
		}
	}

	acc := tx.Accounting()
	if acc.AcceptedCount == 0 || acc.RejectedCount == 0 || acc.RollbackCount == 0 {
		t.Fatalf("expected non-zero accepted/rejected/rollback counts, got: %+v", acc)
	}
}

func TestMTPTransaction_AccountingAndReceipt(t *testing.T) {
	cfg := MTPTransactionConfig{
		HiddenSize:    4,
		VocabSize:     50,
		MaxDraftDepth: 3,
		Backend:       Qwen38MTPBackendMetal,
	}

	initState := &MTPState{
		Position:  0,
		KV:        [][]float32{{1, 2, 3, 4}},
		Recurrent: [][]float32{{0.1, 0.2}},
		Conv:      [][]float32{{0.01}},
		Hidden:    [][]float32{{1, 2, 3, 4}},
	}

	tx := NewMTPTransactionWithState(initState, cfg)
	defer tx.Close()

	_, _ = tx.BeginRound()
	_ = tx.Propose([]int{5, 10}, [][]float32{{1, 2, 3, 4}})
	_ = tx.AppendStep([][]float32{{1}}, [][]float32{{2}}, [][]float32{{3}}, []float32{1, 2, 3, 4})
	_ = tx.AppendStep([][]float32{{4}}, [][]float32{{5}}, [][]float32{{6}}, []float32{1, 2, 3, 4})

	accepted, _ := tx.VerifyTokens([]int{5, 10})
	_ = tx.Commit(accepted)

	acc := tx.Accounting()
	if acc.AcceptedCount != 2 {
		t.Errorf("AcceptedCount = %d, want 2", acc.AcceptedCount)
	}
	if acc.DraftTimeNS == 0 && acc.VerifyTimeNS == 0 {
		// Time can be zero on very fast clocks, but TotalOverheadNS should be tracked
	}

	receipt := tx.Receipt()
	if err := receipt.Validate(); err != nil {
		t.Fatalf("Receipt.Validate() failed on successful transaction: %v", err)
	}
	if receipt.Outcome != Qwen38MTPOutcomeSucceeded {
		t.Errorf("receipt.Outcome = %q, want %q", receipt.Outcome, Qwen38MTPOutcomeSucceeded)
	}
	if receipt.Engine != Qwen38EngineMTP {
		t.Errorf("receipt.Engine = %q, want %q", receipt.Engine, Qwen38EngineMTP)
	}

	// Downgraded transaction receipt
	txDown := NewMTPTransaction(cfg)
	defer txDown.Close()
	_ = txDown.Downgrade(MTPDowngradeMemoryPressure, "headroom depleted")
	receiptDown := txDown.Receipt()
	if err := receiptDown.Validate(); err != nil {
		t.Fatalf("Receipt.Validate() failed on downgraded transaction: %v", err)
	}
	if receiptDown.Outcome != Qwen38MTPOutcomeTargetOnly {
		t.Errorf("receiptDown.Outcome = %q, want %q", receiptDown.Outcome, Qwen38MTPOutcomeTargetOnly)
	}
	if receiptDown.Engine != Qwen38EngineTargetDecode {
		t.Errorf("receiptDown.Engine = %q, want %q", receiptDown.Engine, Qwen38EngineTargetDecode)
	}
}

func TestMTPTransaction_SessionPrefixSnapshotRollback(t *testing.T) {
	cfg := cfgV(8, 2, 2, 1, 4, 16)
	m := NewSynthetic(cfg)
	session := m.NewSession()
	t.Cleanup(session.Close)

	session.Prefill([]int{1, 2, 3})
	baseLen := session.Cache.Len()

	txCfg := MTPTransactionConfig{
		HiddenSize:    cfg.HiddenSize,
		VocabSize:     cfg.VocabSize,
		MaxDraftDepth: 2,
		Backend:       Qwen38MTPBackendMetal,
	}

	tx, err := NewMTPTransactionWithTarget(session, txCfg)
	if err != nil {
		t.Fatalf("NewMTPTransactionWithTarget error = %v", err)
	}
	defer tx.Close()

	if _, err := tx.BeginRound(); err != nil {
		t.Fatalf("BeginRound error = %v", err)
	}

	// Mutate session cache
	session.Step(4)
	if session.Cache.Len() != baseLen+1 {
		t.Fatalf("session Cache.Len = %d, want %d", session.Cache.Len(), baseLen+1)
	}

	// Rollback
	if err := tx.Rollback(); err != nil {
		t.Fatalf("Rollback error = %v", err)
	}

	if session.Cache.Len() != baseLen {
		t.Errorf("after Rollback session Cache.Len = %d, want %d", session.Cache.Len(), baseLen)
	}
}

func assertNoForeignFallback(t *testing.T, tx *MTPTransaction) {
	t.Helper()
	eng := string(tx.Engine())
	if strings.Contains(strings.ToLower(eng), "llama") {
		t.Fatalf("foreign runtime llama.cpp detected in engine: %q", eng)
	}
	if tx.downgradeErr != nil {
		detail := strings.ToLower(tx.downgradeErr.Detail)
		if strings.Contains(detail, "llama") {
			t.Fatalf("foreign runtime llama.cpp detected in downgrade detail: %q", detail)
		}
	}
	if !errors.Is(tx.downgradeErr, ErrTargetVerificationDowngrade) {
		t.Errorf("downgrade error does not unwrap to ErrTargetVerificationDowngrade")
	}
}
