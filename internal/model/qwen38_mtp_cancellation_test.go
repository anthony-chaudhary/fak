package model

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"
)

func TestQwen38MTP_CancelDuringDrafting(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)

	tx, err := NewMTPTransactionWithTarget(target, MTPTransactionConfig{
		HiddenSize:    32,
		VocabSize:     97,
		MaxDraftDepth: 3,
		Backend:       Qwen38MTPBackendMetal,
	})
	if err != nil {
		t.Fatalf("failed to create transaction: %v", err)
	}

	mgr := NewQwen38MTPCancellationManager(Qwen38MTPBoundsConfig{
		MaxMemoryBytes: 10 * 1024 * 1024,
	})

	// Pre-round checkpoint baseline state
	cp, err := tx.BeginRound()
	if err != nil {
		t.Fatalf("BeginRound failed: %v", err)
	}
	baselinePos := cp.position

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel context immediately to simulate in-flight abort
	cancel()

	_, stepErr := mgr.ExecuteDraftStepWithContext(ctx, tx, func(ctx context.Context, pos int, priorHidden []float32) (kv, recurrent, conv [][]float32, hidden []float32, logits []float32, token int, err error) {
		return nil, nil, nil, nil, nil, 10, nil
	})

	if stepErr == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
	if !errors.Is(stepErr, ErrMTPContextCanceled) {
		t.Fatalf("expected ErrMTPContextCanceled, got %v", stepErr)
	}

	// Invariant 1: State rolled back to baseline
	if tx.State().Position != baselinePos {
		t.Fatalf("position mutated: got %d, want baseline %d", tx.State().Position, baselinePos)
	}

	// Invariant 2: Transaction locks released (must not deadlock)
	lockAcquired := make(chan struct{})
	go func() {
		_ = tx.Accounting()
		_ = tx.DowngradeReason()
		close(lockAcquired)
	}()

	select {
	case <-lockAcquired:
		// success: locks were cleanly released
	case <-time.After(2 * time.Second):
		t.Fatal("deadlock: transaction locks were not released after cancellation")
	}

	// Invariant 3: Marks downgrade reason context_canceled
	if mgr.LastDowngradeReason() != string(MTPDowngradeContextCanceled) {
		t.Fatalf("manager downgrade reason = %q, want %q", mgr.LastDowngradeReason(), MTPDowngradeContextCanceled)
	}
	if tx.DowngradeReason() != MTPDowngradeContextCanceled {
		t.Fatalf("tx downgrade reason = %q, want %q", tx.DowngradeReason(), MTPDowngradeContextCanceled)
	}

	// Invariant 4: Receipt records valid cancellation evidence
	rcpt := mgr.Receipt(tx)
	if err := rcpt.Validate(); err != nil {
		t.Fatalf("cancellation receipt validation failed: %v", err)
	}
	if !rcpt.DraftAborted || !rcpt.StateRolledBack || !rcpt.LocksReleased {
		t.Fatalf("incomplete receipt flags: %+v", rcpt)
	}
}

func TestQwen38MTP_CancelDuringVerification(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)

	tx, err := NewMTPTransactionWithTarget(target, MTPTransactionConfig{
		HiddenSize:    32,
		VocabSize:     97,
		MaxDraftDepth: 3,
		Backend:       Qwen38MTPBackendMetal,
	})
	if err != nil {
		t.Fatalf("failed to create transaction: %v", err)
	}

	mgr := NewQwen38MTPCancellationManager(Qwen38MTPBoundsConfig{
		MaxMemoryBytes: 10 * 1024 * 1024,
	})

	cp, err := tx.BeginRound()
	if err != nil {
		t.Fatalf("BeginRound failed: %v", err)
	}
	baselinePos := cp.position

	// Drafting succeeds under active context
	activeCtx := context.Background()
	dummyHidden := make([][]float32, 1)
	dummyHidden[0] = make([]float32, 32)
	if pErr := tx.Propose([]int{5, 6}, dummyHidden); pErr != nil {
		t.Fatalf("Propose failed: %v", pErr)
	}

	// Context is cancelled before verification phase
	verifyCtx, cancel := context.WithCancel(activeCtx)
	cancel()

	dummyLogits := make([][]float32, 2)
	dummyLogits[0] = make([]float32, 97)
	dummyLogits[1] = make([]float32, 97)

	_, vErr := mgr.VerifyWithContext(verifyCtx, tx, dummyLogits)
	if vErr == nil {
		t.Fatal("expected verification cancellation error, got nil")
	}
	if !errors.Is(vErr, ErrMTPContextCanceled) {
		t.Fatalf("expected ErrMTPContextCanceled, got %v", vErr)
	}

	// State rolled back cleanly
	if tx.State().Position != baselinePos {
		t.Fatalf("state position = %d, want baseline %d", tx.State().Position, baselinePos)
	}

	// Locks released cleanly
	done := make(chan bool, 1)
	go func() {
		_ = tx.Accounting()
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(1 * time.Second):
		t.Fatal("deadlock after verification cancellation")
	}

	if mgr.LastDowngradeReason() != string(MTPDowngradeContextCanceled) {
		t.Fatalf("downgrade reason = %q, want %q", mgr.LastDowngradeReason(), MTPDowngradeContextCanceled)
	}
}

func TestQwen38MTP_CancelInjectedMemoryPressure(t *testing.T) {
	mgr := NewQwen38MTPCancellationManager(Qwen38MTPBoundsConfig{
		MaxMemoryBytes:     1024 * 1024,
		InjectedAllocLimit: 4096, // 4 KB limit for testing
		FailClosedOnOOM:    true,
	})

	// Normal allocation below limit
	if err := mgr.TrackAllocation(2048); err != nil {
		t.Fatalf("TrackAllocation(2048) unexpected error: %v", err)
	}
	if mgr.CurrentAlloc() != 2048 {
		t.Fatalf("CurrentAlloc = %d, want 2048", mgr.CurrentAlloc())
	}

	// Allocation that exceeds injected limit: must fail-closed without panicking
	allocErr := mgr.TrackAllocation(3000) // 2048 + 3000 = 5048 > 4096
	if allocErr == nil {
		t.Fatal("expected memory limit error, got nil")
	}

	var dgErr *MTPDowngradeError
	if !errors.As(allocErr, &dgErr) {
		t.Fatalf("expected *MTPDowngradeError, got %T: %v", allocErr, allocErr)
	}
	if dgErr.Reason != MTPDowngradeMemoryPressure {
		t.Fatalf("downgrade reason = %q, want %q", dgErr.Reason, MTPDowngradeMemoryPressure)
	}

	// Maps to Qwen38MTPMemoryUnsafe
	if dgErr.Reason.ToQwen38DowngradeReason() != Qwen38MTPMemoryUnsafe {
		t.Fatalf("mapped reason = %q, want %q", dgErr.Reason.ToQwen38DowngradeReason(), Qwen38MTPMemoryUnsafe)
	}

	if !mgr.IsDowngraded() {
		t.Fatal("expected manager to be marked as downgraded")
	}
	if mgr.LastDowngradeReason() != string(MTPDowngradeMemoryPressure) {
		t.Fatalf("manager last downgrade reason = %q, want %q", mgr.LastDowngradeReason(), MTPDowngradeMemoryPressure)
	}
}

func TestQwen38MTP_CancelSessionReusabilityCycles(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)

	sess, err := NewQwen38MTPPersistentSession(target)
	if err != nil {
		t.Fatalf("NewQwen38MTPPersistentSession failed: %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })

	mgr := NewQwen38MTPCancellationManager(Qwen38MTPBoundsConfig{
		MaxMemoryBytes: 50 * 1024 * 1024,
	})

	prompt := []int{2, 4, 6, 8}

	// --- Cycle 1: Cancellation during generation ---
	ctx1, cancel1 := context.WithCancel(context.Background())
	cancel1() // pre-cancel
	_, cErr1 := mgr.GenerateWithContext(ctx1, sess, prompt, 4)
	if cErr1 == nil {
		t.Fatal("cycle 1: expected context cancellation error")
	}

	// Invariant: Session must not be permanently wedged or cancelled
	if sess.IsCancelled() {
		t.Fatal("cycle 1: session left in cancelled state after cancellation cycle")
	}

	// --- Cycle 2: Normal generation immediately succeeds ---
	ctx2 := context.Background()
	tokens2, err2 := mgr.GenerateWithContext(ctx2, sess, prompt, 4)
	if err2 != nil {
		t.Fatalf("cycle 2: generation failed after cancellation: %v", err2)
	}
	if len(tokens2) != 4 {
		t.Fatalf("cycle 2: generated tokens len=%d, want 4", len(tokens2))
	}

	// --- Cycle 3: Injected memory limit triggers clean downgrade ---
	mgr.SetInjectedAllocLimit(100) // very small limit
	ctx3 := context.Background()
	_, _, err3 := mgr.ExecuteSessionRoundWithContext(ctx3, sess)
	if err3 == nil {
		t.Fatal("cycle 3: expected memory pressure error under injected limit")
	}
	if mgr.LastDowngradeReason() != string(MTPDowngradeMemoryPressure) {
		t.Fatalf("cycle 3: downgrade reason = %q, want %q", mgr.LastDowngradeReason(), MTPDowngradeMemoryPressure)
	}

	// Reset manager limits for clean recovery
	mgr.Reset()
	mgr.SetInjectedAllocLimit(0)

	// --- Cycle 4: Immediate reusability and parity check ---
	// Reset session conversation state
	if rErr := sess.Reset(); rErr != nil {
		t.Fatalf("cycle 4: session reset failed: %v", rErr)
	}

	ctx4 := context.Background()
	tokens4, err4 := mgr.GenerateWithContext(ctx4, sess, prompt, 4)
	if err4 != nil {
		t.Fatalf("cycle 4: generation failed: %v", err4)
	}
	if len(tokens4) != 4 {
		t.Fatalf("cycle 4: generated tokens len=%d, want 4", len(tokens4))
	}

	// Compare tokens4 with an independent reference target decode to prove zero corruption
	refTarget := m.NewSession()
	t.Cleanup(refTarget.Close)
	refLogits := refTarget.Prefill(prompt)
	var refTokens []int
	for i := 0; i < 4; i++ {
		next := argmaxF32(refLogits)
		refTokens = append(refTokens, next)
		refLogits = refTarget.Step(next)
	}

	if !reflect.DeepEqual(tokens4, refTokens) {
		t.Fatalf("cycle 4: token corruption detected! generated=%v, ref=%v", tokens4, refTokens)
	}
}

func TestQwen38MTP_CancelConcurrentAccessSafety(t *testing.T) {
	mgr := NewQwen38MTPCancellationManager(Qwen38MTPBoundsConfig{
		MaxMemoryBytes: 10 * 1024 * 1024,
	})

	var wg sync.WaitGroup
	workers := 16

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				_ = mgr.TrackAllocation(int64(j * 10))
				_ = mgr.CurrentAlloc()
				_ = mgr.PeakAlloc()
				mgr.ReleaseAllocation(int64(j * 5))
				if j%10 == 0 {
					_ = mgr.CheckContext(context.Background())
				}
			}
		}(i)
	}

	wg.Wait()
	if mgr.CurrentAlloc() < 0 {
		t.Fatalf("negative alloc bytes: %d", mgr.CurrentAlloc())
	}
}
