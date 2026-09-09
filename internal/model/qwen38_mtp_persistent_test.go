package model

import (
	"errors"
	"reflect"
	"testing"
)

func TestQwen38MTPPersistentPromptReplacementMatchesFresh(t *testing.T) {
	for _, tc := range []struct {
		name   string
		prompt []int
	}{
		{"divergent", []int{0, 1, 7}},
		{"strict-prefix", []int{0, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := qwen38HybridMTPEnabledSyntheticModel(t)
			target, fresh := m.NewSession(), m.NewSession()
			t.Cleanup(target.Close)
			t.Cleanup(fresh.Close)
			fresh.captureTargetHidden = true
			cfg := DefaultQwen38MTPPersistentConfig()
			cfg.PromptCache = NewMTPPromptCache()
			sess, err := NewQwen38MTPPersistentSession(target, cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = sess.Close() })
			if _, err := sess.Prefill([]int{0, 1, 2, 3}); err != nil {
				t.Fatal(err)
			}

			// The oracle never sees the discarded tokens or the persistent wrapper.
			// Cache equality includes K/Kraw/V, positions, lineage, recurrence and
			// convolution windows; hidden rows and their token history must match too.
			check := func(prompt []int, wantLogits []float32) {
				t.Helper()
				got, err := sess.Prefill(prompt)
				if err != nil {
					t.Fatal(err)
				}
				assertQwen35MTPTargetStateEqual(t, target, fresh)
				assertFloat32BitsEqual(t, "replacement logits", got, wantLogits)
				assertFloat32BitsEqual(t, "stored logits", sess.lastLogits, wantLogits)
				if !reflect.DeepEqual(sess.CommittedTokens(), prompt) {
					t.Fatalf("committed tokens = %v, want %v", sess.CommittedTokens(), prompt)
				}
			}
			check(tc.prompt, fresh.Prefill(tc.prompt))
			check(append(append([]int(nil), tc.prompt...), 6), fresh.Step(6))
		})
	}
}

func TestQwen38MTP_Persistent_MultiTurnGreedyParityAndContextExtension(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)

	// Session A: MTP Accelerated Persistent Session
	tMTP := m.NewSession()
	tMTP.captureTargetHidden = true
	t.Cleanup(tMTP.Close)
	mtpSess, err := NewQwen38MTPPersistentSession(tMTP)
	if err != nil {
		t.Fatalf("failed to create persistent MTP session: %v", err)
	}
	t.Cleanup(func() { _ = mtpSess.Close() })

	if !mtpSess.Eligibility().Eligible {
		t.Fatalf("expected MTP eligible session, got reason %q", mtpSess.Eligibility().DowngradeReason)
	}

	// Session B: Target-Only Persistent Session (Reference)
	tTarget := m.NewSession()
	tTarget.captureTargetHidden = true
	t.Cleanup(tTarget.Close)
	cfgTargetOnly := DefaultQwen38MTPPersistentConfig()
	cfgTargetOnly.DisableMTP = true
	targetSess, err := NewQwen38MTPPersistentSession(tTarget, cfgTargetOnly)
	if err != nil {
		t.Fatalf("failed to create persistent target-only session: %v", err)
	}
	t.Cleanup(func() { _ = targetSess.Close() })

	// --- Turn 1 ---
	turn1Prompt := []int{0, 1, 2, 3}
	genLen := 5

	genMTP1, err := mtpSess.Generate(turn1Prompt, genLen)
	if err != nil {
		t.Fatalf("MTP turn 1 generate failed: %v", err)
	}

	genTarget1, err := targetSess.Generate(turn1Prompt, genLen)
	if err != nil {
		t.Fatalf("Target turn 1 generate failed: %v", err)
	}

	if !reflect.DeepEqual(genMTP1, genTarget1) {
		t.Fatalf("turn 1 greedy parity mismatch: MTP=%v, Target=%v", genMTP1, genTarget1)
	}
	if mtpSess.TurnCount() != 1 || targetSess.TurnCount() != 1 {
		t.Fatalf("turn count mismatch: MTP=%d, Target=%d, want 1", mtpSess.TurnCount(), targetSess.TurnCount())
	}

	// --- Turn 2 (Context Extension / Continuation) ---
	// Conversation continues: previous prompt + turn 1 response + user reply
	turn2Prompt := append(append([]int(nil), turn1Prompt...), genMTP1...)
	turn2Prompt = append(turn2Prompt, 7, 8)

	genMTP2, err := mtpSess.Generate(turn2Prompt, genLen)
	if err != nil {
		t.Fatalf("MTP turn 2 generate failed: %v", err)
	}

	genTarget2, err := targetSess.Generate(turn2Prompt, genLen)
	if err != nil {
		t.Fatalf("Target turn 2 generate failed: %v", err)
	}

	if !reflect.DeepEqual(genMTP2, genTarget2) {
		t.Fatalf("turn 2 greedy parity mismatch: MTP=%v, Target=%v", genMTP2, genTarget2)
	}
	if mtpSess.TurnCount() != 2 || targetSess.TurnCount() != 2 {
		t.Fatalf("turn count mismatch: MTP=%d, Target=%d, want 2", mtpSess.TurnCount(), targetSess.TurnCount())
	}

	// Assert exact cache prefix retention: cache length must match total conversation tokens
	wantTotalTokens := len(turn2Prompt) + genLen
	if mtpSess.Target().Cache.Len() != wantTotalTokens {
		t.Fatalf("MTP session cache len=%d, want %d", mtpSess.Target().Cache.Len(), wantTotalTokens)
	}
	if targetSess.Target().Cache.Len() != wantTotalTokens {
		t.Fatalf("Target session cache len=%d, want %d", targetSess.Target().Cache.Len(), wantTotalTokens)
	}
}

func TestQwen38MTP_Persistent_PartialAcceptRollbackAndZeroStateLeaks(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)

	sess, err := NewQwen38MTPPersistentSession(target)
	if err != nil {
		t.Fatalf("failed to create persistent session: %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })

	prompt := []int{0, 1}
	if _, err := sess.Prefill(prompt); err != nil {
		t.Fatalf("prefill failed: %v", err)
	}

	initLen := sess.Target().Cache.Len()
	if initLen != len(prompt) {
		t.Fatalf("initial cache len=%d, want %d", initLen, len(prompt))
	}

	// Execute a step round
	emitted, receipt, err := sess.StepRound()
	if err != nil {
		t.Fatalf("StepRound failed: %v", err)
	}

	if len(emitted) == 0 {
		t.Fatal("expected at least 1 emitted token")
	}

	// The target cache length must have grown by EXACTLY len(emitted).
	// No speculative tokens beyond emitted may leak into the cache!
	expectedLen := initLen + len(emitted)
	actualLen := sess.Target().Cache.Len()
	if actualLen != expectedLen {
		t.Fatalf("state leak detected: cache len=%d, want %d (emitted=%d)", actualLen, expectedLen, len(emitted))
	}

	// Target hidden history must also strictly match cache length.
	sess.Target().targetHiddenMu.RLock()
	hiddenLen := len(sess.Target().targetHiddenTokens)
	sess.Target().targetHiddenMu.RUnlock()

	if hiddenLen != expectedLen {
		t.Fatalf("hidden history state leak: hiddenTokens=%d, want %d", hiddenLen, expectedLen)
	}

	if receipt != nil {
		if receipt.AcceptedCount > 0 && len(receipt.AcceptedTokens) != receipt.AcceptedCount {
			t.Fatalf("receipt accepted mismatch: count=%d, tokens=%v", receipt.AcceptedCount, receipt.AcceptedTokens)
		}
	}
}

func TestQwen38MTP_Persistent_Cancellation(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)

	sess, err := NewQwen38MTPPersistentSession(target)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })

	// Cancel session before prefill
	sess.Cancel()
	if !sess.IsCancelled() {
		t.Fatal("expected session to be cancelled")
	}

	_, err = sess.Prefill([]int{0, 1, 2})
	if !errors.Is(err, ErrQwen38MTPCancelled) {
		t.Fatalf("prefill error = %v, want ErrQwen38MTPCancelled", err)
	}

	_, err = sess.Generate([]int{0, 1}, 3)
	if !errors.Is(err, ErrQwen38MTPCancelled) {
		t.Fatalf("generate error = %v, want ErrQwen38MTPCancelled", err)
	}

	// Reset cancellation
	sess.ResetCancel()
	if sess.IsCancelled() {
		t.Fatal("expected cancellation to be cleared")
	}

	_, err = sess.Prefill([]int{0, 1, 2})
	if err != nil {
		t.Fatalf("prefill failed after reset cancel: %v", err)
	}
}

func TestQwen38MTP_Persistent_Reset(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)

	sess, err := NewQwen38MTPPersistentSession(target)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })

	// Run turn 1
	turn1, err := sess.Generate([]int{0, 1, 2}, 4)
	if err != nil {
		t.Fatalf("turn 1 generate failed: %v", err)
	}
	if len(turn1) != 4 {
		t.Fatalf("turn 1 tokens len=%d, want 4", len(turn1))
	}
	if sess.Target().Cache.Len() == 0 {
		t.Fatal("cache empty after turn 1")
	}

	// Reset session
	if err := sess.Reset(); err != nil {
		t.Fatalf("reset failed: %v", err)
	}

	if len(sess.CommittedTokens()) != 0 {
		t.Fatalf("committed tokens len=%d after reset, want 0", len(sess.CommittedTokens()))
	}
	if sess.Target().Cache.Len() != 0 {
		t.Fatalf("target cache len=%d after reset, want 0", sess.Target().Cache.Len())
	}
	if sess.TurnCount() != 0 {
		t.Fatalf("turn count=%d after reset, want 0", sess.TurnCount())
	}

	// Run fresh turn after reset
	promptNew := []int{1, 2, 3}
	genNew, err := sess.Generate(promptNew, 4)
	if err != nil {
		t.Fatalf("generate after reset failed: %v", err)
	}

	// Compare with reference session
	ref := m.NewSession()
	ref.captureTargetHidden = true
	t.Cleanup(ref.Close)
	cfgRef := DefaultQwen38MTPPersistentConfig()
	cfgRef.DisableMTP = true
	refSess, err := NewQwen38MTPPersistentSession(ref, cfgRef)
	if err != nil {
		t.Fatalf("failed to create ref session: %v", err)
	}
	t.Cleanup(func() { _ = refSess.Close() })

	genRef, err := refSess.Generate(promptNew, 4)
	if err != nil {
		t.Fatalf("ref generate failed: %v", err)
	}

	if !reflect.DeepEqual(genNew, genRef) {
		t.Fatalf("output after reset mismatch: got %v, want %v", genNew, genRef)
	}
}

func TestQwen38MTP_Persistent_PromptCacheCoexistence(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)

	promptCache := NewMTPPromptCache()
	cfg := DefaultQwen38MTPPersistentConfig()
	cfg.PromptCache = promptCache

	sess, err := NewQwen38MTPPersistentSession(target, cfg)
	if err != nil {
		t.Fatalf("failed to create session with prompt cache: %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })

	prompt := []int{0, 1, 2, 3}
	if _, err := sess.Prefill(prompt); err != nil {
		t.Fatalf("first prefill failed: %v", err)
	}

	if promptCache.Len() != 1 {
		t.Fatalf("prompt cache len=%d, want 1", promptCache.Len())
	}

	// Reset and prefill with cache match
	if err := sess.Reset(); err != nil {
		t.Fatalf("reset failed: %v", err)
	}

	if _, err := sess.Prefill(prompt); err != nil {
		t.Fatalf("second prefill with cache hit failed: %v", err)
	}

	if sess.Target().Cache.Len() != len(prompt) {
		t.Fatalf("cache len=%d after cache hit prefill, want %d", sess.Target().Cache.Len(), len(prompt))
	}
}
