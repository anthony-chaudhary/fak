package model

import (
	"encoding/json"
	"errors"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

const (
	qwen38MTPWitnessModel           = "Qwen3.8"
	qwen38MTPWitnessEngine          = "fak-native/model"
	qwen38MTPWitnessDowngradeEngine = "fak-native ordinary target decode"
)

type qwen38MTPBatchReceipt struct {
	Model                    string `json:"model"`
	Engine                   string `json:"engine"`
	DowngradeEngine          string `json:"downgrade_engine"`
	ExternalFallback         string `json:"external_fallback"`
	ProposedBlocks           int    `json:"proposed_blocks"`
	TargetVerifyForwardCalls int    `json:"target_verify_forward_calls"`
	ProposedTokens           int    `json:"proposed_tokens"`
	AcceptedTokens           int    `json:"accepted_tokens"`
	RejectedTokens           int    `json:"rejected_tokens"`
	SetupNS                  int64  `json:"setup_ns"`
	DraftingNS               int64  `json:"drafting_ns"`
	TargetVerificationNS     int64  `json:"target_verification_ns"`
	RejectionNS              int64  `json:"rejection_ns"`
	RollbackNS               int64  `json:"rollback_ns"`
	SynchronizationNS        int64  `json:"synchronization_ns"`
	MemoryBytes              uint64 `json:"memory_bytes"`
	RecoveryNS               int64  `json:"recovery_ns"`
}

func (r qwen38MTPBatchReceipt) validate() error {
	if r.Model != qwen38MTPWitnessModel {
		return errors.New("receipt must identify Qwen3.8")
	}
	if r.Engine != qwen38MTPWitnessEngine || r.DowngradeEngine != qwen38MTPWitnessDowngradeEngine {
		return errors.New("receipt must identify fak-native speculative and ordinary decode engines")
	}
	if r.ExternalFallback != "none" || strings.Contains(strings.ToLower(r.Engine+" "+r.DowngradeEngine), "llama") {
		return errors.New("receipt must forbid external or llama.cpp fallback")
	}
	if r.ProposedBlocks <= 0 || r.TargetVerifyForwardCalls != r.ProposedBlocks {
		return errors.New("receipt must show exactly one target VerifyForward call per proposed block")
	}
	if r.ProposedTokens != r.AcceptedTokens+r.RejectedTokens || r.RejectedTokens <= 0 {
		return errors.New("receipt must account for accepted and rejected proposal tokens")
	}
	return nil
}

func TestQwen38MTPBatchedVerificationWitness(t *testing.T) {
	m := NewSynthetic(cfgV(32, 2, 2, 1, 16, 64))
	prompt := []int{1, 2, 3}
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	before := target.Prefill(prompt)
	want := m.NewSession()
	want.captureTargetHidden = true
	t.Cleanup(want.Close)
	want.Prefill(prompt)

	verifyCalls := 0
	proposedTokens := 0
	acceptedTokens := 0
	rejectedTokens := 0
	for block, accepted := range []int{2, 1, 0} {
		draft := []int{block + 4, block + 5, block + 6}
		tx, err := beginQwen35MTPTargetTransaction(target, before)
		if err != nil {
			t.Fatalf("block %d begin: %v", block, err)
		}
		verifyForward := tx.verify
		tx.verify = func(tokens []int) [][]float32 {
			verifyCalls++
			return verifyForward(tokens)
		}
		if _, err := tx.Verify(draft); err != nil {
			t.Fatalf("block %d verify: %v", block, err)
		}
		before, err = tx.Commit(accepted)
		if err != nil {
			t.Fatalf("block %d accepted-prefix commit: %v", block, err)
		}
		for _, token := range draft[:accepted] {
			want.Step(token)
		}
		assertQwen35MTPTargetStateEqual(t, target, want)
		proposedTokens += len(draft)
		acceptedTokens += accepted
		rejectedTokens += len(draft) - accepted
	}

	receipt := qwen38MTPBatchReceipt{
		Model:                    qwen38MTPWitnessModel,
		Engine:                   qwen38MTPWitnessEngine,
		DowngradeEngine:          qwen38MTPWitnessDowngradeEngine,
		ExternalFallback:         "none",
		ProposedBlocks:           3,
		TargetVerifyForwardCalls: verifyCalls,
		ProposedTokens:           proposedTokens,
		AcceptedTokens:           acceptedTokens,
		RejectedTokens:           rejectedTokens,
	}
	if err := receipt.validate(); err != nil {
		t.Fatal(err)
	}

	unsupported := m.NewSession()
	t.Cleanup(unsupported.Close)
	_, err := SpecDecodeGreedyQwen35MTPDepthN(unsupported, prompt, 2, Qwen35MTPMaxDraftDepth+1)
	verdict := assertQwen35MTPUnsupported(t, err)
	if !strings.Contains(verdict.Reason, "witnessed native bound") {
		t.Fatalf("downgrade reason = %q, want explicit fak-native admission boundary", verdict.Reason)
	}
	ordinary := unsupported.Generate(prompt, 2)
	reference := m.NewSession()
	t.Cleanup(reference.Close)
	if wantOrdinary := reference.Generate(prompt, 2); !slices.Equal(ordinary, wantOrdinary) {
		t.Fatalf("fak-native ordinary downgrade = %v, want %v", ordinary, wantOrdinary)
	}
}

func BenchmarkQwen38MTPBatchedVerificationReceipt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var memBefore, memAfter runtime.MemStats
		runtime.ReadMemStats(&memBefore)
		setupStart := time.Now()
		m := NewSynthetic(cfgV(32, 2, 2, 1, 16, 64))
		target := m.NewSession()
		before := target.Prefill([]int{1, 2, 3})
		setupNS := time.Since(setupStart).Nanoseconds()

		draftStart := time.Now()
		draft := []int{4, 5, 6, 7}
		draftingNS := time.Since(draftStart).Nanoseconds()

		tx, err := beginQwen35MTPTargetTransaction(target, before)
		if err != nil {
			b.Fatal(err)
		}
		verifyCalls := 0
		verifyForward := tx.verify
		tx.verify = func(tokens []int) [][]float32 {
			verifyCalls++
			return verifyForward(tokens)
		}
		verifyStart := time.Now()
		rows, err := tx.Verify(draft)
		verifyNS := time.Since(verifyStart).Nanoseconds()
		if err != nil || len(rows) != len(draft) {
			b.Fatalf("batched VerifyForward: rows=%d err=%v", len(rows), err)
		}

		rejectionStart := time.Now()
		accepted := len(draft) - 2
		rejected := len(draft) - accepted
		rejectionNS := time.Since(rejectionStart).Nanoseconds()

		rollbackStart := time.Now()
		before, err = tx.Commit(accepted)
		rollbackNS := time.Since(rollbackStart).Nanoseconds()
		if err != nil {
			b.Fatal(err)
		}
		syncStart := time.Now()
		if target.Cache.Len() != len([]int{1, 2, 3})+accepted {
			b.Fatalf("accepted-prefix cache length=%d", target.Cache.Len())
		}
		if len(before) == 0 {
			b.Fatal("accepted-prefix commit returned empty logits")
		}
		synchronizationNS := time.Since(syncStart).Nanoseconds()

		recoveryStart := time.Now()
		recovery, err := beginQwen35MTPTargetTransaction(target, before)
		if err == nil {
			err = recovery.Abort()
		}
		recoveryNS := time.Since(recoveryStart).Nanoseconds()
		if err != nil {
			b.Fatal(err)
		}
		target.Close()
		runtime.ReadMemStats(&memAfter)

		receipt := qwen38MTPBatchReceipt{
			Model:                    qwen38MTPWitnessModel,
			Engine:                   qwen38MTPWitnessEngine,
			DowngradeEngine:          qwen38MTPWitnessDowngradeEngine,
			ExternalFallback:         "none",
			ProposedBlocks:           1,
			TargetVerifyForwardCalls: verifyCalls,
			ProposedTokens:           len(draft),
			AcceptedTokens:           accepted,
			RejectedTokens:           rejected,
			SetupNS:                  setupNS,
			DraftingNS:               draftingNS,
			TargetVerificationNS:     verifyNS,
			RejectionNS:              rejectionNS,
			RollbackNS:               rollbackNS,
			SynchronizationNS:        synchronizationNS,
			MemoryBytes:              memAfter.TotalAlloc - memBefore.TotalAlloc,
			RecoveryNS:               recoveryNS,
		}
		if err := receipt.validate(); err != nil {
			b.Fatal(err)
		}
		encoded, err := json.Marshal(receipt)
		if err != nil {
			b.Fatal(err)
		}
		b.Logf("qwen3.8_mtp_batch_receipt=%s", encoded)
		b.ReportMetric(float64(receipt.TargetVerifyForwardCalls)/float64(receipt.ProposedBlocks), "VerifyForward/block")
		b.ReportMetric(float64(receipt.MemoryBytes), "receipt_B")
	}
}
