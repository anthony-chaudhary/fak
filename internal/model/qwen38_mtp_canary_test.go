package model

import (
	"strings"
	"testing"
	"time"
)

func registerQwen38CanaryTestEvidence(t *testing.T, mgr *Qwen38MTPCanaryManager, receiptID string, envelope Qwen38CanaryEnvelope) {
	t.Helper()
	now := time.Now().UTC()
	err := mgr.RegisterCanaryEvidence(Qwen38MTPCanaryEvidence{
		Receipt: Qwen38MTPCanaryReceipt{
			SchemaVersion:   Qwen38MTPCanaryReceiptSchema,
			ReceiptID:       receiptID,
			DefaultOn:       true,
			Engine:          Qwen38EngineMTP,
			Envelope:        envelope,
			Speedup:         1.1,
			TokensProduced:  2,
			TokensProposed:  1,
			TokensAccepted:  1,
			DowngradeReason: Qwen38MTPEligible,
			CircuitStatus:   CanaryCircuitClosed,
			LatencyNS:       Qwen38MTPLatencyNS{Setup: 1, Draft: 1, Verify: 1, Total: 3},
			MemoryBytes:     Qwen38MTPMemoryBytes{DraftWorkspace: 1, VerifyWorkspace: 1, Peak: 1},
		},
		ObservedAt: now.Add(-time.Hour),
		ValidUntil: now.Add(time.Hour),
	})
	if err != nil {
		t.Fatalf("register canary evidence: %v", err)
	}
}

func TestQwen38MTP_Canary_InsideCertifiedEnvelope(t *testing.T) {
	mgr := NewQwen38MTPCanaryManager()

	hashStr := strings.Repeat("ab", 32)

	req := Qwen38CanaryRequest{
		ModelReady:    true,
		OperatorOptIn: false, // Default-on does not require manual operator flag!
		Envelope: Qwen38CanaryEnvelope{
			ModelFamily:   "Qwen3.8",
			Format:        Qwen38MTPFormatQ4K,
			Backend:       Qwen38MTPBackendMetal,
			HeadroomBytes: 3 * 1024 * 1024 * 1024, // 3GB > 2GB
			ArtifactHash:  hashStr,
			DraftDepth:    3,
		},
	}
	req.EvidenceReceiptID = "q4k-metal-receipt"
	registerQwen38CanaryTestEvidence(t, mgr, req.EvidenceReceiptID, req.Envelope)

	dec := mgr.EvaluateCanary(req)
	if !dec.InsideEnvelope {
		t.Fatalf("expected InsideEnvelope = true, rejection: %s", dec.RejectionReason)
	}
	if !dec.CanaryDefaultOn {
		t.Fatal("expected CanaryDefaultOn = true without operator flag")
	}
	if dec.Engine != Qwen38EngineMTP {
		t.Fatalf("Engine = %q, want %q", dec.Engine, Qwen38EngineMTP)
	}
	if dec.DowngradeReason != Qwen38MTPEligible {
		t.Fatalf("DowngradeReason = %q, want %q", dec.DowngradeReason, Qwen38MTPEligible)
	}

	// Test F32 and cpu-native backend inside envelope
	reqF32 := req
	reqF32.Envelope.Format = Qwen38MTPFormatF32
	reqF32.Envelope.Backend = Qwen38MTPBackendCPU
	reqF32.EvidenceReceiptID = "f32-cpu-receipt"
	registerQwen38CanaryTestEvidence(t, mgr, reqF32.EvidenceReceiptID, reqF32.Envelope)
	decF32 := mgr.EvaluateCanary(reqF32)
	if !decF32.CanaryDefaultOn || decF32.Engine != Qwen38EngineMTP {
		t.Fatalf("F32 cpu-native expected default-on inside envelope, got %v, %s", decF32.CanaryDefaultOn, decF32.RejectionReason)
	}
}

func TestQwen38MTP_Canary_OutsideEnvelopeBoundaries(t *testing.T) {
	mgr := NewQwen38MTPCanaryManager()

	validHash := strings.Repeat("ab", 32)

	baseEnv := Qwen38CanaryEnvelope{
		ModelFamily:   "Qwen3.8",
		Format:        Qwen38MTPFormatQ4K,
		Backend:       Qwen38MTPBackendMetal,
		HeadroomBytes: 4 * 1024 * 1024 * 1024,
		ArtifactHash:  validHash,
		DraftDepth:    2,
	}
	const receiptID = "boundary-receipt"
	registerQwen38CanaryTestEvidence(t, mgr, receiptID, baseEnv)

	cases := []struct {
		name       string
		mutate     func(e *Qwen38CanaryEnvelope)
		optIn      bool
		wantEngine Qwen38MTPEngine
		wantOptIn  bool
	}{
		{
			name:       "model family mismatch",
			mutate:     func(e *Qwen38CanaryEnvelope) { e.ModelFamily = "Llama-3" },
			optIn:      false,
			wantEngine: Qwen38EngineTargetDecode,
			wantOptIn:  false,
		},
		{
			name:       "unsupported format BF16",
			mutate:     func(e *Qwen38CanaryEnvelope) { e.Format = Qwen38MTPFormatBF16 },
			optIn:      false,
			wantEngine: Qwen38EngineTargetDecode,
			wantOptIn:  false,
		},
		{
			name:       "unsupported backend cuda",
			mutate:     func(e *Qwen38CanaryEnvelope) { e.Backend = "cuda" },
			optIn:      false,
			wantEngine: Qwen38EngineTargetDecode,
			wantOptIn:  false,
		},
		{
			name:       "headroom below 2GB",
			mutate:     func(e *Qwen38CanaryEnvelope) { e.HeadroomBytes = 1 * 1024 * 1024 * 1024 },
			optIn:      false,
			wantEngine: Qwen38EngineTargetDecode,
			wantOptIn:  false,
		},
		{
			name: "unproven artifact hash",
			mutate: func(e *Qwen38CanaryEnvelope) {
				e.ArtifactHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
			},
			optIn:      false,
			wantEngine: Qwen38EngineTargetDecode,
			wantOptIn:  false,
		},
		{
			name:       "draft depth 0 outside [1, 4]",
			mutate:     func(e *Qwen38CanaryEnvelope) { e.DraftDepth = 0 },
			optIn:      false,
			wantEngine: Qwen38EngineTargetDecode,
			wantOptIn:  false,
		},
		{
			name:       "draft depth 5 outside [1, 4]",
			mutate:     func(e *Qwen38CanaryEnvelope) { e.DraftDepth = 5 },
			optIn:      false,
			wantEngine: Qwen38EngineTargetDecode,
			wantOptIn:  false,
		},
		{
			name:       "outside envelope with explicit operator opt-in",
			mutate:     func(e *Qwen38CanaryEnvelope) { e.HeadroomBytes = 1 * 1024 * 1024 * 1024 },
			optIn:      true,
			wantEngine: Qwen38EngineMTP,
			wantOptIn:  true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := baseEnv
			tc.mutate(&env)
			req := Qwen38CanaryRequest{
				Envelope:          env,
				EvidenceReceiptID: receiptID,
				OperatorOptIn:     tc.optIn,
				ModelReady:        true,
			}
			dec := mgr.EvaluateCanary(req)
			if dec.CanaryDefaultOn {
				t.Fatal("expected CanaryDefaultOn = false for outside-envelope condition")
			}
			if dec.Engine != tc.wantEngine {
				t.Fatalf("dec.Engine = %q, want %q", dec.Engine, tc.wantEngine)
			}
			if dec.OptInActive != tc.wantOptIn {
				t.Fatalf("dec.OptInActive = %v, want %v", dec.OptInActive, tc.wantOptIn)
			}
		})
	}
}

func TestQwen38MTP_Canary_CircuitBreakerTripOnDivergence(t *testing.T) {
	mgr := NewQwen38MTPCanaryManager()

	validHash := strings.Repeat("ab", 32)

	req := Qwen38CanaryRequest{
		ModelReady:    true,
		OperatorOptIn: false,
		Envelope: Qwen38CanaryEnvelope{
			ModelFamily:   "Qwen3.8",
			Format:        Qwen38MTPFormatQ4K,
			Backend:       Qwen38MTPBackendMetal,
			HeadroomBytes: 3 * 1024 * 1024 * 1024,
			ArtifactHash:  validHash,
			DraftDepth:    2,
		},
	}
	req.EvidenceReceiptID = "circuit-receipt"
	registerQwen38CanaryTestEvidence(t, mgr, req.EvidenceReceiptID, req.Envelope)

	// 1. Before divergence: default-on is active
	dec1 := mgr.EvaluateCanary(req)
	if !dec1.CanaryDefaultOn || dec1.Engine != Qwen38EngineMTP {
		t.Fatalf("expected initial default-on, got %v", dec1.CanaryDefaultOn)
	}

	// 2. Correctness divergence occurs!
	mgr.RecordDivergence("speculative token argmax diverged from target logits at pos 42")

	// 3. Verify circuit breaker state
	status, reason := mgr.CircuitStatus()
	if status != CanaryCircuitTripped {
		t.Fatalf("circuit status = %q, want %q", status, CanaryCircuitTripped)
	}
	if reason == "" {
		t.Fatal("expected non-empty circuit trip reason")
	}

	// 4. Subsequent requests inside envelope must be revoked and downgraded to target-only decode
	dec2 := mgr.EvaluateCanary(req)
	if dec2.CanaryDefaultOn {
		t.Fatal("CanaryDefaultOn remained true after circuit breaker trip")
	}
	if !dec2.CircuitTripped {
		t.Fatal("dec2.CircuitTripped expected true")
	}
	if dec2.Engine != Qwen38EngineTargetDecode {
		t.Fatalf("dec2.Engine = %q, want %q", dec2.Engine, Qwen38EngineTargetDecode)
	}
	if dec2.DowngradeReason != Qwen38MTPCorrectnessDiverged {
		t.Fatalf("dec2.DowngradeReason = %q, want %q", dec2.DowngradeReason, Qwen38MTPCorrectnessDiverged)
	}

	// 5. Reset restores canary operation
	mgr.ResetCircuitBreaker()
	dec3 := mgr.EvaluateCanary(req)
	if !dec3.CanaryDefaultOn || dec3.Engine != Qwen38EngineMTP {
		t.Fatalf("expected reset to restore canary default-on, got %v", dec3.CanaryDefaultOn)
	}
}

func TestQwen38MTP_Canary_ReceiptValidation(t *testing.T) {
	mgr := NewQwen38MTPCanaryManager()

	validHash := strings.Repeat("ab", 32)

	req := Qwen38CanaryRequest{
		ModelReady:    true,
		OperatorOptIn: false,
		Envelope: Qwen38CanaryEnvelope{
			ModelFamily:   "Qwen3.8",
			Format:        Qwen38MTPFormatQ4K,
			Backend:       Qwen38MTPBackendMetal,
			HeadroomBytes: 3 * 1024 * 1024 * 1024,
			ArtifactHash:  validHash,
			DraftDepth:    2,
		},
	}
	req.EvidenceReceiptID = "receipt-validation-source"
	registerQwen38CanaryTestEvidence(t, mgr, req.EvidenceReceiptID, req.Envelope)

	dec := mgr.EvaluateCanary(req)

	lat := Qwen38MTPLatencyNS{
		Setup:    5,
		Draft:    20,
		Verify:   10,
		Rollback: 0,
		Sync:     5,
		Recovery: 0,
		Total:    40,
	}
	mem := Qwen38MTPMemoryBytes{
		DraftWorkspace:  1024,
		VerifyWorkspace: 512,
		RollbackState:   256,
		Peak:            4096,
	}

	receipt, err := mgr.EmitReceipt("canary-receipt-001", dec, 1.35, 10, 8, lat, mem)
	if err != nil {
		t.Fatalf("EmitReceipt failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt.Validate() failed: %v", err)
	}

	// Verify invalid receipt rejected: speedup <= 1.0 on default-on
	invalidReceipt := *receipt
	invalidReceipt.Speedup = 0.95
	if err := invalidReceipt.Validate(); err == nil {
		t.Fatal("expected error on speedup <= 1.0 for default-on canary receipt")
	}

	// Verify invalid receipt rejected: divergence > 0 on default-on
	invalidReceipt2 := *receipt
	invalidReceipt2.DivergenceCount = 1
	if err := invalidReceipt2.Validate(); err == nil {
		t.Fatal("expected error on divergence > 0 for default-on canary receipt")
	}
}
