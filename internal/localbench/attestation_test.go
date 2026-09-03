package localbench

import (
	"crypto/ed25519"
	"path/filepath"
	"testing"
	"time"
)

func sampleBindings(receiptDigest string) AttestationBindings {
	return AttestationBindings{
		ReceiptDigest: receiptDigest,
		ModelArtifact: ModelArtifactBinding{
			Name:   "unsloth/Qwen3.8-27B-GGUF",
			Digest: "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
			Format: "gguf",
		},
		Quantization: QuantizationBinding{
			Format:        "Q4_K_M",
			BitsPerWeight: 4.5,
			Details:       "type=Q4_K_M hybrid-metal stream-quant",
		},
		Benchmark: BenchmarkBinding{
			ID:       "modelbench",
			Workload: "prompt-heavy-e2e",
		},
		Quality: QualityBinding{
			EvalKind:     "exact_match",
			Threshold:    "pass",
			ResultDigest: "sha256:4d62b9a76d8b3c9f8e7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e",
			Passed:       true,
		},
		Execution: ExecutionBinding{
			Engine:         "fak-native",
			Backend:        "metal",
			Runtime:        "metal/qwen35-hybrid-session-v1",
			Fallback:       "none",
			ModuleRevision: "internal/model@r448+g8145dc0bea",
		},
	}
}

func TestAttestationSignAndVerify(t *testing.T) {
	r := sampleReceipt(t)
	pub, priv, err := GenerateSigningKey()
	if err != nil {
		t.Fatal(err)
	}

	bindings := sampleBindings(r.Integrity.SHA256)
	createdAt := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)
	expiresAt := createdAt.Add(365 * 24 * time.Hour)

	env, err := SignReceipt(r, bindings, priv, "operator-key-1", createdAt, expiresAt)
	if err != nil {
		t.Fatalf("SignReceipt: %v", err)
	}

	// 1. Verify without trust store -> TrustSelfSigned
	status, err := VerifyAttestation(env, nil, createdAt.Add(time.Hour))
	if err != nil {
		t.Fatalf("VerifyAttestation self-signed: %v", err)
	}
	if status != TrustSelfSigned {
		t.Fatalf("expected TrustSelfSigned, got %s", status)
	}

	// 2. Verify with trust store containing public key -> TrustVerified
	store := NewTrustStore()
	store.AddTrustedKey("operator-key-1", pub)
	status, err = VerifyAttestation(env, store, createdAt.Add(time.Hour))
	if err != nil {
		t.Fatalf("VerifyAttestation verified: %v", err)
	}
	if status != TrustVerified {
		t.Fatalf("expected TrustVerified, got %s", status)
	}

	// 3. Round-trip file read/verify
	tmp := filepath.Join(t.TempDir(), "attested.json")
	if err := WriteAttestationEnvelope(tmp, env); err != nil {
		t.Fatal(err)
	}
	readR, readEnv, readStatus, err := ReadReceiptOrEnvelope(tmp, store, createdAt.Add(time.Hour))
	if err != nil {
		t.Fatalf("ReadReceiptOrEnvelope: %v", err)
	}
	if readR == nil || readEnv == nil || readStatus != TrustVerified {
		t.Fatalf("read failed: status=%s, env=%v", readStatus, readEnv)
	}
	if readR.Integrity.SHA256 != r.Integrity.SHA256 {
		t.Fatalf("receipt digest mismatch: got %s, want %s", readR.Integrity.SHA256, r.Integrity.SHA256)
	}
}

func TestAttestationExpiration(t *testing.T) {
	r := sampleReceipt(t)
	_, priv, err := GenerateSigningKey()
	if err != nil {
		t.Fatal(err)
	}

	bindings := sampleBindings(r.Integrity.SHA256)
	createdAt := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)
	expiresAt := createdAt.Add(24 * time.Hour)

	env, err := SignReceipt(r, bindings, priv, "exp-key", createdAt, expiresAt)
	if err != nil {
		t.Fatal(err)
	}

	// Before expiry -> OK
	status, err := VerifyAttestation(env, nil, createdAt.Add(time.Hour))
	if err != nil || status != TrustSelfSigned {
		t.Fatalf("unexpected verification before expiry: status=%s err=%v", status, err)
	}

	// After expiry -> TrustExpired
	status, err = VerifyAttestation(env, nil, expiresAt.Add(time.Minute))
	if status != TrustExpired {
		t.Fatalf("expected TrustExpired, got %s (err: %v)", status, err)
	}
}

func TestAttestationKeyRevocation(t *testing.T) {
	r := sampleReceipt(t)
	pub, priv, err := GenerateSigningKey()
	if err != nil {
		t.Fatal(err)
	}

	bindings := sampleBindings(r.Integrity.SHA256)
	createdAt := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)

	env, err := SignReceipt(r, bindings, priv, "compromised-key", createdAt, time.Time{})
	if err != nil {
		t.Fatal(err)
	}

	store := NewTrustStore()
	store.AddTrustedKey("compromised-key", pub)

	status, err := VerifyAttestation(env, store, createdAt.Add(time.Hour))
	if err != nil || status != TrustVerified {
		t.Fatalf("expected TrustVerified before revocation: %v", err)
	}

	// Revoke the key
	store.RevokeKey("compromised-key", "key leaked", createdAt.Add(2*time.Hour))

	status, err = VerifyAttestation(env, store, createdAt.Add(3*time.Hour))
	if status != TrustRevoked {
		t.Fatalf("expected TrustRevoked, got %s (err: %v)", status, err)
	}
}

func TestAttestationTamperRejection(t *testing.T) {
	r := sampleReceipt(t)
	_, priv, err := GenerateSigningKey()
	if err != nil {
		t.Fatal(err)
	}

	bindings := sampleBindings(r.Integrity.SHA256)
	env, err := SignReceipt(r, bindings, priv, "key-1", time.Time{}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}

	// Case 1: Tamper model artifact name
	t.Run("tamper_model_name", func(t *testing.T) {
		tampered := env
		tampered.Attestation.Bindings.ModelArtifact.Name = "forged/model"
		status, err := VerifyAttestation(tampered, nil, time.Time{})
		if status != TrustInvalid || err == nil {
			t.Fatalf("expected TrustInvalid on tampered model name, got %s err=%v", status, err)
		}
	})

	// Case 2: Tamper model artifact digest
	t.Run("tamper_model_digest", func(t *testing.T) {
		tampered := env
		tampered.Attestation.Bindings.ModelArtifact.Digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
		status, err := VerifyAttestation(tampered, nil, time.Time{})
		if status != TrustInvalid || err == nil {
			t.Fatalf("expected TrustInvalid on tampered model digest, got %s err=%v", status, err)
		}
	})

	// Case 3: Tamper quantization format
	t.Run("tamper_quant_format", func(t *testing.T) {
		tampered := env
		tampered.Attestation.Bindings.Quantization.Format = "Q8_0"
		status, err := VerifyAttestation(tampered, nil, time.Time{})
		if status != TrustInvalid || err == nil {
			t.Fatalf("expected TrustInvalid on tampered quant format, got %s err=%v", status, err)
		}
	})

	// Case 4: Tamper execution backend
	t.Run("tamper_backend", func(t *testing.T) {
		tampered := env
		tampered.Attestation.Bindings.Execution.Backend = "cuda"
		status, err := VerifyAttestation(tampered, nil, time.Time{})
		if status != TrustInvalid || err == nil {
			t.Fatalf("expected TrustInvalid on tampered backend, got %s err=%v", status, err)
		}
	})

	// Case 5: Tamper inner receipt
	t.Run("tamper_inner_receipt", func(t *testing.T) {
		tampered := env
		tampered.Receipt.ExitStatus = 1
		status, err := VerifyAttestation(tampered, nil, time.Time{})
		if status != TrustInvalid || err == nil {
			t.Fatalf("expected TrustInvalid on tampered inner receipt, got %s err=%v", status, err)
		}
	})

	// Case 6: Tamper signature
	t.Run("tamper_signature", func(t *testing.T) {
		tampered := env
		tampered.Attestation.Signature = "00" + tampered.Attestation.Signature[2:]
		status, err := VerifyAttestation(tampered, nil, time.Time{})
		if status != TrustInvalid || err == nil {
			t.Fatalf("expected TrustInvalid on tampered signature, got %s err=%v", status, err)
		}
	})
}

func TestNativeInferenceInvariant(t *testing.T) {
	r := sampleReceipt(t)
	_, priv, err := GenerateSigningKey()
	if err != nil {
		t.Fatal(err)
	}

	// 1. Attempt to sign with empty backend under fak-native
	t.Run("empty_backend", func(t *testing.T) {
		b := sampleBindings(r.Integrity.SHA256)
		b.Execution.Backend = ""
		_, err := SignReceipt(r, b, priv, "k", time.Time{}, time.Time{})
		if err == nil {
			t.Fatal("expected error signing fak-native with empty backend")
		}
	})

	// 2. Attempt to sign with empty runtime under fak-native
	t.Run("empty_runtime", func(t *testing.T) {
		b := sampleBindings(r.Integrity.SHA256)
		b.Execution.Runtime = ""
		_, err := SignReceipt(r, b, priv, "k", time.Time{}, time.Time{})
		if err == nil {
			t.Fatal("expected error signing fak-native with empty runtime")
		}
	})

	// 3. Attempt to sign with non-none fallback under fak-native
	t.Run("invalid_fallback", func(t *testing.T) {
		b := sampleBindings(r.Integrity.SHA256)
		b.Execution.Fallback = "external-helper"
		_, err := SignReceipt(r, b, priv, "k", time.Time{}, time.Time{})
		if err == nil {
			t.Fatal("expected error signing fak-native with fallback='external-helper'")
		}
	})

	// 4. Non-fak-native engine (e.g. external baseline) is allowed to carry external label
	t.Run("external_engine", func(t *testing.T) {
		b := sampleBindings(r.Integrity.SHA256)
		b.Execution.Engine = "llamacpp-reference"
		b.Execution.Backend = "cuda"
		b.Execution.Runtime = "llamacpp-b3000"
		b.Execution.Fallback = "none"
		env, err := SignReceipt(r, b, priv, "k", time.Time{}, time.Time{})
		if err != nil {
			t.Fatalf("unexpected error signing reference engine: %v", err)
		}
		status, err := VerifyAttestation(env, nil, time.Time{})
		if err != nil || status != TrustSelfSigned {
			t.Fatalf("unexpected verification for reference engine: status=%s err=%v", status, err)
		}
	})
}

func TestBackwardCompatibilityUnsignedReceipt(t *testing.T) {
	r := sampleReceipt(t)
	tmp := filepath.Join(t.TempDir(), "unsigned.json")
	if err := writeReceipt(tmp, r); err != nil {
		t.Fatal(err)
	}

	readR, readEnv, status, err := ReadReceiptOrEnvelope(tmp, nil, time.Time{})
	if err != nil {
		t.Fatalf("ReadReceiptOrEnvelope on v1 receipt: %v", err)
	}
	if readR == nil {
		t.Fatal("readR is nil")
	}
	if readEnv != nil {
		t.Fatal("expected nil env for unsigned receipt")
	}
	if status != TrustUnsigned {
		t.Fatalf("expected TrustUnsigned, got %s", status)
	}
	if readR.Integrity.SHA256 != r.Integrity.SHA256 {
		t.Fatalf("digest mismatch: got %s, want %s", readR.Integrity.SHA256, r.Integrity.SHA256)
	}
}

func TestInvalidKeySize(t *testing.T) {
	r := sampleReceipt(t)
	b := sampleBindings(r.Integrity.SHA256)
	_, err := SignReceipt(r, b, ed25519.PrivateKey{1, 2, 3}, "k", time.Time{}, time.Time{})
	if err == nil {
		t.Fatal("expected error on invalid key size")
	}
}
