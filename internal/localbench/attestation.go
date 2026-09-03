package localbench

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	AttestationSchema = "fak.local-hardware-benchmark.attestation/v1"
	AttestationAlg    = "ed25519"
)

type TrustStatus string

const (
	TrustUnsigned   TrustStatus = "unsigned"
	TrustSelfSigned TrustStatus = "self_signed"
	TrustVerified   TrustStatus = "verified"
	TrustRevoked    TrustStatus = "revoked"
	TrustExpired    TrustStatus = "expired"
	TrustInvalid    TrustStatus = "invalid"
)

// AttestationEnvelope wraps a sealed v1 benchmark receipt alongside an
// authenticating attestation that binds model artifact identity, quantization,
// benchmark workload, quality results, and execution backend.
type AttestationEnvelope struct {
	Schema      string      `json:"schema"`
	Receipt     Receipt     `json:"receipt"`
	Attestation Attestation `json:"attestation"`
}

type Attestation struct {
	Version   string              `json:"version"`
	KeyID     string              `json:"key_id"`
	PublicKey string              `json:"public_key"`
	Algorithm string              `json:"algorithm"`
	CreatedAt string              `json:"created_at"`
	ExpiresAt string              `json:"expires_at,omitempty"`
	Bindings  AttestationBindings `json:"bindings"`
	Signature string              `json:"signature"`
}

type AttestationBindings struct {
	ReceiptDigest string               `json:"receipt_digest"`
	ModelArtifact ModelArtifactBinding `json:"model_artifact"`
	Quantization  QuantizationBinding  `json:"quantization"`
	Benchmark     BenchmarkBinding     `json:"benchmark"`
	Quality       QualityBinding       `json:"quality"`
	Execution     ExecutionBinding     `json:"execution"`
}

type ModelArtifactBinding struct {
	Name   string `json:"name"`
	Digest string `json:"digest"`
	Format string `json:"format"`
}

type QuantizationBinding struct {
	Format        string  `json:"format"`
	BitsPerWeight float64 `json:"bits_per_weight"`
	Details       string  `json:"details,omitempty"`
}

type BenchmarkBinding struct {
	ID       string `json:"id"`
	Workload string `json:"workload"`
}

type QualityBinding struct {
	EvalKind     string `json:"eval_kind"`
	Threshold    string `json:"threshold"`
	ResultDigest string `json:"result_digest"`
	Passed       bool   `json:"passed"`
}

type ExecutionBinding struct {
	Engine         string `json:"engine"`
	Backend        string `json:"backend"`
	Runtime        string `json:"runtime"`
	Fallback       string `json:"fallback"`
	ModuleRevision string `json:"module_revision"`
}

type keyRetirementRecord struct {
	RevokedAt string `json:"revoked_at"`
	Reason    string `json:"reason"`
}

// TrustStore tracks known operator/fleet public keys and revoked keys for
// attestation verification and key lifecycle rotation.
type TrustStore struct {
	mu          sync.RWMutex
	trustedKeys map[string]ed25519.PublicKey
	revokedKeys map[string]keyRetirementRecord
}

func NewTrustStore() *TrustStore {
	return &TrustStore{
		trustedKeys: make(map[string]ed25519.PublicKey),
		revokedKeys: make(map[string]keyRetirementRecord),
	}
}

func (ts *TrustStore) AddTrustedKey(keyID string, pub ed25519.PublicKey) {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	ts.trustedKeys[keyID] = pub
}

func (ts *TrustStore) RevokeKey(keyID string, reason string, at time.Time) {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	ts.revokedKeys[keyID] = keyRetirementRecord{
		RevokedAt: at.UTC().Format(time.RFC3339),
		Reason:    reason,
	}
}

func (ts *TrustStore) IsTrusted(keyID, pubHex string) bool {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	pub, ok := ts.trustedKeys[keyID]
	if !ok {
		return false
	}
	return strings.EqualFold(hex.EncodeToString(pub), pubHex)
}

func (ts *TrustStore) IsRevoked(keyID string) (bool, keyRetirementRecord) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	rec, ok := ts.revokedKeys[keyID]
	return ok, rec
}

// GenerateSigningKey generates a new Ed25519 keypair for local benchmark attestation.
// Private keys must be saved with strict 0600 file permissions outside of source trees.
func GenerateSigningKey() (ed25519.PublicKey, ed25519.PrivateKey, error) {
	return ed25519.GenerateKey(rand.Reader)
}

// CanonicalBindings produces the deterministic canonical bytes over the attestation bindings.
func CanonicalBindings(b AttestationBindings) ([]byte, error) {
	return json.Marshal(b)
}

// SignReceipt produces an AttestationEnvelope binding an integrity-verified Receipt
// to model artifact, quantization, benchmark workload, quality results, and execution backend.
func SignReceipt(r Receipt, bindings AttestationBindings, priv ed25519.PrivateKey, keyID string, createdAt, expiresAt time.Time) (AttestationEnvelope, error) {
	if err := verify(r); err != nil {
		return AttestationEnvelope{}, fmt.Errorf("cannot attest invalid or tampered receipt: %w", err)
	}
	if len(priv) != ed25519.PrivateKeySize {
		return AttestationEnvelope{}, fmt.Errorf("invalid ed25519 private key length %d, want %d", len(priv), ed25519.PrivateKeySize)
	}

	// Native inference invariant: execution claimed as fak-native must name
	// the native backend, forward runtime, and explicitly state fallback="none".
	if bindings.Execution.Engine == "fak-native" {
		if strings.TrimSpace(bindings.Execution.Backend) == "" {
			return AttestationEnvelope{}, errors.New("fak-native attestation requires non-empty execution.backend")
		}
		if strings.TrimSpace(bindings.Execution.Runtime) == "" {
			return AttestationEnvelope{}, errors.New("fak-native attestation requires non-empty execution.runtime")
		}
		if strings.TrimSpace(bindings.Execution.Fallback) != "none" {
			return AttestationEnvelope{}, fmt.Errorf("fak-native attestation requires execution.fallback='none', got %q", bindings.Execution.Fallback)
		}
	}

	// Receipt digest binding
	bindings.ReceiptDigest = r.Integrity.SHA256

	payload, err := CanonicalBindings(bindings)
	if err != nil {
		return AttestationEnvelope{}, fmt.Errorf("serializing canonical bindings: %w", err)
	}

	sig := ed25519.Sign(priv, payload)
	pub := priv.Public().(ed25519.PublicKey)

	var expiresStr string
	if !expiresAt.IsZero() {
		expiresStr = expiresAt.UTC().Format(time.RFC3339)
	}
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}

	att := Attestation{
		Version:   "v1",
		KeyID:     keyID,
		PublicKey: hex.EncodeToString(pub),
		Algorithm: AttestationAlg,
		CreatedAt: createdAt.UTC().Format(time.RFC3339),
		ExpiresAt: expiresStr,
		Bindings:  bindings,
		Signature: hex.EncodeToString(sig),
	}

	return AttestationEnvelope{
		Schema:      AttestationSchema,
		Receipt:     r,
		Attestation: att,
	}, nil
}

// VerifyAttestation verifies the cryptographic signature of the attestation envelope,
// confirms receipt integrity, checks native-inference invariants, and evaluates
// expiration, revocation, and known-key trust status.
func VerifyAttestation(env AttestationEnvelope, store *TrustStore, now time.Time) (TrustStatus, error) {
	if env.Schema != AttestationSchema {
		return TrustInvalid, fmt.Errorf("unsupported attestation schema %q", env.Schema)
	}
	if err := verify(env.Receipt); err != nil {
		return TrustInvalid, fmt.Errorf("inner receipt integrity failed: %w", err)
	}
	if env.Attestation.Algorithm != AttestationAlg {
		return TrustInvalid, fmt.Errorf("unsupported attestation algorithm %q", env.Attestation.Algorithm)
	}
	if env.Attestation.Version != "v1" {
		return TrustInvalid, fmt.Errorf("unsupported attestation version %q", env.Attestation.Version)
	}

	// Verify receipt digest binding
	if !strings.EqualFold(env.Attestation.Bindings.ReceiptDigest, env.Receipt.Integrity.SHA256) {
		return TrustInvalid, errors.New("attestation receipt digest does not match inner receipt integrity SHA-256")
	}

	// Verify native inference invariant
	if env.Attestation.Bindings.Execution.Engine == "fak-native" {
		if strings.TrimSpace(env.Attestation.Bindings.Execution.Backend) == "" ||
			strings.TrimSpace(env.Attestation.Bindings.Execution.Runtime) == "" ||
			strings.TrimSpace(env.Attestation.Bindings.Execution.Fallback) != "none" {
			return TrustInvalid, errors.New("fak-native attestation requires non-empty backend, runtime, and fallback=none")
		}
	}

	// Check expiration
	if env.Attestation.ExpiresAt != "" {
		exp, err := time.Parse(time.RFC3339, env.Attestation.ExpiresAt)
		if err != nil {
			return TrustInvalid, fmt.Errorf("invalid expires_at timestamp: %w", err)
		}
		if now.IsZero() {
			now = time.Now().UTC()
		}
		if now.After(exp) {
			return TrustExpired, fmt.Errorf("attestation expired at %s", env.Attestation.ExpiresAt)
		}
	}

	// Check revocation
	if store != nil {
		if isRevoked, rec := store.IsRevoked(env.Attestation.KeyID); isRevoked {
			return TrustRevoked, fmt.Errorf("attestation signing key %q was revoked at %s (reason: %s)", env.Attestation.KeyID, rec.RevokedAt, rec.Reason)
		}
	}

	pubBytes, err := hex.DecodeString(env.Attestation.PublicKey)
	if err != nil || len(pubBytes) != ed25519.PublicKeySize {
		return TrustInvalid, errors.New("invalid public key encoding in attestation")
	}
	sigBytes, err := hex.DecodeString(env.Attestation.Signature)
	if err != nil || len(sigBytes) != ed25519.SignatureSize {
		return TrustInvalid, errors.New("invalid signature encoding in attestation")
	}

	payload, err := CanonicalBindings(env.Attestation.Bindings)
	if err != nil {
		return TrustInvalid, fmt.Errorf("canonical payload serialization error: %w", err)
	}

	if !ed25519.Verify(pubBytes, payload, sigBytes) {
		return TrustInvalid, errors.New("attestation signature verification failed (tampered or forged)")
	}

	if store != nil && store.IsTrusted(env.Attestation.KeyID, env.Attestation.PublicKey) {
		return TrustVerified, nil
	}

	return TrustSelfSigned, nil
}

// WriteAttestationEnvelope serializes an attestation envelope to disk.
func WriteAttestationEnvelope(path string, env AttestationEnvelope) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	return writeJSON(f, env)
}

// ReadReceiptOrEnvelope parses and verifies either a raw v1 receipt or an attested v1 envelope.
func ReadReceiptOrEnvelope(path string, store *TrustStore, now time.Time) (*Receipt, *AttestationEnvelope, TrustStatus, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, TrustInvalid, err
	}

	var schemaCheck struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &schemaCheck); err != nil {
		return nil, nil, TrustInvalid, err
	}

	switch schemaCheck.Schema {
	case receiptSchema:
		var r Receipt
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&r); err != nil {
			return nil, nil, TrustInvalid, err
		}
		var trailing any
		if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
			if err == nil {
				return nil, nil, TrustInvalid, errors.New("receipt contains trailing JSON data")
			}
			return nil, nil, TrustInvalid, fmt.Errorf("receipt trailing data: %w", err)
		}
		if err := verify(r); err != nil {
			return nil, nil, TrustInvalid, err
		}
		return &r, nil, TrustUnsigned, nil

	case AttestationSchema:
		var env AttestationEnvelope
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&env); err != nil {
			return nil, nil, TrustInvalid, err
		}
		var trailing any
		if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
			if err == nil {
				return nil, nil, TrustInvalid, errors.New("attestation contains trailing JSON data")
			}
			return nil, nil, TrustInvalid, fmt.Errorf("attestation trailing data: %w", err)
		}
		status, err := VerifyAttestation(env, store, now)
		if err != nil && status == TrustInvalid {
			return nil, nil, status, err
		}
		return &env.Receipt, &env, status, err

	default:
		return nil, nil, TrustInvalid, fmt.Errorf("unsupported schema %q", schemaCheck.Schema)
	}
}
