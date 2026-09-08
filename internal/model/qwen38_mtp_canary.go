package model

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Qwen38MTPCanaryReceiptSchema is the canonical schema for Qwen3.8 MTP canary receipts.
const Qwen38MTPCanaryReceiptSchema = "fak/qwen38-mtp-canary-receipt/v1"

// MinCanaryMemoryHeadroomBytes defines the 2GB minimum headroom floor required for certified canary admission.
const MinCanaryMemoryHeadroomBytes uint64 = 2 * 1024 * 1024 * 1024 // 2GB

// Qwen38MTPBackendCPU represents the cpu-native test and verification backend.
const Qwen38MTPBackendCPU Qwen38MTPBackend = "cpu-native"

// CanaryCircuitStatus tracks whether the safety ratchet circuit breaker is closed or tripped.
type CanaryCircuitStatus string

const (
	CanaryCircuitClosed  CanaryCircuitStatus = "closed"
	CanaryCircuitTripped CanaryCircuitStatus = "tripped"
)

// Qwen38CanaryEnvelope defines the certified operational envelope for canary default-on MTP.
type Qwen38CanaryEnvelope struct {
	ModelFamily   string                `json:"model_family"`
	Format        Qwen38MTPTensorFormat `json:"format"`
	Backend       Qwen38MTPBackend      `json:"backend"`
	HeadroomBytes uint64                `json:"headroom_bytes"`
	ArtifactHash  string                `json:"artifact_hash"`
	DraftDepth    int                   `json:"draft_depth"`
}

// Qwen38CanaryRequest describes an incoming inference request for canary adjudication.
type Qwen38CanaryRequest struct {
	Envelope          Qwen38CanaryEnvelope `json:"envelope"`
	EvidenceReceiptID string               `json:"evidence_receipt_id,omitempty"`
	OperatorOptIn     bool                 `json:"operator_opt_in"`
	ModelReady        bool                 `json:"model_ready"`
}

// Qwen38CanaryDecision captures the result of canary envelope evaluation.
type Qwen38CanaryDecision struct {
	CanaryDefaultOn bool                     `json:"canary_default_on"`
	OptInActive     bool                     `json:"opt_in_active"`
	Engine          Qwen38MTPEngine          `json:"engine"`
	DowngradeReason Qwen38MTPDowngradeReason `json:"downgrade_reason,omitempty"`
	InsideEnvelope  bool                     `json:"inside_envelope"`
	CircuitTripped  bool                     `json:"circuit_tripped"`
	RejectionReason string                   `json:"rejection_reason,omitempty"`
	Envelope        Qwen38CanaryEnvelope     `json:"envelope"`
}

// Qwen38MTPCanaryReceipt witnesses default selection, fak-native engine, zero divergence,
// and net end-to-end acceleration.
type Qwen38MTPCanaryReceipt struct {
	SchemaVersion   string                   `json:"schema_version"`
	ReceiptID       string                   `json:"receipt_id"`
	DefaultOn       bool                     `json:"default_on"`
	Engine          Qwen38MTPEngine          `json:"engine"`
	Envelope        Qwen38CanaryEnvelope     `json:"envelope"`
	DivergenceCount int64                    `json:"divergence_count"`
	ErrorCount      int64                    `json:"error_count"`
	Speedup         float64                  `json:"speedup"`
	TokensProduced  int                      `json:"tokens_produced"`
	TokensAccepted  int                      `json:"tokens_accepted"`
	TokensProposed  int                      `json:"tokens_proposed"`
	DowngradeReason Qwen38MTPDowngradeReason `json:"downgrade_reason,omitempty"`
	CircuitStatus   CanaryCircuitStatus      `json:"circuit_status"`
	LatencyNS       Qwen38MTPLatencyNS       `json:"latency_ns"`
	MemoryBytes     Qwen38MTPMemoryBytes     `json:"memory_bytes"`
}

// Qwen38MTPCanaryEvidence binds a validated performance/correctness receipt to
// its observation and expiry window. The receipt carries both identities the
// gate relies on: ReceiptID identifies the witness, while Envelope.ArtifactHash
// identifies the exact model artifact the witness measured.
type Qwen38MTPCanaryEvidence struct {
	Receipt    Qwen38MTPCanaryReceipt `json:"receipt"`
	ObservedAt time.Time              `json:"observed_at"`
	ValidUntil time.Time              `json:"valid_until"`
}

// Validate rejects evidence that is not a successful, time-bounded MTP witness
// for a content-addressed artifact. Expiry is evaluated at admission time so a
// once-valid witness cannot silently remain authoritative forever.
func (e Qwen38MTPCanaryEvidence) Validate() error {
	if err := e.Receipt.Validate(); err != nil {
		return fmt.Errorf("model: invalid MTP canary evidence receipt: %w", err)
	}
	if !e.Receipt.DefaultOn || e.Receipt.Engine != Qwen38EngineMTP {
		return errors.New("model: MTP canary evidence must witness successful default-on fak-native MTP")
	}
	if _, ok := normalizeCanaryArtifactHash(e.Receipt.Envelope.ArtifactHash); !ok {
		return fmt.Errorf("model: MTP canary evidence artifact hash %q is not SHA-256", e.Receipt.Envelope.ArtifactHash)
	}
	if e.ObservedAt.IsZero() || e.ValidUntil.IsZero() || !e.ValidUntil.After(e.ObservedAt) {
		return errors.New("model: MTP canary evidence requires a non-empty validity window")
	}
	return nil
}

// Validate verifies that the canary receipt adheres to all schema and safety invariants.
func (r Qwen38MTPCanaryReceipt) Validate() error {
	if r.SchemaVersion != Qwen38MTPCanaryReceiptSchema {
		return fmt.Errorf("model: canary receipt schema %q, want %q", r.SchemaVersion, Qwen38MTPCanaryReceiptSchema)
	}
	if r.ReceiptID == "" {
		return errors.New("model: canary receipt receipt_id is empty")
	}
	if !validQwen38MTPEngine(r.Engine) {
		return fmt.Errorf("model: canary receipt invalid engine %q", r.Engine)
	}

	if r.DefaultOn {
		if r.Engine != Qwen38EngineMTP {
			return fmt.Errorf("model: default-on canary must execute %q, got %q", Qwen38EngineMTP, r.Engine)
		}
		if r.DivergenceCount != 0 {
			return fmt.Errorf("model: default-on canary requires zero divergence, got %d", r.DivergenceCount)
		}
		if r.Speedup <= 1.0 {
			return fmt.Errorf("model: default-on canary requires net speedup > 1.0, got %g", r.Speedup)
		}
		if r.CircuitStatus != CanaryCircuitClosed {
			return fmt.Errorf("model: default-on canary circuit status %q != %q", r.CircuitStatus, CanaryCircuitClosed)
		}
		if r.DowngradeReason != Qwen38MTPEligible {
			return fmt.Errorf("model: default-on canary unexpected downgrade reason %q", r.DowngradeReason)
		}
	} else {
		if r.Engine == Qwen38EngineTargetDecode && r.DowngradeReason == Qwen38MTPEligible {
			return errors.New("model: target decode receipt requires a non-empty downgrade reason")
		}
	}

	if r.DivergenceCount > 0 {
		if r.CircuitStatus != CanaryCircuitTripped {
			return fmt.Errorf("model: non-zero divergence requires circuit status %q, got %q", CanaryCircuitTripped, r.CircuitStatus)
		}
		if r.DefaultOn {
			return errors.New("model: canary cannot be default-on with non-zero divergence")
		}
	}

	if err := r.LatencyNS.validate(); err != nil {
		return fmt.Errorf("model: canary receipt invalid latency: %w", err)
	}
	if err := r.MemoryBytes.validate(); err != nil {
		return fmt.Errorf("model: canary receipt invalid memory: %w", err)
	}

	return nil
}

// Qwen38MTPCanaryManager enforces certified canary envelopes, automated default-on activation,
// and safety ratchet circuit breaking.
type Qwen38MTPCanaryManager struct {
	mu                 sync.RWMutex
	evidenceByReceipt  map[string]Qwen38MTPCanaryEvidence
	circuitStatus      CanaryCircuitStatus
	circuitTripReason  string
	totalEvaluations   int64
	canaryDefaultCount int64
	optInCount         int64
	targetOnlyCount    int64
	divergenceCount    int64
	errorCount         int64
}

// NewQwen38MTPCanaryManager initializes an empty, fail-closed evidence registry
// and a closed circuit. No artifact is trusted until a validated witness is
// registered; in particular, a literal label can never manufacture admission.
func NewQwen38MTPCanaryManager() *Qwen38MTPCanaryManager {
	return &Qwen38MTPCanaryManager{
		evidenceByReceipt: make(map[string]Qwen38MTPCanaryEvidence),
		circuitStatus:     CanaryCircuitClosed,
	}
}

// RegisterCanaryEvidence adds a validated receipt and its bounded freshness
// window to the admission registry. Receipt IDs are immutable identities: a
// second registration may refresh the identical witness, but cannot rebind the
// ID to another artifact or envelope.
func (m *Qwen38MTPCanaryManager) RegisterCanaryEvidence(evidence Qwen38MTPCanaryEvidence) error {
	if err := evidence.Validate(); err != nil {
		return err
	}
	receiptID := strings.TrimSpace(evidence.Receipt.ReceiptID)
	m.mu.Lock()
	defer m.mu.Unlock()
	if previous, ok := m.evidenceByReceipt[receiptID]; ok && !canaryEvidenceIdentityMatches(previous, evidence) {
		return fmt.Errorf("model: MTP canary receipt identity %q is already bound to another envelope", receiptID)
	}
	m.evidenceByReceipt[receiptID] = evidence
	return nil
}

func canaryEvidenceIdentityMatches(a, b Qwen38MTPCanaryEvidence) bool {
	aHash, aOK := normalizeCanaryArtifactHash(a.Receipt.Envelope.ArtifactHash)
	bHash, bOK := normalizeCanaryArtifactHash(b.Receipt.Envelope.ArtifactHash)
	return aOK && bOK && strings.TrimSpace(a.Receipt.ReceiptID) == strings.TrimSpace(b.Receipt.ReceiptID) &&
		aHash == bHash &&
		strings.EqualFold(strings.TrimSpace(a.Receipt.Envelope.ModelFamily), strings.TrimSpace(b.Receipt.Envelope.ModelFamily)) &&
		a.Receipt.Envelope.Format == b.Receipt.Envelope.Format &&
		a.Receipt.Envelope.Backend == b.Receipt.Envelope.Backend &&
		a.Receipt.Envelope.HeadroomBytes == b.Receipt.Envelope.HeadroomBytes &&
		a.Receipt.Envelope.DraftDepth == b.Receipt.Envelope.DraftDepth
}

// TripCircuitBreaker immediately trips the safety circuit breaker, revoking canary default status.
func (m *Qwen38MTPCanaryManager) TripCircuitBreaker(reason string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.circuitStatus = CanaryCircuitTripped
	if reason == "" {
		reason = "correctness_diverged"
	}
	m.circuitTripReason = reason
}

// ResetCircuitBreaker resets the circuit breaker to closed and clears divergence counters.
func (m *Qwen38MTPCanaryManager) ResetCircuitBreaker() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.circuitStatus = CanaryCircuitClosed
	m.circuitTripReason = ""
	m.divergenceCount = 0
}

// RecordDivergence records a correctness divergence and trips the circuit breaker immediately.
func (m *Qwen38MTPCanaryManager) RecordDivergence(detail string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.divergenceCount++
	m.circuitStatus = CanaryCircuitTripped
	m.circuitTripReason = detail
}

// RecordStep records step outcome and trips circuit breaker if divergence is observed.
func (m *Qwen38MTPCanaryManager) RecordStep(diverged bool, errored bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if errored {
		m.errorCount++
	}
	if diverged {
		m.divergenceCount++
		m.circuitStatus = CanaryCircuitTripped
		m.circuitTripReason = "divergence_during_speculative_verification"
	}
}

// CircuitStatus returns the current safety ratchet circuit breaker state.
func (m *Qwen38MTPCanaryManager) CircuitStatus() (CanaryCircuitStatus, string) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.circuitStatus, m.circuitTripReason
}

// EvaluateCanary evaluates an incoming request against the certified canary envelope.
func (m *Qwen38MTPCanaryManager) EvaluateCanary(req Qwen38CanaryRequest) Qwen38CanaryDecision {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.totalEvaluations++
	env := req.Envelope

	// 1. Safety Ratchet Circuit Breaker
	if m.circuitStatus == CanaryCircuitTripped {
		m.targetOnlyCount++
		return Qwen38CanaryDecision{
			CanaryDefaultOn: false,
			Engine:          Qwen38EngineTargetDecode,
			DowngradeReason: Qwen38MTPCorrectnessDiverged,
			InsideEnvelope:  false,
			CircuitTripped:  true,
			RejectionReason: fmt.Sprintf("safety ratchet circuit breaker tripped: %s", m.circuitTripReason),
			Envelope:        env,
		}
	}

	// 2. Validate Certified Envelope Requirements
	var rejectionReasons []string

	// Model family check ("Qwen3.8")
	if !strings.EqualFold(env.ModelFamily, "Qwen3.8") {
		rejectionReasons = append(rejectionReasons, fmt.Sprintf("model family %q != Qwen3.8", env.ModelFamily))
	}

	// Format check (F32 or Q4_K)
	if env.Format != Qwen38MTPFormatF32 && env.Format != Qwen38MTPFormatQ4K {
		rejectionReasons = append(rejectionReasons, fmt.Sprintf("format %q unsupported (must be F32 or Q4_K)", env.Format))
	}

	// Backend check ("metal" or "cpu-native")
	if env.Backend != Qwen38MTPBackendMetal && env.Backend != Qwen38MTPBackendCPU && env.Backend != "metal" && env.Backend != "cpu-native" {
		rejectionReasons = append(rejectionReasons, fmt.Sprintf("backend %q unsupported for canary", env.Backend))
	}

	// Memory headroom check (> 2GB)
	if env.HeadroomBytes <= MinCanaryMemoryHeadroomBytes {
		rejectionReasons = append(rejectionReasons, fmt.Sprintf("headroom %d <= 2GB threshold", env.HeadroomBytes))
	}

	// Evidence check: a request names the exact receipt whose validated envelope
	// authorized this artifact. Unknown IDs are missing evidence; a known receipt
	// with another artifact/envelope is mismatched; an expired receipt is stale.
	evidenceReason := Qwen38MTPEligible
	receiptID := strings.TrimSpace(req.EvidenceReceiptID)
	evidence, found := m.evidenceByReceipt[receiptID]
	switch {
	case receiptID == "" || !found:
		evidenceReason = Qwen38MTPEvidenceMissing
		rejectionReasons = append(rejectionReasons, "witnessed canary evidence is missing")
	case !time.Now().Before(evidence.ValidUntil):
		evidenceReason = Qwen38MTPEvidenceStale
		rejectionReasons = append(rejectionReasons, fmt.Sprintf("canary evidence receipt %q is stale", receiptID))
	case !canaryEnvelopesMatch(evidence.Receipt.Envelope, env):
		evidenceReason = Qwen38MTPEvidenceMismatch
		rejectionReasons = append(rejectionReasons, fmt.Sprintf("canary evidence receipt %q does not match the requested envelope", receiptID))
	}

	// Bounded draft depth (1 <= K <= 4)
	if env.DraftDepth < 1 || env.DraftDepth > 4 {
		rejectionReasons = append(rejectionReasons, fmt.Sprintf("draft depth %d outside [1, 4]", env.DraftDepth))
	}

	if !req.ModelReady {
		rejectionReasons = append(rejectionReasons, "model tensors or runtime unready")
	}

	// Inside certified envelope?
	if len(rejectionReasons) == 0 {
		m.canaryDefaultCount++
		return Qwen38CanaryDecision{
			CanaryDefaultOn: true,
			Engine:          Qwen38EngineMTP,
			DowngradeReason: Qwen38MTPEligible,
			InsideEnvelope:  true,
			CircuitTripped:  false,
			Envelope:        env,
		}
	}

	rejectionText := strings.Join(rejectionReasons, "; ")

	// Outside envelope: check if operator opted in
	if req.OperatorOptIn && req.ModelReady {
		m.optInCount++
		return Qwen38CanaryDecision{
			CanaryDefaultOn: false,
			OptInActive:     true,
			Engine:          Qwen38EngineMTP,
			DowngradeReason: Qwen38MTPEligible,
			InsideEnvelope:  false,
			CircuitTripped:  false,
			RejectionReason: rejectionText,
			Envelope:        env,
		}
	}

	m.targetOnlyCount++
	downgradeReason := Qwen38MTPQualityOutsideEnvelope
	if evidenceReason != Qwen38MTPEligible {
		downgradeReason = evidenceReason
	}
	return Qwen38CanaryDecision{
		CanaryDefaultOn: false,
		Engine:          Qwen38EngineTargetDecode,
		DowngradeReason: downgradeReason,
		InsideEnvelope:  false,
		CircuitTripped:  false,
		RejectionReason: rejectionText,
		Envelope:        env,
	}
}

// Evidence-specific downgrade reasons preserve the distinction between absent,
// wrong, and expired authority while keeping every fallback on fak-native target
// decode. They intentionally do not imply a different execution engine.
const (
	Qwen38MTPEvidenceMissing  Qwen38MTPDowngradeReason = "canary_evidence_missing"
	Qwen38MTPEvidenceMismatch Qwen38MTPDowngradeReason = "canary_evidence_mismatch"
	Qwen38MTPEvidenceStale    Qwen38MTPDowngradeReason = "canary_evidence_stale"
)

func normalizeCanaryArtifactHash(hash string) (string, bool) {
	clean := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(hash)), "sha256:")
	decoded, err := hex.DecodeString(clean)
	return clean, err == nil && len(decoded) == sha256.Size
}

func canaryEnvelopesMatch(witnessed, requested Qwen38CanaryEnvelope) bool {
	knownHash, hashOK := normalizeCanaryArtifactHash(witnessed.ArtifactHash)
	requestedHash, requestedOK := normalizeCanaryArtifactHash(requested.ArtifactHash)
	return hashOK && requestedOK && knownHash == requestedHash &&
		strings.EqualFold(strings.TrimSpace(witnessed.ModelFamily), strings.TrimSpace(requested.ModelFamily)) &&
		witnessed.Format == requested.Format && witnessed.Backend == requested.Backend &&
		witnessed.DraftDepth == requested.DraftDepth && requested.HeadroomBytes >= witnessed.HeadroomBytes
}

// EmitReceipt constructs and validates a Qwen38MTPCanaryReceipt witnessing the run.
func (m *Qwen38MTPCanaryManager) EmitReceipt(
	receiptID string,
	dec Qwen38CanaryDecision,
	speedup float64,
	tokensProposed, tokensAccepted int,
	lat Qwen38MTPLatencyNS,
	mem Qwen38MTPMemoryBytes,
) (*Qwen38MTPCanaryReceipt, error) {
	m.mu.RLock()
	cStatus := m.circuitStatus
	divCount := m.divergenceCount
	errCount := m.errorCount
	m.mu.RUnlock()

	receipt := &Qwen38MTPCanaryReceipt{
		SchemaVersion:   Qwen38MTPCanaryReceiptSchema,
		ReceiptID:       receiptID,
		DefaultOn:       dec.CanaryDefaultOn,
		Engine:          dec.Engine,
		Envelope:        dec.Envelope,
		DivergenceCount: divCount,
		ErrorCount:      errCount,
		Speedup:         speedup,
		TokensProduced:  tokensAccepted + 1,
		TokensAccepted:  tokensAccepted,
		TokensProposed:  tokensProposed,
		DowngradeReason: dec.DowngradeReason,
		CircuitStatus:   cStatus,
		LatencyNS:       lat,
		MemoryBytes:     mem,
	}

	if err := receipt.Validate(); err != nil {
		return nil, fmt.Errorf("model: canary receipt validation failed: %w", err)
	}

	return receipt, nil
}
