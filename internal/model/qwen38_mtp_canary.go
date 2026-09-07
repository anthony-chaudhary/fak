package model

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"
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
	Envelope      Qwen38CanaryEnvelope `json:"envelope"`
	OperatorOptIn bool                 `json:"operator_opt_in"`
	ModelReady    bool                 `json:"model_ready"`
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
	mu                   sync.RWMutex
	provenArtifactHashes map[string]bool
	circuitStatus        CanaryCircuitStatus
	circuitTripReason    string
	totalEvaluations     int64
	canaryDefaultCount   int64
	optInCount           int64
	targetOnlyCount      int64
	divergenceCount      int64
	errorCount           int64
}

// NewQwen38MTPCanaryManager initializes a canary manager with default proven hashes and closed circuit.
func NewQwen38MTPCanaryManager() *Qwen38MTPCanaryManager {
	m := &Qwen38MTPCanaryManager{
		provenArtifactHashes: make(map[string]bool),
		circuitStatus:        CanaryCircuitClosed,
	}
	// Seed canonical proven Qwen3.8 artifact baseline hash
	h := sha256.Sum256([]byte("qwen3.8-mtp-certified-v1"))
	m.provenArtifactHashes[hex.EncodeToString(h[:])] = true
	return m
}

// RegisterProvenArtifact adds a verified artifact SHA256 hash to the certified set.
func (m *Qwen38MTPCanaryManager) RegisterProvenArtifact(hash string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.provenArtifactHashes[strings.ToLower(strings.TrimSpace(hash))] = true
}

// IsArtifactProven checks whether an artifact hash is in the proven set.
func (m *Qwen38MTPCanaryManager) IsArtifactProven(hash string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.provenArtifactHashes[strings.ToLower(strings.TrimSpace(hash))]
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

	// Artifact hash check
	cleanHash := strings.ToLower(strings.TrimSpace(env.ArtifactHash))
	if cleanHash == "" || !m.provenArtifactHashes[cleanHash] {
		rejectionReasons = append(rejectionReasons, fmt.Sprintf("artifact hash %q is unproven", env.ArtifactHash))
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
	return Qwen38CanaryDecision{
		CanaryDefaultOn: false,
		Engine:          Qwen38EngineTargetDecode,
		DowngradeReason: Qwen38MTPQualityOutsideEnvelope,
		InsideEnvelope:  false,
		CircuitTripped:  false,
		RejectionReason: rejectionText,
		Envelope:        env,
	}
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
