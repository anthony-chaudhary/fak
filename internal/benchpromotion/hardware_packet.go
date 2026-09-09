package benchpromotion

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// Candidate disposition constants for multi-fidelity promotion.
const (
	DispositionHardware       = "HARDWARE"
	DispositionAbstain        = "ABSTAIN"
	DispositionPruned         = "PRUNED"
	DispositionSimulationOnly = "SIMULATION_ONLY"
)

const (
	// SchemaVersion is the current schema version for hardware-ready packets.
	SchemaVersion = "fak-hardware-packet/1"

	// HardwarePacketSchemaVersion is an alias for SchemaVersion.
	HardwarePacketSchemaVersion = SchemaVersion

	// EngineFakNative is the required engine for native hardware execution.
	EngineFakNative = "fak-native"
)

// HardwarePacketRequest defines the input parameters for creating a HardwareReadyPacket.
type HardwarePacketRequest struct {
	Disposition           string
	CandidateDigest       string
	EvidenceDigest        string
	HardwareClass         string
	Engine                string // MUST be "fak-native"
	Model                 string
	Workload              string
	QualityGate           string
	ObjectiveSLO          string
	AuthorizedBudget      string
	LeaseLane             string
	CaptureCommand        string
	ExpectedReceiptSchema string
}

// HardwareReadyPacket represents an immutable, digest-bound packet for moving
// a surviving simulated candidate onto sanctioned hardware.
type HardwareReadyPacket struct {
	SchemaVersion         string `json:"schema_version"`
	CandidateDigest       string `json:"candidate_digest"`
	EvidenceDigest        string `json:"evidence_digest"`
	HardwareClass         string `json:"hardware_class"`
	Engine                string `json:"engine"`
	Model                 string `json:"model"`
	Workload              string `json:"workload"`
	QualityGate           string `json:"quality_gate"`
	ObjectiveSLO          string `json:"objective_slo"`
	AuthorizedBudget      string `json:"authorized_budget"`
	LeaseLane             string `json:"lease_lane"`
	CaptureCommand        string `json:"capture_command"`
	ExpectedReceiptSchema string `json:"expected_receipt_schema"`
	PacketDigest          string `json:"packet_digest"` // sha256 of canonical JSON excluding packet_digest
}

// packetDigestPayload represents the fields serialized to compute PacketDigest.
type packetDigestPayload struct {
	SchemaVersion         string `json:"schema_version"`
	CandidateDigest       string `json:"candidate_digest"`
	EvidenceDigest        string `json:"evidence_digest"`
	HardwareClass         string `json:"hardware_class"`
	Engine                string `json:"engine"`
	Model                 string `json:"model"`
	Workload              string `json:"workload"`
	QualityGate           string `json:"quality_gate"`
	ObjectiveSLO          string `json:"objective_slo"`
	AuthorizedBudget      string `json:"authorized_budget"`
	LeaseLane             string `json:"lease_lane"`
	CaptureCommand        string `json:"capture_command"`
	ExpectedReceiptSchema string `json:"expected_receipt_schema"`
}

// CanonicalJSON returns the deterministic JSON bytes of the packet excluding PacketDigest.
func (p *HardwareReadyPacket) CanonicalJSON() ([]byte, error) {
	if p == nil {
		return nil, fmt.Errorf("benchpromotion: hardware packet is nil")
	}
	payload := packetDigestPayload{
		SchemaVersion:         p.SchemaVersion,
		CandidateDigest:       p.CandidateDigest,
		EvidenceDigest:        p.EvidenceDigest,
		HardwareClass:         p.HardwareClass,
		Engine:                p.Engine,
		Model:                 p.Model,
		Workload:              p.Workload,
		QualityGate:           p.QualityGate,
		ObjectiveSLO:          p.ObjectiveSLO,
		AuthorizedBudget:      p.AuthorizedBudget,
		LeaseLane:             p.LeaseLane,
		CaptureCommand:        p.CaptureCommand,
		ExpectedReceiptSchema: p.ExpectedReceiptSchema,
	}
	return json.Marshal(payload)
}

func computePacketDigest(p *HardwareReadyPacket) (string, error) {
	data, err := p.CanonicalJSON()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// EmitHardwarePacket validates the request, constructs a HardwareReadyPacket with SchemaVersion
// "fak-hardware-packet/1", and computes its canonical PacketDigest.
func EmitHardwarePacket(req HardwarePacketRequest) (*HardwareReadyPacket, error) {
	if req.Disposition != DispositionHardware {
		return nil, fmt.Errorf("benchpromotion: runnable packet refused for non-hardware disposition %q", req.Disposition)
	}
	if req.Engine != EngineFakNative {
		return nil, fmt.Errorf("benchpromotion: engine must be fak-native, got %q", req.Engine)
	}

	if strings.TrimSpace(req.CandidateDigest) == "" {
		return nil, fmt.Errorf("benchpromotion: missing required field %q", "candidate_digest")
	}
	if strings.TrimSpace(req.EvidenceDigest) == "" {
		return nil, fmt.Errorf("benchpromotion: missing required field %q", "evidence_digest")
	}
	if strings.TrimSpace(req.HardwareClass) == "" {
		return nil, fmt.Errorf("benchpromotion: missing required field %q", "hardware_class")
	}
	if strings.TrimSpace(req.Model) == "" {
		return nil, fmt.Errorf("benchpromotion: missing required field %q", "model")
	}
	if strings.TrimSpace(req.Workload) == "" {
		return nil, fmt.Errorf("benchpromotion: missing required field %q", "workload")
	}
	if strings.TrimSpace(req.CaptureCommand) == "" {
		return nil, fmt.Errorf("benchpromotion: missing required field %q", "capture_command")
	}
	if strings.TrimSpace(req.ExpectedReceiptSchema) == "" {
		return nil, fmt.Errorf("benchpromotion: missing required field %q", "expected_receipt_schema")
	}
	if strings.TrimSpace(req.AuthorizedBudget) == "" {
		return nil, fmt.Errorf("benchpromotion: missing required field %q", "authorized_budget")
	}
	if strings.TrimSpace(req.LeaseLane) == "" {
		return nil, fmt.Errorf("benchpromotion: missing required field %q", "lease_lane")
	}

	packet := &HardwareReadyPacket{
		SchemaVersion:         SchemaVersion,
		CandidateDigest:       req.CandidateDigest,
		EvidenceDigest:        req.EvidenceDigest,
		HardwareClass:         req.HardwareClass,
		Engine:                req.Engine,
		Model:                 req.Model,
		Workload:              req.Workload,
		QualityGate:           req.QualityGate,
		ObjectiveSLO:          req.ObjectiveSLO,
		AuthorizedBudget:      req.AuthorizedBudget,
		LeaseLane:             req.LeaseLane,
		CaptureCommand:        req.CaptureCommand,
		ExpectedReceiptSchema: req.ExpectedReceiptSchema,
	}

	digest, err := computePacketDigest(packet)
	if err != nil {
		return nil, fmt.Errorf("benchpromotion: compute packet digest: %w", err)
	}
	packet.PacketDigest = digest

	return packet, nil
}

// VerifyHardwarePacket checks that the packet is non-nil, satisfies schema and engine invariants,
// contains all required fields, and has a valid matching PacketDigest.
func VerifyHardwarePacket(p *HardwareReadyPacket) error {
	if p == nil {
		return fmt.Errorf("benchpromotion: hardware packet is nil")
	}
	if p.SchemaVersion != SchemaVersion {
		return fmt.Errorf("benchpromotion: invalid schema version %q, expected %q", p.SchemaVersion, SchemaVersion)
	}
	if p.Engine != EngineFakNative {
		return fmt.Errorf("benchpromotion: engine must be fak-native, got %q", p.Engine)
	}
	if strings.TrimSpace(p.CandidateDigest) == "" {
		return fmt.Errorf("benchpromotion: missing required field %q", "candidate_digest")
	}
	if strings.TrimSpace(p.EvidenceDigest) == "" {
		return fmt.Errorf("benchpromotion: missing required field %q", "evidence_digest")
	}
	if strings.TrimSpace(p.HardwareClass) == "" {
		return fmt.Errorf("benchpromotion: missing required field %q", "hardware_class")
	}
	if strings.TrimSpace(p.Model) == "" {
		return fmt.Errorf("benchpromotion: missing required field %q", "model")
	}
	if strings.TrimSpace(p.Workload) == "" {
		return fmt.Errorf("benchpromotion: missing required field %q", "workload")
	}
	if strings.TrimSpace(p.CaptureCommand) == "" {
		return fmt.Errorf("benchpromotion: missing required field %q", "capture_command")
	}
	if strings.TrimSpace(p.ExpectedReceiptSchema) == "" {
		return fmt.Errorf("benchpromotion: missing required field %q", "expected_receipt_schema")
	}
	if strings.TrimSpace(p.AuthorizedBudget) == "" {
		return fmt.Errorf("benchpromotion: missing required field %q", "authorized_budget")
	}
	if strings.TrimSpace(p.LeaseLane) == "" {
		return fmt.Errorf("benchpromotion: missing required field %q", "lease_lane")
	}
	if strings.TrimSpace(p.PacketDigest) == "" {
		return fmt.Errorf("benchpromotion: missing packet digest")
	}

	expectedDigest, err := computePacketDigest(p)
	if err != nil {
		return fmt.Errorf("benchpromotion: compute packet digest: %w", err)
	}
	if p.PacketDigest != expectedDigest {
		return fmt.Errorf("benchpromotion: packet digest mismatch: expected %s, got %s", expectedDigest, p.PacketDigest)
	}

	return nil
}
