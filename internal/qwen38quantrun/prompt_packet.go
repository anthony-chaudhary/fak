package qwen38quantrun

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
)

const (
	PromptTokenPacketSchema = "fak.qwen38.prompt-token-packet.v1"
)

// ContextBudget defines the token and byte capacity constraints for a trial.
type ContextBudget struct {
	ContextTokens      int    `json:"context_tokens"`
	ContextBudgetBytes uint64 `json:"context_budget_bytes,omitempty"`
}

// GenerationControls bounds decoding parameters to ensure deterministic comparison.
type GenerationControls struct {
	Temperature     float64  `json:"temperature"`
	TopP            float64  `json:"top_p"`
	TopK            int      `json:"top_k,omitempty"`
	MaxOutputTokens int      `json:"max_output_tokens,omitempty"`
	StopTokens      []string `json:"stop_tokens,omitempty"`
	StopTokenIDs    []int    `json:"stop_token_ids,omitempty"`
}

// PromptTokenPacket freezes the exact prompt tokenization and generation envelope
// across candidate and comparator arms for AMD product-path trials.
type PromptTokenPacket struct {
	Schema             string             `json:"schema"`
	PacketID           string             `json:"packet_id,omitempty"`
	ArtifactSHA256     string             `json:"artifact_sha256"`
	TokenizerIdentity  string             `json:"tokenizer_identity,omitempty"`
	TokenizerDigest    string             `json:"tokenizer_digest"`
	PromptTokenIDs     []int              `json:"prompt_token_ids"`
	StopTokens         []string           `json:"stop_tokens,omitempty"`
	StopTokenIDs       []int              `json:"stop_token_ids,omitempty"`
	ContextBudget      ContextBudget      `json:"context_budget"`
	GenerationControls GenerationControls `json:"generation_controls"`
	PacketDigest       string             `json:"packet_digest,omitempty"`
}

func normalizePacketStopTokens(p *PromptTokenPacket) {
	if len(p.StopTokens) == 0 && len(p.GenerationControls.StopTokens) > 0 {
		p.StopTokens = append([]string(nil), p.GenerationControls.StopTokens...)
	} else if len(p.GenerationControls.StopTokens) == 0 && len(p.StopTokens) > 0 {
		p.GenerationControls.StopTokens = append([]string(nil), p.StopTokens...)
	}
	if len(p.StopTokenIDs) == 0 && len(p.GenerationControls.StopTokenIDs) > 0 {
		p.StopTokenIDs = append([]int(nil), p.GenerationControls.StopTokenIDs...)
	} else if len(p.GenerationControls.StopTokenIDs) == 0 && len(p.StopTokenIDs) > 0 {
		p.GenerationControls.StopTokenIDs = append([]int(nil), p.StopTokenIDs...)
	}
}

func validatePromptPacketFields(p PromptTokenPacket) error {
	if p.Schema != PromptTokenPacketSchema {
		return fmt.Errorf("prompt packet schema %q must be %q", p.Schema, PromptTokenPacketSchema)
	}
	if !validOracleSHA256(p.ArtifactSHA256) {
		return fmt.Errorf("prompt packet artifact_sha256 %q is not a valid 64-char hex SHA-256", p.ArtifactSHA256)
	}
	if p.TokenizerDigest == "" {
		return errors.New("prompt packet tokenizer_digest is required")
	}
	if len(p.PromptTokenIDs) == 0 {
		return errors.New("prompt packet prompt_token_ids must not be empty")
	}
	for i, id := range p.PromptTokenIDs {
		if id < 0 {
			return fmt.Errorf("prompt_token_ids contains negative token ID %d at index %d", id, i)
		}
	}
	if p.ContextBudget.ContextTokens <= 0 {
		return errors.New("context_budget.context_tokens must be positive")
	}
	if len(p.PromptTokenIDs) > p.ContextBudget.ContextTokens {
		return fmt.Errorf("prompt_token_ids length %d exceeds context budget %d", len(p.PromptTokenIDs), p.ContextBudget.ContextTokens)
	}
	if math.IsNaN(p.GenerationControls.Temperature) || math.IsInf(p.GenerationControls.Temperature, 0) || p.GenerationControls.Temperature < 0 {
		return errors.New("generation_controls.temperature must be finite and non-negative")
	}
	if math.IsNaN(p.GenerationControls.TopP) || math.IsInf(p.GenerationControls.TopP, 0) || p.GenerationControls.TopP < 0 || p.GenerationControls.TopP > 1.0 {
		return errors.New("generation_controls.top_p must be between 0.0 and 1.0")
	}
	return nil
}

// ComputePromptPacketDigest computes the canonical SHA-256 digest of a prompt packet.
func ComputePromptPacketDigest(p PromptTokenPacket) (string, error) {
	clone := p
	clone.PacketDigest = ""
	normalizePacketStopTokens(&clone)
	raw, err := canonicalJSON(clone)
	if err != nil {
		return "", fmt.Errorf("canonical JSON: %w", err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

// FreezePromptPacket validates packet fields, computes its canonical digest, and returns the sealed packet.
func FreezePromptPacket(p PromptTokenPacket) (PromptTokenPacket, error) {
	if err := validatePromptPacketFields(p); err != nil {
		return PromptTokenPacket{}, err
	}
	digest, err := ComputePromptPacketDigest(p)
	if err != nil {
		return PromptTokenPacket{}, err
	}
	sealed := p
	normalizePacketStopTokens(&sealed)
	sealed.PacketDigest = digest
	return sealed, nil
}

// VerifyPromptPacket validates fields and verifies that the packet digest matches its contents without tampering.
func VerifyPromptPacket(p PromptTokenPacket) error {
	if err := validatePromptPacketFields(p); err != nil {
		return err
	}
	if p.PacketDigest == "" {
		return errors.New("prompt packet is not frozen: missing packet_digest")
	}
	expected, err := ComputePromptPacketDigest(p)
	if err != nil {
		return err
	}
	if p.PacketDigest != expected {
		return fmt.Errorf("prompt packet digest mismatch: tampered or corrupted packet (got %s want %s)", p.PacketDigest, expected)
	}
	return nil
}

// ExportPromptPacket exports a verified, frozen prompt packet into indented JSON bytes.
func ExportPromptPacket(p PromptTokenPacket) ([]byte, error) {
	if err := VerifyPromptPacket(p); err != nil {
		return nil, fmt.Errorf("cannot export invalid or unfrozen packet: %w", err)
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

// ImportPromptPacket strictly deserializes and verifies a frozen prompt packet.
func ImportPromptPacket(data []byte) (PromptTokenPacket, error) {
	var p PromptTokenPacket
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&p); err != nil {
		return PromptTokenPacket{}, fmt.Errorf("decode prompt packet: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return PromptTokenPacket{}, errors.New("multiple JSON values in prompt packet")
		}
		return PromptTokenPacket{}, err
	}
	if err := VerifyPromptPacket(p); err != nil {
		return PromptTokenPacket{}, err
	}
	return p, nil
}

// WritePromptPacketFile writes an exported prompt packet to disk.
func WritePromptPacketFile(path string, p PromptTokenPacket) error {
	data, err := ExportPromptPacket(p)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// ReadPromptPacketFile reads and strictly verifies a frozen prompt packet from disk.
func ReadPromptPacketFile(path string) (PromptTokenPacket, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return PromptTokenPacket{}, fmt.Errorf("read prompt packet file: %w", err)
	}
	return ImportPromptPacket(raw)
}

// ValidatePromptPacketAttestation asserts that candidate and reference prompt packets are both valid and attest the identical sequence and envelope.
func ValidatePromptPacketAttestation(candidate, reference PromptTokenPacket) error {
	if err := VerifyPromptPacket(candidate); err != nil {
		return fmt.Errorf("candidate prompt packet invalid: %w", err)
	}
	if err := VerifyPromptPacket(reference); err != nil {
		return fmt.Errorf("reference prompt packet invalid: %w", err)
	}
	if candidate.ArtifactSHA256 != reference.ArtifactSHA256 {
		return fmt.Errorf("artifact SHA-256 mismatch: candidate=%s reference=%s", candidate.ArtifactSHA256, reference.ArtifactSHA256)
	}
	if candidate.TokenizerDigest != reference.TokenizerDigest {
		return fmt.Errorf("tokenizer digest mismatch: candidate=%s reference=%s", candidate.TokenizerDigest, reference.TokenizerDigest)
	}
	if !slices.Equal(candidate.PromptTokenIDs, reference.PromptTokenIDs) {
		return errors.New("prompt token IDs mismatch between candidate and reference")
	}
	if !slices.Equal(candidate.StopTokens, reference.StopTokens) || !slices.Equal(candidate.StopTokenIDs, reference.StopTokenIDs) {
		return errors.New("stop tokens mismatch between candidate and reference")
	}
	if candidate.ContextBudget.ContextTokens != reference.ContextBudget.ContextTokens ||
		candidate.ContextBudget.ContextBudgetBytes != reference.ContextBudget.ContextBudgetBytes {
		return errors.New("context budget mismatch between candidate and reference")
	}
	if candidate.GenerationControls.Temperature != reference.GenerationControls.Temperature ||
		candidate.GenerationControls.TopP != reference.GenerationControls.TopP ||
		candidate.GenerationControls.TopK != reference.GenerationControls.TopK ||
		candidate.GenerationControls.MaxOutputTokens != reference.GenerationControls.MaxOutputTokens {
		return errors.New("generation controls mismatch between candidate and reference")
	}
	if candidate.PacketDigest != reference.PacketDigest {
		return fmt.Errorf("packet digest mismatch: candidate=%s reference=%s", candidate.PacketDigest, reference.PacketDigest)
	}
	return nil
}

// ValidateArmPromptPacketAttestation verifies that candidate and reference receipts attest the same prompt sequence and envelope.
func ValidateArmPromptPacketAttestation(candidate, reference AMDArmReceipt) error {
	if candidate.Engine != "fak-native" || candidate.ComparatorOnly || candidate.FallbackActive {
		return errors.New("candidate arm must be fak-native with no fallback and not comparator-only")
	}
	if candidate.ArtifactSHA256 != reference.ArtifactSHA256 {
		return fmt.Errorf("artifact SHA-256 mismatch: candidate=%s reference=%s", candidate.ArtifactSHA256, reference.ArtifactSHA256)
	}
	if candidate.TokenizerDigest != "" && reference.TokenizerDigest != "" && candidate.TokenizerDigest != reference.TokenizerDigest {
		return fmt.Errorf("tokenizer digest mismatch: candidate=%s reference=%s", candidate.TokenizerDigest, reference.TokenizerDigest)
	}
	if !slices.Equal(candidate.PromptTokenIDs, reference.PromptTokenIDs) {
		return errors.New("prompt token IDs mismatch between arms")
	}
	if candidate.PromptPacketDigest != "" && reference.PromptPacketDigest != "" && candidate.PromptPacketDigest != reference.PromptPacketDigest {
		return fmt.Errorf("prompt packet digest mismatch: candidate=%s reference=%s", candidate.PromptPacketDigest, reference.PromptPacketDigest)
	}
	if candidate.ContextTokens != reference.ContextTokens || candidate.ContextBudgetBytes != reference.ContextBudgetBytes {
		return errors.New("context budget mismatch between arms")
	}
	if candidate.Temperature != reference.Temperature || candidate.TopP != reference.TopP {
		return errors.New("generation envelope mismatch between arms")
	}
	if candidate.PromptPacket != nil && reference.PromptPacket != nil {
		if err := ValidatePromptPacketAttestation(*candidate.PromptPacket, *reference.PromptPacket); err != nil {
			return fmt.Errorf("prompt packet attestation failed: %w", err)
		}
	} else if candidate.PromptPacket != nil || reference.PromptPacket != nil {
		return errors.New("one arm carries prompt_packet but the other does not")
	}
	return nil
}
