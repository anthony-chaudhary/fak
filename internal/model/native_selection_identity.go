package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	// NativeSelectionIdentitySchemaV1 fixes the functional axes covered by the
	// first native kernel-selection identity contract.
	NativeSelectionIdentitySchemaV1 = "fak.kernel-selection/v1"

	NativeSelectionQuantizationF32  = "F32"
	NativeSelectionQuantizationQ8_0 = "Q8_0"
	NativeSelectionQuantizationQ4K  = "Q4_K"
)

// NativeSelectionIdentity is the decoded, versioned functional identity of one
// resolved native execution path. ModelRef is the planner-visible model
// reference; it is deliberately not a content-derived artifact digest.
type NativeSelectionIdentity struct {
	Schema              string `json:"schema"`
	ModelRef            string `json:"model_ref"`
	Backend             string `json:"backend"`
	ForwardPath         string `json:"forward_path"`
	Quantization        string `json:"quantization"`
	PrefillChunkTokens  int    `json:"prefill_chunk_tokens"`
	CPUOffloadExperts   int    `json:"cpu_offload_experts"`
	Q4KGateUpOutputSlab bool   `json:"q4k_gate_up_output_slab"`
}

// CanonicalJSON returns the stable v1 encoding used by Digest. The fixed struct
// is intentional: maps and reflection-discovered fields would make identity
// ordering and schema membership implicit.
func (id NativeSelectionIdentity) CanonicalJSON() ([]byte, error) {
	if err := id.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(id)
}

// Digest returns the canonical lowercase SHA-256 identity.
func (id NativeSelectionIdentity) Digest() (string, error) {
	canonical, err := id.CanonicalJSON()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canonical)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// Validate rejects incomplete or contradictory identities instead of issuing a
// stable digest for an execution envelope the planner could not have selected.
func (id NativeSelectionIdentity) Validate() error {
	if id.Schema != NativeSelectionIdentitySchemaV1 {
		return fmt.Errorf("kernel selection identity schema %q: want %q", id.Schema, NativeSelectionIdentitySchemaV1)
	}
	if strings.TrimSpace(id.ModelRef) == "" {
		return fmt.Errorf("kernel selection identity model_ref is empty")
	}
	if strings.TrimSpace(id.Backend) == "" {
		return fmt.Errorf("kernel selection identity backend is empty")
	}
	if strings.TrimSpace(id.ForwardPath) == "" {
		return fmt.Errorf("kernel selection identity forward_path is empty")
	}
	switch id.Quantization {
	case NativeSelectionQuantizationF32, NativeSelectionQuantizationQ8_0, NativeSelectionQuantizationQ4K:
	default:
		return fmt.Errorf("kernel selection identity quantization %q is unknown", id.Quantization)
	}
	if id.PrefillChunkTokens < 0 {
		return fmt.Errorf("kernel selection identity prefill_chunk_tokens %d is negative", id.PrefillChunkTokens)
	}
	if id.CPUOffloadExperts < 0 {
		return fmt.Errorf("kernel selection identity cpu_offload_experts %d is negative", id.CPUOffloadExperts)
	}
	if id.Quantization != NativeSelectionQuantizationQ4K && id.PrefillChunkTokens != 0 {
		return fmt.Errorf("kernel selection identity prefill chunk requires Q4_K quantization")
	}
	if id.Quantization != NativeSelectionQuantizationQ4K && id.Q4KGateUpOutputSlab {
		return fmt.Errorf("kernel selection identity Q4_K gate/up output slab requires Q4_K quantization")
	}
	return nil
}
