package orchestration

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const ChildContextReceiptSchema = "fak.child-context.receipt.v1"

// ChildContextDeclaration binds the selected child declaration to both the
// child node and the immutable parent plan that declared it.
type ChildContextDeclaration struct {
	ChildNodeID      string `json:"child_node_id"`
	ParentPlanDigest string `json:"parent_plan_digest"`
	Digest           string `json:"digest"`
}

// ChildContextBudget is the capacity reserved before launching the child.
type ChildContextBudget struct {
	Workers int   `json:"workers"`
	Tokens  int64 `json:"tokens"`
}

// ChildContextReceipt is the immutable, parent-derived semantic context for
// one child launch. It intentionally carries no process or liveness fields.
type ChildContextReceipt struct {
	Schema                              string                  `json:"schema"`
	ParentRunID                         string                  `json:"parent_run_id"`
	ParentSessionID                     string                  `json:"parent_session_id"`
	ChildNodeID                         string                  `json:"child_node_id"`
	ParentPlanDigest                    string                  `json:"parent_plan_digest"`
	ChildDeclaration                    ChildContextDeclaration `json:"child_declaration"`
	AccessDeclarationDigest             string                  `json:"access_declaration_digest"`
	CapabilityEnvelopeDigest            string                  `json:"capability_envelope_digest"`
	ReservedBudget                      ChildContextBudget      `json:"reserved_budget"`
	WorkspaceStateEpoch                 string                  `json:"workspace_state_epoch"`
	InputArtifactRefs                   []string                `json:"input_artifact_refs"`
	ExpectedOutputWitnessContractDigest string                  `json:"expected_output_witness_contract_digest"`
}

var ErrInvalidChildContextReceipt = errors.New("invalid child context receipt")

// Validate enforces the canonical receipt contract without consulting storage
// or importing the implementations that own the referenced identities.
func (r ChildContextReceipt) Validate() error {
	if r.Schema != ChildContextReceiptSchema {
		return fmt.Errorf("%w: unsupported schema", ErrInvalidChildContextReceipt)
	}
	for name, value := range map[string]string{
		"parent run id":         r.ParentRunID,
		"parent session id":     r.ParentSessionID,
		"child node id":         r.ChildNodeID,
		"workspace state epoch": r.WorkspaceStateEpoch,
	} {
		if !canonicalIdentifier(value) {
			return fmt.Errorf("%w: %s is required and must be canonical", ErrInvalidChildContextReceipt, name)
		}
	}
	for name, value := range map[string]string{
		"parent plan digest":                      r.ParentPlanDigest,
		"child declaration digest":                r.ChildDeclaration.Digest,
		"access declaration digest":               r.AccessDeclarationDigest,
		"capability envelope digest":              r.CapabilityEnvelopeDigest,
		"expected output/witness contract digest": r.ExpectedOutputWitnessContractDigest,
	} {
		if !canonicalSHA256(value) {
			return fmt.Errorf("%w: %s must be a lowercase SHA-256 digest", ErrInvalidChildContextReceipt, name)
		}
	}
	if r.ChildDeclaration.ChildNodeID != r.ChildNodeID || r.ChildDeclaration.ParentPlanDigest != r.ParentPlanDigest {
		return fmt.Errorf("%w: child declaration is not bound to the child and parent plan", ErrInvalidChildContextReceipt)
	}
	if r.ReservedBudget.Workers <= 0 || r.ReservedBudget.Tokens <= 0 {
		return fmt.Errorf("%w: reserved worker and token budgets must be positive", ErrInvalidChildContextReceipt)
	}
	if r.InputArtifactRefs == nil {
		return fmt.Errorf("%w: input artifact refs must be an array", ErrInvalidChildContextReceipt)
	}
	previous := ""
	for _, ref := range r.InputArtifactRefs {
		if !canonicalIdentifier(ref) || (previous != "" && ref <= previous) {
			return fmt.Errorf("%w: input artifact refs must be canonical, unique, and sorted", ErrInvalidChildContextReceipt)
		}
		previous = ref
	}
	return nil
}

// CanonicalJSON returns the single supported JSON representation.
func (r ChildContextReceipt) CanonicalJSON() ([]byte, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(r)
}

// Digest returns the lowercase SHA-256 digest of CanonicalJSON.
func (r ChildContextReceipt) Digest() (string, error) {
	encoded, err := r.CanonicalJSON()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

// DecodeChildContextReceiptJSON accepts exactly one canonical receipt. Unknown
// fields, trailing values, alternate field ordering, and insignificant
// whitespace are rejected rather than assigned a second identity.
func DecodeChildContextReceiptJSON(data []byte) (ChildContextReceipt, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var receipt ChildContextReceipt
	if err := decoder.Decode(&receipt); err != nil {
		return ChildContextReceipt{}, ErrInvalidChildContextReceipt
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return ChildContextReceipt{}, ErrInvalidChildContextReceipt
	}
	canonical, err := receipt.CanonicalJSON()
	if err != nil || !bytes.Equal(data, canonical) {
		return ChildContextReceipt{}, ErrInvalidChildContextReceipt
	}
	return receipt, nil
}

func canonicalIdentifier(value string) bool {
	return value != "" && strings.TrimSpace(value) == value
}

func canonicalSHA256(value string) bool {
	if len(value) != sha256.Size*2 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
